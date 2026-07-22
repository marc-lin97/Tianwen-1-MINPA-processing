"""Independent streaming audit for the MINPA all-energy MSE cell archive."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .tw1_all_energy_mse_grid_records import (
    COMMON_RECORD_FIELDS,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_QUALITY_ALGORITHM_VERSION,
    EXPECTED_RECORD_FIELDS,
    FIELD_METADATA,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SPECIES,
    SPECIES_RECORD_FIELDS,
    MseGridSpec,
    NvProduct,
    _cell_metadata,
    _root_schema,
    bin_mse_positions,
    build_compound_records,
    canonical_row_count,
    cell_file_path,
    discover_nv_products,
    load_all_energy_day_chunk,
)
from .constants import DEFAULT_TW1_MOMAG_ROOT, DEFAULT_TW1_R_ROOT


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def expected_field_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for output_name in EXPECTED_RECORD_FIELDS:
        source_name = output_name
        for spec in SPECIES.values():
            prefix = f"{spec['prefix']}_"
            if output_name.startswith(prefix):
                source_name = output_name[len(prefix) :]
                break
        units, description = FIELD_METADATA[source_name]
        metadata[output_name] = {"units": units, "description": description}
        if "mse" in source_name:
            metadata[output_name]["coordinate_system"] = "MSE"
    return metadata


def cleanup_empty_staging_groups(output_root: Path) -> dict[str, int]:
    """Remove empty transaction groups only; never remove incomplete content."""
    files_seen = removed = nonempty = 0
    for path in sorted((Path(output_root) / "cells").rglob("*.h5")):
        files_seen += 1
        with h5py.File(path, "a", libver="latest") as handle:
            if "_staging" not in handle:
                continue
            if len(handle["_staging"]) != 0:
                nonempty += 1
                continue
            del handle["_staging"]
            removed += 1
    return {
        "files_seen": files_seen,
        "empty_staging_groups_removed": removed,
        "nonempty_staging_groups": nonempty,
    }


def _equal_records(expected: np.void, actual: np.void) -> list[str]:
    mismatches: list[str] = []
    for name in EXPECTED_RECORD_FIELDS:
        left = expected[name]
        right = actual[name]
        if np.issubdtype(np.asarray(left).dtype, np.floating):
            if not bool(np.isclose(left, right, rtol=0.0, atol=1.0e-10, equal_nan=True)):
                mismatches.append(name)
        elif left != right:
            mismatches.append(name)
    return mismatches


def audit_tw1_all_energy_mse_grid_records(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    momag_root: Path = DEFAULT_TW1_MOMAG_ROOT,
    r_root: Path = DEFAULT_TW1_R_ROOT,
    *,
    dates: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    expected_days: int | None = None,
    expected_excluded_days: int | None = None,
    expected_quality_algorithm_version: str | None = EXPECTED_QUALITY_ALGORITHM_VERSION,
    grid: MseGridSpec | None = None,
    position_tolerance_s: float = 5.0,
    rotation_gap_s: float = 8.0,
    rotation_atol: float = 1.0e-6,
    source_samples_per_day: int = 3,
    require_no_staging: bool = True,
    max_reported_errors: int = 100,
    write_summary: bool = True,
) -> dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    momag_root = Path(momag_root)
    r_root = Path(r_root)
    grid = grid or MseGridSpec()
    errors: list[str] = []
    error_count = 0

    def fail(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < max_reported_errors:
            errors.append(message)

    products = discover_nv_products(
        input_root, dates=dates, start_date=start_date, end_date=end_date
    )
    processable: list[NvProduct] = []
    excluded_products: list[NvProduct] = []
    for product in products:
        momag = momag_root / f"Bss{product.date}.mat"
        rotation = r_root / f"R_MSO2MSE_Tianwen-1_{product.date}.mat"
        (processable if momag.exists() and rotation.exists() else excluded_products).append(product)
    processable_dates = {product.date for product in processable}
    if expected_days is not None and len(processable) != expected_days:
        fail(f"processable source day count {len(processable)} != expected {expected_days}")
    if expected_excluded_days is not None and len(excluded_products) != expected_excluded_days:
        fail(f"excluded source day count {len(excluded_products)} != expected {expected_excluded_days}")

    coverage_path = output_root / "logs" / "geometry_coverage.json"
    coverage: dict[str, Any] = {}
    if not coverage_path.exists():
        fail(f"missing geometry coverage report: {coverage_path}")
    else:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        if int(coverage.get("excluded_geometry_days", -1)) != len(excluded_products):
            fail("geometry coverage excluded-day count differs from source discovery")
        reported_dates = {item.get("date") for item in coverage.get("excluded", [])}
        if reported_dates != {product.date for product in excluded_products}:
            fail("geometry coverage excluded-date list differs from source discovery")
        excluded_rows = sum(canonical_row_count(product.path, product.date)[0] for product in excluded_products)
        if int(coverage.get("excluded_canonical_utc_rows", -1)) != excluded_rows:
            fail("geometry coverage excluded canonical-row count differs from source inputs")

    schema_path = output_root / "grid_schema.json"
    required_schema = _root_schema(grid, rotation_gap_s, position_tolerance_s)
    if not schema_path.exists():
        fail(f"missing grid schema: {schema_path}")
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        for key, expected in required_schema.items():
            if schema.get(key) != expected:
                fail(f"grid schema mismatch for {key}")

    day_summary_dir = output_root / "logs" / "days"
    summary_paths = sorted(day_summary_dir.glob("*.json"))
    summary_dates = {path.stem for path in summary_paths}
    missing_summaries = sorted(processable_dates - summary_dates)
    extra_summaries = sorted(summary_dates - processable_dates)
    if missing_summaries:
        fail(f"missing day summaries count={len(missing_summaries)} first={missing_summaries[:10]}")
    if extra_summaries:
        fail(f"extra day summaries count={len(extra_summaries)} first={extra_summaries[:10]}")

    summaries: dict[str, dict[str, Any]] = {}
    expected_rows_by_date: dict[str, int] = {}
    expected_groups_by_date: dict[str, int] = {}
    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        date = path.stem
        summaries[date] = summary
        if summary.get("date") != date or summary.get("status") != "complete":
            fail(f"invalid day summary identity/status: {path}")
        expected_rows_by_date[date] = int(summary.get("rows_finite_position_in_grid", -1))
        expected_groups_by_date[date] = int(summary.get("nonempty_cells", -1))
        if len(summary.get("cell_files", [])) != expected_groups_by_date[date]:
            fail(f"{date} cell_files count differs from nonempty_cells")
        for attr in ("source_nv_mso2_file", "source_momag_file", "rotation_file"):
            source = Path(str(summary.get(attr, "")))
            if not source.exists():
                fail(f"{date} missing provenance path {attr}={source}")

    latest_path = output_root / "logs" / "latest_run_summary.json"
    latest: dict[str, Any] = {}
    if not latest_path.exists():
        fail(f"missing final run summary: {latest_path}")
    else:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        complete = int(latest.get("processed_days", -1)) + int(latest.get("already_complete_days", -1))
        if int(latest.get("requested_days", -1)) != len(processable) or complete != len(processable):
            fail("final run summary complete/requested counts differ from processable inputs")

    metadata_expected = expected_field_metadata()
    row_counts: defaultdict[str, int] = defaultdict(int)
    group_counts: defaultdict[str, int] = defaultdict(int)
    total_rows = total_groups = 0
    staging_groups = nonempty_staging_groups = 0
    max_rm_error = 0.0
    max_orth_error = 0.0
    max_det_error = 0.0
    algorithm_versions: set[str] = set()
    science_counts = {
        spec["prefix"]: {
            "positive_density": 0,
            "finite_velocity": 0,
            "status_1": 0,
            "status_4": 0,
            "status_5": 0,
            "quality_available": 0,
            "quality_flagged": 0,
        }
        for spec in SPECIES.values()
    }

    cell_paths = sorted((output_root / "cells").rglob("*.h5"))
    for path in cell_paths:
        try:
            with h5py.File(path, "r") as handle:
                flat_bin = int(handle.attrs.get("flat_bin", -1))
                if flat_bin < 0 or flat_bin >= int(np.prod(grid.shape)):
                    fail(f"invalid flat_bin in {path}")
                    continue
                if path.resolve() != cell_file_path(output_root, flat_bin, grid).resolve():
                    fail(f"cell path disagrees with flat_bin: {path}")
                for key, expected in _cell_metadata(flat_bin, grid).items():
                    actual = handle.attrs.get(key)
                    if isinstance(expected, float):
                        if actual is None or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-10):
                            fail(f"{path}: cell metadata mismatch for {key}")
                    elif actual != expected:
                        fail(f"{path}: cell metadata mismatch for {key}")
                if "_staging" in handle:
                    staging_groups += 1
                    if len(handle["_staging"]) != 0:
                        nonempty_staging_groups += 1
                        fail(f"nonempty staging group in {path}")
                    elif require_no_staging:
                        fail(f"empty staging group remains in {path}")
                if "days" not in handle:
                    fail(f"missing days group in {path}")
                    continue
                for date, day in handle["days"].items():
                    total_groups += 1
                    group_counts[date] += 1
                    if date not in processable_dates:
                        fail(f"unexpected day {date} in {path}")
                    if int(day.attrs.get("complete", 0)) != 1 or "records" not in day:
                        fail(f"incomplete or missing records for {date} in {path}")
                        continue
                    records = day["records"]
                    if records.dtype.names != EXPECTED_RECORD_FIELDS:
                        fail(f"record fields differ for {date} in {path}")
                        continue
                    rows = int(records.shape[0])
                    total_rows += rows
                    row_counts[date] += rows
                    if int(day.attrs.get("row_count", -1)) != rows:
                        fail(f"row_count mismatch for {date} in {path}")
                    try:
                        metadata = json.loads(_text(records.attrs["field_metadata_json"]))
                    except Exception as exc:
                        fail(f"invalid field metadata for {date} in {path}: {exc}")
                    else:
                        if metadata != metadata_expected:
                            fail(f"field metadata differs for {date} in {path}")
                    version = _text(day.attrs.get("quality_flag_algorithm_version", ""))
                    algorithm_versions.add(version)
                    if expected_quality_algorithm_version is not None and version != expected_quality_algorithm_version:
                        fail(f"quality algorithm version differs for {date} in {path}")
                    if "placeholder" not in _text(day.attrs.get("quality_flag_missing_semantics", "")):
                        fail(f"quality missing semantics absent for {date} in {path}")
                    orth = float(day.attrs.get("rotation_orthonormal_max_error", math.inf))
                    det = float(day.attrs.get("rotation_determinant_max_error", math.inf))
                    max_orth_error = max(max_orth_error, orth)
                    max_det_error = max(max_det_error, det)
                    if orth > rotation_atol or det > rotation_atol:
                        fail(f"rotation validity tolerance exceeded for {date} in {path}")

                    epoch = records["epoch_unix_s"][:]
                    day_start = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}", "D")
                    if np.any(epoch.astype("datetime64[s]").astype("datetime64[D]") != day_start):
                        fail(f"noncanonical UTC epoch stored for {date} in {path}")
                    position_km = np.column_stack([records[f"pos_mse_{axis}_km"][:] for axis in "xyz"])
                    position_rm = np.column_stack([records[f"pos_mse_{axis}_rm"][:] for axis in "xyz"])
                    if rows:
                        rm_error = float(np.max(np.abs(position_km / grid.mars_radius_km - position_rm)))
                        max_rm_error = max(max_rm_error, rm_error)
                        if rm_error > 1.0e-10:
                            fail(f"position km/Rm inconsistency for {date} in {path}")
                        calculated_bin, _ = bin_mse_positions(position_km, grid)
                        if np.any(calculated_bin != flat_bin):
                            fail(f"wrong grid assignment for {date} in {path}")
                    if not np.all(np.isin(records["r_matrix_valid_flag"][:], [0, 1])):
                        fail(f"nonbinary rotation valid flag for {date} in {path}")
                    if not np.all(np.isin(records["r_matrix_within_tolerance_flag"][:], [0, 1])):
                        fail(f"nonbinary rotation tolerance flag for {date} in {path}")
                    fov = records["mse_plus_z_in_minpa_fov_flag"][:]
                    if np.any(np.isfinite(fov) & ~np.isin(fov, [0.0, 1.0])):
                        fail(f"nonbinary finite FOV flag for {date} in {path}")

                    for spec in SPECIES.values():
                        prefix = spec["prefix"]
                        density = records[f"{prefix}_density_cm3"][:]
                        velocity = np.column_stack(
                            [records[f"{prefix}_v_mse_{axis}_km_s"][:] for axis in "xyz"]
                        )
                        density_valid = (np.isfinite(density) & (density > 0.0)).astype(np.int8)
                        velocity_valid = np.all(np.isfinite(velocity), axis=1).astype(np.int8)
                        if not np.array_equal(records[f"{prefix}_density_valid_flag"][:], density_valid):
                            fail(f"{prefix} density-valid mismatch for {date} in {path}")
                        if not np.array_equal(records[f"{prefix}_velocity_valid_flag"][:], velocity_valid):
                            fail(f"{prefix} velocity-valid mismatch for {date} in {path}")
                        expected_status = np.full(rows, 4, dtype=np.int16)
                        expected_status[(density_valid == 1) & (velocity_valid == 0)] = 5
                        expected_status[(density_valid == 1) & (velocity_valid == 1)] = 1
                        status = records[f"{prefix}_processing_status_code"][:]
                        if not np.array_equal(status, expected_status):
                            fail(f"{prefix} status mismatch for {date} in {path}")
                        available = records[f"{prefix}_quality_flag_available_flag"][:]
                        matched = records[f"{prefix}_quality_flag_epoch_matched_flag"][:]
                        bitmask = records[f"{prefix}_quality_flag_bitmask"][:].astype(np.uint32)
                        bits = np.column_stack(
                            [
                                records[f"{prefix}_quality_flag_bit{index}_{name}"][:]
                                for index, name in (
                                    (1, "sparse_caution"),
                                    (2, "sparse_invalid"),
                                    (3, "uv_contamination"),
                                    (4, "evenodd_error"),
                                    (5, "high_channel"),
                                )
                            ]
                        )
                        if not (
                            np.all(np.isin(available, [0, 1]))
                            and np.all(np.isin(matched, [0, 1]))
                            and np.all(np.isin(bits, [0, 1]))
                        ):
                            fail(f"nonbinary {prefix} quality fields for {date} in {path}")
                        expected_bits = np.column_stack(
                            [((bitmask >> index) & 1).astype(np.int8) for index in range(5)]
                        )
                        if not np.array_equal(bits[available == 1], expected_bits[available == 1]):
                            fail(f"{prefix} quality bits disagree with bitmask for {date} in {path}")
                        counts = science_counts[prefix]
                        counts["positive_density"] += int(np.count_nonzero(density_valid))
                        counts["finite_velocity"] += int(np.count_nonzero(velocity_valid))
                        counts["status_1"] += int(np.count_nonzero(status == 1))
                        counts["status_4"] += int(np.count_nonzero(status == 4))
                        counts["status_5"] += int(np.count_nonzero(status == 5))
                        counts["quality_available"] += int(np.count_nonzero(available == 1))
                        counts["quality_flagged"] += int(np.count_nonzero((available == 1) & (bitmask != 0)))
        except OSError as exc:
            fail(f"cannot open HDF5 file {path}: {exc}")

    for date in sorted(processable_dates | summary_dates | set(row_counts) | set(group_counts)):
        if row_counts.get(date, 0) != expected_rows_by_date.get(date, -1):
            fail(f"{date} HDF5 rows {row_counts.get(date, 0)} != summary {expected_rows_by_date.get(date, -1)}")
        if group_counts.get(date, 0) != expected_groups_by_date.get(date, -1):
            fail(f"{date} HDF5 groups {group_counts.get(date, 0)} != summary {expected_groups_by_date.get(date, -1)}")
    expected_total_rows = sum(expected_rows_by_date.values())
    expected_total_groups = sum(expected_groups_by_date.values())
    if total_rows != expected_total_rows:
        fail(f"global HDF5 rows {total_rows} != summary total {expected_total_rows}")
    if total_groups != expected_total_groups:
        fail(f"global HDF5 groups {total_groups} != summary total {expected_total_groups}")

    source_rows_total = source_in_grid_total = source_sample_rows = 0
    source_sample_mismatches = 0
    for product in processable:
        if product.date not in summaries:
            continue
        try:
            chunk = load_all_energy_day_chunk(
                product,
                momag_root,
                r_root,
                grid,
                position_tolerance_s=position_tolerance_s,
                rotation_gap_s=rotation_gap_s,
            )
        except Exception as exc:
            fail(f"cannot independently reconstruct source day {product.date}: {exc}")
            continue
        source_rows_total += chunk.rows_total
        source_in_grid_total += chunk.rows_in_grid
        summary = summaries[product.date]
        if chunk.rows_total != int(summary.get("rows_canonical_utc", -1)):
            fail(f"{product.date} canonical source rows differ from summary")
        if chunk.rows_in_grid != int(summary.get("rows_finite_position_in_grid", -1)):
            fail(f"{product.date} reconstructed in-grid rows differ from summary")
        valid_rows = np.flatnonzero(chunk.flat_bin >= 0)
        if valid_rows.size == 0 or source_samples_per_day <= 0:
            continue
        count = min(int(source_samples_per_day), int(valid_rows.size))
        sample_positions = np.unique(np.linspace(0, valid_rows.size - 1, count, dtype=int))
        for row in valid_rows[sample_positions]:
            source_sample_rows += 1
            flat_bin = int(chunk.flat_bin[row])
            path = cell_file_path(output_root, flat_bin, grid)
            try:
                with h5py.File(path, "r") as handle:
                    records = handle[f"days/{product.date}/records"][:]
            except Exception as exc:
                fail(f"cannot read source-sample target {product.date} in {path}: {exc}")
                source_sample_mismatches += 1
                continue
            matches = np.flatnonzero(records["epoch_unix_s"] == chunk.common["epoch_unix_s"][row])
            if matches.size != 1:
                fail(f"{product.date} sampled epoch appears {matches.size} times in expected cell")
                source_sample_mismatches += 1
                continue
            expected_record, _ = build_compound_records(chunk, np.asarray([row]))
            mismatch_fields = _equal_records(expected_record[0], records[int(matches[0])])
            if mismatch_fields:
                fail(f"{product.date} source sample differs in fields {mismatch_fields[:10]}")
                source_sample_mismatches += 1

    if source_rows_total != sum(int(item.get("rows_canonical_utc", 0)) for item in summaries.values()):
        fail("global independently reconstructed canonical-row total differs from summaries")
    if source_in_grid_total != expected_total_rows:
        fail("global independently reconstructed in-grid total differs from HDF5 summaries")

    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "errors": errors,
        "requirements": {
            "input_product_days": len(products),
            "processable_geometry_days": len(processable),
            "excluded_geometry_days": len(excluded_products),
            "day_summaries": len(summary_paths),
            "missing_day_summaries": len(missing_summaries),
            "extra_day_summaries": len(extra_summaries),
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "record_field_count": len(EXPECTED_RECORD_FIELDS),
            "grid_shape_xyz": list(grid.shape),
            "grid_min_rm": grid.min_rm,
            "grid_max_rm": grid.max_rm,
            "grid_step_rm": grid.step_rm,
            "mars_radius_km": grid.mars_radius_km,
            "quality_algorithm_versions": sorted(algorithm_versions),
        },
        "storage": {
            "hdf5_files": len(cell_paths),
            "cell_day_groups": total_groups,
            "records": total_rows,
            "expected_cell_day_groups_from_summaries": expected_total_groups,
            "expected_records_from_summaries": expected_total_rows,
            "staging_groups": staging_groups,
            "nonempty_staging_groups": nonempty_staging_groups,
        },
        "source_reconstruction": {
            "canonical_rows": source_rows_total,
            "in_grid_rows": source_in_grid_total,
            "sample_rows_compared": source_sample_rows,
            "sample_mismatches": source_sample_mismatches,
            "sample_policy": "equally spaced canonical in-grid rows for every processable UTC day",
        },
        "validation_maxima": {
            "position_km_to_rm_error": max_rm_error,
            "rotation_orthonormal_error": max_orth_error,
            "rotation_determinant_error": max_det_error,
        },
        "science_counts": science_counts,
        "density_aggregation_policy": "per-record densities retained; no cross-epoch sum is interpreted as local density",
        "final_run_summary": {
            "requested_days": latest.get("requested_days"),
            "processed_days": latest.get("processed_days"),
            "already_complete_days": latest.get("already_complete_days"),
            "canonical_utc_rows": latest.get("canonical_utc_rows"),
            "in_grid_rows": latest.get("in_grid_rows"),
        },
    }
    if write_summary:
        _atomic_json(output_root / "logs" / "full_audit_summary.json", report)
    return report
