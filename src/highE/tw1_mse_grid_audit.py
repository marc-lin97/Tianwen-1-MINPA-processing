"""Streaming completion audit for Tianwen-1 MINPA MSE grid records."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat

from .tw1_mse_grid_records import (
    DATASET_METADATA,
    MseGridSpec,
    QUALITY_DATASET_NAMES,
    SCHEMA_VERSION,
    bin_mse_positions,
    cell_file_path,
    discover_daily_pairs,
)


COMMON_RECORD_FIELDS = (
    "epoch_unix_s",
    "pos_mse_x_km",
    "pos_mse_y_km",
    "pos_mse_z_km",
    "pos_mse_x_rm",
    "pos_mse_y_rm",
    "pos_mse_z_rm",
    "mse_plus_z_in_minpa_fov_flag",
    "r_matrix_match_dt_s",
    "r_matrix_valid_flag",
    "r_matrix_within_tolerance_flag",
)
SPECIES_RECORD_FIELDS = (
    "density_cm3",
    "v_mse_x_km_s",
    "v_mse_y_km_s",
    "v_mse_z_km_s",
    "processing_status_code",
    "density_valid_flag",
    "velocity_valid_flag",
    *QUALITY_DATASET_NAMES,
)
EXPECTED_RECORD_FIELDS = COMMON_RECORD_FIELDS + tuple(
    f"{prefix}_{field}"
    for prefix in ("oplus", "o2plus")
    for field in SPECIES_RECORD_FIELDS
)
EXPECTED_QUALITY_ALGORITHM_VERSION = "2026-07-13-seven-band-all-species-v2-mode12-product-time"


def _attribute_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _mat_info_value(info: Any, name: str, default: Any) -> Any:
    if info is None or not hasattr(info, name):
        return default
    value = getattr(info, name)
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def audit_tw1_day_summary_provenance(
    input_root: Path,
    output_root: Path,
    *,
    expected_days: int | None = None,
    expected_quality_algorithm_version: str | None = EXPECTED_QUALITY_ALGORITHM_VERSION,
    coordinate_atol: float = 1.0e-8,
    rotation_atol: float = 1.0e-6,
    quality_tolerance_s: float = 0.001,
    max_reported_errors: int = 100,
    write_summary: bool = True,
) -> dict[str, Any]:
    """Audit all daily summaries, including dates with no in-grid cell group."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    errors: list[str] = []
    error_count = 0

    def fail(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < max_reported_errors:
            errors.append(message)

    pairs = discover_daily_pairs(input_root)
    pair_by_date = {pair.date: pair for pair in pairs}
    expected_dates = set(pair_by_date)
    if expected_days is not None and len(pairs) != expected_days:
        fail(f"source paired date count {len(pairs)} != expected {expected_days}")
    summary_paths = sorted((output_root / "logs" / "days").glob("*.json"))
    summary_by_date = {path.stem: path for path in summary_paths}
    missing = sorted(expected_dates - set(summary_by_date))
    extra = sorted(set(summary_by_date) - expected_dates)
    if missing:
        fail(f"missing day summaries: {missing[:10]} (count={len(missing)})")
    if extra:
        fail(f"extra day summaries: {extra[:10]} (count={len(extra)})")

    source_paths: set[str] = set()
    quality_versions: set[str] = set()
    total_rows = total_in_grid = total_groups = zero_cell_days = 0
    maxima = {
        "position_recompute_error_km": 0.0,
        "oplus_velocity_recompute_error_km_s": 0.0,
        "o2plus_velocity_recompute_error_km_s": 0.0,
        "rotation_orthonormal_error": 0.0,
        "rotation_determinant_error": 0.0,
    }
    validation_mapping = {
        "max_abs_position_recompute_error_km": ("position_recompute_error_km", coordinate_atol),
        "max_abs_oplus_velocity_recompute_error_km_s": ("oplus_velocity_recompute_error_km_s", coordinate_atol),
        "max_abs_o2plus_velocity_recompute_error_km_s": ("o2plus_velocity_recompute_error_km_s", coordinate_atol),
        "rotation_orthonormal_max_error": ("rotation_orthonormal_error", rotation_atol),
        "rotation_determinant_max_error": ("rotation_determinant_error", rotation_atol),
    }
    for date in sorted(expected_dates & set(summary_by_date)):
        path = summary_by_date[date]
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"cannot read day summary {path}: {exc}")
            continue
        if summary.get("date") != date or summary.get("status") != "complete":
            fail(f"invalid day summary identity/status: {path}")
        pair = pair_by_date[date]
        sources = summary.get("source_files", {})
        expected_species_paths = {"Oplus": pair.oplus, "O2plus": pair.o2plus}
        for species, expected_path in expected_species_paths.items():
            actual = Path(str(sources.get(species, "")))
            source_paths.add(str(actual))
            if not actual.exists() or actual.resolve() != expected_path.resolve():
                fail(f"{date} {species} source path mismatch or missing: {actual}")
        quality_path = Path(str(sources.get("NV_MSO_2", "")))
        rotation_path = Path(str(summary.get("rotation_file", "")))
        for label, source_path in (("NV_MSO_2", quality_path), ("rotation", rotation_path)):
            source_paths.add(str(source_path))
            if not source_path.exists():
                fail(f"{date} missing {label} source path: {source_path}")
        if quality_path.exists():
            try:
                data = loadmat(
                    quality_path,
                    variable_names=["MINPA_quality_flag_info"],
                    squeeze_me=True,
                    struct_as_record=False,
                )
                info = data.get("MINPA_quality_flag_info")
                version = str(_mat_info_value(info, "algorithm_version", "unknown"))
                tolerance = float(_mat_info_value(info, "time_match_tolerance_s", math.nan))
                missing_semantics = str(_mat_info_value(info, "missing_semantics", ""))
                quality_versions.add(version)
                if expected_quality_algorithm_version is not None and version != expected_quality_algorithm_version:
                    fail(f"{date} NV algorithm version {version!r} differs from expected")
                if not math.isclose(tolerance, quality_tolerance_s, rel_tol=0.0, abs_tol=1.0e-12):
                    fail(f"{date} NV time tolerance {tolerance!r} differs from expected")
                if "placeholder" not in missing_semantics:
                    fail(f"{date} NV missing semantics do not preserve placeholder meaning")
            except Exception as exc:
                fail(f"cannot inspect NV metadata for {date}: {exc}")

        rows_total = int(summary.get("rows_total", -1))
        rows_in_grid = int(summary.get("rows_finite_position_in_grid", -1))
        groups = int(summary.get("nonempty_cells", -1))
        cell_files = summary.get("cell_files", [])
        if min(rows_total, rows_in_grid, groups) < 0 or rows_in_grid > rows_total:
            fail(f"{date} invalid row/group counts")
        if not isinstance(cell_files, list) or len(cell_files) != groups:
            fail(f"{date} cell_files count differs from nonempty_cells")
        total_rows += max(rows_total, 0)
        total_in_grid += max(rows_in_grid, 0)
        total_groups += max(groups, 0)
        zero_cell_days += int(groups == 0)

        validation = summary.get("validation", {})
        for source_name, (report_name, tolerance) in validation_mapping.items():
            try:
                value = float(validation[source_name])
            except (KeyError, TypeError, ValueError):
                fail(f"{date} missing/invalid validation field {source_name}")
                continue
            if math.isfinite(value):
                maxima[report_name] = max(maxima[report_name], value)
                if value > tolerance:
                    fail(f"{date} validation field {source_name}={value:g} exceeds {tolerance:g}")
            elif rows_in_grid > 0:
                fail(f"{date} non-finite validation field {source_name} with in-grid records")

    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "errors": errors,
        "source_paired_dates": len(pairs),
        "day_summaries": len(summary_paths),
        "missing_day_summaries": len(missing),
        "extra_day_summaries": len(extra),
        "unique_daily_source_paths_checked": len(source_paths),
        "quality_algorithm_versions": sorted(quality_versions),
        "zero_cell_days": zero_cell_days,
        "rows_total": total_rows,
        "rows_in_grid": total_in_grid,
        "cell_day_groups": total_groups,
        "validation_maxima": maxima,
    }
    if write_summary:
        _atomic_json(output_root / "logs" / "day_summary_provenance_audit.json", report)
    return report


def cleanup_empty_staging_groups(output_root: Path) -> dict[str, int]:
    """Remove only empty transaction groups; never alter incomplete content."""
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
    return {"files_seen": files_seen, "empty_staging_groups_removed": removed, "nonempty_staging_groups": nonempty}


def _expected_field_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for output_name in EXPECTED_RECORD_FIELDS:
        source_name = output_name
        for prefix in ("oplus_", "o2plus_"):
            if output_name.startswith(prefix):
                source_name = output_name[len(prefix) :]
                break
        units, description = DATASET_METADATA[source_name]
        metadata[output_name] = {"units": units, "description": description}
        if "mse" in source_name:
            metadata[output_name]["coordinate_system"] = "MSE"
    return metadata


def _grid_metadata(flat_bin: int, grid: MseGridSpec) -> dict[str, Any]:
    ix, iy, iz = np.unravel_index(flat_bin, grid.shape)
    return {
        "schema_version": SCHEMA_VERSION,
        "mission": "Tianwen-1",
        "instrument": "MINPA",
        "coordinate_system": "MSE",
        "flat_bin": flat_bin,
        "ix": int(ix),
        "iy": int(iy),
        "iz": int(iz),
        "grid_min_rm": grid.min_rm,
        "grid_max_rm": grid.max_rm,
        "grid_step_rm": grid.step_rm,
        "mars_radius_km": grid.mars_radius_km,
        "x_center_rm": grid.center_rm(int(ix)),
        "y_center_rm": grid.center_rm(int(iy)),
        "z_center_rm": grid.center_rm(int(iz)),
        "x_edge_min_rm": grid.edge_rm(int(ix))[0],
        "x_edge_max_rm": grid.edge_rm(int(ix))[1],
        "y_edge_min_rm": grid.edge_rm(int(iy))[0],
        "y_edge_max_rm": grid.edge_rm(int(iy))[1],
        "z_edge_min_rm": grid.edge_rm(int(iz))[0],
        "z_edge_max_rm": grid.edge_rm(int(iz))[1],
        "high_energy_min_eV": 1000.0,
    }


def audit_tw1_mse_grid_records(
    input_root: Path,
    output_root: Path,
    *,
    expected_days: int | None = None,
    expected_quality_algorithm_version: str | None = EXPECTED_QUALITY_ALGORITHM_VERSION,
    coordinate_atol: float = 1.0e-8,
    rotation_atol: float = 1.0e-6,
    require_no_staging: bool = True,
    max_reported_errors: int = 100,
    write_summary: bool = True,
) -> dict[str, Any]:
    """Audit every cell-day and row without loading all grid files together."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    grid = MseGridSpec()
    errors: list[str] = []
    error_count = 0

    def fail(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < max_reported_errors:
            errors.append(message)

    pairs = discover_daily_pairs(input_root)
    source_dates = [pair.date for pair in pairs]
    source_date_set = set(source_dates)
    if expected_days is not None and len(source_dates) != expected_days:
        fail(f"source paired date count {len(source_dates)} != expected {expected_days}")

    day_summary_dir = output_root / "logs" / "days"
    summary_paths = sorted(day_summary_dir.glob("*.json"))
    summary_dates = {path.stem for path in summary_paths}
    missing_summaries = sorted(source_date_set - summary_dates)
    extra_summaries = sorted(summary_dates - source_date_set)
    if missing_summaries:
        fail(f"missing day summaries: {missing_summaries[:10]} (count={len(missing_summaries)})")
    if extra_summaries:
        fail(f"extra day summaries: {extra_summaries[:10]} (count={len(extra_summaries)})")

    expected_rows_by_date: dict[str, int] = {}
    expected_groups_by_date: dict[str, int] = {}
    summary_cell_file_counts: dict[str, int] = {}
    for path in summary_paths:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - exercised by production audit
            fail(f"cannot read day summary {path}: {exc}")
            continue
        date = path.stem
        if summary.get("date") != date or summary.get("status") != "complete":
            fail(f"invalid day summary identity/status: {path}")
        expected_rows_by_date[date] = int(summary.get("rows_finite_position_in_grid", -1))
        expected_groups_by_date[date] = int(summary.get("nonempty_cells", -1))
        cell_files = summary.get("cell_files", [])
        summary_cell_file_counts[date] = len(cell_files) if isinstance(cell_files, list) else -1
        if summary_cell_file_counts[date] != expected_groups_by_date[date]:
            fail(f"{date} summary cell_files count differs from nonempty_cells")

    latest_path = output_root / "logs" / "latest_run_summary.json"
    latest: dict[str, Any] = {}
    if not latest_path.exists():
        fail(f"missing final run summary: {latest_path}")
    else:
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"cannot read final run summary {latest_path}: {exc}")
        if latest:
            requested = int(latest.get("requested_days", -1))
            complete = int(latest.get("processed_days", -1)) + int(latest.get("already_complete_days", -1))
            if requested != len(source_dates) or complete != len(source_dates):
                fail(f"final run summary day counts requested={requested}, complete={complete}, source={len(source_dates)}")

    schema_path = output_root / "grid_schema.json"
    if not schema_path.exists():
        fail(f"missing grid schema: {schema_path}")
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        expected_grid = grid.as_dict()
        for key, expected in {
            "schema_version": SCHEMA_VERSION,
            "mission": "Tianwen-1",
            "instrument": "MINPA",
            "coordinate_system": "MSE",
            "high_energy_selection": "raw MINPA energy > 1000 eV",
        }.items():
            if schema.get(key) != expected:
                fail(f"grid schema {key}={schema.get(key)!r} != {expected!r}")
        if schema.get("grid") != expected_grid:
            fail("grid schema grid definition differs from required [-5,5], 0.1 Rm, Rm=3397 km")
        quality_schema = schema.get("quality_flags", {})
        if "NV_MSO_2" not in str(quality_schema.get("source", "")):
            fail("grid schema does not identify NV_MSO_2 quality source")
        if "placeholder" not in str(quality_schema.get("missing_semantics", "")):
            fail("grid schema does not preserve unavailable-flag placeholder semantics")

    expected_field_metadata = _expected_field_metadata()
    row_counts: defaultdict[str, int] = defaultdict(int)
    group_counts: defaultdict[str, int] = defaultdict(int)
    algorithm_versions: set[str] = set()
    source_paths_checked: set[str] = set()
    total_rows = 0
    total_groups = 0
    staging_groups = 0
    nonempty_staging_groups = 0
    max_position_rm_km_error = 0.0
    max_position_recompute_error = 0.0
    max_velocity_recompute_error = {"oplus": 0.0, "o2plus": 0.0}
    max_rotation_orthonormal_error = 0.0
    max_rotation_determinant_error = 0.0
    density_stats = {
        "oplus": {"finite_count": 0, "min_cm3": math.inf, "max_cm3": -math.inf},
        "o2plus": {"finite_count": 0, "min_cm3": math.inf, "max_cm3": -math.inf},
    }
    flagged_available_rows = {"oplus": 0, "o2plus": 0}
    available_rows = {"oplus": 0, "o2plus": 0}

    cell_paths = sorted((output_root / "cells").rglob("*.h5"))
    for path in cell_paths:
        try:
            with h5py.File(path, "r") as handle:
                flat_bin = int(handle.attrs.get("flat_bin", -1))
                if flat_bin < 0 or flat_bin >= int(np.prod(grid.shape)):
                    fail(f"invalid flat_bin in {path}: {flat_bin}")
                    continue
                expected_path = cell_file_path(output_root, flat_bin, grid)
                if path.resolve() != expected_path.resolve():
                    fail(f"cell filename/path disagrees with flat_bin: {path}")
                for key, expected in _grid_metadata(flat_bin, grid).items():
                    actual = handle.attrs.get(key)
                    if isinstance(expected, float):
                        if actual is None or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-10):
                            fail(f"{path}: grid metadata mismatch for {key}")
                    elif actual != expected:
                        fail(f"{path}: grid metadata mismatch for {key}")
                if "_staging" in handle:
                    staging_groups += 1
                    if len(handle["_staging"]) != 0:
                        nonempty_staging_groups += 1
                        fail(f"nonempty _staging group in {path}")
                    elif require_no_staging:
                        fail(f"empty _staging group remains in {path}")
                if "days" not in handle:
                    fail(f"missing days group in {path}")
                    continue
                for date, day in handle["days"].items():
                    total_groups += 1
                    group_counts[date] += 1
                    if date not in source_date_set:
                        fail(f"unexpected date group {date} in {path}")
                    if int(day.attrs.get("complete", 0)) != 1:
                        fail(f"incomplete day group {date} in {path}")
                    if "records" not in day:
                        fail(f"missing records dataset for {date} in {path}")
                        continue
                    records = day["records"]
                    names = records.dtype.names
                    if names != EXPECTED_RECORD_FIELDS:
                        fail(f"record fields differ in {path}/days/{date}: {names}")
                        continue
                    rows = int(records.shape[0])
                    total_rows += rows
                    row_counts[date] += rows
                    if int(day.attrs.get("row_count", -1)) != rows:
                        fail(f"row_count attribute mismatch for {date} in {path}")
                    try:
                        field_metadata = json.loads(_attribute_text(records.attrs["field_metadata_json"]))
                    except Exception as exc:
                        fail(f"invalid field metadata for {date} in {path}: {exc}")
                    else:
                        if field_metadata != expected_field_metadata:
                            fail(f"field metadata mismatch for {date} in {path}")

                    for attr_name in ("source_oplus_file", "source_o2plus_file", "rotation_file", "quality_flag_file"):
                        source = _attribute_text(day.attrs.get(attr_name, ""))
                        if source not in source_paths_checked:
                            source_paths_checked.add(source)
                            if not source or not Path(source).exists():
                                fail(f"missing provenance file {attr_name}={source!r}")
                    version = _attribute_text(day.attrs.get("quality_flag_algorithm_version", ""))
                    algorithm_versions.add(version)
                    if expected_quality_algorithm_version is not None and version != expected_quality_algorithm_version:
                        fail(f"quality algorithm version {version!r} differs in {path}/days/{date}")
                    if "placeholder" not in _attribute_text(day.attrs.get("quality_flag_missing_semantics", "")):
                        fail(f"missing NV unavailable semantics in {path}/days/{date}")

                    position_error = float(day.attrs.get("max_abs_position_recompute_error_km", math.inf))
                    oplus_error = float(day.attrs.get("max_abs_oplus_velocity_recompute_error_km_s", math.inf))
                    o2plus_error = float(day.attrs.get("max_abs_o2plus_velocity_recompute_error_km_s", math.inf))
                    orth_error = float(day.attrs.get("rotation_orthonormal_max_error", math.inf))
                    det_error = float(day.attrs.get("rotation_determinant_max_error", math.inf))
                    max_position_recompute_error = max(max_position_recompute_error, position_error)
                    max_velocity_recompute_error["oplus"] = max(max_velocity_recompute_error["oplus"], oplus_error)
                    max_velocity_recompute_error["o2plus"] = max(max_velocity_recompute_error["o2plus"], o2plus_error)
                    max_rotation_orthonormal_error = max(max_rotation_orthonormal_error, orth_error)
                    max_rotation_determinant_error = max(max_rotation_determinant_error, det_error)
                    if position_error > coordinate_atol or oplus_error > coordinate_atol or o2plus_error > coordinate_atol:
                        fail(f"coordinate recompute tolerance exceeded for {date} in {path}")
                    if orth_error > rotation_atol or det_error > rotation_atol:
                        fail(f"rotation validity tolerance exceeded for {date} in {path}")

                    position_rm = np.column_stack([records[f"pos_mse_{axis}_rm"][:] for axis in "xyz"])
                    position_km = np.column_stack([records[f"pos_mse_{axis}_km"][:] for axis in "xyz"])
                    if rows:
                        rm_km_error = float(np.max(np.abs(position_km / grid.mars_radius_km - position_rm)))
                        max_position_rm_km_error = max(max_position_rm_km_error, rm_km_error)
                        if rm_km_error > coordinate_atol:
                            fail(f"position km/Rm inconsistency for {date} in {path}")
                        calculated_bins, _ = bin_mse_positions(position_km, grid)
                        if np.any(calculated_bins != flat_bin):
                            fail(f"row assigned to wrong grid cell for {date} in {path}")

                    for species in ("oplus", "o2plus"):
                        density = records[f"{species}_density_cm3"][:]
                        finite_density = np.isfinite(density)
                        density_valid = records[f"{species}_density_valid_flag"][:]
                        if not np.array_equal(density_valid, finite_density.astype(np.int8)):
                            fail(f"{species} density-valid flag mismatch for {date} in {path}")
                        if np.any(density[finite_density] < 0.0):
                            fail(f"negative finite {species} density for {date} in {path}")
                        if np.any(finite_density):
                            finite_values = density[finite_density]
                            stats = density_stats[species]
                            stats["finite_count"] += int(finite_values.size)
                            stats["min_cm3"] = min(float(stats["min_cm3"]), float(np.min(finite_values)))
                            stats["max_cm3"] = max(float(stats["max_cm3"]), float(np.max(finite_values)))
                        velocity = np.column_stack([records[f"{species}_v_mse_{axis}_km_s"][:] for axis in "xyz"])
                        velocity_valid = records[f"{species}_velocity_valid_flag"][:]
                        expected_velocity_valid = np.all(np.isfinite(velocity), axis=1).astype(np.int8)
                        if not np.array_equal(velocity_valid, expected_velocity_valid):
                            fail(f"{species} velocity-valid flag mismatch for {date} in {path}")

                        available = records[f"{species}_quality_flag_available_flag"][:]
                        matched = records[f"{species}_quality_flag_epoch_matched_flag"][:]
                        bitmask = records[f"{species}_quality_flag_bitmask"][:].astype(np.uint32)
                        bits = np.column_stack(
                            [records[f"{species}_quality_flag_bit{bit}_{name}"][:] for bit, name in (
                                (1, "sparse_caution"),
                                (2, "sparse_invalid"),
                                (3, "uv_contamination"),
                                (4, "evenodd_error"),
                                (5, "high_channel"),
                            )]
                        )
                        if not (np.all(np.isin(available, [0, 1])) and np.all(np.isin(matched, [0, 1])) and np.all(np.isin(bits, [0, 1]))):
                            fail(f"non-binary {species} quality fields for {date} in {path}")
                        usable = available == 1
                        if np.any(usable & (matched != 1)):
                            fail(f"available but epoch-unmatched {species} quality flag for {date} in {path}")
                        expected_bits = np.column_stack([((bitmask >> index) & 1).astype(np.int8) for index in range(5)])
                        if not np.array_equal(bits[usable], expected_bits[usable]):
                            fail(f"{species} NV bit columns disagree with bitmask for {date} in {path}")
                        available_rows[species] += int(np.count_nonzero(usable))
                        flagged_available_rows[species] += int(np.count_nonzero(usable & (bitmask != 0)))
        except OSError as exc:
            fail(f"cannot open HDF5 file {path}: {exc}")

    for date in sorted(source_date_set | summary_dates | set(row_counts) | set(group_counts)):
        if row_counts.get(date, 0) != expected_rows_by_date.get(date, -1):
            fail(f"{date} HDF5 row total {row_counts.get(date, 0)} != summary {expected_rows_by_date.get(date, -1)}")
        if group_counts.get(date, 0) != expected_groups_by_date.get(date, -1):
            fail(f"{date} HDF5 cell-day count {group_counts.get(date, 0)} != summary {expected_groups_by_date.get(date, -1)}")

    expected_total_rows = sum(value for value in expected_rows_by_date.values() if value >= 0)
    expected_total_groups = sum(value for value in expected_groups_by_date.values() if value >= 0)
    if total_rows != expected_total_rows:
        fail(f"global HDF5 row total {total_rows} != day-summary total {expected_total_rows}")
    if total_groups != expected_total_groups:
        fail(f"global HDF5 cell-day total {total_groups} != day-summary total {expected_total_groups}")

    for stats in density_stats.values():
        if stats["finite_count"] == 0:
            stats["min_cm3"] = None
            stats["max_cm3"] = None

    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "errors": errors,
        "requirements": {
            "source_paired_dates": len(source_dates),
            "day_summaries": len(summary_paths),
            "missing_day_summaries": len(missing_summaries),
            "extra_day_summaries": len(extra_summaries),
            "grid_shape_xyz": list(grid.shape),
            "grid_min_rm": grid.min_rm,
            "grid_max_rm": grid.max_rm,
            "grid_step_rm": grid.step_rm,
            "mars_radius_km": grid.mars_radius_km,
            "record_field_count": len(EXPECTED_RECORD_FIELDS),
            "quality_algorithm_versions": sorted(algorithm_versions),
        },
        "storage": {
            "hdf5_files": len(cell_paths),
            "cell_day_groups": total_groups,
            "records": total_rows,
            "expected_cell_day_groups_from_day_summaries": expected_total_groups,
            "expected_records_from_day_summaries": expected_total_rows,
            "staging_groups": staging_groups,
            "nonempty_staging_groups": nonempty_staging_groups,
        },
        "validation_maxima": {
            "position_km_to_rm_error": max_position_rm_km_error,
            "position_recompute_error_km": max_position_recompute_error,
            "oplus_velocity_recompute_error_km_s": max_velocity_recompute_error["oplus"],
            "o2plus_velocity_recompute_error_km_s": max_velocity_recompute_error["o2plus"],
            "rotation_orthonormal_error": max_rotation_orthonormal_error,
            "rotation_determinant_error": max_rotation_determinant_error,
        },
        "science_counts": {
            "density": density_stats,
            "quality_available_rows": available_rows,
            "quality_flagged_available_rows": flagged_available_rows,
        },
        "provenance_unique_source_files_checked": len(source_paths_checked),
        "final_run_summary": {
            "requested_days": latest.get("requested_days"),
            "processed_days": latest.get("processed_days"),
            "already_complete_days": latest.get("already_complete_days"),
        },
        "density_aggregation_policy": "per-record densities retained; no cross-epoch sum is interpreted as local density",
    }
    if write_summary:
        _atomic_json(output_root / "logs" / "full_audit_summary.json", report)
    return report
