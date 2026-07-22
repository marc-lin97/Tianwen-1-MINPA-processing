"""Classify full-energy MINPA moments by +Xb_MSE cones and plot MSE X-Z maps."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constants import DEFAULT_TW1_MOMAG_ROOT, DEFAULT_TW1_NV_MSO2_ROOT, DEFAULT_TW1_R_ROOT
from .tw1_all_energy_mse_grid_records import (
    NvProduct,
    canonical_row_count,
    discover_nv_products,
    load_all_energy_day_chunk,
)
from .tw1_mse_grid_records import MseGridSpec
from .tw1_mse_xz_fov_stats import (
    CATEGORY_DEFINITIONS,
    CATEGORY_IDS,
    XbClassification,
    _coordinate_slug,
    _initialize_accumulators,
    _products_from_dense,
    _reduce_selected_rows,
    _valid_species_masks,
    classify_xb_mse,
    create_classified_plots,
    select_y_grid_centers,
)
from .tw1_mse_xz_stats import (
    ALL_ENERGY_SPECIES,
    ALL_ENERGY_SPECIES_LABELS,
    STAT_NAMES,
    _atomic_json,
    _atomic_npz,
    _finite_percentile,
)


SPECIES_MAP = (("hplus", "Hplus"), ("oplus", "Oplus"), ("o2plus", "O2plus"))


@dataclass(frozen=True)
class AllEnergyClassifiedDay:
    date: str
    source_file: str
    rows_total: int
    rows_in_grid: int
    xb_valid_rows: int
    xb_valid_in_grid_rows: int
    category_rows: dict[str, int]
    category_in_grid_rows: dict[str, int]
    xb_valid_but_outside_rows: int
    xb_valid_in_grid_but_outside_rows: int
    validation: dict[str, float]
    aggregates: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]


def aggregate_all_energy_classified_chunk(
    chunk: Any,
    classification: XbClassification,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Reduce one full-energy day into sparse sufficient statistics for both cones."""
    if classification.valid.shape != chunk.flat_bin.shape:
        raise ValueError("Xb classification length differs from the all-energy rows")
    output: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for category in CATEGORY_IDS:
        output[category] = {}
        category_mask = classification.categories[category]
        for output_species, chunk_species in SPECIES_MAP:
            values = chunk.species[chunk_species]
            density = np.asarray(values["density_cm3"], dtype=float)
            velocity = np.column_stack(
                [np.asarray(values[f"v_mse_{axis}_km_s"], dtype=float) for axis in "xyz"]
            )
            density_ok, velocity_ok = _valid_species_masks(chunk, chunk_species)
            output[category][output_species] = _reduce_selected_rows(
                chunk.flat_bin,
                density,
                velocity,
                density_ok & category_mask,
                velocity_ok & category_mask,
            )
    return output


def _process_all_energy_day(
    arguments: tuple[NvProduct, str, str, MseGridSpec, float, float]
) -> AllEnergyClassifiedDay:
    product, momag_root_text, r_root_text, grid, position_tolerance_s, rotation_gap_s = arguments
    chunk = load_all_energy_day_chunk(
        product,
        Path(momag_root_text),
        Path(r_root_text),
        grid,
        position_tolerance_s=position_tolerance_s,
        rotation_gap_s=rotation_gap_s,
    )
    xb_mse = np.column_stack([chunk.common[f"xb_mse_{axis}"] for axis in "xyz"])
    rotation_valid = chunk.common["r_matrix_within_tolerance_flag"] == 1
    xb_mse = np.asarray(xb_mse, dtype=float)
    xb_mse[~rotation_valid] = np.nan
    classification = classify_xb_mse(xb_mse)
    in_grid = chunk.flat_bin >= 0
    classified = np.logical_or.reduce(list(classification.categories.values()))
    category_rows = {
        category: int(np.count_nonzero(classification.categories[category]))
        for category in CATEGORY_IDS
    }
    category_in_grid_rows = {
        category: int(np.count_nonzero(classification.categories[category] & in_grid))
        for category in CATEGORY_IDS
    }
    valid_norm = np.linalg.norm(xb_mse[classification.valid], axis=1)
    norm_error = (
        float(np.max(np.abs(valid_norm - 1.0))) if valid_norm.size else math.nan
    )
    validation = {
        "max_abs_xb_mse_unit_norm_error": norm_error,
        "rotation_orthonormal_max_error": float(chunk.validation["rotation_orthonormal_max_error"]),
        "rotation_determinant_max_error": float(chunk.validation["rotation_determinant_max_error"]),
        "max_abs_rotation_match_dt_s": float(chunk.validation["max_abs_rotation_match_dt_s"]),
        "cone_overlap_rows": float(
            np.count_nonzero(
                classification.categories[CATEGORY_IDS[0]]
                & classification.categories[CATEGORY_IDS[1]]
            )
        ),
    }
    return AllEnergyClassifiedDay(
        date=product.date,
        source_file=str(product.path),
        rows_total=chunk.rows_total,
        rows_in_grid=chunk.rows_in_grid,
        xb_valid_rows=int(np.count_nonzero(classification.valid)),
        xb_valid_in_grid_rows=int(np.count_nonzero(classification.valid & in_grid)),
        category_rows=category_rows,
        category_in_grid_rows=category_in_grid_rows,
        xb_valid_but_outside_rows=int(np.count_nonzero(classification.valid & ~classified)),
        xb_valid_in_grid_but_outside_rows=int(
            np.count_nonzero(classification.valid & in_grid & ~classified)
        ),
        validation=validation,
        aggregates=aggregate_all_energy_classified_chunk(chunk, classification),
    )


def _merge_day(
    accumulators: dict[str, dict[str, dict[str, np.ndarray]]],
    daily: AllEnergyClassifiedDay,
) -> None:
    for category in CATEGORY_IDS:
        for species in ALL_ENERGY_SPECIES:
            flat, statistics = daily.aggregates[category][species]
            for row, name in enumerate(STAT_NAMES):
                values = statistics[row]
                if name.endswith("count"):
                    values = values.astype(np.int64)
                np.add.at(accumulators[category][species][name], flat, values)


def _processable_products(
    input_root: Path,
    momag_root: Path,
    r_root: Path,
    *,
    max_days: int | None,
) -> tuple[list[NvProduct], list[dict[str, Any]], int]:
    products = discover_nv_products(input_root)
    if not products:
        raise FileNotFoundError(f"No NV_YYYYMMDD.mat products found under {input_root}")
    processable: list[NvProduct] = []
    excluded: list[dict[str, Any]] = []
    for product in products:
        momag_path = momag_root / f"Bss{product.date}.mat"
        rotation_path = r_root / f"R_MSO2MSE_Tianwen-1_{product.date}.mat"
        if momag_path.exists() and rotation_path.exists():
            processable.append(product)
        else:
            canonical_rows, source_rows = canonical_row_count(product.path, product.date)
            excluded.append(
                {
                    "date": product.date,
                    "source_rows": source_rows,
                    "canonical_utc_rows_excluded": canonical_rows,
                    "momag_exists": momag_path.exists(),
                    "rotation_exists": rotation_path.exists(),
                }
            )
    total_processable = len(processable)
    if max_days is not None:
        processable = processable[: int(max_days)]
    if not processable:
        raise FileNotFoundError("No all-energy day has both MOMAG and R_MSO2MSE geometry")
    return processable, excluded, total_processable


def run_tw1_all_energy_mse_xz_xb_statistics(
    input_root: Path = DEFAULT_TW1_NV_MSO2_ROOT,
    output_root: Path = Path(
        r"D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_all_energy_mse_xz_xb_classes_y_m0p5_p0p5_5Rm"
    ),
    *,
    momag_root: Path = DEFAULT_TW1_MOMAG_ROOT,
    r_root: Path = DEFAULT_TW1_R_ROOT,
    grid: MseGridSpec | None = None,
    workers: int = 1,
    max_days: int | None = None,
    write_plots: bool = True,
    progress_every: int = 25,
    position_tolerance_s: float = 5.0,
    rotation_gap_s: float = 8.0,
    y_min_rm: float = -0.5,
    y_max_rm: float = 0.5,
) -> dict[str, Any]:
    """Generate full-energy H+/O+/O2+ X-Z maps for both +Xb_MSE cones."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    momag_root = Path(momag_root)
    r_root = Path(r_root)
    grid = MseGridSpec() if grid is None else grid
    products, excluded, processable_day_count = _processable_products(
        input_root, momag_root, r_root, max_days=max_days
    )
    accumulators = _initialize_accumulators(
        grid.size**3, species_names=ALL_ENERGY_SPECIES
    )
    arguments = [
        (product, str(momag_root), str(r_root), grid, position_tolerance_s, rotation_gap_s)
        for product in products
    ]
    daily_summaries: list[AllEnergyClassifiedDay] = []
    if workers <= 1:
        iterator: Iterable[AllEnergyClassifiedDay] = map(_process_all_energy_day, arguments)
        for completed, daily in enumerate(iterator, start=1):
            _merge_day(accumulators, daily)
            daily_summaries.append(daily)
            if progress_every > 0 and (completed % progress_every == 0 or completed == len(products)):
                print(f"classified all-energy MINPA days: {completed}/{len(products)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_process_all_energy_day, arguments, chunksize=1)
            for completed, daily in enumerate(iterator, start=1):
                _merge_day(accumulators, daily)
                daily_summaries.append(daily)
                if progress_every > 0 and (completed % progress_every == 0 or completed == len(products)):
                    print(f"classified all-energy MINPA days: {completed}/{len(products)}", flush=True)

    centers = grid.min_rm + (np.arange(grid.size, dtype=float) + 0.5) * grid.step_rm
    edges = grid.min_rm + np.arange(grid.size + 1, dtype=float) * grid.step_rm
    y_indices, y_centers = select_y_grid_centers(
        grid, y_min_rm=y_min_rm, y_max_rm=y_max_rm
    )
    projection_suffix = f"y_{_coordinate_slug(y_min_rm)}_{_coordinate_slug(y_max_rm)}"
    projection_description = (
        f"record-weighted aggregation over Y grid centers in [{y_min_rm:g}, {y_max_rm:g}] Rm"
    )
    common = {
        "axis_x_rm": centers,
        "axis_y_rm": centers,
        "axis_z_rm": centers,
        "grid_edges_rm": edges,
        "mars_radius_km": np.asarray(grid.mars_radius_km),
        "energy_selection": np.asarray("all NV_MSO_2 energy bins; no >1 keV threshold"),
        "quality_policy": np.asarray("NaN invalid; NV_MSO_2 bits 2-5 invalid; bit 1 retained"),
        "projection_method": np.asarray(projection_description),
        "projection_y_indices": y_indices,
        "projection_y_centers_rm": y_centers,
        "projection_y_min_rm": np.asarray(y_min_rm),
        "projection_y_max_rm": np.asarray(y_max_rm),
    }
    classified_xz: dict[str, dict[str, np.ndarray]] = {}
    data_products: list[str] = []
    for category in CATEGORY_IDS:
        sparse, xz = _products_from_dense(
            accumulators[category],
            grid,
            y_min_rm=y_min_rm,
            y_max_rm=y_max_rm,
            species_names=ALL_ENERGY_SPECIES,
        )
        classified_xz[category] = xz
        metadata = {
            **common,
            "category_id": np.asarray(category),
            "category_definition": np.asarray(CATEGORY_DEFINITIONS[category]["plain"]),
        }
        stem = f"tw1_minpa_all_energy_{category}"
        sparse_path = output_root / "data" / f"{stem}_mse_3d_cell_means_valid.npz"
        xz_path = output_root / "data" / f"{stem}_mse_xz_{projection_suffix}_means_valid.npz"
        _atomic_npz(sparse_path, **metadata, **sparse)
        _atomic_npz(xz_path, **metadata, **xz)
        data_products.extend([str(sparse_path), str(xz_path)])

    plots, color_limits = (
        create_classified_plots(
            classified_xz,
            edges,
            output_root / "figures",
            y_min_rm=y_min_rm,
            y_max_rm=y_max_rm,
            species_names=ALL_ENERGY_SPECIES,
            species_labels=ALL_ENERGY_SPECIES_LABELS,
            product_slug="tw1_minpa_all_energy",
            energy_title="all energy bins",
        )
        if write_plots
        else ([], {})
    )

    category_summary: dict[str, Any] = {}
    for category in CATEGORY_IDS:
        category_summary[category] = {
            "definition": CATEGORY_DEFINITIONS[category]["plain"],
            "species": {},
        }
        values = classified_xz[category]
        for species in ALL_ENERGY_SPECIES:
            density = values[f"{species}_mean_density_cm3"]
            speed = values[f"{species}_mean_speed_km_s"]
            flux = values[f"{species}_density_speed_flux_cm2_s"]
            category_summary[category]["species"][species] = {
                "density_records": int(np.sum(values[f"{species}_density_count"], dtype=np.int64)),
                "velocity_records": int(np.sum(values[f"{species}_velocity_count"], dtype=np.int64)),
                "xz_pixels_with_density": int(np.count_nonzero(np.isfinite(density))),
                "xz_pixels_with_velocity": int(np.count_nonzero(np.isfinite(speed))),
                "mean_density_positive_min_cm3": _finite_percentile(density, 0.0, positive=True),
                "mean_density_positive_max_cm3": _finite_percentile(density, 100.0, positive=True),
                "mean_speed_positive_min_km_s": _finite_percentile(speed, 0.0, positive=True),
                "mean_speed_positive_max_km_s": _finite_percentile(speed, 100.0, positive=True),
                "density_speed_flux_positive_min_cm2_s": _finite_percentile(flux, 0.0, positive=True),
                "density_speed_flux_positive_max_cm2_s": _finite_percentile(flux, 100.0, positive=True),
            }

    summary: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "momag_root": str(momag_root),
        "r_mso2mse_root": str(r_root),
        "energy_selection": "all NV_MSO_2 energy bins; no >1 keV threshold",
        "input_product_days": processable_day_count + len(excluded),
        "processable_geometry_days": processable_day_count,
        "days_processed": len(daily_summaries),
        "excluded_geometry_days": len(excluded),
        "excluded_canonical_utc_rows": int(
            sum(item["canonical_utc_rows_excluded"] for item in excluded)
        ),
        "excluded_geometry": excluded,
        "workers": int(workers),
        "rows_total": int(sum(day.rows_total for day in daily_summaries)),
        "rows_in_grid": int(sum(day.rows_in_grid for day in daily_summaries)),
        "xb_valid_rows": int(sum(day.xb_valid_rows for day in daily_summaries)),
        "xb_valid_in_grid_rows": int(sum(day.xb_valid_in_grid_rows for day in daily_summaries)),
        "xb_category_rows": {
            category: int(sum(day.category_rows[category] for day in daily_summaries))
            for category in CATEGORY_IDS
        },
        "xb_category_in_grid_rows": {
            category: int(sum(day.category_in_grid_rows[category] for day in daily_summaries))
            for category in CATEGORY_IDS
        },
        "xb_valid_but_outside_both_cones_rows": int(
            sum(day.xb_valid_but_outside_rows for day in daily_summaries)
        ),
        "xb_valid_in_grid_but_outside_both_cones_rows": int(
            sum(day.xb_valid_in_grid_but_outside_rows for day in daily_summaries)
        ),
        "classification": {
            "vector_chain": "Xb_MSE = R_MSO2MSE * Xb_MSO",
            "source_axis": "canonical UTC rows of NV_MSO_2.Xb_MSO",
            "half_angle_deg": 45.0,
            "boundary": "inclusive",
            "categories_are_mutually_exclusive": True,
            "outside_both_cones": "reported in coverage, excluded from both statistics",
        },
        "filters": {
            "non_finite": "invalid for affected moment; non-finite Xb is unclassified",
            "quality": "NV_MSO_2 bits 2-5 invalid; bit 1 retained; unavailable flags invalid",
            "processing": "processing_status_code == 1",
            "rotation": f"finite nearest R_MSO2MSE within {rotation_gap_s:g} s",
        },
        "aggregation": {
            "three_dimensional": "per-record arithmetic mean inside each 0.1 Rm cell",
            "xz_projection": projection_description,
            "projection_y_indices": y_indices.tolist(),
            "projection_y_centers_rm": y_centers.tolist(),
            "projection_y_bounds_rm": [y_min_rm, y_max_rm],
            "density": "mean of per-record density, not a sum over epochs",
            "speed": "mean of per-record Euclidean MSE speed magnitude",
            "flux": "mean density_cm-3 * mean speed_km_s * 1e5, in cm-2 s-1",
        },
        "grid": grid.as_dict(),
        "validation_maxima": {
            name: float(np.nanmax([day.validation[name] for day in daily_summaries]))
            for name in (
                "max_abs_xb_mse_unit_norm_error",
                "rotation_orthonormal_max_error",
                "rotation_determinant_max_error",
                "max_abs_rotation_match_dt_s",
                "cone_overlap_rows",
            )
        },
        "shared_plot_color_limits": {
            key: list(value) for key, value in color_limits.items()
        },
        "data_products": data_products,
        "plot_products": plots,
        "categories": category_summary,
        "daily_summaries": [
            {
                "date": day.date,
                "source_file": day.source_file,
                "rows_total": day.rows_total,
                "rows_in_grid": day.rows_in_grid,
                "xb_valid_rows": day.xb_valid_rows,
                "xb_valid_in_grid_rows": day.xb_valid_in_grid_rows,
                "category_rows": day.category_rows,
                "category_in_grid_rows": day.category_in_grid_rows,
                "xb_valid_but_outside_rows": day.xb_valid_but_outside_rows,
                "xb_valid_in_grid_but_outside_rows": day.xb_valid_in_grid_but_outside_rows,
                "validation": day.validation,
            }
            for day in daily_summaries
        ],
    }
    _atomic_json(
        output_root
        / "logs"
        / f"tw1_minpa_all_energy_mse_xz_xb_classified_{projection_suffix}_summary.json",
        summary,
    )
    return summary
