"""Rebin classified Tianwen-1 MINPA MSE sufficient statistics."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .tw1_mse_grid_records import MseGridSpec
from .tw1_mse_xz_fov_stats import (
    CATEGORY_DEFINITIONS,
    CATEGORY_IDS,
    _coordinate_slug,
    _products_from_dense,
    create_classified_plots,
    select_y_grid_centers,
)
from .tw1_mse_xz_stats import (
    SPECIES,
    STAT_NAMES,
    _atomic_json,
    _atomic_npz,
    _finite_percentile,
    _safe_mean,
)


def _validate_aligned_grids(source_grid: MseGridSpec, target_grid: MseGridSpec) -> int:
    """Return the integer coarsening ratio for two edge-aligned grids."""
    if not (
        np.isclose(source_grid.min_rm, target_grid.min_rm)
        and np.isclose(source_grid.max_rm, target_grid.max_rm)
        and np.isclose(source_grid.mars_radius_km, target_grid.mars_radius_km)
    ):
        raise ValueError("Source and target grids must share bounds and Mars radius")
    ratio_float = target_grid.step_rm / source_grid.step_rm
    ratio = int(round(ratio_float))
    if ratio < 1 or not np.isclose(ratio_float, ratio, rtol=0.0, atol=1.0e-12):
        raise ValueError("Target grid step must be an integer multiple of source grid step")
    if source_grid.size != target_grid.size * ratio:
        raise ValueError("Source grid size must be exactly divisible by the target grid size")
    return ratio


def rebin_sparse_sufficient_statistics(
    source: Mapping[str, np.ndarray],
    source_grid: MseGridSpec,
    target_grid: MseGridSpec,
) -> dict[str, dict[str, np.ndarray]]:
    """Sum sparse sufficient statistics into an aligned coarser 3-D grid."""
    ratio = _validate_aligned_grids(source_grid, target_grid)
    source_indices = [np.asarray(source[name], dtype=np.int64) for name in ("ix", "iy", "iz")]
    row_count = source_indices[0].size
    if any(index.shape != (row_count,) for index in source_indices):
        raise ValueError("Sparse source coordinate arrays must be one-dimensional and equal length")
    if any(np.any((index < 0) | (index >= source_grid.size)) for index in source_indices):
        raise ValueError("Sparse source coordinate lies outside the declared source grid")

    target_indices = [index // ratio for index in source_indices]
    target_flat = np.ravel_multi_index(tuple(target_indices), target_grid.shape)
    rebinned: dict[str, dict[str, np.ndarray]] = {}
    for species in SPECIES:
        rebinned[species] = {}
        for name in STAT_NAMES:
            key = f"{species}_{name}"
            values = np.asarray(source[key])
            if values.shape != (row_count,):
                raise ValueError(f"{key} does not match sparse source coordinates")
            dtype = np.int64 if name.endswith("count") else np.float64
            target = np.zeros(target_grid.size**3, dtype=dtype)
            np.add.at(target, target_flat, values.astype(dtype, copy=False))
            rebinned[species][name] = target
    return rebinned


def project_sparse_statistics_to_coarse_xz(
    source: Mapping[str, np.ndarray],
    source_grid: MseGridSpec,
    target_grid: MseGridSpec,
    *,
    y_min_rm: float,
    y_max_rm: float,
) -> dict[str, np.ndarray]:
    """Project an exact source-grid Y slab while coarsening only X and Z."""
    ratio = _validate_aligned_grids(source_grid, target_grid)
    source_y_indices, _ = select_y_grid_centers(
        source_grid, y_min_rm=y_min_rm, y_max_rm=y_max_rm
    )
    ix = np.asarray(source["ix"], dtype=np.int64)
    iy = np.asarray(source["iy"], dtype=np.int64)
    iz = np.asarray(source["iz"], dtype=np.int64)
    row_count = ix.size
    if iy.shape != (row_count,) or iz.shape != (row_count,):
        raise ValueError("Sparse source coordinate arrays must be one-dimensional and equal length")
    selected = np.isin(iy, source_y_indices)
    flat_xz = (ix[selected] // ratio) * target_grid.size + (iz[selected] // ratio)

    xz: dict[str, np.ndarray] = {}
    for species in SPECIES:
        for name in STAT_NAMES:
            key = f"{species}_{name}"
            values = np.asarray(source[key])
            if values.shape != (row_count,):
                raise ValueError(f"{key} does not match sparse source coordinates")
            target = np.zeros(
                target_grid.size**2,
                dtype=np.int64 if name.endswith("count") else np.float64,
            )
            np.add.at(target, flat_xz, values[selected])
            xz[key] = target.reshape(target_grid.size, target_grid.size)
        xz[f"{species}_mean_density_cm3"] = _safe_mean(
            xz[f"{species}_density_sum_cm3"], xz[f"{species}_density_count"]
        )
        for output, numerator in (
            ("mean_vx_km_s", "vx_sum_km_s"),
            ("mean_vy_km_s", "vy_sum_km_s"),
            ("mean_vz_km_s", "vz_sum_km_s"),
            ("mean_speed_km_s", "speed_sum_km_s"),
        ):
            xz[f"{species}_{output}"] = _safe_mean(
                xz[f"{species}_{numerator}"], xz[f"{species}_velocity_count"]
            )
        xz[f"{species}_density_speed_flux_cm2_s"] = (
            xz[f"{species}_mean_density_cm3"]
            * xz[f"{species}_mean_speed_km_s"]
            * 1.0e5
        )
    return xz


def _read_source_summary(source_root: Path) -> tuple[Path | None, dict[str, Any]]:
    summaries = sorted((source_root / "logs").glob("tw1_minpa_mse_xz_fov_classified_*_summary.json"))
    if len(summaries) != 1:
        return None, {}
    path = summaries[0]
    with path.open("r", encoding="utf-8") as handle:
        return path, json.load(handle)


def _category_summary(classified_xz: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in CATEGORY_IDS:
        result[category] = {"definition": CATEGORY_DEFINITIONS[category]["plain"], "species": {}}
        for species in SPECIES:
            values = classified_xz[category]
            density = values[f"{species}_mean_density_cm3"]
            speed = values[f"{species}_mean_speed_km_s"]
            flux = values[f"{species}_density_speed_flux_cm2_s"]
            result[category]["species"][species] = {
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
    return result


def run_rebin_classified_statistics(
    source_root: Path,
    output_root: Path,
    *,
    source_grid: MseGridSpec | None = None,
    target_grid: MseGridSpec | None = None,
    y_min_rm: float = -0.5,
    y_max_rm: float = 0.5,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Rebin 0.1 Rm classified products and generate 0.2 Rm X-Z products."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    source_grid = MseGridSpec(step_rm=0.1) if source_grid is None else source_grid
    target_grid = MseGridSpec(step_rm=0.2) if target_grid is None else target_grid
    ratio = _validate_aligned_grids(source_grid, target_grid)
    target_centers = target_grid.min_rm + (np.arange(target_grid.size) + 0.5) * target_grid.step_rm
    target_edges = target_grid.min_rm + np.arange(target_grid.size + 1) * target_grid.step_rm
    source_y_indices, source_y_centers = select_y_grid_centers(
        source_grid, y_min_rm=y_min_rm, y_max_rm=y_max_rm
    )
    projection_suffix = f"y_{_coordinate_slug(y_min_rm)}_{_coordinate_slug(y_max_rm)}"
    source_summary_path, source_summary = _read_source_summary(source_root)

    classified_xz: dict[str, dict[str, np.ndarray]] = {}
    data_products: list[str] = []
    conservation: dict[str, Any] = {}
    for category in CATEGORY_IDS:
        source_path = source_root / "data" / f"tw1_minpa_{category}_mse_3d_cell_means_valid.npz"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with np.load(source_path) as source:
            expected_edges = source_grid.min_rm + np.arange(source_grid.size + 1) * source_grid.step_rm
            if not np.allclose(source["grid_edges_rm"], expected_edges, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"Source grid metadata does not describe a {source_grid.step_rm:g} Rm grid")
            accumulators = rebin_sparse_sufficient_statistics(source, source_grid, target_grid)
            xz = project_sparse_statistics_to_coarse_xz(
                source,
                source_grid,
                target_grid,
                y_min_rm=y_min_rm,
                y_max_rm=y_max_rm,
            )
            conservation[category] = {}
            for species in SPECIES:
                conservation[category][species] = {}
                for name in STAT_NAMES:
                    source_total = np.sum(source[f"{species}_{name}"], dtype=np.float64)
                    target_total = np.sum(accumulators[species][name], dtype=np.float64)
                    if not np.isclose(source_total, target_total, rtol=1.0e-13, atol=1.0e-9):
                        raise ValueError(f"Rebin conservation failed for {category} {species} {name}")
                    conservation[category][species][name] = {
                        "source_total": float(source_total),
                        "target_total": float(target_total),
                    }

        sparse, _ = _products_from_dense(accumulators, target_grid)
        classified_xz[category] = xz
        metadata = {
            "axis_x_rm": target_centers,
            "axis_y_rm": target_centers,
            "axis_z_rm": target_centers,
            "grid_edges_rm": target_edges,
            "mars_radius_km": np.asarray(target_grid.mars_radius_km),
            "source_grid_step_rm": np.asarray(source_grid.step_rm),
            "target_grid_step_rm": np.asarray(target_grid.step_rm),
            "source_product": np.asarray(str(source_path)),
            "rebin_method": np.asarray("sum counts and physical sums, then recompute record-weighted means"),
            "quality_policy": np.asarray("inherited: NaN invalid; NV_MSO_2 bits 2-5 invalid; bit 1 retained"),
            "projection_method": np.asarray(
                "select source-grid Y cells inside the requested slab, then coarsen X and Z"
            ),
            "projection_grid_step_rm": np.asarray(source_grid.step_rm),
            "projection_y_indices": source_y_indices,
            "projection_y_centers_rm": source_y_centers,
            "projection_y_min_rm": np.asarray(y_min_rm),
            "projection_y_max_rm": np.asarray(y_max_rm),
            "category_id": np.asarray(category),
            "category_definition": np.asarray(CATEGORY_DEFINITIONS[category]["plain"]),
        }
        sparse_path = output_root / "data" / f"tw1_minpa_{category}_mse_3d_grid0p2_cell_means_valid.npz"
        xz_path = output_root / "data" / f"tw1_minpa_{category}_mse_xz_grid0p2_{projection_suffix}_means_valid.npz"
        _atomic_npz(sparse_path, **metadata, **sparse)
        _atomic_npz(xz_path, **metadata, **xz)
        data_products.extend([str(sparse_path), str(xz_path)])

    plots, color_limits = (
        create_classified_plots(
            classified_xz,
            target_edges,
            output_root / "figures",
            y_min_rm=y_min_rm,
            y_max_rm=y_max_rm,
            grid_step_rm=target_grid.step_rm,
        )
        if write_plots
        else ([], {})
    )
    summary: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "source_root": str(source_root),
        "source_summary": None if source_summary_path is None else str(source_summary_path),
        "output_root": str(output_root),
        "processing_history": (
            f"rebinned edge-aligned {source_grid.step_rm:g} Rm sufficient statistics to "
            f"{target_grid.step_rm:g} Rm by summing counts and physical sums; means and flux recomputed"
        ),
        "source_grid": source_grid.as_dict(),
        "grid": target_grid.as_dict(),
        "coarsening_ratio": ratio,
        "classification": source_summary.get("classification"),
        "filters": source_summary.get("filters"),
        "aggregation": {
            "three_dimensional": "record-weighted means recomputed after sufficient-statistic rebinning",
            "xz_projection": (
                f"source {source_grid.step_rm:g} Rm Y cells satisfying {y_min_rm:g} <= Y_MSE <= "
                f"{y_max_rm:g} Rm, followed by {target_grid.step_rm:g} Rm X-Z coarsening"
            ),
            "projection_selection_stage": "before 3-D coarsening, preserving the exact source-grid Y slab",
            "projection_grid_step_rm": source_grid.step_rm,
            "projection_y_indices": source_y_indices.tolist(),
            "projection_y_centers_rm": source_y_centers.tolist(),
            "projection_y_bounds_rm": [y_min_rm, y_max_rm],
            "density": "sum rebinned per-record density values / rebinned valid density record count",
            "velocity": "sum rebinned MSE velocity components / rebinned valid velocity record count",
            "speed": "sum rebinned per-record Euclidean MSE speed magnitudes / valid velocity record count",
            "flux": "projected mean density_cm-3 * projected mean speed_km_s * 1e5, in cm-2 s-1",
        },
        "conservation": conservation,
        "shared_plot_color_limits": {key: list(value) for key, value in color_limits.items()},
        "data_products": data_products,
        "plot_products": plots,
        "categories": _category_summary(classified_xz),
    }
    summary_path = (
        output_root
        / "logs"
        / f"tw1_minpa_mse_xz_fov_classified_grid0p2_{projection_suffix}_summary.json"
    )
    _atomic_json(summary_path, summary)
    return summary
