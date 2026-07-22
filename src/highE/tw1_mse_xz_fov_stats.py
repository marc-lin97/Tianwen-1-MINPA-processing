"""Classify Tianwen-1 MINPA records by body +X axis and map MSE X-Z means."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat

from .coordinates import apply_rotation, interpolate_matrices_nearest
from .tw1_minpa import load_tw1_mso2mse, tw1_rotation_file
from .tw1_mse_grid_records import (
    DailyGridChunk,
    DailyProductPair,
    MseGridSpec,
    discover_daily_pairs,
    load_daily_grid_chunk,
    prepare_rotation_time_axis,
)
from .tw1_mse_xz_stats import (
    SPECIES,
    SPECIES_LABELS,
    STAT_NAMES,
    _atomic_json,
    _atomic_npz,
    _finite_percentile,
    _plot_map,
    _safe_mean,
)


CATEGORY_DEFINITIONS = {
    "xb_plus_z_le45": {
        "label": r"$\angle(+X_b,+Z_{MSE})\leq45^\circ$",
        "plain": "angle(+Xb_MSE,+Z_MSE) <= 45 deg",
    },
    "xb_minus_z_le45": {
        "label": r"$\angle(+X_b,-Z_{MSE})\leq45^\circ$",
        "plain": "angle(+Xb_MSE,-Z_MSE) <= 45 deg",
    },
}
CATEGORY_IDS = tuple(CATEGORY_DEFINITIONS)


@dataclass(frozen=True)
class XbClassification:
    xb_mse: np.ndarray
    angle_plus_z_deg: np.ndarray
    angle_minus_z_deg: np.ndarray
    valid: np.ndarray
    categories: dict[str, np.ndarray]


@dataclass(frozen=True)
class DailyClassifiedStatistics:
    date: str
    rows_total: int
    rows_in_grid: int
    xb_valid_rows: int
    category_rows: dict[str, int]
    unclassified_xb_rows: int
    xb_validation: dict[str, float]
    aggregates: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]


def _nearest_rows(source_time: np.ndarray, target_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_time, dtype=float).reshape(-1)
    target = np.asarray(target_time, dtype=float).reshape(-1)
    if source.size < 1 or not np.all(np.isfinite(source)):
        raise ValueError("Xb_MSO time axis is empty or non-finite")
    order = np.argsort(source, kind="stable")
    source = source[order]
    if source.size > 1 and np.any(np.diff(source) <= 0.0):
        raise ValueError("Xb_MSO time axis is not strictly increasing")
    insertion = np.searchsorted(source, target)
    if source.size == 1:
        nearest_sorted = np.zeros(target.size, dtype=np.int64)
    else:
        insertion = np.clip(insertion, 1, source.size - 1)
        left = insertion - 1
        choose_right = np.abs(source[insertion] - target) < np.abs(source[left] - target)
        nearest_sorted = np.where(choose_right, insertion, left)
    return order[nearest_sorted], target - source[nearest_sorted]


def classify_xb_mse(xb_mse: np.ndarray, *, half_angle_deg: float = 45.0) -> XbClassification:
    """Classify finite +Xb_MSE unit directions into the +/-Z_MSE cones."""
    vectors = np.asarray(xb_mse, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"Expected Xb_MSE shape (N,3), received {vectors.shape}")
    if not (0.0 < half_angle_deg < 90.0):
        raise ValueError("half_angle_deg must lie strictly between 0 and 90 degrees")
    norm = np.linalg.norm(vectors, axis=1)
    valid = np.all(np.isfinite(vectors), axis=1) & np.isfinite(norm) & (norm > 0.0)
    unit_z = np.full(norm.shape, np.nan, dtype=float)
    unit_z[valid] = vectors[valid, 2] / norm[valid]
    angle_plus = np.full(norm.shape, np.nan, dtype=float)
    angle_minus = np.full(norm.shape, np.nan, dtype=float)
    angle_plus[valid] = np.degrees(np.arccos(np.clip(unit_z[valid], -1.0, 1.0)))
    angle_minus[valid] = np.degrees(np.arccos(np.clip(-unit_z[valid], -1.0, 1.0)))
    plus = valid & (angle_plus <= half_angle_deg)
    minus = valid & (angle_minus <= half_angle_deg)
    if np.any(plus & minus):
        raise ValueError("The +Z and -Z Xb cones unexpectedly overlap")
    return XbClassification(
        xb_mse=vectors,
        angle_plus_z_deg=angle_plus,
        angle_minus_z_deg=angle_minus,
        valid=valid,
        categories={"xb_plus_z_le45": plus, "xb_minus_z_le45": minus},
    )


def load_and_transform_xb_mse(
    pair: DailyProductPair,
    target_epoch: np.ndarray,
    *,
    r_root: Path,
    nv_mso2_root: Path,
    xb_time_tolerance_s: float = 0.001,
    rotation_gap_s: float = 8.0,
    comparison_max_angle_deg: float = 0.1,
) -> tuple[XbClassification, dict[str, float]]:
    """Load the epoch-native +Xb_MSE axis and validate it by an independent chain."""
    target = np.asarray(target_epoch, dtype=float).reshape(-1)
    nv_path = Path(nv_mso2_root) / f"NV_{pair.date}.mat"
    nv = loadmat(nv_path, variable_names=["Xb_MSO"], squeeze_me=True, struct_as_record=False)
    xb_source = np.asarray(nv["Xb_MSO"], dtype=float)
    if xb_source.ndim != 2 or xb_source.shape[1] != 4:
        raise ValueError(f"Unexpected Xb_MSO shape in {nv_path}: {xb_source.shape}")
    nearest, xb_dt = _nearest_rows(xb_source[:, 0], target)
    xb_mso = xb_source[nearest, 1:4]
    xb_epoch_matched = np.abs(xb_dt) <= xb_time_tolerance_s

    rotation_path = tw1_rotation_file(pair.date, Path(r_root))
    rotation = load_tw1_mso2mse(rotation_path)
    rotation_time, matrices, rotation_info = prepare_rotation_time_axis(
        rotation["time_s"], rotation["R_mso_to_mse"]
    )
    matched_rotation, rotation_dt = interpolate_matrices_nearest(rotation_time, matrices, target)
    independently_transformed = apply_rotation(matched_rotation, xb_mso)
    transform_valid = (
        xb_epoch_matched
        & (np.abs(rotation_dt) <= rotation_gap_s)
        & np.all(np.isfinite(matched_rotation), axis=(1, 2))
        & np.all(np.isfinite(xb_mso), axis=1)
        & np.all(np.isfinite(independently_transformed), axis=1)
    )

    axes_fields = ["epoch_unix_s"] + [f"inst_x_axis_mse_{axis}" for axis in "xyz"]
    daily = loadmat(pair.oplus, variable_names=axes_fields, squeeze_me=True, struct_as_record=False)
    daily_epoch = np.asarray(daily["epoch_unix_s"], dtype=float).reshape(-1)
    if not np.array_equal(target, daily_epoch, equal_nan=True):
        raise ValueError(f"Stored +Xb_MSE epoch differs from daily product epoch for {pair.date}")
    stored = np.column_stack(
        [np.asarray(daily[f"inst_x_axis_mse_{axis}"], dtype=float).reshape(-1) for axis in "xyz"]
    )
    axis_valid = (
        (np.abs(rotation_dt) <= rotation_gap_s)
        & np.all(np.isfinite(matched_rotation), axis=(1, 2))
        & np.all(np.isfinite(stored), axis=1)
    )
    xb_mse = stored.copy()
    xb_mse[~axis_valid] = np.nan
    classification = classify_xb_mse(xb_mse)

    compare = transform_valid & axis_valid
    comparison_error = (
        0.0
        if not np.any(compare)
        else float(np.max(np.abs(independently_transformed[compare] - stored[compare])))
    )
    if np.any(compare):
        transformed_unit = independently_transformed[compare] / np.linalg.norm(
            independently_transformed[compare], axis=1, keepdims=True
        )
        stored_unit = stored[compare] / np.linalg.norm(stored[compare], axis=1, keepdims=True)
        angular_difference = np.degrees(
            np.arccos(np.clip(np.sum(transformed_unit * stored_unit, axis=1), -1.0, 1.0))
        )
        maximum_angular_difference = float(np.max(angular_difference))
        independent_classes = classify_xb_mse(independently_transformed[compare])
        stored_classes = classify_xb_mse(stored[compare])
        class_disagreements = float(
            sum(
                np.count_nonzero(
                    independent_classes.categories[category] != stored_classes.categories[category]
                )
                for category in CATEGORY_IDS
            )
        )
    else:
        maximum_angular_difference = 0.0
        class_disagreements = 0.0
    if maximum_angular_difference > comparison_max_angle_deg:
        raise ValueError(
            f"Independent Xb_MSO->MSE chain differs from the epoch-native daily +Xb axis by "
            f"{maximum_angular_difference:.6g} deg for {pair.date}"
        )
    norm_error = (
        0.0
        if not np.any(axis_valid)
        else float(np.max(np.abs(np.linalg.norm(xb_mse[axis_valid], axis=1) - 1.0)))
    )
    if norm_error > 1.0e-8:
        raise ValueError(f"Xb_MSE unit-vector error {norm_error:.6g} for {pair.date}")
    validation = {
        "xb_epoch_matched_rows": float(np.count_nonzero(xb_epoch_matched)),
        "xb_transform_valid_rows": float(np.count_nonzero(transform_valid)),
        "xb_epoch_native_axis_valid_rows": float(np.count_nonzero(axis_valid)),
        "xb_stored_comparison_rows": float(np.count_nonzero(compare)),
        "max_abs_xb_epoch_match_dt_s": (
            float(np.max(np.abs(xb_dt[xb_epoch_matched]))) if np.any(xb_epoch_matched) else math.nan
        ),
        "max_abs_rotation_match_dt_s_for_valid_xb": (
            float(np.max(np.abs(rotation_dt[transform_valid]))) if np.any(transform_valid) else math.nan
        ),
        "max_abs_xb_mse_stored_difference": comparison_error,
        "max_xb_independent_chain_angular_difference_deg": maximum_angular_difference,
        "xb_independent_chain_category_disagreements": class_disagreements,
        "max_abs_xb_mse_unit_norm_error": norm_error,
        **rotation_info,
    }
    return classification, validation


def _valid_species_masks(chunk: DailyGridChunk, species_id: str) -> tuple[np.ndarray, np.ndarray]:
    values = chunk.species[species_id]
    density = np.asarray(values["density_cm3"], dtype=float)
    velocity = np.column_stack([np.asarray(values[f"v_mse_{axis}_km_s"], dtype=float) for axis in "xyz"])
    bitmask = np.asarray(values["quality_flag_bitmask"], dtype=np.uint32)
    base = (
        (chunk.flat_bin >= 0)
        & (chunk.common["r_matrix_within_tolerance_flag"] == 1)
        & (values["processing_status_code"] == 1)
        & (values["quality_flag_available_flag"] == 1)
        & (values["quality_flag_epoch_matched_flag"] == 1)
        & ((bitmask & np.uint32(0b11110)) == 0)
    )
    density_ok = base & (values["density_valid_flag"] == 1) & np.isfinite(density) & (density > 0.0)
    velocity_ok = density_ok & (values["velocity_valid_flag"] == 1) & np.all(np.isfinite(velocity), axis=1)
    return density_ok, velocity_ok


def _reduce_selected_rows(
    flat_bin: np.ndarray,
    density: np.ndarray,
    velocity: np.ndarray,
    density_mask: np.ndarray,
    velocity_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.union1d(flat_bin[density_mask], flat_bin[velocity_mask]).astype(np.int64)
    statistics = np.zeros((len(STAT_NAMES), selected.size), dtype=np.float64)
    if selected.size == 0:
        return selected, statistics
    density_indices = np.searchsorted(selected, flat_bin[density_mask])
    np.add.at(statistics[0], density_indices, 1.0)
    np.add.at(statistics[1], density_indices, density[density_mask])
    velocity_indices = np.searchsorted(selected, flat_bin[velocity_mask])
    np.add.at(statistics[2], velocity_indices, 1.0)
    chosen_velocity = velocity[velocity_mask]
    for offset in range(3):
        np.add.at(statistics[3 + offset], velocity_indices, chosen_velocity[:, offset])
    np.add.at(statistics[6], velocity_indices, np.linalg.norm(chosen_velocity, axis=1))
    return selected, statistics


def aggregate_classified_daily_chunk(
    chunk: DailyGridChunk,
    classification: XbClassification,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Reduce one day into sparse sufficient statistics for both Xb cones."""
    if classification.valid.shape != chunk.flat_bin.shape:
        raise ValueError("Xb classification length differs from the daily particle rows")
    output: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for category in CATEGORY_IDS:
        output[category] = {}
        category_mask = classification.categories[category]
        for output_species, chunk_species in (("oplus", "Oplus"), ("o2plus", "O2plus")):
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


def _process_day(arguments: tuple[DailyProductPair, str, str, MseGridSpec, float, float]) -> DailyClassifiedStatistics:
    pair, r_root_text, nv_root_text, grid, xb_tolerance_s, rotation_gap_s = arguments
    r_root = Path(r_root_text)
    nv_root = Path(nv_root_text)
    chunk = load_daily_grid_chunk(
        pair,
        r_root,
        grid,
        nv_mso2_root=nv_root,
        quality_flag_tolerance_s=xb_tolerance_s,
        rotation_gap_s=rotation_gap_s,
    )
    classification, validation = load_and_transform_xb_mse(
        pair,
        chunk.common["epoch_unix_s"],
        r_root=r_root,
        nv_mso2_root=nv_root,
        xb_time_tolerance_s=xb_tolerance_s,
        rotation_gap_s=rotation_gap_s,
    )
    categories = {name: int(np.count_nonzero(mask)) for name, mask in classification.categories.items()}
    classified = np.logical_or.reduce(list(classification.categories.values()))
    return DailyClassifiedStatistics(
        date=pair.date,
        rows_total=chunk.rows_total,
        rows_in_grid=chunk.rows_in_grid,
        xb_valid_rows=int(np.count_nonzero(classification.valid)),
        category_rows=categories,
        unclassified_xb_rows=int(np.count_nonzero(classification.valid & ~classified)),
        xb_validation=validation,
        aggregates=aggregate_classified_daily_chunk(chunk, classification),
    )


def _initialize_accumulators(
    cell_count: int,
    *,
    species_names: tuple[str, ...] = SPECIES,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    return {
        category: {
            species: {
                name: np.zeros(cell_count, dtype=np.int64 if name.endswith("count") else np.float64)
                for name in STAT_NAMES
            }
            for species in species_names
        }
        for category in CATEGORY_IDS
    }


def _merge_daily(
    accumulators: dict[str, dict[str, dict[str, np.ndarray]]],
    daily: DailyClassifiedStatistics,
) -> None:
    for category in CATEGORY_IDS:
        for species in SPECIES:
            flat, statistics = daily.aggregates[category][species]
            for row, name in enumerate(STAT_NAMES):
                if name.endswith("count"):
                    np.add.at(accumulators[category][species][name], flat, statistics[row].astype(np.int64))
                else:
                    np.add.at(accumulators[category][species][name], flat, statistics[row])


def _products_from_dense(
    category_accum: dict[str, dict[str, np.ndarray]],
    grid: MseGridSpec,
    *,
    y_min_rm: float | None = None,
    y_max_rm: float | None = None,
    species_names: tuple[str, ...] = SPECIES,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    occupied = np.zeros(grid.size**3, dtype=bool)
    for species in species_names:
        occupied |= category_accum[species]["density_count"] > 0
        occupied |= category_accum[species]["velocity_count"] > 0
    flat = np.flatnonzero(occupied)
    ix, iy, iz = np.unravel_index(flat, grid.shape)
    sparse: dict[str, np.ndarray] = {
        "flat_bin": flat.astype(np.int32),
        "ix": ix.astype(np.int16),
        "iy": iy.astype(np.int16),
        "iz": iz.astype(np.int16),
    }
    xz: dict[str, np.ndarray] = {}
    y_indices, _ = select_y_grid_centers(grid, y_min_rm=y_min_rm, y_max_rm=y_max_rm)
    for species in species_names:
        values = category_accum[species]
        for name in STAT_NAMES:
            sparse[f"{species}_{name}"] = values[name][flat]
            xz[f"{species}_{name}"] = values[name].reshape(grid.shape)[:, y_indices, :].sum(axis=1)
        sparse[f"{species}_mean_density_cm3"] = _safe_mean(
            sparse[f"{species}_density_sum_cm3"], sparse[f"{species}_density_count"]
        )
        xz[f"{species}_mean_density_cm3"] = _safe_mean(
            xz[f"{species}_density_sum_cm3"], xz[f"{species}_density_count"]
        )
        for output, numerator in (
            ("mean_vx_km_s", "vx_sum_km_s"),
            ("mean_vy_km_s", "vy_sum_km_s"),
            ("mean_vz_km_s", "vz_sum_km_s"),
            ("mean_speed_km_s", "speed_sum_km_s"),
        ):
            sparse[f"{species}_{output}"] = _safe_mean(
                sparse[f"{species}_{numerator}"], sparse[f"{species}_velocity_count"]
            )
            xz[f"{species}_{output}"] = _safe_mean(
                xz[f"{species}_{numerator}"], xz[f"{species}_velocity_count"]
            )
        sparse[f"{species}_density_speed_flux_cm2_s"] = (
            sparse[f"{species}_mean_density_cm3"]
            * sparse[f"{species}_mean_speed_km_s"]
            * 1.0e5
        )
        xz[f"{species}_density_speed_flux_cm2_s"] = (
            xz[f"{species}_mean_density_cm3"]
            * xz[f"{species}_mean_speed_km_s"]
            * 1.0e5
        )
    return sparse, xz


def select_y_grid_centers(
    grid: MseGridSpec,
    *,
    y_min_rm: float | None,
    y_max_rm: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Y indices/centers included in an inclusive MSE projection slab."""
    centers = grid.min_rm + (np.arange(grid.size, dtype=float) + 0.5) * grid.step_rm
    if y_min_rm is None and y_max_rm is None:
        return np.arange(grid.size, dtype=np.int64), centers
    if y_min_rm is None or y_max_rm is None:
        raise ValueError("y_min_rm and y_max_rm must either both be set or both be omitted")
    if not (np.isfinite(y_min_rm) and np.isfinite(y_max_rm) and y_min_rm <= y_max_rm):
        raise ValueError("Y projection bounds must be finite and ordered")
    tolerance = max(1.0, abs(y_min_rm), abs(y_max_rm)) * 1.0e-12
    selected = np.flatnonzero((centers >= y_min_rm - tolerance) & (centers <= y_max_rm + tolerance))
    if selected.size == 0:
        raise ValueError(f"No Y grid centers lie inside [{y_min_rm:g}, {y_max_rm:g}] Rm")
    return selected.astype(np.int64), centers[selected]


def _coordinate_slug(value: float) -> str:
    return ("p" if value >= 0 else "m") + f"{abs(value):g}".replace(".", "p")


def _common_color_limits(
    classified_xz: dict[str, dict[str, np.ndarray]],
    *,
    species_names: tuple[str, ...] = SPECIES,
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for species in species_names:
        variables = {
            "density_log10": [
                np.where(values[f"{species}_mean_density_cm3"] > 0, np.log10(values[f"{species}_mean_density_cm3"]), np.nan)
                for values in classified_xz.values()
            ],
            "speed_log10": [
                np.where(values[f"{species}_mean_speed_km_s"] > 0, np.log10(values[f"{species}_mean_speed_km_s"]), np.nan)
                for values in classified_xz.values()
            ],
            "flux_log10": [
                np.where(
                    values[f"{species}_density_speed_flux_cm2_s"] > 0,
                    np.log10(values[f"{species}_density_speed_flux_cm2_s"]),
                    np.nan,
                )
                for values in classified_xz.values()
            ],
            "vx": [values[f"{species}_mean_vx_km_s"] for values in classified_xz.values()],
            "vy": [values[f"{species}_mean_vy_km_s"] for values in classified_xz.values()],
            "vz": [values[f"{species}_mean_vz_km_s"] for values in classified_xz.values()],
        }
        for variable, arrays in variables.items():
            finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
            if finite.size == 0:
                raise ValueError(f"No finite classified values for {species} {variable}")
            if variable in {"vx", "vy", "vz"}:
                limit = float(np.percentile(np.abs(finite), 98.0))
                if not np.isfinite(limit) or limit <= 0.0:
                    limit = float(np.max(np.abs(finite))) or 1.0
                limits[f"{species}_{variable}"] = (-limit, limit)
            else:
                low, high = np.percentile(finite, [2.0, 98.0])
                if not high > low:
                    high = low + max(abs(low) * 1.0e-6, 1.0e-12)
                limits[f"{species}_{variable}"] = (float(low), float(high))
    return limits


def create_classified_plots(
    classified_xz: dict[str, dict[str, np.ndarray]],
    grid_edges: np.ndarray,
    output_dir: Path,
    *,
    y_min_rm: float | None,
    y_max_rm: float | None,
    grid_step_rm: float | None = None,
    species_names: tuple[str, ...] = SPECIES,
    species_labels: dict[str, str] | None = None,
    product_slug: str = "tw1_minpa",
    energy_title: str = "",
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]]]:
    limits = _common_color_limits(classified_xz, species_names=species_names)
    products: list[dict[str, Any]] = []
    if y_min_rm is None:
        projection_label = "all-Y X-Z projection"
        filename_suffix = "all_y"
    else:
        projection_label = rf"${y_min_rm:g}\leq Y_{{MSE}}\leq {y_max_rm:g}\ R_M$"
        filename_suffix = f"y_{_coordinate_slug(y_min_rm)}_{_coordinate_slug(y_max_rm)}"
    if grid_step_rm is not None:
        projection_label += rf", $\Delta={grid_step_rm:g}\ R_M$"
    if energy_title:
        projection_label += f", {energy_title}"
    labels = SPECIES_LABELS if species_labels is None else species_labels
    for category, xz in classified_xz.items():
        category_label = CATEGORY_DEFINITIONS[category]["label"]
        for species in species_names:
            species_label = labels[species]
            density = xz[f"{species}_mean_density_cm3"]
            speed = xz[f"{species}_mean_speed_km_s"]
            flux = xz[f"{species}_density_speed_flux_cm2_s"]
            definitions = (
                ("density_log10", np.where(density > 0, np.log10(density), np.nan), r"$\log_{10}\langle n\rangle$ (cm$^{-3}$)", "viridis", False),
                ("speed_log10", np.where(speed > 0, np.log10(speed), np.nan), r"$\log_{10}\langle |V|\rangle$ (km s$^{-1}$)", "magma", False),
                ("flux_log10", np.where(flux > 0, np.log10(flux), np.nan), r"$\log_{10}(\langle n\rangle\langle|V|\rangle)$ (cm$^{-2}$ s$^{-1}$)", "cividis", False),
                ("vx", xz[f"{species}_mean_vx_km_s"], r"$\langle V_x\rangle$ (km s$^{-1}$)", "RdBu_r", True),
                ("vy", xz[f"{species}_mean_vy_km_s"], r"$\langle V_y\rangle$ (km s$^{-1}$)", "RdBu_r", True),
                ("vz", xz[f"{species}_mean_vz_km_s"], r"$\langle V_z\rangle$ (km s$^{-1}$)", "RdBu_r", True),
            )
            for variable, values, colorbar, cmap, symmetric in definitions:
                stem = output_dir / f"{product_slug}_{category}_{species}_{variable}_mse_xz_{filename_suffix}_valid"
                product = _plot_map(
                    values,
                    grid_edges,
                    stem,
                    title=rf"{species_label}, {category_label}, {projection_label}",
                    colorbar_label=colorbar,
                    cmap=cmap,
                    symmetric=symmetric,
                    color_limits=limits[f"{species}_{variable}"],
                )
                products.append({"category": category, "species": species, "variable": variable, **product})
    return products, limits


def run_tw1_mse_xz_fov_statistics(
    input_root: Path,
    output_root: Path,
    *,
    r_root: Path,
    nv_mso2_root: Path,
    grid: MseGridSpec | None = None,
    workers: int = 1,
    max_days: int | None = None,
    write_plots: bool = True,
    progress_every: int = 25,
    xb_time_tolerance_s: float = 0.001,
    rotation_gap_s: float = 8.0,
    y_min_rm: float | None = None,
    y_max_rm: float | None = None,
) -> dict[str, Any]:
    """Process daily MINPA pairs and generate both +Xb_MSE cone products."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    grid = MseGridSpec() if grid is None else grid
    pairs = discover_daily_pairs(input_root)
    if max_days is not None:
        pairs = pairs[:max_days]
    if not pairs:
        raise FileNotFoundError(f"No paired Tianwen-1 daily products under {input_root}")
    accumulators = _initialize_accumulators(grid.size**3)
    arguments = [
        (pair, str(r_root), str(nv_mso2_root), grid, xb_time_tolerance_s, rotation_gap_s)
        for pair in pairs
    ]
    daily_summaries: list[DailyClassifiedStatistics] = []
    if workers <= 1:
        iterator: Iterable[DailyClassifiedStatistics] = map(_process_day, arguments)
        for completed, daily in enumerate(iterator, start=1):
            _merge_daily(accumulators, daily)
            daily_summaries.append(daily)
            if progress_every > 0 and (completed % progress_every == 0 or completed == len(pairs)):
                print(f"classified MINPA days: {completed}/{len(pairs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_process_day, arguments, chunksize=1)
            for completed, daily in enumerate(iterator, start=1):
                _merge_daily(accumulators, daily)
                daily_summaries.append(daily)
                if progress_every > 0 and (completed % progress_every == 0 or completed == len(pairs)):
                    print(f"classified MINPA days: {completed}/{len(pairs)}", flush=True)

    centers = grid.min_rm + (np.arange(grid.size, dtype=float) + 0.5) * grid.step_rm
    edges = grid.min_rm + np.arange(grid.size + 1, dtype=float) * grid.step_rm
    y_indices, y_centers = select_y_grid_centers(grid, y_min_rm=y_min_rm, y_max_rm=y_max_rm)
    projection_suffix = (
        "all_y"
        if y_min_rm is None
        else f"y_{_coordinate_slug(y_min_rm)}_{_coordinate_slug(y_max_rm)}"
    )
    projection_description = (
        "record-weighted aggregation over all Y cells"
        if y_min_rm is None
        else f"record-weighted aggregation over Y grid centers in [{y_min_rm:g}, {y_max_rm:g}] Rm"
    )
    common = {
        "axis_x_rm": centers,
        "axis_y_rm": centers,
        "axis_z_rm": centers,
        "grid_edges_rm": edges,
        "mars_radius_km": np.asarray(grid.mars_radius_km),
        "quality_policy": np.asarray("NaN invalid; NV_MSO_2 bits 2-5 invalid; bit 1 retained"),
        "projection_method": np.asarray(projection_description),
        "projection_y_indices": y_indices,
        "projection_y_centers_rm": y_centers,
        "projection_y_min_rm": np.asarray(np.nan if y_min_rm is None else y_min_rm),
        "projection_y_max_rm": np.asarray(np.nan if y_max_rm is None else y_max_rm),
    }
    classified_xz: dict[str, dict[str, np.ndarray]] = {}
    data_products: list[str] = []
    for category in CATEGORY_IDS:
        sparse, xz = _products_from_dense(
            accumulators[category], grid, y_min_rm=y_min_rm, y_max_rm=y_max_rm
        )
        classified_xz[category] = xz
        metadata = {
            **common,
            "category_id": np.asarray(category),
            "category_definition": np.asarray(CATEGORY_DEFINITIONS[category]["plain"]),
        }
        sparse_path = output_root / "data" / f"tw1_minpa_{category}_mse_3d_cell_means_valid.npz"
        xz_path = output_root / "data" / f"tw1_minpa_{category}_mse_xz_{projection_suffix}_means_valid.npz"
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
        )
        if write_plots
        else ([], {})
    )
    category_summary: dict[str, Any] = {}
    for category in CATEGORY_IDS:
        category_summary[category] = {"definition": CATEGORY_DEFINITIONS[category]["plain"], "species": {}}
        for species in SPECIES:
            values = classified_xz[category]
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

    maxima_names = (
        "max_abs_xb_epoch_match_dt_s",
        "max_abs_rotation_match_dt_s_for_valid_xb",
        "max_abs_xb_mse_stored_difference",
        "max_xb_independent_chain_angular_difference_deg",
        "max_abs_xb_mse_unit_norm_error",
    )
    validation_maxima = {
        name: float(np.nanmax([daily.xb_validation[name] for daily in daily_summaries])) for name in maxima_names
    }
    summary: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "r_mso2mse_root": str(r_root),
        "nv_mso2_root": str(nv_mso2_root),
        "days_processed": len(daily_summaries),
        "workers": workers,
        "rows_total": int(sum(day.rows_total for day in daily_summaries)),
        "rows_in_grid": int(sum(day.rows_in_grid for day in daily_summaries)),
        "xb_valid_rows": int(sum(day.xb_valid_rows for day in daily_summaries)),
        "xb_category_rows": {
            category: int(sum(day.category_rows[category] for day in daily_summaries)) for category in CATEGORY_IDS
        },
        "xb_valid_but_outside_both_cones_rows": int(sum(day.unclassified_xb_rows for day in daily_summaries)),
        "classification": {
            "vector_chain": "Xb_MSE = R_MSO2MSE * Xb_MSO; Xb_MSO = R_MSO_from_body * [1,0,0]^T",
            "authoritative_epoch_axis": "daily high-energy inst_x_axis_mse_x/y/z (orbiter-body +Xb at each particle epoch)",
            "independent_validation_axis": "NV_MSO_2.Xb_MSO rotated with the nearest valid R_MSO2MSE matrix",
            "half_angle_deg": 45.0,
            "boundary": "inclusive",
            "categories_are_mutually_exclusive": True,
            "outside_both_cones": "retained in coverage counts but excluded from both classified statistics",
        },
        "filters": {
            "non_finite": "invalid for the affected density/vector statistic; Xb NaN is unclassified",
            "quality": "NV_MSO_2 bits 2-5 invalid; bit 1 retained; unavailable or epoch-unmatched flags invalid",
            "processing": "processing_status_code == 1",
            "rotation": f"finite nearest R_MSO2MSE within {rotation_gap_s:g} s",
            "fov": "the two Xb_MSE cone categories replace the previous FOV grouping",
        },
        "aggregation": {
            "three_dimensional": "per-record arithmetic mean inside each 0.1 Rm cell",
            "xz_projection": projection_description,
            "projection_y_indices": y_indices.tolist(),
            "projection_y_centers_rm": y_centers.tolist(),
            "projection_y_bounds_rm": None if y_min_rm is None else [y_min_rm, y_max_rm],
            "density": "mean of per-record density, not a sum over epochs",
            "speed": "mean of per-record Euclidean MSE speed magnitude",
            "flux": "density-speed flux = projected mean density_cm-3 * projected mean speed_km_s * 1e5, in cm-2 s-1",
        },
        "grid": grid.as_dict(),
        "xb_validation_maxima": validation_maxima,
        "xb_independent_chain_category_disagreements": int(
            sum(
                day.xb_validation["xb_independent_chain_category_disagreements"]
                for day in daily_summaries
            )
        ),
        "shared_plot_color_limits": {key: list(value) for key, value in color_limits.items()},
        "data_products": data_products,
        "plot_products": plots,
        "categories": category_summary,
        "daily_summaries": [
            {
                "date": day.date,
                "rows_total": day.rows_total,
                "rows_in_grid": day.rows_in_grid,
                "xb_valid_rows": day.xb_valid_rows,
                "category_rows": day.category_rows,
                "unclassified_xb_rows": day.unclassified_xb_rows,
                "xb_validation": day.xb_validation,
            }
            for day in daily_summaries
        ],
    }
    _atomic_json(
        output_root / "logs" / f"tw1_minpa_mse_xz_fov_classified_{projection_suffix}_summary.json",
        summary,
    )
    return summary
