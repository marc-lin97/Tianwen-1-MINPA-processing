"""Stream Tianwen-1 MINPA daily products into per-cell MSE HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.io import loadmat

from .constants import DEFAULT_OUTPUT_ROOT, DEFAULT_TW1_NV_MSO2_ROOT, DEFAULT_TW1_R_ROOT, MATCH_MAX_GAP_S
from .coordinates import apply_rotation, interpolate_matrices_nearest
from .tw1_minpa import load_tw1_mso2mse, tw1_rotation_file


DEFAULT_INPUT_ROOT = DEFAULT_OUTPUT_ROOT / "Tianwen-1"
DEFAULT_GRID_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "processed_mse_spatial_stats" / "tw1_minpa_grid_records_5Rm"
SPECIES_IDS = ("Oplus", "O2plus")
SCHEMA_VERSION = 3


@dataclass(frozen=True)
class MseGridSpec:
    """Cartesian MSE grid defined by cell edges in Mars radii."""

    min_rm: float = -5.0
    max_rm: float = 5.0
    step_rm: float = 0.1
    mars_radius_km: float = 3397.0

    def __post_init__(self) -> None:
        if not (self.max_rm > self.min_rm and self.step_rm > 0.0 and self.mars_radius_km > 0.0):
            raise ValueError("Grid bounds, step, and Mars radius must be positive and ordered")
        cells = (self.max_rm - self.min_rm) / self.step_rm
        if not math.isclose(cells, round(cells), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Grid range must be exactly divisible by grid step")

    @property
    def size(self) -> int:
        return int(round((self.max_rm - self.min_rm) / self.step_rm))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.size, self.size, self.size)

    def center_rm(self, index: int) -> float:
        return self.min_rm + (float(index) + 0.5) * self.step_rm

    def edge_rm(self, index: int) -> tuple[float, float]:
        low = self.min_rm + float(index) * self.step_rm
        return low, low + self.step_rm

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_rm": self.min_rm,
            "max_rm": self.max_rm,
            "step_rm": self.step_rm,
            "mars_radius_km": self.mars_radius_km,
            "shape_xyz": list(self.shape),
            "boundary_convention": "closed at -5 Rm and +5 Rm; +5 Rm is assigned to the final cell",
        }


@dataclass(frozen=True)
class DailyProductPair:
    date: str
    oplus: Path
    o2plus: Path


@dataclass
class DailyGridChunk:
    date: str
    source_files: dict[str, str]
    rotation_file: str
    quality_flag_file: str
    quality_flag_metadata: dict[str, Any]
    common: dict[str, np.ndarray]
    species: dict[str, dict[str, np.ndarray]]
    flat_bin: np.ndarray
    validation: dict[str, float]

    @property
    def rows_total(self) -> int:
        return int(self.common["epoch_unix_s"].size)

    @property
    def rows_in_grid(self) -> int:
        return int(np.count_nonzero(self.flat_bin >= 0))

    @property
    def memory_bytes(self) -> int:
        arrays = list(self.common.values()) + [self.flat_bin]
        for values in self.species.values():
            arrays.extend(values.values())
        return int(sum(np.asarray(value).nbytes for value in arrays))


COMMON_FIELDS = (
    "epoch_unix_s",
    "pos_mso_x_km",
    "pos_mso_y_km",
    "pos_mso_z_km",
    "pos_mse_x_km",
    "pos_mse_y_km",
    "pos_mse_z_km",
    "mse_plus_z_in_minpa_fov_flag",
    "r_matrix_match_dt_s",
)
SPECIES_FIELDS = (
    "density_cm3",
    "v_mso_x_km_s",
    "v_mso_y_km_s",
    "v_mso_z_km_s",
    "v_mse_x_km_s",
    "v_mse_y_km_s",
    "v_mse_z_km_s",
    "processing_status_code",
)


DATASET_METADATA: dict[str, tuple[str, str]] = {
    "epoch_unix_s": ("s since 1970-01-01T00:00:00 UTC", "UTC epoch represented as Unix seconds"),
    "pos_mse_x_km": ("km", "Spacecraft MSE X position recomputed from MSO"),
    "pos_mse_y_km": ("km", "Spacecraft MSE Y position recomputed from MSO"),
    "pos_mse_z_km": ("km", "Spacecraft MSE Z position recomputed from MSO"),
    "pos_mse_x_rm": ("Rm", "Spacecraft MSE X position using Rm=3397 km"),
    "pos_mse_y_rm": ("Rm", "Spacecraft MSE Y position using Rm=3397 km"),
    "pos_mse_z_rm": ("Rm", "Spacecraft MSE Z position using Rm=3397 km"),
    "mse_plus_z_in_minpa_fov_flag": ("1", "1 inside MINPA FOV, 0 outside, NaN undetermined"),
    "r_matrix_match_dt_s": ("s", "Particle epoch minus nearest R_MSO2MSE epoch"),
    "r_matrix_valid_flag": ("1", "1 when all nine matched R_MSO2MSE elements are finite"),
    "r_matrix_within_tolerance_flag": (
        "1",
        f"1 when the matrix is finite and abs(match dt) <= {MATCH_MAX_GAP_S:g} s",
    ),
    "density_cm3": ("cm^-3", "Per-record density for raw MINPA energy >1000 eV"),
    "v_mse_x_km_s": ("km s^-1", "Density-weighted high-energy ion MSE X velocity"),
    "v_mse_y_km_s": ("km s^-1", "Density-weighted high-energy ion MSE Y velocity"),
    "v_mse_z_km_s": ("km s^-1", "Density-weighted high-energy ion MSE Z velocity"),
    "processing_status_code": ("1", "Status code copied from daily species product"),
    "density_valid_flag": ("1", "1 when density is finite"),
    "velocity_valid_flag": ("1", "1 when all three MSE velocity components are finite"),
    "quality_flag_bitmask": ("1", "Five-bit MINPA quality issue mask copied from NV_MSO_2"),
    "quality_flag_available_flag": (
        "1",
        "1 when the NV_MSO_2 species flag is available; 0 means numeric bitmask zero is only a placeholder",
    ),
    "quality_flag_epoch_matched_flag": ("1", "1 when high-energy epoch matched NV_MSO_2 within tolerance"),
    "quality_flag_match_dt_s": ("s", "High-energy epoch minus matched NV_MSO_2 quality-flag epoch"),
    "quality_flag_bit1_sparse_caution": ("1", "Valid raw 2-D channel count is 5-10; use with caution"),
    "quality_flag_bit2_sparse_invalid": ("1", "Valid raw 2-D channel count is below 5; invalid sample"),
    "quality_flag_bit3_uv_contamination": ("1", "More than 80% of 1-D DEF channels exceed 1e5"),
    "quality_flag_bit4_evenodd_error": ("1", "Odd/even peak-valley acquisition error in a tested energy band"),
    "quality_flag_bit5_high_channel": ("1", "At least one 2-D DEF channel exceeds 1e10; distorted/caution"),
}


QUALITY_DATASET_NAMES = (
    "quality_flag_bitmask",
    "quality_flag_available_flag",
    "quality_flag_epoch_matched_flag",
    "quality_flag_match_dt_s",
    "quality_flag_bit1_sparse_caution",
    "quality_flag_bit2_sparse_invalid",
    "quality_flag_bit3_uv_contamination",
    "quality_flag_bit4_evenodd_error",
    "quality_flag_bit5_high_channel",
)


def _flatten(data: dict[str, Any], field: str, path: Path) -> np.ndarray:
    if field not in data:
        raise KeyError(f"Missing {field!r} in {path}")
    return np.asarray(data[field]).reshape(-1)


def _vectors(data: dict[str, Any], prefix: str, suffix: str, path: Path) -> np.ndarray:
    return np.column_stack([_flatten(data, f"{prefix}_{axis}_{suffix}", path).astype(float) for axis in "xyz"])


def _equal_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and bool(np.array_equal(left, right, equal_nan=True))


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def _quality_info_value(info: Any, name: str, default: Any) -> Any:
    if info is None or not hasattr(info, name):
        return default
    value = getattr(info, name)
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def load_nv_mso2_quality_flags(
    path: Path,
    target_epoch: np.ndarray,
    *,
    tolerance_s: float = 0.001,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    fields: list[str] = ["MINPA_quality_flag_info"]
    for suffix in ("O", "O2"):
        fields.extend(
            [
                f"quality_flag_{suffix}_TW1",
                f"quality_flag_bit_{suffix}_TW1",
                f"quality_flag_available_{suffix}_TW1",
            ]
        )
    data = loadmat(path, variable_names=fields, squeeze_me=True, struct_as_record=False)
    info = data.get("MINPA_quality_flag_info")
    file_tolerance = float(_quality_info_value(info, "time_match_tolerance_s", tolerance_s))
    if not math.isclose(file_tolerance, tolerance_s, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"NV_MSO_2 tolerance {file_tolerance:g} s differs from requested {tolerance_s:g} s: {path}")
    result: dict[str, dict[str, np.ndarray]] = {}
    reference_time: np.ndarray | None = None
    for species_id, suffix in (("Oplus", "O"), ("O2plus", "O2")):
        flag = np.asarray(data[f"quality_flag_{suffix}_TW1"], dtype=float)
        bits = np.asarray(data[f"quality_flag_bit_{suffix}_TW1"], dtype=float)
        available = np.asarray(data[f"quality_flag_available_{suffix}_TW1"], dtype=float)
        if flag.ndim != 2 or flag.shape[1] != 2 or bits.shape != (flag.shape[0], 6) or available.shape != flag.shape:
            raise ValueError(f"Unexpected {species_id} NV_MSO_2 flag shape in {path}")
        source_time = flag[:, 0]
        if not (_equal_with_nan(source_time, bits[:, 0]) and _equal_with_nan(source_time, available[:, 0])):
            raise ValueError(f"Inconsistent {species_id} NV_MSO_2 flag time columns in {path}")
        if source_time.size < 1 or (source_time.size > 1 and not np.all(np.diff(source_time) > 0.0)):
            raise ValueError(f"NV_MSO_2 quality flag time is not strictly increasing in {path}")
        if reference_time is None:
            reference_time = source_time
        elif not _equal_with_nan(reference_time, source_time):
            raise ValueError(f"O and O2 NV_MSO_2 flag epochs differ in {path}")
        insertion = np.searchsorted(source_time, target_epoch)
        if source_time.size == 1:
            nearest = np.zeros(target_epoch.size, dtype=np.int64)
        else:
            insertion = np.clip(insertion, 1, source_time.size - 1)
            left = insertion - 1
            choose_right = np.abs(source_time[insertion] - target_epoch) < np.abs(source_time[left] - target_epoch)
            nearest = np.where(choose_right, insertion, left)
        match_dt = target_epoch - source_time[nearest]
        matched = np.abs(match_dt) <= tolerance_s
        mapped_flag = np.zeros(target_epoch.size, dtype=np.uint32)
        mapped_available = np.zeros(target_epoch.size, dtype=np.int8)
        mapped_bits = np.zeros((target_epoch.size, 5), dtype=np.int8)
        mapped_flag[matched] = np.rint(flag[nearest[matched], 1]).astype(np.uint32)
        mapped_available[matched] = (available[nearest[matched], 1] > 0.5).astype(np.int8)
        mapped_bits[matched] = np.rint(bits[nearest[matched], 1:6]).astype(np.int8)
        expected_bits = np.column_stack(
            [((mapped_flag >> np.uint32(index)) & np.uint32(1)).astype(np.int8) for index in range(5)]
        )
        usable = matched & (mapped_available == 1)
        if not np.array_equal(mapped_bits[usable], expected_bits[usable]):
            raise ValueError(f"{species_id} NV_MSO_2 bit columns disagree with bitmask in {path}")
        result[species_id] = {
            "quality_flag_bitmask": mapped_flag,
            "quality_flag_available_flag": mapped_available,
            "quality_flag_epoch_matched_flag": matched.astype(np.int8),
            "quality_flag_match_dt_s": np.where(matched, match_dt, np.nan),
            "quality_flag_bit1_sparse_caution": mapped_bits[:, 0],
            "quality_flag_bit2_sparse_invalid": mapped_bits[:, 1],
            "quality_flag_bit3_uv_contamination": mapped_bits[:, 2],
            "quality_flag_bit4_evenodd_error": mapped_bits[:, 3],
            "quality_flag_bit5_high_channel": mapped_bits[:, 4],
        }
    metadata = {
        "algorithm_version": str(_quality_info_value(info, "algorithm_version", "unknown")),
        "time_match_tolerance_s": file_tolerance,
        "flag_encoding": str(_quality_info_value(info, "flag_encoding", "five-bit MINPA quality issue mask")),
        "missing_semantics": str(
            _quality_info_value(
                info,
                "missing_semantics",
                "quality_flag_available=false means numeric zero is a placeholder, not a valid quality judgment",
            )
        ),
    }
    return result, metadata


def _validate_recomputed(name: str, calculated: np.ndarray, stored: np.ndarray, atol: float) -> float:
    if calculated.shape != stored.shape:
        raise ValueError(f"{name} shape mismatch: {calculated.shape} != {stored.shape}")
    if not np.array_equal(np.isfinite(calculated), np.isfinite(stored)):
        raise ValueError(f"{name} finite-value mask differs from stored daily product")
    error = _max_abs_difference(calculated, stored)
    if error > atol:
        raise ValueError(f"{name} maximum absolute error {error:.6g} exceeds tolerance {atol:.6g}")
    return error


def prepare_rotation_time_axis(
    time_s: np.ndarray,
    matrices: np.ndarray,
    *,
    duplicate_matrix_atol: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Sort rotation epochs and remove only physically equivalent duplicates."""
    time = np.asarray(time_s, dtype=float).reshape(-1)
    rotation = np.asarray(matrices, dtype=float)
    if rotation.shape != (time.size, 3, 3) or time.size < 2:
        raise ValueError(f"Unexpected R_MSO2MSE shape: time={time.shape}, matrix={rotation.shape}")
    if not np.all(np.isfinite(time)):
        raise ValueError("R_MSO2MSE time contains non-finite values")
    order = np.argsort(time, kind="stable")
    reordered = not np.array_equal(order, np.arange(time.size))
    sorted_time = time[order]
    sorted_rotation = rotation[order]
    duplicate_positions = np.flatnonzero(np.diff(sorted_time) == 0.0)
    duplicate_max_error = 0.0
    preferred_more_finite = 0
    conflicting_complete_groups = 0
    unique_times: list[float] = []
    unique_matrices: list[np.ndarray] = []
    start = 0
    while start < sorted_time.size:
        stop = start + 1
        while stop < sorted_time.size and sorted_time[stop] == sorted_time[start]:
            stop += 1
        block = sorted_rotation[start:stop]
        finite_scores = np.count_nonzero(np.isfinite(block), axis=(1, 2))
        best_score = int(np.max(finite_scores))
        candidates = np.flatnonzero(finite_scores == best_score)
        if int(finite_scores[0]) < best_score:
            preferred_more_finite += 1
        reference = block[int(candidates[0])]
        for candidate_index in candidates[1:]:
            candidate = block[int(candidate_index)]
            if not np.array_equal(np.isfinite(reference), np.isfinite(candidate)):
                conflicting_complete_groups += 1
                continue
            error = _max_abs_difference(reference, candidate)
            duplicate_max_error = max(duplicate_max_error, error)
            if error > duplicate_matrix_atol:
                conflicting_complete_groups += 1
        unique_times.append(float(sorted_time[start]))
        unique_matrices.append(reference)
        start = stop
    unique_time = np.asarray(unique_times, dtype=float)
    unique_rotation = np.asarray(unique_matrices, dtype=float)
    if not np.all(np.diff(unique_time) > 0.0):
        raise ValueError("R_MSO2MSE time could not be made strictly increasing")
    return unique_time, unique_rotation, {
        "rotation_time_duplicates_removed": float(duplicate_positions.size),
        "rotation_duplicate_matrix_max_error": duplicate_max_error,
        "rotation_duplicate_groups_preferred_more_finite": float(preferred_more_finite),
        "rotation_duplicate_conflicting_complete_groups": float(conflicting_complete_groups),
        "rotation_time_reordered_flag": float(reordered),
    }


def discover_daily_pairs(
    input_root: Path,
    *,
    dates: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[DailyProductPair]:
    input_root = Path(input_root)
    if dates is None:
        candidates = sorted(input_root.rglob("tw1_minpa_highE_Oplus_????????.mat"))
    else:
        candidates = [input_root / str(date)[:4] / f"tw1_minpa_highE_Oplus_{date}.mat" for date in dates]
    pairs: list[DailyProductPair] = []
    for oplus in candidates:
        match = re.search(r"_(\d{8})\.mat$", oplus.name)
        if not match or not oplus.exists():
            continue
        date = match.group(1)
        if start_date is not None and date < start_date:
            continue
        if end_date is not None and date > end_date:
            continue
        o2plus = oplus.with_name(oplus.name.replace("_Oplus_", "_O2plus_"))
        if not o2plus.exists():
            raise FileNotFoundError(f"Missing paired O2+ product for {date}: {o2plus}")
        pairs.append(DailyProductPair(date, oplus, o2plus))
    return sorted(pairs, key=lambda item: item.date)


def bin_mse_positions(position_mse_km: np.ndarray, grid: MseGridSpec) -> tuple[np.ndarray, np.ndarray]:
    position_rm = np.asarray(position_mse_km, dtype=float) / grid.mars_radius_km
    finite = np.all(np.isfinite(position_rm), axis=1)
    inside = finite & np.all((position_rm >= grid.min_rm) & (position_rm <= grid.max_rm), axis=1)
    indices = np.full(position_rm.shape, -1, dtype=np.int32)
    raw = np.floor((position_rm[inside] - grid.min_rm) / grid.step_rm).astype(np.int32)
    indices[inside] = np.clip(raw, 0, grid.size - 1)
    flat = np.full(position_rm.shape[0], -1, dtype=np.int64)
    if np.any(inside):
        flat[inside] = np.ravel_multi_index(indices[inside].T, grid.shape)
    return flat, position_rm


def load_daily_grid_chunk(
    pair: DailyProductPair,
    r_root: Path,
    grid: MseGridSpec,
    *,
    nv_mso2_root: Path = DEFAULT_TW1_NV_MSO2_ROOT,
    quality_flag_tolerance_s: float = 0.001,
    coordinate_atol: float = 1.0e-8,
    rotation_gap_s: float = MATCH_MAX_GAP_S,
) -> DailyGridChunk:
    variables = COMMON_FIELDS + SPECIES_FIELDS
    products = {
        "Oplus": loadmat(pair.oplus, variable_names=variables, squeeze_me=True, struct_as_record=False),
        "O2plus": loadmat(pair.o2plus, variable_names=variables, squeeze_me=True, struct_as_record=False),
    }
    oplus = products["Oplus"]
    o2plus = products["O2plus"]
    epoch = _flatten(oplus, "epoch_unix_s", pair.oplus).astype(float)
    epoch_o2 = _flatten(o2plus, "epoch_unix_s", pair.o2plus).astype(float)
    if not _equal_with_nan(epoch, epoch_o2):
        raise ValueError(f"O+ and O2+ epoch mismatch for {pair.date}")
    if epoch.size > 1 and not np.all(np.diff(epoch) > 0.0):
        raise ValueError(f"Epoch is not strictly increasing for {pair.date}")

    pos_mso = _vectors(oplus, "pos_mso", "km", pair.oplus)
    stored_pos_mse = _vectors(oplus, "pos_mse", "km", pair.oplus)
    if not _equal_with_nan(pos_mso, _vectors(o2plus, "pos_mso", "km", pair.o2plus)):
        raise ValueError(f"O+ and O2+ MSO position mismatch for {pair.date}")
    if not _equal_with_nan(stored_pos_mse, _vectors(o2plus, "pos_mse", "km", pair.o2plus)):
        raise ValueError(f"O+ and O2+ stored MSE position mismatch for {pair.date}")

    fov = _flatten(oplus, "mse_plus_z_in_minpa_fov_flag", pair.oplus).astype(float)
    if not _equal_with_nan(fov, _flatten(o2plus, "mse_plus_z_in_minpa_fov_flag", pair.o2plus).astype(float)):
        raise ValueError(f"O+ and O2+ FOV flag mismatch for {pair.date}")

    quality_flag_path = Path(nv_mso2_root) / f"NV_{pair.date}.mat"
    if not quality_flag_path.exists():
        raise FileNotFoundError(f"Missing NV_MSO_2 quality flag file for {pair.date}: {quality_flag_path}")
    quality_flags, quality_flag_metadata = load_nv_mso2_quality_flags(
        quality_flag_path,
        epoch,
        tolerance_s=quality_flag_tolerance_s,
    )

    rotation_path = tw1_rotation_file(pair.date, Path(r_root))
    rotation = load_tw1_mso2mse(rotation_path)
    rotation_time = np.asarray(rotation["time_s"], dtype=float)
    matrices = np.asarray(rotation["R_mso_to_mse"], dtype=float)
    try:
        rotation_time, matrices, rotation_axis_validation = prepare_rotation_time_axis(rotation_time, matrices)
    except ValueError as exc:
        raise ValueError(f"Invalid R_MSO2MSE time axis for {pair.date}: {exc}") from exc
    matched, match_dt = interpolate_matrices_nearest(rotation_time, matrices, epoch)
    finite_rotation = np.all(np.isfinite(matched), axis=(1, 2))
    if np.any(finite_rotation):
        valid_matrices = matched[finite_rotation]
        identity_error = float(np.max(np.abs(valid_matrices @ np.swapaxes(valid_matrices, 1, 2) - np.eye(3))))
        determinant_error = float(np.max(np.abs(np.linalg.det(valid_matrices) - 1.0)))
        if identity_error > 1.0e-6 or determinant_error > 1.0e-6:
            raise ValueError(f"R_MSO2MSE is not a proper rotation for {pair.date}")
    else:
        identity_error = math.nan
        determinant_error = math.nan

    pos_mse = apply_rotation(matched, pos_mso)
    pos_error = _validate_recomputed("MSE position", pos_mse, stored_pos_mse, coordinate_atol)
    species: dict[str, dict[str, np.ndarray]] = {}
    velocity_errors: dict[str, float] = {}
    for species_id, path in (("Oplus", pair.oplus), ("O2plus", pair.o2plus)):
        data = products[species_id]
        velocity_mso = _vectors(data, "v_mso", "km_s", path)
        stored_velocity_mse = _vectors(data, "v_mse", "km_s", path)
        velocity_mse = apply_rotation(matched, velocity_mso)
        velocity_errors[species_id] = _validate_recomputed(
            f"{species_id} MSE velocity", velocity_mse, stored_velocity_mse, coordinate_atol
        )
        density = _flatten(data, "density_cm3", path).astype(float)
        status = _flatten(data, "processing_status_code", path).astype(np.int16)
        if not (density.size == status.size == epoch.size == velocity_mse.shape[0]):
            raise ValueError(f"Inconsistent {species_id} row count for {pair.date}")
        species[species_id] = {
            "density_cm3": density,
            "v_mse_x_km_s": velocity_mse[:, 0],
            "v_mse_y_km_s": velocity_mse[:, 1],
            "v_mse_z_km_s": velocity_mse[:, 2],
            "processing_status_code": status,
            "density_valid_flag": np.isfinite(density).astype(np.int8),
            "velocity_valid_flag": np.all(np.isfinite(velocity_mse), axis=1).astype(np.int8),
            **quality_flags[species_id],
        }

    flat_bin, position_rm = bin_mse_positions(pos_mse, grid)
    within_tolerance = finite_rotation & (np.abs(match_dt) <= float(rotation_gap_s))
    common = {
        "epoch_unix_s": epoch,
        "pos_mse_x_km": pos_mse[:, 0],
        "pos_mse_y_km": pos_mse[:, 1],
        "pos_mse_z_km": pos_mse[:, 2],
        "pos_mse_x_rm": position_rm[:, 0],
        "pos_mse_y_rm": position_rm[:, 1],
        "pos_mse_z_rm": position_rm[:, 2],
        "mse_plus_z_in_minpa_fov_flag": fov,
        "r_matrix_match_dt_s": match_dt,
        "r_matrix_valid_flag": finite_rotation.astype(np.int8),
        "r_matrix_within_tolerance_flag": within_tolerance.astype(np.int8),
    }
    validation = {
        "max_abs_position_recompute_error_km": pos_error,
        "max_abs_oplus_velocity_recompute_error_km_s": velocity_errors["Oplus"],
        "max_abs_o2plus_velocity_recompute_error_km_s": velocity_errors["O2plus"],
        "max_abs_rotation_match_dt_s": float(np.max(np.abs(match_dt))) if match_dt.size else math.nan,
        "rotation_orthonormal_max_error": identity_error,
        "rotation_determinant_max_error": determinant_error,
        "finite_rotation_rows": float(np.count_nonzero(finite_rotation)),
        "rotation_within_tolerance_rows": float(np.count_nonzero(within_tolerance)),
        **rotation_axis_validation,
    }
    return DailyGridChunk(
        date=pair.date,
        source_files={"Oplus": str(pair.oplus), "O2plus": str(pair.o2plus), "NV_MSO_2": str(quality_flag_path)},
        rotation_file=str(rotation_path),
        quality_flag_file=str(quality_flag_path),
        quality_flag_metadata=quality_flag_metadata,
        common=common,
        species=species,
        flat_bin=flat_bin,
        validation=validation,
    )


def _format_center(value: float) -> str:
    return f"{value:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


def cell_file_path(output_root: Path, flat_bin: int, grid: MseGridSpec) -> Path:
    ix, iy, iz = np.unravel_index(int(flat_bin), grid.shape)
    center = tuple(grid.center_rm(index) for index in (ix, iy, iz))
    name = (
        f"cell_ix{ix:03d}_iy{iy:03d}_iz{iz:03d}"
        f"__x{_format_center(center[0])}_y{_format_center(center[1])}_z{_format_center(center[2])}Rm.h5"
    )
    return Path(output_root) / "cells" / f"ix{ix:03d}" / name


def _set_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    dataset = group.create_dataset(name, data=array, compression="gzip", compression_opts=4, shuffle=True)
    units, description = DATASET_METADATA[name]
    dataset.attrs["units"] = units
    dataset.attrs["description"] = description
    if "mse" in name:
        dataset.attrs["coordinate_system"] = "MSE"


def build_compound_records(chunk: DailyGridChunk, rows: np.ndarray) -> tuple[np.ndarray, dict[str, dict[str, str]]]:
    """Pack one cell-day into one HDF5 compound dataset with named fields."""
    selected = np.asarray(rows, dtype=np.int64)
    fields: list[tuple[str, np.dtype[Any]]] = []
    sources: list[tuple[str, str, np.ndarray]] = []
    metadata: dict[str, dict[str, str]] = {}
    for name, values in chunk.common.items():
        output_name = name
        array = np.asarray(values)
        fields.append((output_name, array.dtype))
        sources.append((output_name, name, array))
    for species_id, prefix in (("Oplus", "oplus"), ("O2plus", "o2plus")):
        for name, values in chunk.species[species_id].items():
            output_name = f"{prefix}_{name}"
            array = np.asarray(values)
            fields.append((output_name, array.dtype))
            sources.append((output_name, name, array))
    records = np.empty(selected.size, dtype=np.dtype(fields))
    for output_name, source_name, array in sources:
        records[output_name] = array[selected]
        units, description = DATASET_METADATA[source_name]
        metadata[output_name] = {"units": units, "description": description}
        if "mse" in source_name:
            metadata[output_name]["coordinate_system"] = "MSE"
    return records, metadata


def _ensure_cell_metadata(handle: h5py.File, flat_bin: int, grid: MseGridSpec) -> None:
    ix, iy, iz = np.unravel_index(int(flat_bin), grid.shape)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "mission": "Tianwen-1",
        "instrument": "MINPA",
        "coordinate_system": "MSE",
        "flat_bin": int(flat_bin),
        "ix": int(ix),
        "iy": int(iy),
        "iz": int(iz),
        "grid_min_rm": grid.min_rm,
        "grid_max_rm": grid.max_rm,
        "grid_step_rm": grid.step_rm,
        "mars_radius_km": grid.mars_radius_km,
        "x_center_rm": grid.center_rm(ix),
        "y_center_rm": grid.center_rm(iy),
        "z_center_rm": grid.center_rm(iz),
        "x_edge_min_rm": grid.edge_rm(ix)[0],
        "x_edge_max_rm": grid.edge_rm(ix)[1],
        "y_edge_min_rm": grid.edge_rm(iy)[0],
        "y_edge_max_rm": grid.edge_rm(iy)[1],
        "z_edge_min_rm": grid.edge_rm(iz)[0],
        "z_edge_max_rm": grid.edge_rm(iz)[1],
        "high_energy_min_eV": 1000.0,
    }
    if "schema_version" not in handle.attrs:
        for key, value in expected.items():
            handle.attrs[key] = value
        handle.attrs["created_utc"] = datetime.now(UTC).isoformat()
    else:
        for key, value in expected.items():
            actual = handle.attrs.get(key)
            if isinstance(value, float):
                if actual is None or not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1.0e-10):
                    raise ValueError(f"Grid metadata mismatch for {handle.filename}: {key}")
            elif actual != value:
                raise ValueError(f"Grid metadata mismatch for {handle.filename}: {key}")
    handle.attrs["updated_utc"] = datetime.now(UTC).isoformat()


def write_cell_day(
    path: Path,
    chunk: DailyGridChunk,
    rows: np.ndarray,
    flat_bin: int,
    grid: MseGridSpec,
    *,
    overwrite: bool = False,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a", libver="latest") as handle:
        _ensure_cell_metadata(handle, flat_bin, grid)
        days = handle.require_group("days")
        staging = handle.require_group("_staging")
        if chunk.date in days:
            if not overwrite:
                return "skipped"
            del days[chunk.date]
        if chunk.date in staging:
            del staging[chunk.date]
        day_group = staging.create_group(chunk.date)
        day_group.attrs["complete"] = np.int8(0)
        day_group.attrs["row_count"] = int(rows.size)
        day_group.attrs["source_oplus_file"] = chunk.source_files["Oplus"]
        day_group.attrs["source_o2plus_file"] = chunk.source_files["O2plus"]
        day_group.attrs["rotation_file"] = chunk.rotation_file
        day_group.attrs["quality_flag_file"] = chunk.quality_flag_file
        day_group.attrs["quality_flag_algorithm_version"] = chunk.quality_flag_metadata["algorithm_version"]
        day_group.attrs["quality_flag_time_match_tolerance_s"] = chunk.quality_flag_metadata["time_match_tolerance_s"]
        day_group.attrs["quality_flag_encoding"] = chunk.quality_flag_metadata["flag_encoding"]
        day_group.attrs["quality_flag_missing_semantics"] = chunk.quality_flag_metadata["missing_semantics"]
        day_group.attrs["rotation_matching"] = "nearest epoch"
        for key, value in chunk.validation.items():
            day_group.attrs[key] = value

        records, field_metadata = build_compound_records(chunk, rows)
        dataset = day_group.create_dataset(
            "records",
            data=records,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        dataset.attrs["layout"] = "HDF5 compound records; one row per MINPA epoch"
        dataset.attrs["field_metadata_json"] = json.dumps(field_metadata, ensure_ascii=False, sort_keys=True)
        day_group.attrs["complete"] = np.int8(1)
        handle.flush()
        handle.move(f"_staging/{chunk.date}", f"days/{chunk.date}")
        if len(handle["_staging"]) == 0:
            del handle["_staging"]
        handle.flush()
    return "written"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _ensure_root_schema(
    output_root: Path,
    grid: MseGridSpec,
    rotation_gap_s: float,
    quality_flag_tolerance_s: float,
) -> None:
    path = Path(output_root) / "grid_schema.json"
    required = {
        "schema_version": SCHEMA_VERSION,
        "mission": "Tianwen-1",
        "instrument": "MINPA",
        "input_product": "daily high-energy O+ and O2+ moment MAT files",
        "high_energy_selection": "raw MINPA energy > 1000 eV",
        "coordinate_system": "MSE",
        "rotation_matching": "nearest daily R_MSO2MSE epoch",
        "rotation_match_tolerance_s": float(rotation_gap_s),
        "quality_flags": {
            "source": "daily NV_MSO_2 quality_flag_O_TW1 and quality_flag_O2_TW1",
            "time_match_tolerance_s": float(quality_flag_tolerance_s),
            "encoding": "bit1 sparse caution; bit2 sparse invalid; bit3 UV contamination; bit4 even/odd error; bit5 high channel",
            "missing_semantics": "available=0 means bitmask zero is only a placeholder",
        },
        "grid": grid.as_dict(),
        "storage": "one HDF5 file per non-empty grid cell; one group and one compound records dataset per UTC date",
        "record_policy": "all records with finite MSE position inside the grid; invalid moments remain NaN with status flags",
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in required}
        if comparable != required:
            raise ValueError(f"Existing output schema is incompatible: {path}")
    else:
        _atomic_json(path, {**required, "created_utc": datetime.now(UTC).isoformat()})


def process_daily_pair(
    pair: DailyProductPair,
    output_root: Path,
    r_root: Path,
    nv_mso2_root: Path,
    grid: MseGridSpec,
    *,
    quality_flag_tolerance_s: float = 0.001,
    coordinate_atol: float = 1.0e-8,
    rotation_gap_s: float = MATCH_MAX_GAP_S,
    overwrite: bool = False,
) -> dict[str, Any]:
    chunk = load_daily_grid_chunk(
        pair,
        r_root,
        grid,
        nv_mso2_root=nv_mso2_root,
        quality_flag_tolerance_s=quality_flag_tolerance_s,
        coordinate_atol=coordinate_atol,
        rotation_gap_s=rotation_gap_s,
    )
    valid_rows = np.flatnonzero(chunk.flat_bin >= 0)
    order = valid_rows[np.argsort(chunk.flat_bin[valid_rows], kind="stable")]
    bins = chunk.flat_bin[order]
    unique_bins, starts, counts = np.unique(bins, return_index=True, return_counts=True)
    written = skipped = 0
    cell_files: list[str] = []
    for flat_bin, start, count in zip(unique_bins, starts, counts):
        rows = order[int(start) : int(start + count)]
        path = cell_file_path(output_root, int(flat_bin), grid)
        result = write_cell_day(path, chunk, rows, int(flat_bin), grid, overwrite=overwrite)
        written += int(result == "written")
        skipped += int(result == "skipped")
        cell_files.append(str(path))
    return {
        "date": pair.date,
        "completed_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "source_files": chunk.source_files,
        "rotation_file": chunk.rotation_file,
        "rows_total": chunk.rows_total,
        "rows_finite_position_in_grid": chunk.rows_in_grid,
        "rows_outside_or_invalid_position": chunk.rows_total - chunk.rows_in_grid,
        "nonempty_cells": int(unique_bins.size),
        "cell_day_groups_written": written,
        "cell_day_groups_skipped": skipped,
        "working_memory_bytes": chunk.memory_bytes,
        "validation": chunk.validation,
        "cell_files": cell_files,
    }


def run_tw1_mse_grid_records(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_GRID_OUTPUT_ROOT,
    r_root: Path = DEFAULT_TW1_R_ROOT,
    nv_mso2_root: Path = DEFAULT_TW1_NV_MSO2_ROOT,
    *,
    dates: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    grid: MseGridSpec | None = None,
    quality_flag_tolerance_s: float = 0.001,
    coordinate_atol: float = 1.0e-8,
    rotation_gap_s: float = MATCH_MAX_GAP_S,
    min_free_gb: float = 20.0,
    overwrite: bool = False,
    max_days: int | None = None,
) -> dict[str, Any]:
    grid = grid or MseGridSpec()
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_root_schema(output_root, grid, rotation_gap_s, quality_flag_tolerance_s)
    pairs = discover_daily_pairs(input_root, dates=dates, start_date=start_date, end_date=end_date)
    if max_days is not None:
        pairs = pairs[: int(max_days)]
    if not pairs:
        raise FileNotFoundError(f"No paired Tianwen-1 daily products found under {input_root}")

    started = datetime.now(UTC)
    day_summaries: list[dict[str, Any]] = []
    max_memory_bytes = 0
    for pair in pairs:
        summary_path = output_root / "logs" / "days" / f"{pair.date}.json"
        if summary_path.exists() and not overwrite:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                day_summaries.append({"date": pair.date, "status": "already_complete"})
                continue
        free_bytes = shutil.disk_usage(output_root).free
        if free_bytes < float(min_free_gb) * (1024**3):
            raise OSError(f"Free space {free_bytes / 1024**3:.2f} GiB is below --min-free-gb={min_free_gb:g}")
        summary = process_daily_pair(
            pair,
            output_root,
            Path(r_root),
            Path(nv_mso2_root),
            grid,
            quality_flag_tolerance_s=quality_flag_tolerance_s,
            coordinate_atol=coordinate_atol,
            rotation_gap_s=rotation_gap_s,
            overwrite=overwrite,
        )
        max_memory_bytes = max(max_memory_bytes, int(summary["working_memory_bytes"]))
        _atomic_json(summary_path, summary)
        day_summaries.append({key: value for key, value in summary.items() if key != "cell_files"})
        _atomic_json(
            output_root / "logs" / "progress.json",
            {
                "updated_utc": datetime.now(UTC).isoformat(),
                "requested_days": len(pairs),
                "visited_days": len(day_summaries),
                "complete_or_already_complete_days": int(
                    sum(item["status"] in ("complete", "already_complete") for item in day_summaries)
                ),
                "last_completed_date": pair.date,
                "maximum_daily_working_memory_bytes": max_memory_bytes,
                "free_space_gb": shutil.disk_usage(output_root).free / (1024**3),
            },
        )

    completed = datetime.now(UTC)
    run_summary = {
        "created_utc": completed.isoformat(),
        "started_utc": started.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "rotation_root": str(r_root),
        "nv_mso2_root": str(nv_mso2_root),
        "grid": grid.as_dict(),
        "coordinate_validation_atol": coordinate_atol,
        "rotation_match_tolerance_s": rotation_gap_s,
        "quality_flag_match_tolerance_s": quality_flag_tolerance_s,
        "min_free_gb": min_free_gb,
        "requested_days": len(pairs),
        "processed_days": int(sum(item["status"] == "complete" for item in day_summaries)),
        "already_complete_days": int(sum(item["status"] == "already_complete" for item in day_summaries)),
        "maximum_daily_working_memory_bytes": max_memory_bytes,
        "free_space_gb_after": shutil.disk_usage(output_root).free / (1024**3),
        "days": day_summaries,
    }
    _atomic_json(output_root / "logs" / "latest_run_summary.json", run_summary)
    return run_summary
