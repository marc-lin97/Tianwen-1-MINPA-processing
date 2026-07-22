"""Archive Tianwen-1 MINPA ``NV_MSO_2`` all-energy moments by MSE cell.

The source product contains one density and one MSO bulk-velocity vector per
species and epoch.  This module preserves those per-record moments, rotates
the vectors and spacecraft position to MSE, and writes one HDF5 file per
non-empty three-dimensional spatial cell.  It never sums density over epochs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

from .constants import DEFAULT_TW1_MOMAG_ROOT, DEFAULT_TW1_NV_MSO2_ROOT, DEFAULT_TW1_R_ROOT, MATCH_MAX_GAP_S
from .coordinates import apply_rotation, interpolate_matrices_nearest, nearest_indices
from .tw1_minpa import load_tw1_mso2mse, minpa_theta_edges_deg, tw1_rotation_file
from .tw1_mse_grid_records import MseGridSpec, bin_mse_positions, cell_file_path, prepare_rotation_time_axis


DEFAULT_INPUT_ROOT = DEFAULT_TW1_NV_MSO2_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_all_energy_grid_records_5Rm"
)
SCHEMA_NAME = "tw1_minpa_all_energy_mse_grid_records"
SCHEMA_VERSION = 1
EXPECTED_QUALITY_ALGORITHM_VERSION = "2026-07-13-seven-band-all-species-v2-mode12-product-time"
POSITION_MATCH_TOLERANCE_S = 5.0

SPECIES: dict[str, dict[str, str]] = {
    "Hplus": {"prefix": "hplus", "suffix": "H", "density": "NH_TW1", "velocity": "VH_TW1_MSO"},
    "Oplus": {"prefix": "oplus", "suffix": "O", "density": "NO_TW1", "velocity": "VO_TW1_MSO"},
    "O2plus": {"prefix": "o2plus", "suffix": "O2", "density": "NO2_TW1", "velocity": "VO2_TW1_MSO"},
}

QUALITY_FIELDS = (
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
    *QUALITY_FIELDS,
)

EXPECTED_RECORD_FIELDS = COMMON_RECORD_FIELDS + tuple(
    f"{spec['prefix']}_{field}" for spec in SPECIES.values() for field in SPECIES_RECORD_FIELDS
)

FIELD_METADATA: dict[str, tuple[str, str]] = {
    "epoch_unix_s": ("s since 1970-01-01T00:00:00 UTC", "Authoritative NH_TW1 Unix epoch"),
    "pos_mse_x_km": ("km", "Spacecraft MSE X position from nearest MOMAG P_TW1 and R_MSO2MSE"),
    "pos_mse_y_km": ("km", "Spacecraft MSE Y position from nearest MOMAG P_TW1 and R_MSO2MSE"),
    "pos_mse_z_km": ("km", "Spacecraft MSE Z position from nearest MOMAG P_TW1 and R_MSO2MSE"),
    "pos_mse_x_rm": ("Rm", "Spacecraft MSE X position using Rm=3397 km"),
    "pos_mse_y_rm": ("Rm", "Spacecraft MSE Y position using Rm=3397 km"),
    "pos_mse_z_rm": ("Rm", "Spacecraft MSE Z position using Rm=3397 km"),
    "mse_plus_z_in_minpa_fov_flag": (
        "1",
        "1 when MSE +Z is inside the mode-specific MINPA ion FOV reconstructed from Xb_MSO",
    ),
    "r_matrix_match_dt_s": ("s", "Particle epoch minus nearest R_MSO2MSE epoch"),
    "r_matrix_valid_flag": ("1", "1 when all nine matched R_MSO2MSE elements are finite"),
    "r_matrix_within_tolerance_flag": (
        "1",
        f"1 when the matrix is finite and abs(match dt) <= {MATCH_MAX_GAP_S:g} s",
    ),
    "density_cm3": ("cm^-3", "Per-record all-energy density copied from NV_MSO_2"),
    "v_mse_x_km_s": ("km s^-1", "All-energy ion MSE X bulk velocity rotated from NV_MSO_2 MSO"),
    "v_mse_y_km_s": ("km s^-1", "All-energy ion MSE Y bulk velocity rotated from NV_MSO_2 MSO"),
    "v_mse_z_km_s": ("km s^-1", "All-energy ion MSE Z bulk velocity rotated from NV_MSO_2 MSO"),
    "processing_status_code": (
        "1",
        "Derived compatibility status: 1=density and velocity valid, 5=density only, 4=invalid or zero density",
    ),
    "density_valid_flag": ("1", "1 when density is finite and strictly positive"),
    "velocity_valid_flag": ("1", "1 when all three MSE velocity components are finite"),
    "quality_flag_bitmask": ("1", "Five-bit species-specific MINPA quality issue mask from NV_MSO_2"),
    "quality_flag_available_flag": (
        "1",
        "1 when the NV_MSO_2 quality flag is available; zero mask is a placeholder when unavailable",
    ),
    "quality_flag_epoch_matched_flag": ("1", "1 when the row-aligned quality epoch equals NH_TW1 epoch"),
    "quality_flag_match_dt_s": ("s", "NH_TW1 epoch minus row-aligned quality epoch"),
    "quality_flag_bit1_sparse_caution": ("1", "Valid raw 2-D channel count is 5-10; caution"),
    "quality_flag_bit2_sparse_invalid": ("1", "Valid raw 2-D channel count is below 5; invalid"),
    "quality_flag_bit3_uv_contamination": ("1", "More than 80% of 1-D DEF channels exceed 1e5"),
    "quality_flag_bit4_evenodd_error": ("1", "Odd/even peak-valley acquisition error"),
    "quality_flag_bit5_high_channel": ("1", "At least one 2-D DEF channel exceeds 1e10"),
}


@dataclass(frozen=True)
class NvProduct:
    date: str
    path: Path


@dataclass
class AllEnergyDayChunk:
    date: str
    source_file: str
    momag_file: str
    rotation_file: str
    product_metadata: dict[str, Any]
    quality_metadata: dict[str, Any]
    common: dict[str, np.ndarray]
    species: dict[str, dict[str, np.ndarray]]
    flat_bin: np.ndarray
    validation: dict[str, Any]

    @property
    def rows_total(self) -> int:
        return int(self.common["epoch_unix_s"].size)

    @property
    def rows_in_grid(self) -> int:
        return int(np.count_nonzero(self.flat_bin >= 0))

    @property
    def memory_bytes(self) -> int:
        arrays = list(self.common.values()) + [self.flat_bin]
        for fields in self.species.values():
            arrays.extend(fields.values())
        return int(sum(np.asarray(value).nbytes for value in arrays))


def _unwrap_mat_struct(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0]
    return value


def _info_value(info: Any, name: str, default: Any) -> Any:
    value = _unwrap_mat_struct(info)
    if value is None or not hasattr(value, name):
        return default
    result = getattr(value, name)
    if isinstance(result, np.ndarray) and result.size == 1:
        return result.reshape(-1)[0].item()
    return result


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _as_text(value.reshape(-1)[0])
    return str(value)


def _date_bounds(date: str) -> tuple[float, float]:
    start = datetime.strptime(date, "%Y%m%d").replace(tzinfo=UTC)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def _require_matrix(data: dict[str, Any], name: str, columns: int, path: Path) -> np.ndarray:
    if name not in data:
        raise KeyError(f"Missing {name!r} in {path}")
    array = np.asarray(data[name], dtype=float)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"Unexpected {name} shape {array.shape} in {path}; expected (n,{columns})")
    return array


def _validate_row_time(name: str, values: np.ndarray, epoch: np.ndarray, path: Path, *, allow_nan: bool) -> None:
    if values.shape[0] != epoch.size:
        raise ValueError(f"{name} row count differs from NH_TW1 in {path}")
    time = np.asarray(values[:, 0], dtype=float)
    if allow_nan:
        finite = np.isfinite(time)
        if not np.array_equal(time[finite], epoch[finite]):
            raise ValueError(f"Finite {name} epochs differ from NH_TW1 row epochs in {path}")
    elif not np.array_equal(time, epoch):
        raise ValueError(f"{name} epochs differ from NH_TW1 in {path}")


def discover_nv_products(
    input_root: Path = DEFAULT_INPUT_ROOT,
    *,
    dates: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[NvProduct]:
    root = Path(input_root)
    if dates is None:
        paths = sorted(root.glob("NV_????????.mat"))
    else:
        paths = [root / f"NV_{date}.mat" for date in dates]
    products: list[NvProduct] = []
    for path in paths:
        match = re.fullmatch(r"NV_(\d{8})\.mat", path.name)
        if not match or not path.exists():
            continue
        date = match.group(1)
        if start_date is not None and date < start_date:
            continue
        if end_date is not None and date > end_date:
            continue
        products.append(NvProduct(date, path))
    return sorted(products, key=lambda product: product.date)


def _nv_variable_names() -> list[str]:
    names = ["Xb_MSO", "mode_TW1", "NV_MSO_2_info", "MINPA_quality_flag_info"]
    for spec in SPECIES.values():
        names.extend(
            [
                spec["density"],
                spec["velocity"],
                f"quality_flag_{spec['suffix']}_TW1",
                f"quality_flag_bit_{spec['suffix']}_TW1",
                f"quality_flag_available_{spec['suffix']}_TW1",
            ]
        )
    return names


def canonical_row_count(path: Path, date: str) -> tuple[int, int]:
    data = loadmat(path, variable_names=["NH_TW1"], squeeze_me=False, struct_as_record=False)
    density = _require_matrix(data, "NH_TW1", 2, path)
    epoch = density[:, 0]
    start, end = _date_bounds(date)
    return int(np.count_nonzero((epoch >= start) & (epoch < end))), int(epoch.size)


def _load_position(path: Path, target_epoch: np.ndarray, tolerance_s: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = loadmat(path, variable_names=["P_TW1", "Pitch_TW1"], squeeze_me=False, struct_as_record=False)
    position = _require_matrix(data, "P_TW1", 4, path)
    pitch = _require_matrix(data, "Pitch_TW1", 2, path)
    if position.shape[0] != pitch.shape[0] or not np.array_equal(position[:, 0], pitch[:, 0], equal_nan=True):
        raise ValueError(f"P_TW1 and Pitch_TW1 time axes differ in {path}")
    source_time = position[:, 0]
    if not np.all(np.isfinite(source_time)):
        raise ValueError(f"P_TW1 time contains non-finite values in {path}")
    order = np.argsort(source_time, kind="stable")
    source_time = source_time[order]
    values = np.where(np.abs(position[order, 1:4]) > 1.0e29, np.nan, position[order, 1:4])
    duplicate = np.r_[False, np.diff(source_time) == 0.0]
    if np.any(duplicate):
        keep = ~duplicate
        source_time = source_time[keep]
        values = values[keep]
    if source_time.size < 2 or not np.all(np.diff(source_time) > 0.0):
        raise ValueError(f"P_TW1 time could not be made strictly increasing in {path}")
    indices, match_dt = nearest_indices(source_time, target_epoch)
    matched = values[indices].astype(float)
    finite = np.all(np.isfinite(matched), axis=1)
    within = finite & (np.abs(match_dt) <= float(tolerance_s))
    matched[~within] = np.nan
    return matched, match_dt, {
        "position_time_reordered_flag": int(not np.array_equal(order, np.arange(order.size))),
        "position_time_duplicates_removed": int(np.count_nonzero(duplicate)),
        "finite_position_match_rows": int(np.count_nonzero(finite)),
        "position_within_tolerance_rows": int(np.count_nonzero(within)),
        "max_abs_position_match_dt_s": float(np.max(np.abs(match_dt))) if match_dt.size else math.nan,
    }


def _quality_fields(
    data: dict[str, Any],
    spec: dict[str, str],
    epoch: np.ndarray,
    rows: np.ndarray,
    path: Path,
) -> dict[str, np.ndarray]:
    suffix = spec["suffix"]
    flag = _require_matrix(data, f"quality_flag_{suffix}_TW1", 2, path)
    bits = _require_matrix(data, f"quality_flag_bit_{suffix}_TW1", 6, path)
    available = _require_matrix(data, f"quality_flag_available_{suffix}_TW1", 2, path)
    for name, values in (("quality flag", flag), ("quality bits", bits), ("quality availability", available)):
        _validate_row_time(f"{suffix} {name}", values, epoch, path, allow_nan=False)
    flag_value = np.asarray(flag[rows, 1], dtype=np.uint32)
    bit_value = np.asarray(bits[rows, 1:6], dtype=float)
    available_value = np.asarray(available[rows, 1], dtype=float)
    if not (np.all(np.isin(bit_value, [0.0, 1.0])) and np.all(np.isin(available_value, [0.0, 1.0]))):
        raise ValueError(f"Non-binary {suffix} quality fields in {path}")
    reconstructed = np.zeros(flag_value.size, dtype=np.uint32)
    for index in range(5):
        reconstructed |= bit_value[:, index].astype(np.uint32) << index
    usable = available_value == 1.0
    if not np.array_equal(reconstructed[usable], flag_value[usable]):
        raise ValueError(f"{suffix} quality bit columns disagree with bitmask in {path}")
    return {
        "quality_flag_bitmask": flag_value,
        "quality_flag_available_flag": available_value.astype(np.int8),
        "quality_flag_epoch_matched_flag": np.ones(flag_value.size, dtype=np.int8),
        "quality_flag_match_dt_s": np.zeros(flag_value.size, dtype=float),
        "quality_flag_bit1_sparse_caution": bit_value[:, 0].astype(np.int8),
        "quality_flag_bit2_sparse_invalid": bit_value[:, 1].astype(np.int8),
        "quality_flag_bit3_uv_contamination": bit_value[:, 2].astype(np.int8),
        "quality_flag_bit4_evenodd_error": bit_value[:, 3].astype(np.int8),
        "quality_flag_bit5_high_channel": bit_value[:, 4].astype(np.int8),
    }


def _fov_flag_from_xb_mso(mode: np.ndarray, xb_mso: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xb_mse = apply_rotation(rotation, xb_mso)
    norm = np.linalg.norm(xb_mse, axis=1)
    finite = np.all(np.isfinite(xb_mse), axis=1) & np.isfinite(norm) & (norm > 0.0)
    theta = np.full(mode.size, np.nan, dtype=float)
    theta[finite] = np.degrees(np.arccos(np.clip(-xb_mse[finite, 2] / norm[finite], -1.0, 1.0)))
    flag = np.full(mode.size, np.nan, dtype=float)
    for mode_value in np.unique(mode[np.isfinite(mode)]).astype(int):
        theta_min, theta_max = minpa_theta_edges_deg(int(mode_value))
        selected = finite & (mode == mode_value)
        flag[selected] = ((theta[selected] >= theta_min) & (theta[selected] <= theta_max)).astype(float)
    return flag, theta


def load_all_energy_day_chunk(
    product: NvProduct,
    momag_root: Path,
    r_root: Path,
    grid: MseGridSpec,
    *,
    position_tolerance_s: float = POSITION_MATCH_TOLERANCE_S,
    rotation_gap_s: float = MATCH_MAX_GAP_S,
) -> AllEnergyDayChunk:
    data = loadmat(product.path, variable_names=_nv_variable_names(), squeeze_me=False, struct_as_record=False)
    nh = _require_matrix(data, "NH_TW1", 2, product.path)
    epoch_all = np.asarray(nh[:, 0], dtype=float)
    if epoch_all.size < 1 or not np.all(np.isfinite(epoch_all)) or not np.all(np.diff(epoch_all) > 0.0):
        raise ValueError(f"NH_TW1 epoch must be finite and strictly increasing in {product.path}")
    start, end = _date_bounds(product.date)
    rows = np.flatnonzero((epoch_all >= start) & (epoch_all < end))
    epoch = epoch_all[rows]
    if epoch.size < 1:
        raise ValueError(f"No canonical UTC-date rows in {product.path}")

    mode_all = _require_matrix(data, "mode_TW1", 2, product.path)
    xb_all = _require_matrix(data, "Xb_MSO", 4, product.path)
    _validate_row_time("mode_TW1", mode_all, epoch_all, product.path, allow_nan=False)
    _validate_row_time("Xb_MSO", xb_all, epoch_all, product.path, allow_nan=True)
    mode = np.asarray(mode_all[rows, 1], dtype=float)
    if not np.all(np.isfinite(mode)) or np.any(mode != np.rint(mode)):
        raise ValueError(f"Non-integer or non-finite mode_TW1 in canonical rows of {product.path}")
    xb_mso = np.asarray(xb_all[rows, 1:4], dtype=float)

    momag_path = Path(momag_root) / f"Bss{product.date}.mat"
    rotation_path = tw1_rotation_file(product.date, Path(r_root))
    if not momag_path.exists():
        raise FileNotFoundError(momag_path)
    position_mso, position_dt, position_validation = _load_position(
        momag_path, epoch, position_tolerance_s
    )

    rotation_data = load_tw1_mso2mse(rotation_path)
    rotation_time, matrices, rotation_axis_validation = prepare_rotation_time_axis(
        rotation_data["time_s"], rotation_data["R_mso_to_mse"]
    )
    matched_rotation, rotation_dt = interpolate_matrices_nearest(rotation_time, matrices, epoch)
    finite_rotation = np.all(np.isfinite(matched_rotation), axis=(1, 2))
    within_rotation_gap = finite_rotation & (np.abs(rotation_dt) <= float(rotation_gap_s))
    if np.any(finite_rotation):
        valid_rotation = matched_rotation[finite_rotation]
        orthonormal_error = float(
            np.max(np.abs(valid_rotation @ np.swapaxes(valid_rotation, 1, 2) - np.eye(3)))
        )
        determinant_error = float(np.max(np.abs(np.linalg.det(valid_rotation) - 1.0)))
        if orthonormal_error > 1.0e-6 or determinant_error > 1.0e-6:
            raise ValueError(f"R_MSO2MSE is not a proper rotation for {product.date}")
    else:
        orthonormal_error = math.nan
        determinant_error = math.nan

    position_mse = apply_rotation(matched_rotation, position_mso)
    xb_mse = apply_rotation(matched_rotation, xb_mso)
    fov_flag, fov_theta = _fov_flag_from_xb_mso(mode, xb_mso, matched_rotation)
    species: dict[str, dict[str, np.ndarray]] = {}
    for species_id, spec in SPECIES.items():
        density_matrix = _require_matrix(data, spec["density"], 2, product.path)
        velocity_matrix = _require_matrix(data, spec["velocity"], 4, product.path)
        _validate_row_time(spec["density"], density_matrix, epoch_all, product.path, allow_nan=False)
        _validate_row_time(spec["velocity"], velocity_matrix, epoch_all, product.path, allow_nan=True)
        density = np.asarray(density_matrix[rows, 1], dtype=float)
        velocity_mso = np.asarray(velocity_matrix[rows, 1:4], dtype=float)
        velocity_mse = apply_rotation(matched_rotation, velocity_mso)
        density_valid = np.isfinite(density) & (density > 0.0)
        velocity_valid = np.all(np.isfinite(velocity_mse), axis=1)
        status = np.full(epoch.size, 4, dtype=np.int16)
        status[density_valid & ~velocity_valid] = 5
        status[density_valid & velocity_valid] = 1
        species[species_id] = {
            "density_cm3": density,
            "v_mse_x_km_s": velocity_mse[:, 0],
            "v_mse_y_km_s": velocity_mse[:, 1],
            "v_mse_z_km_s": velocity_mse[:, 2],
            "processing_status_code": status,
            "density_valid_flag": density_valid.astype(np.int8),
            "velocity_valid_flag": velocity_valid.astype(np.int8),
            **_quality_fields(data, spec, epoch_all, rows, product.path),
        }

    flat_bin, position_rm = bin_mse_positions(position_mse, grid)
    common = {
        "epoch_unix_s": epoch,
        "pos_mse_x_km": position_mse[:, 0],
        "pos_mse_y_km": position_mse[:, 1],
        "pos_mse_z_km": position_mse[:, 2],
        "pos_mse_x_rm": position_rm[:, 0],
        "pos_mse_y_rm": position_rm[:, 1],
        "pos_mse_z_rm": position_rm[:, 2],
        "mse_plus_z_in_minpa_fov_flag": fov_flag,
        # In-memory geometry used by classified statistics. These fields are
        # intentionally not part of COMMON_RECORD_FIELDS, so the audited
        # 59-field per-cell archive remains unchanged.
        "xb_mse_x": xb_mse[:, 0],
        "xb_mse_y": xb_mse[:, 1],
        "xb_mse_z": xb_mse[:, 2],
        "r_matrix_match_dt_s": rotation_dt,
        "r_matrix_valid_flag": finite_rotation.astype(np.int8),
        "r_matrix_within_tolerance_flag": within_rotation_gap.astype(np.int8),
    }

    product_info = data.get("NV_MSO_2_info")
    quality_info = data.get("MINPA_quality_flag_info")
    product_metadata = {
        "coordinate_system": _as_text(_info_value(product_info, "coordinate_system", "MSO")),
        "density_processing": _as_text(_info_value(product_info, "density_processing", "NV_MSO_2 density")),
        "velocity_processing": _as_text(_info_value(product_info, "velocity_processing", "NV_MSO_2 MSO velocity")),
        "source_nv_mso": _as_text(_info_value(product_info, "source_nv_mso", "")),
        "source_day_spe": _as_text(_info_value(product_info, "source_day_spe", "")),
        "source_momag": _as_text(_info_value(product_info, "source_momag", "")),
    }
    quality_metadata = {
        "algorithm_version": _as_text(_info_value(quality_info, "algorithm_version", "unknown")),
        "time_match_tolerance_s": float(_info_value(quality_info, "time_match_tolerance_s", 0.001)),
        "flag_encoding": _as_text(_info_value(quality_info, "flag_encoding", "five-bit species mask")),
        "missing_semantics": _as_text(
            _info_value(
                quality_info,
                "missing_semantics",
                "quality_flag_available=false means numeric zero is a placeholder",
            )
        ),
    }
    validation: dict[str, Any] = {
        "source_rows_total": int(epoch_all.size),
        "canonical_utc_rows": int(epoch.size),
        "noncanonical_overlap_rows_excluded": int(epoch_all.size - epoch.size),
        "rows_finite_position_in_grid": int(np.count_nonzero(flat_bin >= 0)),
        "rows_outside_or_invalid_position": int(epoch.size - np.count_nonzero(flat_bin >= 0)),
        "finite_rotation_rows": int(np.count_nonzero(finite_rotation)),
        "rotation_within_tolerance_rows": int(np.count_nonzero(within_rotation_gap)),
        "max_abs_rotation_match_dt_s": float(np.max(np.abs(rotation_dt))) if rotation_dt.size else math.nan,
        "rotation_orthonormal_max_error": orthonormal_error,
        "rotation_determinant_max_error": determinant_error,
        "finite_fov_flag_rows": int(np.count_nonzero(np.isfinite(fov_flag))),
        "fov_inside_rows": int(np.count_nonzero(fov_flag == 1.0)),
        "fov_theta_min_deg": float(np.nanmin(fov_theta)) if np.any(np.isfinite(fov_theta)) else math.nan,
        "fov_theta_max_deg": float(np.nanmax(fov_theta)) if np.any(np.isfinite(fov_theta)) else math.nan,
        **position_validation,
        **rotation_axis_validation,
    }
    return AllEnergyDayChunk(
        date=product.date,
        source_file=str(product.path),
        momag_file=str(momag_path),
        rotation_file=str(rotation_path),
        product_metadata=product_metadata,
        quality_metadata=quality_metadata,
        common=common,
        species=species,
        flat_bin=flat_bin,
        validation=validation,
    )


def build_compound_records(
    chunk: AllEnergyDayChunk, rows: np.ndarray
) -> tuple[np.ndarray, dict[str, dict[str, str]]]:
    selected = np.asarray(rows, dtype=np.int64)
    fields: list[tuple[str, np.dtype[Any]]] = []
    sources: list[tuple[str, str, np.ndarray]] = []
    metadata: dict[str, dict[str, str]] = {}
    for name in COMMON_RECORD_FIELDS:
        array = np.asarray(chunk.common[name])
        fields.append((name, array.dtype))
        sources.append((name, name, array))
    for species_id, spec in SPECIES.items():
        for name in SPECIES_RECORD_FIELDS:
            output_name = f"{spec['prefix']}_{name}"
            array = np.asarray(chunk.species[species_id][name])
            fields.append((output_name, array.dtype))
            sources.append((output_name, name, array))
    records = np.empty(selected.size, dtype=np.dtype(fields))
    for output_name, source_name, array in sources:
        records[output_name] = array[selected]
        units, description = FIELD_METADATA[source_name]
        metadata[output_name] = {"units": units, "description": description}
        if "mse" in source_name:
            metadata[output_name]["coordinate_system"] = "MSE"
    if records.dtype.names != EXPECTED_RECORD_FIELDS:
        raise RuntimeError("Internal all-energy compound record field order changed")
    return records, metadata


def _cell_metadata(flat_bin: int, grid: MseGridSpec) -> dict[str, Any]:
    ix, iy, iz = np.unravel_index(int(flat_bin), grid.shape)
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": "Tianwen-1",
        "instrument": "MINPA",
        "input_product": "NV_MSO_2",
        "energy_selection": "all energy bins represented by NV_MSO_2 moments",
        "coordinate_system": "MSE",
        "flat_bin": int(flat_bin),
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
    }


def _ensure_cell_metadata(handle: h5py.File, flat_bin: int, grid: MseGridSpec) -> None:
    expected = _cell_metadata(flat_bin, grid)
    if "schema_name" not in handle.attrs:
        for key, value in expected.items():
            handle.attrs[key] = value
        handle.attrs["created_utc"] = datetime.now(UTC).isoformat()
    else:
        for key, value in expected.items():
            actual = handle.attrs.get(key)
            if isinstance(value, float):
                if actual is None or not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1.0e-10):
                    raise ValueError(f"Cell metadata mismatch for {handle.filename}: {key}")
            elif actual != value:
                raise ValueError(f"Cell metadata mismatch for {handle.filename}: {key}")
    handle.attrs["updated_utc"] = datetime.now(UTC).isoformat()


def write_cell_day(
    path: Path,
    chunk: AllEnergyDayChunk,
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
        day = staging.create_group(chunk.date)
        day.attrs["complete"] = np.int8(0)
        day.attrs["row_count"] = int(rows.size)
        day.attrs["source_nv_mso2_file"] = chunk.source_file
        day.attrs["source_momag_file"] = chunk.momag_file
        day.attrs["rotation_file"] = chunk.rotation_file
        day.attrs["canonical_row_rule"] = "keep only epochs whose true UTC date equals NV filename date"
        day.attrs["density_processing"] = chunk.product_metadata["density_processing"]
        day.attrs["velocity_processing"] = chunk.product_metadata["velocity_processing"]
        day.attrs["position_matching"] = f"nearest P_TW1 epoch within {POSITION_MATCH_TOLERANCE_S:g} s"
        day.attrs["rotation_matching"] = "nearest actual-UTC-day R_MSO2MSE epoch"
        day.attrs["quality_flag_algorithm_version"] = chunk.quality_metadata["algorithm_version"]
        day.attrs["quality_flag_time_match_tolerance_s"] = chunk.quality_metadata["time_match_tolerance_s"]
        day.attrs["quality_flag_encoding"] = chunk.quality_metadata["flag_encoding"]
        day.attrs["quality_flag_missing_semantics"] = chunk.quality_metadata["missing_semantics"]
        day.attrs["processing_status_definition"] = "1=positive finite density and finite MSE velocity; 5=density only; 4=invalid or zero density"
        for key, value in chunk.validation.items():
            day.attrs[key] = value
        records, field_metadata = build_compound_records(chunk, rows)
        dataset = day.create_dataset(
            "records", data=records, compression="gzip", compression_opts=4, shuffle=True
        )
        dataset.attrs["layout"] = "HDF5 compound records; one row per canonical NV_MSO_2 epoch"
        dataset.attrs["field_metadata_json"] = json.dumps(field_metadata, ensure_ascii=False, sort_keys=True)
        day.attrs["complete"] = np.int8(1)
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


def _root_schema(grid: MseGridSpec, rotation_gap_s: float, position_tolerance_s: float) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": "Tianwen-1",
        "instrument": "MINPA",
        "input_product": "NV_MSO_2 all-energy density and MSO bulk velocity moments",
        "energy_selection": "all energy bins represented by NV_MSO_2; no >1 keV threshold",
        "coordinate_system": "MSE",
        "authoritative_time": "NH_TW1[:,0] Unix seconds UTC; velocity rows remain row-aligned even when their time is NaN",
        "canonical_row_rule": "keep only epochs whose true UTC date equals NV filename date",
        "position_matching": {"source": "MOMAG P_TW1", "method": "nearest", "tolerance_s": position_tolerance_s},
        "rotation_matching": {"source": "R_MSO2MSE", "method": "nearest", "quality_tolerance_s": rotation_gap_s},
        "species": ["H+", "O+", "O2+"],
        "record_field_count": len(EXPECTED_RECORD_FIELDS),
        "status_definition": {"1": "density and velocity valid", "5": "density only", "4": "invalid or zero density"},
        "quality_flags": {
            "source": "row-aligned species-specific NV_MSO_2 five-bit masks",
            "encoding": "bit1 sparse caution; bit2 sparse invalid; bit3 UV contamination; bit4 even/odd error; bit5 high channel",
            "missing_semantics": "available=0 means numeric zero is only a placeholder",
        },
        "grid": grid.as_dict(),
        "storage": "one HDF5 file per non-empty MSE cell; one compound records dataset per true UTC date",
        "record_policy": "all finite-position in-grid records retained; moment and quality failures remain represented by values and flags",
        "density_aggregation_policy": "per-record density retained; no cross-epoch sum is interpreted as local density",
    }


def _ensure_root_schema(
    output_root: Path, grid: MseGridSpec, rotation_gap_s: float, position_tolerance_s: float
) -> None:
    path = Path(output_root) / "grid_schema.json"
    required = _root_schema(grid, rotation_gap_s, position_tolerance_s)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in required}
        if comparable != required:
            raise ValueError(f"Existing output schema is incompatible: {path}")
    else:
        _atomic_json(path, {**required, "created_utc": datetime.now(UTC).isoformat()})


def process_product(
    product: NvProduct,
    output_root: Path,
    momag_root: Path,
    r_root: Path,
    grid: MseGridSpec,
    *,
    position_tolerance_s: float = POSITION_MATCH_TOLERANCE_S,
    rotation_gap_s: float = MATCH_MAX_GAP_S,
    overwrite: bool = False,
) -> dict[str, Any]:
    chunk = load_all_energy_day_chunk(
        product,
        momag_root,
        r_root,
        grid,
        position_tolerance_s=position_tolerance_s,
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
        "date": product.date,
        "completed_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "source_nv_mso2_file": chunk.source_file,
        "source_momag_file": chunk.momag_file,
        "rotation_file": chunk.rotation_file,
        "quality_flag_algorithm_version": chunk.quality_metadata["algorithm_version"],
        "rows_source_file": int(chunk.validation["source_rows_total"]),
        "rows_canonical_utc": chunk.rows_total,
        "rows_noncanonical_overlap_excluded": int(chunk.validation["noncanonical_overlap_rows_excluded"]),
        "rows_finite_position_in_grid": chunk.rows_in_grid,
        "rows_outside_or_invalid_position": chunk.rows_total - chunk.rows_in_grid,
        "nonempty_cells": int(unique_bins.size),
        "cell_day_groups_written": written,
        "cell_day_groups_skipped": skipped,
        "working_memory_bytes": chunk.memory_bytes,
        "validation": chunk.validation,
        "cell_files": cell_files,
    }


def run_tw1_all_energy_mse_grid_records(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    momag_root: Path = DEFAULT_TW1_MOMAG_ROOT,
    r_root: Path = DEFAULT_TW1_R_ROOT,
    *,
    dates: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    grid: MseGridSpec | None = None,
    position_tolerance_s: float = POSITION_MATCH_TOLERANCE_S,
    rotation_gap_s: float = MATCH_MAX_GAP_S,
    min_free_gb: float = 20.0,
    overwrite: bool = False,
    max_days: int | None = None,
) -> dict[str, Any]:
    grid = grid or MseGridSpec()
    input_root = Path(input_root)
    output_root = Path(output_root)
    momag_root = Path(momag_root)
    r_root = Path(r_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_root_schema(output_root, grid, rotation_gap_s, position_tolerance_s)
    products = discover_nv_products(
        input_root, dates=dates, start_date=start_date, end_date=end_date
    )
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
                    "source_nv_mso2_file": str(product.path),
                    "source_rows": source_rows,
                    "canonical_utc_rows_excluded": canonical_rows,
                    "momag_file": str(momag_path),
                    "momag_exists": momag_path.exists(),
                    "rotation_file": str(rotation_path),
                    "rotation_exists": rotation_path.exists(),
                    "reason": "cannot assign MSE position without both MOMAG P_TW1 and R_MSO2MSE",
                }
            )
    if max_days is not None:
        processable = processable[: int(max_days)]
    coverage = {
        "created_utc": datetime.now(UTC).isoformat(),
        "input_product_days": len(products),
        "processable_geometry_days": len(processable),
        "excluded_geometry_days": len(excluded),
        "excluded_canonical_utc_rows": int(sum(item["canonical_utc_rows_excluded"] for item in excluded)),
        "excluded": excluded,
    }
    _atomic_json(output_root / "logs" / "geometry_coverage.json", coverage)
    if not processable:
        raise FileNotFoundError("No selected NV_MSO_2 day has both MOMAG and R_MSO2MSE geometry")

    started = datetime.now(UTC)
    day_summaries: list[dict[str, Any]] = []
    max_memory_bytes = 0
    for product in processable:
        summary_path = output_root / "logs" / "days" / f"{product.date}.json"
        if summary_path.exists() and not overwrite:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                max_memory_bytes = max(max_memory_bytes, int(existing.get("working_memory_bytes", 0)))
                day_summaries.append(existing)
                continue
        free_bytes = shutil.disk_usage(output_root).free
        if free_bytes < float(min_free_gb) * (1024**3):
            raise OSError(
                f"Free space {free_bytes / 1024**3:.2f} GiB is below --min-free-gb={min_free_gb:g}"
            )
        summary = process_product(
            product,
            output_root,
            momag_root,
            r_root,
            grid,
            position_tolerance_s=position_tolerance_s,
            rotation_gap_s=rotation_gap_s,
            overwrite=overwrite,
        )
        max_memory_bytes = max(max_memory_bytes, int(summary["working_memory_bytes"]))
        _atomic_json(summary_path, summary)
        day_summaries.append(summary)
        _atomic_json(
            output_root / "logs" / "progress.json",
            {
                "updated_utc": datetime.now(UTC).isoformat(),
                "input_product_days": len(products),
                "processable_days": len(processable),
                "excluded_geometry_days": len(excluded),
                "complete_days": len(day_summaries),
                "last_completed_date": product.date,
                "maximum_daily_working_memory_bytes": max_memory_bytes,
                "free_space_gb": shutil.disk_usage(output_root).free / (1024**3),
            },
        )

    completed = datetime.now(UTC)
    processed_days = int(sum(item.get("completed_utc", "") >= started.isoformat() for item in day_summaries))
    already_complete_days = len(day_summaries) - processed_days
    run_summary = {
        "created_utc": completed.isoformat(),
        "started_utc": started.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "momag_root": str(momag_root),
        "rotation_root": str(r_root),
        "grid": grid.as_dict(),
        "position_match_tolerance_s": position_tolerance_s,
        "rotation_match_tolerance_s": rotation_gap_s,
        "input_product_days": len(products),
        "requested_days": len(processable),
        "processed_days": processed_days,
        "already_complete_days": already_complete_days,
        "excluded_geometry_days": len(excluded),
        "excluded_canonical_utc_rows": coverage["excluded_canonical_utc_rows"],
        "canonical_utc_rows": int(sum(item["rows_canonical_utc"] for item in day_summaries)),
        "in_grid_rows": int(sum(item["rows_finite_position_in_grid"] for item in day_summaries)),
        "cell_day_groups": int(sum(item["nonempty_cells"] for item in day_summaries)),
        "maximum_daily_working_memory_bytes": max_memory_bytes,
        "free_space_gb_after": shutil.disk_usage(output_root).free / (1024**3),
        "days": [
            {key: value for key, value in item.items() if key != "cell_files"}
            for item in day_summaries
        ],
    }
    _atomic_json(output_root / "logs" / "latest_run_summary.json", run_summary)
    return run_summary
