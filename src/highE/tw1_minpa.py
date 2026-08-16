"""Tianwen-1 MINPA high-energy O+ and O2+ daily moments."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Mapping

import h5py
import numpy as np
from scipy.io import loadmat, savemat

from .constants import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TW1_MINPA_ORI_ROOT,
    DEFAULT_TW1_MINPA_PUBLIC_ROOT,
    DEFAULT_TW1_MOMAG_ROOT,
    DEFAULT_TW1_R_ROOT,
    HIGH_ENERGY_MIN_EV,
    SPECIES,
)
from .coordinates import (
    apply_rotation,
    apply_rotation_to_axes,
    finite_data,
    finite_difference_velocity_km_s,
    interpolate_matrices_nearest,
    tw1_orbiter_to_mso_matrix,
    unix_s_to_utc_text,
)
from .minpa_background import (
    DENOISE_POLICY_VERSION,
    MODE1_SHAPE,
    MinpaBackgroundModel,
    UvDetectionConfig,
    apply_background,
    detect_uv_contamination,
    load_background_model,
    load_project_quality_for_times,
    sha256_file,
)
from .minpa_io import read_public_records as read_public_product_records
from .minpa_modes import angular_geometry, energy_bin_widths_eV, minpa_directions, minpa_to_body_direction, mode_layout, species_mass_indices, split_raw_record
from .minpa_multimode_background import (
    MultimodeBackgroundModel,
    apply_multimode_background,
    load_multimode_background_model,
    mode_channel_shape,
)


EV_J = 1.602176634e-19
MP_KG = 1.67262192369e-27
ATTITUDE_MATCH_TOLERANCE_S = 5.0
RELIABLE_DENSITY_THRESHOLD_CM3 = 0.01
DEFAULT_PROJECT_QUALITY_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "tw1_minpa_quality_flags_all_species"
)
BackgroundPolicy = Literal[
    "none",
    "subtract-and-reject-uv",
    "paper-channel-subtract",
    "multimode-paper-channel-subtract",
    "all-approved-static-channel-subtract",
]


@dataclass(frozen=True)
class Tw1Paths:
    ori_root: Path = DEFAULT_TW1_MINPA_ORI_ROOT
    public_root: Path = DEFAULT_TW1_MINPA_PUBLIC_ROOT
    momag_root: Path = DEFAULT_TW1_MOMAG_ROOT
    r_root: Path = DEFAULT_TW1_R_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    project_quality_root: Path = DEFAULT_PROJECT_QUALITY_ROOT


class SkipDay(RuntimeError):
    """Raised for known data-format gaps that should be skipped in batch mode."""


@dataclass(frozen=True)
class MomagAttitude:
    time: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    yaw: np.ndarray
    position_mso_km: np.ndarray
    spacecraft_velocity_mso_km_s: np.ndarray


def tw1_rotation_file(date: str, root: Path = DEFAULT_TW1_R_ROOT) -> Path:
    path = Path(root) / f"R_MSO2MSE_Tianwen-1_{date}.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_tw1_mso2mse(path: Path) -> dict[str, np.ndarray]:
    data = loadmat(path)
    if {"data1", "data2"} <= data.keys():
        time_s = np.asarray(data["data1"], dtype=float).reshape(-1)
        matrix = np.asarray(data["data2"], dtype=float)
    elif {"t_R_MSO2MSE", "R_MSO2MSE"} <= data.keys():
        time_s = np.asarray(data["t_R_MSO2MSE"], dtype=float).reshape(-1)
        matrix = np.asarray(data["R_MSO2MSE"], dtype=float)
    else:
        raise KeyError(f"Cannot find Tianwen-1 R_MSO2MSE variables in {path}")
    return {"time_s": time_s, "R_mso_to_mse": matrix}


def load_momag_attitude(day: str, momag_root: Path) -> MomagAttitude | None:
    path = Path(momag_root) / f"Bss{day}.mat"
    if not path.exists():
        return None
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    required = {"Pitch_TW1", "Roll_TW1", "Yaw_TW1", "P_TW1"}
    if not required <= data.keys():
        return None
    pitch = finite_data(data["Pitch_TW1"])
    roll = finite_data(data["Roll_TW1"])
    yaw = finite_data(data["Yaw_TW1"])
    position = finite_data(data["P_TW1"])
    time_s = np.asarray(pitch[:, 0], dtype=float)
    position_mso_km = np.asarray(position[:, 1:4], dtype=float)
    return MomagAttitude(
        time=time_s,
        pitch=np.asarray(pitch[:, 1], dtype=float),
        roll=np.asarray(roll[:, 1], dtype=float),
        yaw=np.asarray(yaw[:, 1], dtype=float),
        position_mso_km=position_mso_km,
        spacecraft_velocity_mso_km_s=finite_difference_velocity_km_s(time_s, position_mso_km),
    )


def load_momag_for_day(date: str, momag_root: Path) -> MomagAttitude | None:
    start = datetime.strptime(date, "%Y%m%d").replace(tzinfo=UTC)
    days = [(start + timedelta(days=offset)).strftime("%Y%m%d") for offset in (-1, 0, 1)]
    parts = [att for day in days if (att := load_momag_attitude(day, momag_root)) is not None]
    if not parts:
        return None
    time_s = np.concatenate([part.time for part in parts])
    order = np.argsort(time_s)
    return MomagAttitude(
        time=time_s[order],
        pitch=np.concatenate([part.pitch for part in parts])[order],
        roll=np.concatenate([part.roll for part in parts])[order],
        yaw=np.concatenate([part.yaw for part in parts])[order],
        position_mso_km=np.concatenate([part.position_mso_km for part in parts], axis=0)[order],
        spacecraft_velocity_mso_km_s=np.concatenate([part.spacecraft_velocity_mso_km_s for part in parts], axis=0)[order],
    )


def nearest_attitude(time_s: float, att: MomagAttitude | None) -> tuple[float, float, float, np.ndarray, np.ndarray, float]:
    if att is None or att.time.size == 0:
        return math.nan, math.nan, math.nan, np.full(3, np.nan), np.full(3, np.nan), math.nan
    idx = int(np.searchsorted(att.time, time_s))
    candidates = [i for i in (idx - 1, idx, idx + 1) if 0 <= i < att.time.size]
    best = min(candidates, key=lambda i: abs(att.time[i] - time_s))
    dt = float(abs(att.time[best] - time_s))
    if dt > ATTITUDE_MATCH_TOLERANCE_S:
        return math.nan, math.nan, math.nan, np.full(3, np.nan), np.full(3, np.nan), dt
    return (
        float(att.pitch[best]),
        float(att.roll[best]),
        float(att.yaw[best]),
        att.position_mso_km[best].astype(float),
        att.spacecraft_velocity_mso_km_s[best].astype(float),
        dt,
    )


def date_bounds(date: str) -> tuple[float, float]:
    start = datetime.strptime(date, "%Y%m%d").replace(tzinfo=UTC)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def minpa_mode(path: Path) -> int:
    match = re.search(r"MINPA-MOD(\d+)-", path.name)
    if not match:
        raise ValueError(f"Cannot infer MINPA mode from {path.name}")
    return int(match.group(1))


def _overlap_files(root: Path, date: str, suffix: str) -> list[Path]:
    pattern = re.compile(r"MINPA-MOD(\d+)-.*?_(\d{8})(\d{6})_(\d{8})(\d{6})_")
    out: list[Path] = []
    for path in sorted(Path(root).glob(f"*{suffix}")):
        match = pattern.search(path.name)
        if not match or int(match.group(1)) == 13:
            continue
        if match.group(2) <= date <= match.group(4):
            out.append(path)
    return out


def source_files_for_day(date: str, paths: Tw1Paths) -> tuple[str, list[Path]]:
    ori = _overlap_files(paths.ori_root, date, ".mat")
    if ori:
        return "ori", ori
    public = _overlap_files(paths.public_root, date, ".2B")
    if public:
        return "public", public
    raise FileNotFoundError(f"No Tianwen-1 MINPA ori/public files overlap {date}")


def _decode_matlab_char_array(arr: np.ndarray) -> str:
    values = np.asarray(arr)
    if values.dtype.kind in ("u", "i"):
        return "".join(chr(int(value)) for value in values.reshape(-1) if int(value) != 0)
    return str(values)


def _matlab_char(handle: h5py.File, value: h5py.Reference | np.ndarray | np.generic) -> str:
    if isinstance(value, h5py.Reference):
        arr = np.asarray(handle[value])
    else:
        arr = np.asarray(value)
    if arr.dtype.kind in ("u", "i"):
        return _decode_matlab_char_array(arr)
    return str(arr)


def _matlab_t_values(handle: h5py.File, dataset: h5py.Dataset) -> list[np.ndarray]:
    arr = np.asarray(dataset)
    if arr.dtype == object:
        return [np.asarray(handle[ref]) for ref in arr.reshape(-1)]
    matlab_class = dataset.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, np.bytes_):
        matlab_class = bytes(matlab_class)
    if matlab_class == b"char" and arr.ndim == 2 and arr.shape[1] > 1:
        return [arr[:, idx] for idx in range(arr.shape[1])]
    if arr.ndim == 2 and arr.shape[1] > 1:
        return [arr[:, idx] for idx in range(arr.shape[1])]
    return [arr]


def read_ori_records(path: Path, start_s: float, end_s: float) -> list[tuple[float, np.ndarray]]:
    rows: list[tuple[float, np.ndarray]] = []
    with h5py.File(path, "r") as handle:
        group = handle["tw1_MINPA_data"]
        utc_values = _matlab_t_values(handle, handle[group["value"][0, 0]]["t"])
        ion_values = _matlab_t_values(handle, handle[group["value"][41, 0]]["t"])
        for utc_value, ion_value in zip(utc_values, ion_values):
            utc_text = _matlab_char(handle, utc_value)
            if not utc_text.strip():
                continue
            value = utc_text.rstrip("Z")
            try:
                time_s = datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp()
            except ValueError:
                try:
                    time_s = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC).timestamp()
                except ValueError:
                    continue
            if start_s <= time_s < end_s:
                rows.append((time_s, np.asarray(ion_value, dtype=float).reshape(-1)))
    return rows


def read_public_records(path: Path, start_s: float, end_s: float) -> list[tuple[float, np.ndarray]]:
    return [(record.time_unix_s, record.ion_dpf) for record in read_public_product_records(path, start_s, end_s)]


def split_mode12_records(mode: int, time_s: float, counts: np.ndarray) -> list[tuple[float, np.ndarray]]:
    return split_raw_record(mode, time_s, counts)


def _species_groups(mode: int, species_id: str) -> list[int]:
    return species_mass_indices(mode, {"Oplus": "O+", "O2plus": "O2+"}[species_id]).tolist()


def _rotation_body_to_mso(pitch: float, roll: float, yaw: float) -> np.ndarray:
    return tw1_orbiter_to_mso_matrix(roll, pitch, yaw)


def _directions_body(mode: int) -> tuple[np.ndarray, np.ndarray]:
    _, _, omega = angular_geometry(mode)
    dirs_minpa = minpa_directions(mode)
    dirs_body = minpa_to_body_direction(dirs_minpa)
    return dirs_body, np.asarray(omega, dtype=float)


def minpa_theta_edges_deg(mode: int) -> tuple[float, float]:
    edges = mode_layout(mode).pitch_edges_deg
    if edges.size < 2:
        return math.nan, math.nan
    return float(np.nanmin(edges)), float(np.nanmax(edges))


def mse_plus_z_minpa_fov_flag(mode: int, axes_mse: np.ndarray) -> tuple[float, float, np.ndarray, float, float]:
    target_body = np.asarray(axes_mse, dtype=float)[:, 2]
    theta_min, theta_max = minpa_theta_edges_deg(mode)
    if not np.all(np.isfinite(target_body)) or not np.isfinite(theta_min + theta_max):
        return math.nan, math.nan, target_body, theta_min, theta_max
    norm = float(np.linalg.norm(target_body))
    if not np.isfinite(norm) or norm <= 0.0:
        return math.nan, math.nan, target_body, theta_min, theta_max
    target_body = target_body / norm
    # MINPA ion FOV is 360 deg in azimuth and 90 deg in polar angle. In the
    # validated spacecraft-body convention used here spans the -X_body
    # hemisphere, so theta is the polar angle away from -X_body.
    theta_deg = float(np.degrees(np.arccos(np.clip(-target_body[0], -1.0, 1.0))))
    flag = float(theta_min <= theta_deg <= theta_max)
    return flag, theta_deg, target_body, theta_min, theta_max


def minpa_fov_summary(mode: int, rotation_body_to_frame: np.ndarray | None, frame: str) -> dict[str, float]:
    fields = {
        f"fov_{frame}_attitude_valid": 0.0,
        f"fov_{frame}_solid_angle_sr": math.nan,
        f"fov_{frame}_x_pos_solid_angle_fraction": math.nan,
        f"fov_{frame}_x_neg_solid_angle_fraction": math.nan,
        f"fov_{frame}_y_pos_solid_angle_fraction": math.nan,
        f"fov_{frame}_y_neg_solid_angle_fraction": math.nan,
        f"fov_{frame}_z_pos_solid_angle_fraction": math.nan,
        f"fov_{frame}_z_neg_solid_angle_fraction": math.nan,
    }
    for axis in "xyz":
        fields[f"fov_{frame}_dir_{axis}_weighted_mean"] = math.nan
        fields[f"fov_{frame}_dir_{axis}_min"] = math.nan
        fields[f"fov_{frame}_dir_{axis}_max"] = math.nan
    if rotation_body_to_frame is None:
        return fields
    dirs_body, omega = _directions_body(mode)
    dirs_frame = dirs_body @ rotation_body_to_frame.T
    valid = np.isfinite(omega) & (omega > 0.0) & np.all(np.isfinite(dirs_frame), axis=1)
    if not np.any(valid):
        return fields
    dirs = dirs_frame[valid]
    weights = omega[valid]
    total = float(np.nansum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return fields
    mean_dir = np.nansum(dirs * weights[:, None], axis=0) / total
    fields[f"fov_{frame}_attitude_valid"] = 1.0
    fields[f"fov_{frame}_solid_angle_sr"] = total
    for i, axis in enumerate("xyz"):
        fields[f"fov_{frame}_dir_{axis}_weighted_mean"] = float(mean_dir[i])
        fields[f"fov_{frame}_dir_{axis}_min"] = float(np.nanmin(dirs[:, i]))
        fields[f"fov_{frame}_dir_{axis}_max"] = float(np.nanmax(dirs[:, i]))
        fields[f"fov_{frame}_{axis}_pos_solid_angle_fraction"] = float(np.nansum(weights[dirs[:, i] >= 0.0]) / total)
        fields[f"fov_{frame}_{axis}_neg_solid_angle_fraction"] = float(np.nansum(weights[dirs[:, i] < 0.0]) / total)
    return fields


def high_energy_moment(
    counts: np.ndarray,
    mode: int,
    species_id: str,
    rotation_body_to_mso: np.ndarray | None,
    spacecraft_velocity_mso_km_s: np.ndarray,
    high_energy_min_eV: float,
    add_spacecraft_velocity: bool,
) -> tuple[float, np.ndarray, int, int]:
    layout = mode_layout(mode)
    energy = layout.energy_eV
    masses = layout.mass_amu
    theta, _, omega = angular_geometry(mode)
    groups = _species_groups(mode, species_id)
    if not groups or energy.size == 0 or theta.size == 0:
        return math.nan, np.full(3, np.nan), 0, 2
    expected = energy.size * theta.size * masses.size
    flat = np.asarray(counts, dtype=float).reshape(-1)
    if flat.size != expected:
        return math.nan, np.full(3, np.nan), 0, 6
    energy_mask = energy > high_energy_min_eV
    denergy = energy_bin_widths_eV(mode)[energy_mask]
    if not np.any(energy_mask):
        return math.nan, np.full(3, np.nan), 0, 2
    cube = flat.reshape((energy.size, theta.size, masses.size))
    density_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    dirs_mso = None
    if rotation_body_to_mso is not None and layout.directional_velocity_available:
        dirs_body, _ = _directions_body(mode)
        dirs_mso = dirs_body @ rotation_body_to_mso.T
    for group in groups:
        mass = masses[group]
        speed_km_s = np.sqrt(2.0 * EV_J * energy[energy_mask] / (mass * MP_KG)) / 1000.0
        flux = cube[energy_mask, :, group]
        density = flux * denergy[:, None] * np.asarray(omega, dtype=float)[None, :] / (speed_km_s[:, None] * 1.0e5)
        density_parts.append(density.reshape(-1))
        if dirs_mso is not None:
            velocity = speed_km_s[:, None, None] * dirs_mso[None, :, :]
            velocity_parts.append(velocity.reshape(-1, 3))
    n_cm3 = np.concatenate(density_parts)
    valid = np.isfinite(n_cm3) & (n_cm3 > 0.0)
    if not np.any(valid):
        return math.nan, np.full(3, np.nan), 0, 4
    density_sum = float(np.nansum(n_cm3[valid]))
    if dirs_mso is None or not velocity_parts:
        return density_sum, np.full(3, np.nan), int(np.count_nonzero(valid)), 5
    velocity_mso = np.concatenate(velocity_parts, axis=0)
    valid &= np.all(np.isfinite(velocity_mso), axis=1)
    if not np.any(valid):
        return density_sum, np.full(3, np.nan), 0, 5
    if add_spacecraft_velocity and np.all(np.isfinite(spacecraft_velocity_mso_km_s)):
        velocity_mso = velocity_mso + spacecraft_velocity_mso_km_s[None, :]
    weights = n_cm3[valid]
    bulk = np.nansum(velocity_mso[valid] * weights[:, None], axis=0) / float(np.nansum(weights))
    return density_sum, bulk.astype(float), int(np.count_nonzero(valid)), 1


@dataclass(frozen=True)
class PreparedBackgroundRecords:
    """Background-corrected record payload and auditable per-record diagnostics."""

    corrected_counts: list[np.ndarray]
    background_applied: np.ndarray
    removed_dpf_fraction: np.ndarray
    background_dominated_channel_fraction: np.ndarray
    uv_candidate: np.ndarray
    uv_algorithm_confirmed: np.ndarray
    uv_confirmed: np.ndarray
    uv_project_quality_flag: np.ndarray
    uv_rejected: np.ndarray
    uv_maximum_robust_sigma: np.ndarray
    uv_affected_azimuth_sector_count: np.ndarray
    project_quality_available: np.ndarray
    project_quality_flag: np.ndarray


def background_model_is_provisional(model: MinpaBackgroundModel) -> bool:
    """Return true when any model interval is explicitly labeled provisional."""

    return any(
        "provisional" in str(interval.get("source", "")).lower()
        or "provisional" in str(interval.get("label", "")).lower()
        for interval in model.approved_intervals
    )


def prepare_background_records(
    modes: np.ndarray,
    time_unix_s: np.ndarray,
    counts: list[np.ndarray],
    model: MinpaBackgroundModel,
    *,
    project_quality_root: Path | None = None,
    uv_config: UvDetectionConfig | None = None,
) -> PreparedBackgroundRecords:
    """Correct only Mode-1 records and combine morphology/project UV flags."""

    mode_values = np.asarray(modes, dtype=int).reshape(-1)
    time = np.asarray(time_unix_s, dtype=float).reshape(-1)
    if mode_values.size != time.size or len(counts) != time.size:
        raise ValueError("modes, time_unix_s, and counts must have matching record counts")
    if model.mode != 1 or not model.valid:
        raise ValueError("A valid Mode-1 background model is required")
    n = time.size
    corrected_counts = [np.asarray(value, dtype=float).reshape(-1).copy() for value in counts]
    background_applied = np.zeros(n, dtype=bool)
    removed_fraction = np.full(n, np.nan, dtype=float)
    dominated_fraction = np.full(n, np.nan, dtype=float)
    uv_candidate = np.zeros(n, dtype=bool)
    uv_algorithm_confirmed = np.zeros(n, dtype=bool)
    uv_confirmed = np.zeros(n, dtype=bool)
    uv_project = np.zeros(n, dtype=bool)
    uv_maximum_sigma = np.full(n, np.nan, dtype=float)
    uv_sector_count = np.zeros(n, dtype=np.int16)
    quality_available = np.zeros(n, dtype=bool)
    quality_flag = np.zeros(n, dtype=np.uint32)

    mode1_indices = np.array(
        [
            index
            for index, (mode, value) in enumerate(zip(mode_values, corrected_counts))
            if mode == 1 and value.size == int(np.prod(MODE1_SHAPE))
        ],
        dtype=int,
    )
    if mode1_indices.size:
        cube = np.stack(
            [corrected_counts[index].reshape(MODE1_SHAPE) for index in mode1_indices]
        )
        correction = apply_background(cube, model)
        uv = detect_uv_contamination(
            cube,
            time[mode1_indices],
            model,
            uv_config,
        )
        for local_index, record_index in enumerate(mode1_indices):
            corrected_counts[record_index] = correction.corrected_dpf[local_index].reshape(-1)
        background_applied[mode1_indices] = True
        raw_sum = np.nansum(np.maximum(correction.raw_dpf, 0.0), axis=(1, 2, 3, 4))
        removed_sum = np.nansum(np.maximum(correction.removed_dpf, 0.0), axis=(1, 2, 3, 4))
        removed_fraction[mode1_indices] = np.divide(
            removed_sum,
            raw_sum,
            out=np.full(mode1_indices.size, np.nan, dtype=float),
            where=raw_sum > 0.0,
        )
        dominated_fraction[mode1_indices] = np.mean(
            correction.background_dominated_mask, axis=(1, 2, 3, 4)
        )
        uv_candidate[mode1_indices] = uv.candidate_mask
        uv_algorithm_confirmed[mode1_indices] = uv.confirmed_mask
        uv_confirmed[mode1_indices] = uv.confirmed_mask
        uv_maximum_sigma[mode1_indices] = uv.maximum_robust_sigma
        uv_sector_count[mode1_indices] = uv.affected_azimuth_sector_count

        if project_quality_root is not None and Path(project_quality_root).exists():
            flags, available = load_project_quality_for_times(
                time[mode1_indices], Path(project_quality_root)
            )
            quality_flag[mode1_indices] = flags
            quality_available[mode1_indices] = available
            project_uv_local = available & ((flags & np.uint32(4)) != 0)
            uv_project[mode1_indices] = project_uv_local
            uv_candidate[mode1_indices] |= project_uv_local
            uv_confirmed[mode1_indices] |= project_uv_local

    return PreparedBackgroundRecords(
        corrected_counts=corrected_counts,
        background_applied=background_applied,
        removed_dpf_fraction=removed_fraction,
        background_dominated_channel_fraction=dominated_fraction,
        uv_candidate=uv_candidate,
        uv_algorithm_confirmed=uv_algorithm_confirmed,
        uv_confirmed=uv_confirmed,
        uv_project_quality_flag=uv_project,
        uv_rejected=uv_confirmed.copy(),
        uv_maximum_robust_sigma=uv_maximum_sigma,
        uv_affected_azimuth_sector_count=uv_sector_count,
        project_quality_available=quality_available,
        project_quality_flag=quality_flag,
    )


def prepare_multimode_background_records(
    modes: np.ndarray,
    time_unix_s: np.ndarray,
    counts: list[np.ndarray],
    models: Mapping[int, MultimodeBackgroundModel],
) -> PreparedBackgroundRecords:
    """Correct finalized Mode-4/12 records after any Mode-12 raw split.

    Models are selected strictly by mode. Unsupported and unreviewed mass bins
    remain byte-for-byte numerically equal to the supplied floating values.
    Mode 1 and Mode 7 are intentionally left untouched by this policy.
    """

    mode_values = np.asarray(modes, dtype=int).reshape(-1)
    time = np.asarray(time_unix_s, dtype=float).reshape(-1)
    if mode_values.size != time.size or len(counts) != time.size:
        raise ValueError("modes, time_unix_s, and counts must have matching record counts")
    for mode, model in models.items():
        if int(mode) not in (4, 12) or model.mode != int(mode):
            raise ValueError(f"Multimode bundle key/model mismatch for mode {mode}")
        if not model.valid or model.review_status != "finalized":
            raise ValueError(
                f"Refusing non-finalized Mode {mode} model: {list(model.invalid_reasons)}"
            )
    n = time.size
    corrected = [np.asarray(value, dtype=float).reshape(-1).copy() for value in counts]
    applied = np.zeros(n, dtype=bool)
    removed_fraction = np.full(n, np.nan)
    dominated_fraction = np.full(n, np.nan)
    for mode in (4, 12):
        model = models.get(mode)
        if model is None:
            continue
        expected = int(np.prod(mode_channel_shape(mode)))
        indices = np.asarray(
            [index for index, value in enumerate(corrected) if mode_values[index] == mode and value.size == expected],
            dtype=int,
        )
        if not indices.size:
            continue
        cube = np.stack([corrected[index].reshape(model.shape) for index in indices])
        result = apply_multimode_background(cube, model)
        for local, record_index in enumerate(indices):
            corrected[record_index] = result.corrected_dpf[local].reshape(-1)
        applied[indices] = True
        raw_sum = np.nansum(np.maximum(result.raw_dpf, 0.0), axis=(1, 2, 3, 4))
        removed_sum = np.nansum(np.maximum(result.removed_dpf, 0.0), axis=(1, 2, 3, 4))
        removed_fraction[indices] = np.divide(
            removed_sum,
            raw_sum,
            out=np.full(indices.size, np.nan),
            where=raw_sum > 0.0,
        )
        model_background = np.broadcast_to(model.background_dpf, cube.shape)
        considered = np.isfinite(cube) & np.isfinite(model_background) & np.broadcast_to(model.valid_channel_mask, cube.shape)
        dominated = considered & (cube <= model_background)
        dominated_fraction[indices] = np.divide(
            dominated.sum(axis=(1, 2, 3, 4)),
            considered.sum(axis=(1, 2, 3, 4)),
            out=np.full(indices.size, np.nan),
            where=considered.sum(axis=(1, 2, 3, 4)) > 0,
        )
    zeros = np.zeros(n, dtype=bool)
    return PreparedBackgroundRecords(
        corrected_counts=corrected,
        background_applied=applied,
        removed_dpf_fraction=removed_fraction,
        background_dominated_channel_fraction=dominated_fraction,
        uv_candidate=zeros.copy(),
        uv_algorithm_confirmed=zeros.copy(),
        uv_confirmed=zeros.copy(),
        uv_project_quality_flag=zeros.copy(),
        uv_rejected=zeros.copy(),
        uv_maximum_robust_sigma=np.full(n, np.nan),
        uv_affected_azimuth_sector_count=np.zeros(n, dtype=np.int16),
        project_quality_available=zeros.copy(),
        project_quality_flag=np.zeros(n, dtype=np.uint32),
    )


def load_unified_static_background_bundle(
    path: Path,
) -> tuple[MinpaBackgroundModel, dict[int, MultimodeBackgroundModel]]:
    """Load the frozen, time-invariant Mode-1/4/12 all-approved bundle."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("Unified static background bundle must be frozen")
    if not str(payload.get("temporal_policy", "")).startswith("none"):
        raise ValueError("Unified static background bundle must not use temporal scaling")
    items = payload.get("models", {})
    if set(items) != {"1", "4", "12"}:
        raise ValueError("Unified static bundle must contain exactly Modes 1, 4 and 12")

    def checked_path(item: Mapping[str, Any]) -> Path:
        model_path = Path(str(item["path"]))
        if not model_path.is_absolute():
            model_path = source.parent / model_path
        if sha256_file(model_path) != str(item["sha256"]):
            raise ValueError(f"Unified static model hash mismatch: {model_path}")
        return model_path

    mode1 = load_background_model(checked_path(items["1"]))
    if mode1.mode != 1 or not mode1.valid:
        raise ValueError("Unified static Mode-1 model is invalid")
    multimode: dict[int, MultimodeBackgroundModel] = {}
    for mode in (4, 12):
        model = load_multimode_background_model(checked_path(items[str(mode)]))
        if model.mode != mode or not model.valid or model.review_status != "finalized":
            raise ValueError(f"Unified static Mode-{mode} model is invalid")
        multimode[mode] = model
    return mode1, multimode


def prepare_unified_static_background_records(
    modes: np.ndarray,
    time_unix_s: np.ndarray,
    counts: list[np.ndarray],
    mode1_model: MinpaBackgroundModel,
    multimode_models: Mapping[int, MultimodeBackgroundModel],
    *,
    project_quality_root: Path | None = None,
    uv_config: UvDetectionConfig | None = None,
) -> PreparedBackgroundRecords:
    """Apply static per-channel subtraction to Modes 1, 4 and 12 in one pass."""

    mode1 = prepare_background_records(
        modes, time_unix_s, counts, mode1_model,
        project_quality_root=project_quality_root, uv_config=uv_config,
    )
    multimode = prepare_multimode_background_records(
        modes, time_unix_s, mode1.corrected_counts, multimode_models,
    )
    use_multimode = multimode.background_applied
    return PreparedBackgroundRecords(
        corrected_counts=multimode.corrected_counts,
        background_applied=mode1.background_applied | multimode.background_applied,
        removed_dpf_fraction=np.where(
            use_multimode, multimode.removed_dpf_fraction, mode1.removed_dpf_fraction
        ),
        background_dominated_channel_fraction=np.where(
            use_multimode,
            multimode.background_dominated_channel_fraction,
            mode1.background_dominated_channel_fraction,
        ),
        uv_candidate=mode1.uv_candidate,
        uv_algorithm_confirmed=mode1.uv_algorithm_confirmed,
        uv_confirmed=mode1.uv_confirmed,
        uv_project_quality_flag=mode1.uv_project_quality_flag,
        uv_rejected=mode1.uv_rejected,
        uv_maximum_robust_sigma=mode1.uv_maximum_robust_sigma,
        uv_affected_azimuth_sector_count=mode1.uv_affected_azimuth_sector_count,
        project_quality_available=mode1.project_quality_available,
        project_quality_flag=mode1.project_quality_flag,
    )
def _empty_result(n_rows: int) -> dict[str, np.ndarray]:
    result = {
        "epoch_unix_s": np.full(n_rows, np.nan),
        "record_index": np.full(n_rows, -1, dtype=np.int32),
        "source_file_index": np.full(n_rows, -1, dtype=np.int32),
        "mode": np.full(n_rows, -1, dtype=np.int16),
        "species_code": np.full(n_rows, -1, dtype=np.int16),
        "species_order": np.full(n_rows, -1, dtype=np.int16),
        "energy_bin_count": np.full(n_rows, np.nan),
        "mass_bin_count": np.full(n_rows, np.nan),
        "angle_bin_count": np.full(n_rows, np.nan),
        "mass_window_amu_min": np.full(n_rows, np.nan),
        "mass_window_amu_max": np.full(n_rows, np.nan),
        "density_cm3": np.full(n_rows, np.nan),
        "raw_density_cm3": np.full(n_rows, np.nan),
        "corrected_density_cm3": np.full(n_rows, np.nan),
        "background_density_contribution_cm3": np.full(n_rows, np.nan),
        "background_density_fraction": np.full(n_rows, np.nan),
        "density_valid": np.zeros(n_rows, dtype=np.int16),
        "density_reliable": np.zeros(n_rows, dtype=np.int16),
        "valid_cell_count": np.zeros(n_rows, dtype=np.int32),
        "raw_valid_cell_count": np.zeros(n_rows, dtype=np.int32),
        "corrected_valid_cell_count": np.zeros(n_rows, dtype=np.int32),
        "raw_processing_status_code": np.zeros(n_rows, dtype=np.int16),
        "corrected_processing_status_code": np.zeros(n_rows, dtype=np.int16),
        "background_applied": np.zeros(n_rows, dtype=np.int16),
        "background_model_valid": np.zeros(n_rows, dtype=np.int16),
        "background_removed_dpf_fraction": np.full(n_rows, np.nan),
        "background_dominated_channel_fraction": np.full(n_rows, np.nan),
        "uv_candidate_flag": np.zeros(n_rows, dtype=np.int16),
        "uv_algorithm_confirmed_flag": np.zeros(n_rows, dtype=np.int16),
        "uv_confirmed_flag": np.zeros(n_rows, dtype=np.int16),
        "uv_project_quality_flag": np.zeros(n_rows, dtype=np.int16),
        "uv_rejected_flag": np.zeros(n_rows, dtype=np.int16),
        "uv_maximum_robust_sigma": np.full(n_rows, np.nan),
        "uv_affected_azimuth_sector_count": np.zeros(n_rows, dtype=np.int16),
        "project_quality_available": np.zeros(n_rows, dtype=np.int16),
        "project_quality_flag": np.zeros(n_rows, dtype=np.uint32),
        "background_model_version": np.full(n_rows, "", dtype=object),
        "attitude_match_dt_s": np.full(n_rows, np.nan),
        "r_matrix_match_dt_s": np.full(n_rows, np.nan),
        "processing_status_code": np.zeros(n_rows, dtype=np.int16),
        "pitch_mso_deg": np.full(n_rows, np.nan),
        "roll_mso_deg": np.full(n_rows, np.nan),
        "yaw_mso_deg": np.full(n_rows, np.nan),
        "mse_plus_z_in_minpa_fov_flag": np.full(n_rows, np.nan),
        "mse_plus_z_minpa_theta_deg": np.full(n_rows, np.nan),
        "minpa_fov_theta_min_deg": np.full(n_rows, np.nan),
        "minpa_fov_theta_max_deg": np.full(n_rows, np.nan),
    }
    for axis in "xyz":
        result[f"mse_plus_z_minpa_{axis}"] = np.full(n_rows, np.nan)
    for prefix in ("v_mso", "v_mse", "pos_mso", "pos_mse"):
        unit = "km_s" if prefix.startswith("v_") else "km"
        for axis in "xyz":
            result[f"{prefix}_{axis}_{unit}"] = np.full(n_rows, np.nan)
    for prefix in ("raw_v_mso", "raw_v_mse", "corrected_v_mso", "corrected_v_mse"):
        for axis in "xyz":
            result[f"{prefix}_{axis}_km_s"] = np.full(n_rows, np.nan)
    for frame in ("mso", "mse"):
        result[f"fov_{frame}_attitude_valid"] = np.zeros(n_rows)
        result[f"fov_{frame}_solid_angle_sr"] = np.full(n_rows, np.nan)
        for axis in "xyz":
            result[f"fov_{frame}_dir_{axis}_weighted_mean"] = np.full(n_rows, np.nan)
            result[f"fov_{frame}_dir_{axis}_min"] = np.full(n_rows, np.nan)
            result[f"fov_{frame}_dir_{axis}_max"] = np.full(n_rows, np.nan)
            result[f"fov_{frame}_{axis}_pos_solid_angle_fraction"] = np.full(n_rows, np.nan)
            result[f"fov_{frame}_{axis}_neg_solid_angle_fraction"] = np.full(n_rows, np.nan)
        for inst_axis in "xyz":
            for comp in "xyz":
                result[f"inst_{inst_axis}_axis_{frame}_{comp}"] = np.full(n_rows, np.nan)
    return result


def process_tw1_day(
    date: str,
    paths: Tw1Paths | None = None,
    high_energy_min_eV: float = HIGH_ENERGY_MIN_EV,
    add_spacecraft_velocity: bool = True,
    write_output: bool = True,
    *,
    background_policy: BackgroundPolicy = "none",
    background_model: MinpaBackgroundModel | None = None,
    background_model_path: Path | None = None,
    multimode_background_models: Mapping[int, MultimodeBackgroundModel] | None = None,
    multimode_background_bundle_path: Path | None = None,
    allow_provisional_background_model: bool = False,
    uv_config: UvDetectionConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    paths = paths or Tw1Paths()
    mode1_policies = {"subtract-and-reject-uv", "paper-channel-subtract"}
    multimode_policy = "multimode-paper-channel-subtract"
    unified_static_policy = "all-approved-static-channel-subtract"
    correction_policies = {*mode1_policies, multimode_policy, unified_static_policy}
    if background_policy not in {"none", *correction_policies}:
        raise ValueError(f"Unknown background_policy={background_policy!r}")
    if background_policy in mode1_policies:
        if background_model is None:
            raise ValueError(f"{background_policy} requires an explicit background model")
        if not background_model.valid:
            raise ValueError(f"Refusing invalid background model: {background_model.invalid_reasons}")
        if background_model_is_provisional(background_model) and not allow_provisional_background_model:
            raise ValueError(
                "The selected background model contains provisional intervals; "
                "set allow_provisional_background_model=True only for validation products"
            )
        if (
            background_policy == "paper-channel-subtract"
            and background_model.primary_estimator
            not in ("paper_channel_nonzero_mean", "paper_nonzero_mean")
        ):
            raise ValueError(
                "paper-channel-subtract requires a paper channel nonzero-mean model"
            )
    elif background_policy == multimode_policy:
        if not multimode_background_models:
            raise ValueError(
                "multimode-paper-channel-subtract requires an explicit frozen model bundle"
            )
        for mode, model in multimode_background_models.items():
            if int(mode) not in (4, 12) or model.mode != int(mode):
                raise ValueError(f"Invalid multimode model mapping for mode {mode}")
            if not model.valid or model.review_status != "finalized":
                raise ValueError(
                    f"Refusing provisional/invalid Mode {mode} model: {list(model.invalid_reasons)}"
                )
    elif background_policy == unified_static_policy:
        if background_model is None or not background_model.valid:
            raise ValueError("Unified static subtraction requires a valid Mode-1 model")
        if background_model_is_provisional(background_model):
            raise ValueError("Unified static subtraction refuses provisional Mode-1 models")
        if not multimode_background_models or set(multimode_background_models) != {4, 12}:
            raise ValueError("Unified static subtraction requires frozen Mode-4 and Mode-12 models")
        for mode, model in multimode_background_models.items():
            if model.mode != mode or not model.valid or model.review_status != "finalized":
                raise ValueError(f"Unified static Mode-{mode} model is not finalized")
    source_type, source_files = source_files_for_day(date, paths)
    momag = load_momag_for_day(date, paths.momag_root)
    if momag is None:
        raise SkipDay(f"No Tianwen-1 MOMAG attitude/position for {date} under {paths.momag_root}")
    r_path = tw1_rotation_file(date, paths.r_root)
    rot = load_tw1_mso2mse(r_path)
    start_s, end_s = date_bounds(date)

    raw_records: list[tuple[int, Path, float, np.ndarray]] = []
    for file_index, path in enumerate(source_files):
        mode = minpa_mode(path)
        records = read_ori_records(path, start_s, end_s) if source_type == "ori" else read_public_records(path, start_s, end_s)
        for time_s, counts in records:
            for sub_time_s, sub_counts in split_mode12_records(mode, time_s, counts):
                if start_s <= sub_time_s < end_s:
                    raw_records.append((file_index, path, sub_time_s, sub_counts))
    raw_records.sort(key=lambda item: item[2])
    if not raw_records:
        raise SkipDay(f"No MINPA records inside UTC day {date} from {source_type}")

    record_modes = np.asarray([minpa_mode(item[1]) for item in raw_records], dtype=np.int16)
    record_times = np.asarray([item[2] for item in raw_records], dtype=float)
    record_counts = [np.asarray(item[3], dtype=float).reshape(-1) for item in raw_records]
    if background_policy in mode1_policies:
        assert background_model is not None
        prepared_background = prepare_background_records(
            record_modes,
            record_times,
            record_counts,
            background_model,
            project_quality_root=paths.project_quality_root,
            uv_config=uv_config,
        )
    elif background_policy == multimode_policy:
        assert multimode_background_models is not None
        prepared_background = prepare_multimode_background_records(
            record_modes,
            record_times,
            record_counts,
            multimode_background_models,
        )
    elif background_policy == unified_static_policy:
        assert background_model is not None and multimode_background_models is not None
        prepared_background = prepare_unified_static_background_records(
            record_modes,
            record_times,
            record_counts,
            background_model,
            multimode_background_models,
            project_quality_root=paths.project_quality_root,
            uv_config=uv_config,
        )
    else:
        n_records = len(raw_records)
        prepared_background = PreparedBackgroundRecords(
            corrected_counts=[value.copy() for value in record_counts],
            background_applied=np.zeros(n_records, dtype=bool),
            removed_dpf_fraction=np.full(n_records, np.nan),
            background_dominated_channel_fraction=np.full(n_records, np.nan),
            uv_candidate=np.zeros(n_records, dtype=bool),
            uv_algorithm_confirmed=np.zeros(n_records, dtype=bool),
            uv_confirmed=np.zeros(n_records, dtype=bool),
            uv_project_quality_flag=np.zeros(n_records, dtype=bool),
            uv_rejected=np.zeros(n_records, dtype=bool),
            uv_maximum_robust_sigma=np.full(n_records, np.nan),
            uv_affected_azimuth_sector_count=np.zeros(n_records, dtype=np.int16),
            project_quality_available=np.zeros(n_records, dtype=bool),
            project_quality_flag=np.zeros(n_records, dtype=np.uint32),
        )

    result = _empty_result(len(raw_records) * len(SPECIES))
    times = record_times
    r_mso_to_mse, r_dt = interpolate_matrices_nearest(rot["time_s"], rot["R_mso_to_mse"], times)

    row = 0
    identity_axes = np.eye(3)
    for record_index, (file_index, path, time_s, counts) in enumerate(raw_records):
        mode = minpa_mode(path)
        pitch, roll, yaw, pos_mso, v_sc_mso, attitude_dt = nearest_attitude(time_s, momag)
        attitude_valid = bool(np.isfinite(pitch + roll + yaw))
        rot_body_to_mso = _rotation_body_to_mso(pitch, roll, yaw) if attitude_valid else None
        rot_body_to_mse = r_mso_to_mse[record_index] @ rot_body_to_mso if rot_body_to_mso is not None else None
        pos_mse = r_mso_to_mse[record_index] @ pos_mso if np.all(np.isfinite(pos_mso)) else np.full(3, np.nan)
        axes_mso = identity_axes @ rot_body_to_mso.T if rot_body_to_mso is not None else np.full((3, 3), np.nan)
        axes_mse = apply_rotation_to_axes(r_mso_to_mse[record_index : record_index + 1], axes_mso[None, :, :])[0]
        plus_z_flag, plus_z_theta_deg, plus_z_body, theta_min, theta_max = mse_plus_z_minpa_fov_flag(mode, axes_mse)
        fov_mso = minpa_fov_summary(mode, rot_body_to_mso, "mso")
        fov_mse = minpa_fov_summary(mode, rot_body_to_mse, "mse")
        layout = mode_layout(mode)
        energy = layout.energy_eV
        masses = layout.mass_amu
        angle_count = float(layout.angle_count)
        for species_id, spec in SPECIES.items():
            raw_moment = high_energy_moment(
                counts,
                mode,
                species_id,
                rot_body_to_mso,
                v_sc_mso,
                high_energy_min_eV,
                add_spacecraft_velocity,
            )
            corrected_moment = high_energy_moment(
                prepared_background.corrected_counts[record_index],
                mode,
                species_id,
                rot_body_to_mso,
                v_sc_mso,
                high_energy_min_eV,
                add_spacecraft_velocity,
            )
            if background_policy in mode1_policies or (
                background_policy == unified_static_policy and mode == 1
            ):
                if prepared_background.uv_rejected[record_index]:
                    density = math.nan
                    velocity_mso = np.full(3, np.nan)
                    valid_cell_count = 0
                    status = 7
                else:
                    density, velocity_mso, valid_cell_count, status = corrected_moment
            elif background_policy in {multimode_policy, unified_static_policy}:
                density, velocity_mso, valid_cell_count, status = corrected_moment
            else:
                density, velocity_mso, valid_cell_count, status = raw_moment
            velocity_mse = r_mso_to_mse[record_index] @ velocity_mso if np.all(np.isfinite(velocity_mso)) else np.full(3, np.nan)
            groups = _species_groups(mode, species_id)
            group_masses = masses[groups] if groups else np.array([], dtype=float)
            _fill_row(
                result,
                row,
                record_index,
                file_index,
                time_s,
                mode,
                spec,
                energy.size,
                masses.size,
                angle_count,
                group_masses,
                density,
                velocity_mso,
                velocity_mse,
                pos_mso,
                pos_mse,
                valid_cell_count,
                status,
                pitch,
                roll,
                yaw,
                attitude_dt,
                r_dt[record_index],
                axes_mso,
                axes_mse,
                plus_z_flag,
                plus_z_theta_deg,
                plus_z_body,
                theta_min,
                theta_max,
                fov_mso,
                fov_mse,
            )
            record_background_model = (
                multimode_background_models.get(mode)
                if background_policy in {multimode_policy, unified_static_policy}
                and mode in (4, 12)
                and multimode_background_models is not None
                else background_model
            )
            _fill_background_row(
                result,
                row,
                raw_moment,
                corrected_moment,
                r_mso_to_mse[record_index],
                prepared_background,
                record_index,
                record_background_model,
            )
            row += 1

    summary = _summary(
        date,
        source_type,
        source_files,
        paths,
        r_path,
        result,
        raw_records,
        high_energy_min_eV,
        add_spacecraft_velocity,
        background_policy,
        background_model,
        background_model_path,
        multimode_background_models,
        multimode_background_bundle_path,
        prepared_background,
    )
    summary["runtime"] = {"processing_seconds_before_write": time.perf_counter() - started}
    if write_output:
        suffix = ""
        if background_policy == "subtract-and-reject-uv":
            suffix = "_bgcorr_provisional_v1" if background_model_is_provisional(background_model) else "_bgcorr_v1"
        elif background_policy == "paper-channel-subtract":
            suffix = (
                "_paperbgcorr_provisional_v2"
                if background_model_is_provisional(background_model)
                else "_paperbgcorr_v2"
            )
        elif background_policy == multimode_policy:
            suffix = "_multimodebgcorr_v2"
        elif background_policy == unified_static_policy:
            suffix = "_staticbgcorr_v2_2"
        write_tw1_output(date, paths.output_root, result, summary, product_suffix=suffix)
    return result, summary


def _fill_row(
    result: dict[str, np.ndarray],
    row: int,
    record_index: int,
    file_index: int,
    time_s: float,
    mode: int,
    spec: dict[str, Any],
    energy_count: int,
    mass_count: int,
    angle_count: float,
    group_masses: np.ndarray,
    density: float,
    velocity_mso: np.ndarray,
    velocity_mse: np.ndarray,
    pos_mso: np.ndarray,
    pos_mse: np.ndarray,
    valid_cell_count: int,
    status: int,
    pitch: float,
    roll: float,
    yaw: float,
    attitude_dt: float,
    r_dt: float,
    axes_mso: np.ndarray,
    axes_mse: np.ndarray,
    mse_plus_z_in_minpa_fov_flag: float,
    mse_plus_z_minpa_theta_deg: float,
    mse_plus_z_minpa: np.ndarray,
    minpa_fov_theta_min_deg: float,
    minpa_fov_theta_max_deg: float,
    fov_mso: dict[str, float],
    fov_mse: dict[str, float],
) -> None:
    result["epoch_unix_s"][row] = time_s
    result["record_index"][row] = record_index
    result["source_file_index"][row] = file_index
    result["mode"][row] = mode
    result["species_code"][row] = int(spec["code"])
    result["species_order"][row] = int(spec["code"])
    result["energy_bin_count"][row] = energy_count
    result["mass_bin_count"][row] = mass_count
    result["angle_bin_count"][row] = angle_count
    result["mass_window_amu_min"][row] = float(np.nanmin(group_masses)) if group_masses.size else math.nan
    result["mass_window_amu_max"][row] = float(np.nanmax(group_masses)) if group_masses.size else math.nan
    result["density_cm3"][row] = density
    result["density_valid"][row] = int(np.isfinite(density) and density > 0.0)
    result["density_reliable"][row] = int(np.isfinite(density) and density >= RELIABLE_DENSITY_THRESHOLD_CM3)
    result["valid_cell_count"][row] = valid_cell_count
    result["processing_status_code"][row] = status
    result["pitch_mso_deg"][row] = pitch
    result["roll_mso_deg"][row] = roll
    result["yaw_mso_deg"][row] = yaw
    result["attitude_match_dt_s"][row] = attitude_dt
    result["r_matrix_match_dt_s"][row] = r_dt
    result["mse_plus_z_in_minpa_fov_flag"][row] = mse_plus_z_in_minpa_fov_flag
    result["mse_plus_z_minpa_theta_deg"][row] = mse_plus_z_minpa_theta_deg
    result["minpa_fov_theta_min_deg"][row] = minpa_fov_theta_min_deg
    result["minpa_fov_theta_max_deg"][row] = minpa_fov_theta_max_deg
    for i, axis in enumerate("xyz"):
        result[f"v_mso_{axis}_km_s"][row] = velocity_mso[i]
        result[f"v_mse_{axis}_km_s"][row] = velocity_mse[i]
        result[f"pos_mso_{axis}_km"][row] = pos_mso[i]
        result[f"pos_mse_{axis}_km"][row] = pos_mse[i]
        result[f"mse_plus_z_minpa_{axis}"][row] = mse_plus_z_minpa[i]
    for frame, axes in (("mso", axes_mso), ("mse", axes_mse)):
        for axis_index, axis_name in enumerate("xyz"):
            for comp_index, comp_name in enumerate("xyz"):
                result[f"inst_{axis_name}_axis_{frame}_{comp_name}"][row] = axes[axis_index, comp_index]
    for values in (fov_mso, fov_mse):
        for key, value in values.items():
            result[key][row] = value


def _fill_background_row(
    result: dict[str, np.ndarray],
    row: int,
    raw_moment: tuple[float, np.ndarray, int, int],
    corrected_moment: tuple[float, np.ndarray, int, int],
    rotation_mso_to_mse: np.ndarray,
    prepared: PreparedBackgroundRecords,
    record_index: int,
    model: MinpaBackgroundModel | MultimodeBackgroundModel | None,
) -> None:
    raw_density, raw_velocity_mso, raw_cells, raw_status = raw_moment
    corrected_density, corrected_velocity_mso, corrected_cells, corrected_status = corrected_moment
    raw_velocity_mse = (
        rotation_mso_to_mse @ raw_velocity_mso
        if np.all(np.isfinite(raw_velocity_mso))
        else np.full(3, np.nan)
    )
    corrected_velocity_mse = (
        rotation_mso_to_mse @ corrected_velocity_mso
        if np.all(np.isfinite(corrected_velocity_mso))
        else np.full(3, np.nan)
    )
    result["raw_density_cm3"][row] = raw_density
    result["corrected_density_cm3"][row] = corrected_density
    if np.isfinite(raw_density) and np.isfinite(corrected_density):
        contribution = max(float(raw_density - corrected_density), 0.0)
        result["background_density_contribution_cm3"][row] = contribution
        if raw_density > 0.0:
            result["background_density_fraction"][row] = contribution / raw_density
    result["raw_valid_cell_count"][row] = raw_cells
    result["corrected_valid_cell_count"][row] = corrected_cells
    result["raw_processing_status_code"][row] = raw_status
    result["corrected_processing_status_code"][row] = corrected_status
    result["background_applied"][row] = int(prepared.background_applied[record_index])
    result["background_model_valid"][row] = int(model is not None and model.valid)
    result["background_removed_dpf_fraction"][row] = prepared.removed_dpf_fraction[record_index]
    result["background_dominated_channel_fraction"][row] = (
        prepared.background_dominated_channel_fraction[record_index]
    )
    result["uv_candidate_flag"][row] = int(prepared.uv_candidate[record_index])
    result["uv_algorithm_confirmed_flag"][row] = int(
        prepared.uv_algorithm_confirmed[record_index]
    )
    result["uv_confirmed_flag"][row] = int(prepared.uv_confirmed[record_index])
    result["uv_project_quality_flag"][row] = int(
        prepared.uv_project_quality_flag[record_index]
    )
    result["uv_rejected_flag"][row] = int(prepared.uv_rejected[record_index])
    result["uv_maximum_robust_sigma"][row] = prepared.uv_maximum_robust_sigma[record_index]
    result["uv_affected_azimuth_sector_count"][row] = (
        prepared.uv_affected_azimuth_sector_count[record_index]
    )
    result["project_quality_available"][row] = int(
        prepared.project_quality_available[record_index]
    )
    result["project_quality_flag"][row] = prepared.project_quality_flag[record_index]
    result["background_model_version"][row] = model.algorithm_version if model is not None else ""
    for index, axis in enumerate("xyz"):
        result[f"raw_v_mso_{axis}_km_s"][row] = raw_velocity_mso[index]
        result[f"raw_v_mse_{axis}_km_s"][row] = raw_velocity_mse[index]
        result[f"corrected_v_mso_{axis}_km_s"][row] = corrected_velocity_mso[index]
        result[f"corrected_v_mse_{axis}_km_s"][row] = corrected_velocity_mse[index]


def write_tw1_output(
    date: str,
    output_root: Path,
    result: dict[str, np.ndarray],
    summary: dict[str, Any],
    *,
    product_suffix: str = "",
) -> tuple[dict[str, Path], Path]:
    out_dir = Path(output_root) / "Tianwen-1" / date[:4]
    out_dir.mkdir(parents=True, exist_ok=True)
    species_paths: dict[str, Path] = {}
    species_outputs: dict[str, dict[str, Any]] = {}
    for species_id, spec in SPECIES.items():
        mask = result["species_code"] == int(spec["code"])
        species_result = {key: np.asarray(value)[mask] for key, value in result.items()}
        species_result, duplicate_stats = _drop_duplicate_times(species_result)
        times = species_result["epoch_unix_s"]
        if np.unique(times[np.isfinite(times)]).size != times.size:
            raise ValueError(f"{species_id} output would contain duplicate timestamps for {date}")
        mat_payload = {key: np.asarray(value)[:, None] for key, value in species_result.items()}
        mat_payload["utc"] = unix_s_to_utc_text(times)[:, None]
        mat_payload["species_id"] = np.asarray([species_id] * int(times.size), dtype=object)[:, None]
        mat_payload["species_label"] = np.asarray([spec["label"]] * int(times.size), dtype=object)[:, None]
        mat_path = out_dir / f"tw1_minpa_highE_{species_id}_{date}{product_suffix}.mat"
        savemat(mat_path, mat_payload, do_compression=True)
        species_paths[species_id] = mat_path
        species_outputs[species_id] = {
            "file": str(mat_path),
            "rows": int(times.size),
            "unique_time_count": int(np.unique(times[np.isfinite(times)]).size),
            "pass_rows": int(np.count_nonzero(species_result["processing_status_code"] == 1)),
            **duplicate_stats,
        }
    json_path = out_dir / f"tw1_minpa_highE_{date}{product_suffix}_summary.json"
    summary = {**summary, "species_outputs": species_outputs}
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return species_paths, json_path


def _drop_duplicate_times(species_result: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    times = np.asarray(species_result["epoch_unix_s"], dtype=float)
    _, first_indices, counts = np.unique(times, return_index=True, return_counts=True)
    keep = np.sort(first_indices)
    duplicate_groups = int(np.count_nonzero(counts > 1))
    duplicate_extra_records = int(np.sum(counts[counts > 1] - 1))
    if duplicate_groups == 0:
        return species_result, {"duplicate_time_groups_dropped": 0, "duplicate_extra_records_dropped": 0}
    return {key: np.asarray(value)[keep] for key, value in species_result.items()}, {
        "duplicate_time_groups_dropped": duplicate_groups,
        "duplicate_extra_records_dropped": duplicate_extra_records,
    }


def _summary(
    date: str,
    source_type: str,
    source_files: list[Path],
    paths: Tw1Paths,
    r_path: Path,
    result: dict[str, np.ndarray],
    raw_records: list[tuple[int, Path, float, np.ndarray]],
    high_energy_min_eV: float,
    add_spacecraft_velocity: bool,
    background_policy: BackgroundPolicy,
    background_model: MinpaBackgroundModel | None,
    background_model_path: Path | None,
    multimode_background_models: Mapping[int, MultimodeBackgroundModel] | None,
    multimode_background_bundle_path: Path | None,
    prepared_background: PreparedBackgroundRecords,
) -> dict[str, Any]:
    status, counts = np.unique(result["processing_status_code"], return_counts=True)
    finite_density = result["density_cm3"][np.isfinite(result["density_cm3"])]
    speed_mso = np.linalg.norm(np.column_stack([result["v_mso_x_km_s"], result["v_mso_y_km_s"], result["v_mso_z_km_s"]]), axis=1)
    speed_mse = np.linalg.norm(np.column_stack([result["v_mse_x_km_s"], result["v_mse_y_km_s"], result["v_mse_z_km_s"]]), axis=1)
    speed_diff = np.abs(speed_mso - speed_mse)
    speed_diff = speed_diff[np.isfinite(speed_diff)]
    raw_density = np.asarray(result["raw_density_cm3"], dtype=float)
    corrected_density = np.asarray(result["corrected_density_cm3"], dtype=float)
    comparable_density = np.isfinite(raw_density) & np.isfinite(corrected_density)
    density_monotonic_violations = int(
        np.count_nonzero(comparable_density & (corrected_density > raw_density + 1.0e-12))
    )
    raw_velocity = np.column_stack(
        [result[f"raw_v_mso_{axis}_km_s"] for axis in "xyz"]
    )
    corrected_velocity = np.column_stack(
        [result[f"corrected_v_mso_{axis}_km_s"] for axis in "xyz"]
    )
    raw_speed = np.linalg.norm(raw_velocity, axis=1)
    corrected_speed = np.linalg.norm(corrected_velocity, axis=1)
    comparable_speed = np.isfinite(raw_speed) & np.isfinite(corrected_speed) & (raw_speed > 0.0)
    relative_speed_change = np.abs(corrected_speed - raw_speed) / raw_speed
    model_provisional = (
        background_model_is_provisional(background_model)
        if background_model is not None
        else False
    )
    return {
        "date": date,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_files": {
            "minpa_source_type": source_type,
            "minpa": [str(path) for path in source_files],
            "momag_root": str(paths.momag_root),
            "r_mso2mse": str(r_path),
            "background_model": str(background_model_path) if background_model_path else None,
            "multimode_background_bundle": (
                str(multimode_background_bundle_path)
                if multimode_background_bundle_path
                else None
            ),
        },
        "assumptions": {
            "high_energy_min_eV": high_energy_min_eV,
            "energy_for_threshold_and_moment": "raw MINPA energy table, no spacecraft-potential correction",
            "source_priority": "use result/MINPA/ori if overlapping files exist for the UTC day; otherwise use rawdata/MINPA/public",
            "species_selection": "MINPA mass groups: O+ uses 16 amu group or mode-12 15.56/16.45; O2+ uses 32 amu group or mode-12 30.94/33.17",
            "density_formula": "Ion_Count treated as differential particle flux; dn=Ion_Count*dE*dOmega/v with dE from logarithmic edges of the calibrated energy-center table",
            "coordinate_transform": "MINPA payload vectors first map to spacecraft body coordinates as Xb=-V2,Yb=+V3,Zb=-V1; MOMAG attitude then rotates body to MSO with Rz(-Yaw)*Ry(-Pitch)*Rx(-Roll); R_MSO2MSE rotates MSO vectors to MSE",
            "spacecraft_velocity_added": add_spacecraft_velocity,
            "fov_output": "spacecraft-body coordinate axes in MSO/MSE plus MSE +Z in/out MINPA theta-edge FOV flag; diagnostic solid-angle half-axis fractions are also retained",
            "mse_plus_z_fov_flag": "MINPA ion FOV is 360 deg in azimuth and 90 deg in polar angle, occupying the -X_body hemisphere after the payload-to-body mapping; flag=1 when transformed MSE +Z body-frame theta is inside the mode-specific MINPA pitch/theta edges",
            "minpa_fov_caveat": "MINPA has a half-FOV blind region; density and vector moments are coverage-limited when the relevant ion distribution lies partly outside the observed hemisphere",
            "background_policy": background_policy,
            "background_subtraction": (
                "Mode 1/4/12 static approved-channel backgrounds; DPF_corrected=max(DPF_raw-DPF_background,0); no temporal scaling"
                if background_policy == "all-approved-static-channel-subtract"
                else
                "Mode 4/12 reviewed species mass bins only; DPF_corrected=max(DPF_raw-DPF_background,0) after Mode-12 subrecord splitting"
                if background_policy == "multimode-paper-channel-subtract"
                else "Mode 1 only; DPF_corrected=max(DPF_raw-DPF_background,0) before moment integration"
            ),
            "uv_rejection": "confirmed Mode-1 UV records are rejected as whole records; solar incidence is supporting evidence only",
            "non_mode1_background_behavior": (
                "Mode 7 and all unreviewed Mode-4/12 mass bins remain uncorrected; no temporal scaling or cross-mass proxy"
                if background_policy == "all-approved-static-channel-subtract"
                else
                "Mode 7 and all unreviewed Mode-4/12 mass bins remain uncorrected; no cross-mass proxy is allowed"
                if background_policy == "multimode-paper-channel-subtract"
                else "Mode 4/12 records remain uncorrected because the selected model is Mode 1 only"
            ),
        },
        "background_model": {
            "supplied": background_model is not None,
            "valid": bool(background_model.valid) if background_model is not None else False,
            "algorithm_version": background_model.algorithm_version if background_model is not None else None,
            "primary_estimator": background_model.primary_estimator if background_model is not None else None,
            "denoise_policy_version": (
                str(background_model.provenance.get("denoise_policy_version", DENOISE_POLICY_VERSION))
                if background_model is not None
                else None
            ),
            "provisional": model_provisional,
            "source_hash_count": len(background_model.source_sha256) if background_model is not None else 0,
        },
        "multimode_background_models": {
            str(mode): {
                "valid": bool(model.valid),
                "review_status": model.review_status,
                "algorithm_version": model.algorithm_version,
                "primary_estimator": model.primary_estimator,
            }
            for mode, model in sorted((multimode_background_models or {}).items())
        },
        "records": {
            "minpa_records": len(raw_records),
            "internal_rows_before_species_split": int(result["epoch_unix_s"].size),
            "per_species_rows": len(raw_records),
            "source_file_count": len(source_files),
            "status_counts": {str(int(k)): int(v) for k, v in zip(status, counts)},
            "status_code_meaning": {
                "0": "not_processed",
                "1": "pass",
                "2": "no_high_energy_geometry_cells_for_species",
                "4": "no_positive_density_cells",
                "5": "density_only_no_valid_attitude_velocity",
                "6": "unexpected_counts_shape",
                "7": "record_rejected_as_uv_contamination",
            },
            "background_applied_mode1_records": int(
                np.count_nonzero(
                    prepared_background.background_applied
                    & (
                        np.asarray(
                            [minpa_mode(item[1]) for item in raw_records], dtype=int
                        )
                        == 1
                    )
                )
            ),
            "background_applied_multimode_records": int(
                np.count_nonzero(
                    prepared_background.background_applied
                    & np.isin(
                        np.asarray(
                            [minpa_mode(item[1]) for item in raw_records], dtype=int
                        ),
                        [4, 12],
                    )
                )
            ),
            "uv_candidate_records": int(np.count_nonzero(prepared_background.uv_candidate)),
            "uv_algorithm_confirmed_records": int(
                np.count_nonzero(prepared_background.uv_algorithm_confirmed)
            ),
            "uv_confirmed_records": int(np.count_nonzero(prepared_background.uv_confirmed)),
            "uv_project_quality_records": int(
                np.count_nonzero(prepared_background.uv_project_quality_flag)
            ),
            "uv_rejected_records": int(np.count_nonzero(prepared_background.uv_rejected)),
            "project_quality_available_records": int(
                np.count_nonzero(prepared_background.project_quality_available)
            ),
        },
        "sanity_checks": {
            "density_cm3_median": float(np.nanmedian(finite_density)) if finite_density.size else None,
            "density_cm3_max": float(np.nanmax(finite_density)) if finite_density.size else None,
            "max_abs_speed_mso_mse_difference_km_s": float(np.nanmax(speed_diff)) if speed_diff.size else None,
            "max_abs_r_matrix_match_dt_s": float(np.nanmax(np.abs(result["r_matrix_match_dt_s"]))),
            "max_abs_attitude_match_dt_s": float(np.nanmax(np.abs(result["attitude_match_dt_s"]))),
            "mse_plus_z_in_minpa_fov_fraction": float(np.nanmean(result["mse_plus_z_in_minpa_fov_flag"]))
            if np.any(np.isfinite(result["mse_plus_z_in_minpa_fov_flag"]))
            else None,
            "corrected_density_not_above_raw_violation_count": density_monotonic_violations,
            "raw_density_cm3_median": float(np.nanmedian(raw_density))
            if np.any(np.isfinite(raw_density))
            else None,
            "corrected_density_cm3_median": float(np.nanmedian(corrected_density))
            if np.any(np.isfinite(corrected_density))
            else None,
            "background_density_fraction_median": float(
                np.nanmedian(result["background_density_fraction"])
            )
            if np.any(np.isfinite(result["background_density_fraction"]))
            else None,
            "relative_speed_change_median": float(np.nanmedian(relative_speed_change[comparable_speed]))
            if np.any(comparable_speed)
            else None,
            "relative_speed_change_p95": float(np.nanquantile(relative_speed_change[comparable_speed], 0.95))
            if np.any(comparable_speed)
            else None,
        },
    }
