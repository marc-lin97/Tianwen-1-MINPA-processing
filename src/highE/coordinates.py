"""Coordinate and time utilities for MSO/MSE processing."""

from __future__ import annotations

import numpy as np


def datetime64_to_unix_s(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="datetime64[ns]")
    return (arr - np.datetime64("1970-01-01T00:00:00", "ns")) / np.timedelta64(1, "s")


def unix_s_to_utc_text(values: np.ndarray) -> np.ndarray:
    seconds = np.asarray(values, dtype=float)
    out = []
    for item in seconds:
        if np.isfinite(item):
            out.append(str(np.datetime64(int(round(item * 1_000_000_000)), "ns")))
        else:
            out.append("")
    return np.asarray(out, dtype=object)


def finite_data(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(np.abs(arr) > 1.0e29, np.nan, arr)


def quality_bit_is_set(quality: np.ndarray, bit_index: int) -> np.ndarray:
    values = np.nan_to_num(np.asarray(quality), nan=0).astype(np.int64)
    return (values & (1 << bit_index)) != 0


def bad_quality_mask(quality: np.ndarray, bits: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros_like(np.asarray(quality), dtype=bool)
    for bit in bits:
        mask |= quality_bit_is_set(quality, bit)
    return mask


def energy_to_speed_km_s(energy_eV: np.ndarray, mass_amu: np.ndarray) -> np.ndarray:
    from .constants import AMU_KG, EV_TO_J

    energy_j = np.asarray(energy_eV, dtype=float) * EV_TO_J
    mass_kg = np.asarray(mass_amu, dtype=float) * AMU_KG
    return np.sqrt(2.0 * energy_j / mass_kg) / 1000.0


def density_from_eflux_cm3(
    eflux: np.ndarray,
    energy_eV: np.ndarray,
    denergy_eV: np.ndarray,
    domega_sr: np.ndarray,
    speed_km_s: np.ndarray,
) -> np.ndarray:
    flux = np.asarray(eflux, dtype=float)
    energy = np.asarray(energy_eV, dtype=float)
    width = np.asarray(denergy_eV, dtype=float)
    solid_angle = np.asarray(domega_sr, dtype=float)
    speed = np.asarray(speed_km_s, dtype=float)
    valid = (
        np.isfinite(flux)
        & np.isfinite(energy)
        & np.isfinite(width)
        & np.isfinite(solid_angle)
        & np.isfinite(speed)
        & (flux > 0.0)
        & (energy > 0.0)
        & (width > 0.0)
        & (solid_angle > 0.0)
        & (speed > 0.0)
    )
    density = np.full(np.broadcast_shapes(flux.shape, energy.shape, width.shape, solid_angle.shape, speed.shape), np.nan)
    density[valid] = flux[valid] / energy[valid] * width[valid] * solid_angle[valid] / (speed[valid] * 1.0e5)
    return density


def static_look_direction(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=float))
    phi_rad = np.deg2rad(np.asarray(phi_deg, dtype=float))
    return np.stack(
        [
            np.cos(theta_rad) * np.cos(phi_rad),
            np.cos(theta_rad) * np.sin(phi_rad),
            np.sin(theta_rad),
        ],
        axis=-1,
    )


def mse_plus_z_static_components(axes_mse: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes_mse, dtype=float)
    if axes.shape[-2:] != (3, 3):
        raise ValueError(f"Expected axes_mse with trailing shape (3, 3), got {axes.shape}")
    return axes[..., :, 2]


def static_elevation_deg(vectors_static: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors_static, dtype=float)
    norm = np.linalg.norm(vectors, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_unit = vectors[..., 2] / norm
    z_unit = np.where(np.isfinite(z_unit), np.clip(z_unit, -1.0, 1.0), np.nan)
    return np.rad2deg(np.arcsin(z_unit))


def mse_plus_z_static_fov_flag(
    axes_mse: np.ndarray,
    theta_min_deg: np.ndarray | float,
    theta_max_deg: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_static = mse_plus_z_static_components(axes_mse)
    theta_deg = static_elevation_deg(target_static)
    theta_min = np.asarray(theta_min_deg, dtype=float)
    theta_max = np.asarray(theta_max_deg, dtype=float)
    theta_min, theta_max, theta_eval = np.broadcast_arrays(theta_min, theta_max, theta_deg)
    finite = (
        np.isfinite(theta_eval)
        & np.isfinite(theta_min)
        & np.isfinite(theta_max)
        & np.all(np.isfinite(target_static), axis=-1)
    )
    flag = np.full(theta_eval.shape, np.nan, dtype=float)
    flag[finite] = ((theta_eval[finite] >= theta_min[finite]) & (theta_eval[finite] <= theta_max[finite])).astype(float)
    return flag, theta_deg, target_static


def quaternion_rotate_scalar_first(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=float)
    vec = np.asarray(vectors, dtype=float)
    if quat.shape[-1] != 4:
        return np.full_like(vec, np.nan, dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0.0 or not np.isfinite(norm):
        return np.full_like(vec, np.nan, dtype=float)
    quat = quat / norm
    w = quat[0]
    qvec = quat[1:4]
    return vec + 2.0 * w * np.cross(qvec, vec) + 2.0 * np.cross(qvec, np.cross(qvec, vec))


def quaternion_inverse_rotate_scalar_first(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=float).copy()
    if quat.shape[-1] != 4:
        return np.full_like(np.asarray(vectors, dtype=float), np.nan, dtype=float)
    quat[1:4] *= -1.0
    return quaternion_rotate_scalar_first(quat, vectors)


def finite_difference_velocity_km_s(time_s: np.ndarray, position_km: np.ndarray, max_gap_s: float = 120.0) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    pos = np.asarray(position_km, dtype=float)
    velocity = np.full_like(pos, np.nan, dtype=float)
    if time.size < 2:
        return velocity
    for i in range(time.size):
        if i == 0:
            left, right = 0, 1
        elif i == time.size - 1:
            left, right = time.size - 2, time.size - 1
        else:
            left, right = i - 1, i + 1
        dt = time[right] - time[left]
        if np.isfinite(dt) and 0.0 < dt <= max_gap_s:
            candidate = (pos[right] - pos[left]) / dt
            speed = np.linalg.norm(candidate)
            if np.isfinite(speed) and 0.1 <= speed <= 10.0:
                velocity[i] = candidate
    return velocity


def nearest_indices(source_time_s: np.ndarray, target_time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_time_s, dtype=float)
    target = np.asarray(target_time_s, dtype=float)
    idx = np.searchsorted(source, target)
    idx = np.clip(idx, 1, len(source) - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(source[right] - target) < np.abs(source[left] - target)
    nearest = np.where(choose_right, right, left)
    return nearest, target - source[nearest]


def interpolate_vectors(source_time_s: np.ndarray, vectors: np.ndarray, target_time_s: np.ndarray) -> np.ndarray:
    source = np.asarray(source_time_s, dtype=float)
    target = np.asarray(target_time_s, dtype=float)
    data = np.asarray(vectors, dtype=float)
    out = np.full((target.size, data.shape[1]), np.nan, dtype=float)
    for col in range(data.shape[1]):
        valid = np.isfinite(source) & np.isfinite(data[:, col])
        if np.count_nonzero(valid) >= 2:
            out[:, col] = np.interp(target, source[valid], data[valid, col], left=np.nan, right=np.nan)
    return out


def interpolate_matrices_nearest(
    source_time_s: np.ndarray,
    matrices: np.ndarray,
    target_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    idx, dt = nearest_indices(source_time_s, target_time_s)
    return np.asarray(matrices, dtype=float)[idx], dt


def tw1_orbiter_to_mso_matrix(
    roll_deg: np.ndarray | float,
    pitch_deg: np.ndarray | float,
    yaw_deg: np.ndarray | float,
) -> np.ndarray:
    """Return Tianwen-1 orbiter-body to MSO direction-cosine matrices.

    The MOMAG attitude convention validated against the product's stored MSO
    magnetic field is

        C_MSO_from_Orbiter = Rz(-yaw) @ Ry(-pitch) @ Rx(-roll)

    for right-handed active rotation matrices acting on column vectors.  The
    input angles are in degrees and may be scalars or broadcastable arrays.
    The result has shape ``broadcast_shape + (3, 3)``.
    """
    roll, pitch, yaw = np.broadcast_arrays(
        np.deg2rad(np.asarray(roll_deg, dtype=float)),
        np.deg2rad(np.asarray(pitch_deg, dtype=float)),
        np.deg2rad(np.asarray(yaw_deg, dtype=float)),
    )
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    matrix = np.empty(roll.shape + (3, 3), dtype=float)
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr + sy * cr
    matrix[..., 0, 2] = sy * sr - cy * sp * cr
    matrix[..., 1, 0] = -sy * cp
    matrix[..., 1, 1] = cy * cr - sy * sp * sr
    matrix[..., 1, 2] = sy * sp * cr + cy * sr
    matrix[..., 2, 0] = sp
    matrix[..., 2, 1] = -cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix


def tw1_minpa_legacy_to_probe_step_matrix() -> np.ndarray:
    """Return one ``TW1_MINPA_redeal_3.m`` signed-permutation step.

    For column vectors this implements ``[-V3, -V1, +V2]``.  The mature
    Python/MATLAB MINPA chain applies this step twice to directions expressed
    in the intermediate convention written by ``TW1_MINPA_deal_v5.m``.
    """
    return np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def tw1_minpa_instrument_to_orbiter_matrix() -> np.ndarray:
    """Return the validated MINPA-payload-to-spacecraft-body matrix.

    This compatibility name is retained for existing callers. The mapping was
    selected directly by the local all-species ``NV``/``NV_MSO`` comparison:

        [Xb, Yb, Zb]^T = C_body_from_MINPA [V1, V2, V3]^T

    with Xb = -V2, Yb = +V3, and Zb = -V1.
    """
    return np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def apply_rotation(matrices: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", np.asarray(matrices, dtype=float), np.asarray(vectors, dtype=float))


def apply_rotation_to_axes(matrices: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nkj->nki", np.asarray(matrices, dtype=float), np.asarray(axes, dtype=float))
