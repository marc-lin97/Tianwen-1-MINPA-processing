"""Spectra, ion moments and diagnostic velocity-space projections for MINPA."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coordinates import tw1_orbiter_to_mso_matrix
from .minpa_modes import (
    angular_geometry,
    base_to_legacy_direction,
    energy_bin_edges_eV,
    energy_bin_widths_eV,
    minpa_to_body_direction,
    minpa_directions,
    mode_layout,
    species_mass_indices,
)

EV_J = 1.602176634e-19
MASS_UNIT_KG = 1.67262192369e-27  # preserve the validated MINPA proton-mass convention


@dataclass(frozen=True)
class SpeciesSpectrum:
    mode: int
    species: str
    energy_eV: np.ndarray
    dpf: np.ndarray
    deflux: np.ndarray
    contributing_mass_indices: np.ndarray


@dataclass(frozen=True)
class MomentResult:
    mode: int
    species: str
    density_cm3: float
    velocity_minpa_km_s: np.ndarray
    velocity_body_km_s: np.ndarray
    velocity_mso_km_s: np.ndarray
    scalar_temperature_eV: float
    valid_cell_count: int
    status: str
    energy_range_eV: tuple[float, float]
    spacecraft_velocity_added: bool


@dataclass(frozen=True)
class VelocitySamples:
    velocity_minpa_km_s: np.ndarray
    density_weight_cm3: np.ndarray
    energy_eV: np.ndarray
    mass_amu: np.ndarray
    coverage_velocity_minpa_km_s: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    record_count: int = 1
    angular_samples_per_dimension: int = 1
    energy_samples_per_cell: int = 1


@dataclass(frozen=True)
class VdfProjection:
    x_edges_km_s: np.ndarray
    y_edges_km_s: np.ndarray
    density_per_velocity_area: np.ndarray
    density_integral_cm3: float
    axes: tuple[str, str]
    frame: str
    coverage_mask: np.ndarray
    bulk_velocity_km_s: np.ndarray
    integrated_axis: str
    quantity_label: str = "MINPA reduced density proxy"


def reshape_dpf(values: np.ndarray, mode: int) -> np.ndarray:
    """Return a validated ``[energy, angle, mass]`` DPF cube."""
    layout = mode_layout(mode)
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size != layout.values_per_subrecord:
        raise ValueError(f"Mode {mode} split record expects shape {layout.shape} ({layout.values_per_subrecord} values), got {flat.size}")
    return flat.reshape(layout.shape)


def species_spectrum(values: np.ndarray, mode: int, species: str) -> SpeciesSpectrum:
    """Solid-angle weighted mean DPF and DEF spectrum for one record."""
    layout = mode_layout(mode)
    cube = reshape_dpf(values, mode)
    groups = species_mass_indices(mode, species)
    _, _, omega = angular_geometry(mode)
    if groups.size == 0:
        dpf = np.full(layout.energy_eV.size, np.nan)
    else:
        per_mass = np.stack([np.nansum(cube[:, :, g] * omega[None, :], axis=1) / np.nansum(omega) for g in groups])
        dpf = np.nanmean(per_mass, axis=0)
    return SpeciesSpectrum(mode, species, layout.energy_eV.copy(), dpf, dpf * layout.energy_eV, groups)


def velocity_samples(values: np.ndarray, mode: int, species: str, energy_min_eV: float = 0.0, energy_max_eV: float = np.inf) -> VelocitySamples:
    """Return per-cell velocity and differential density weights.

    The density element is ``DPF*dE*dOmega/v`` using logarithmic bin edges
    inferred from the calibrated energy centers. Speed is converted to cm/s.
    Non-finite, non-positive and uncalibrated ``>=1e8`` DPF cells are assigned
    zero weight.
    """
    layout = mode_layout(mode)
    if not layout.directional_velocity_available:
        raise ValueError(f"Mode {mode} does not resolve 3-D directions; VDF and vector velocity are unavailable")
    cube = reshape_dpf(values, mode)
    groups = species_mass_indices(mode, species)
    if groups.size == 0:
        return VelocitySamples(np.empty((0,3)), np.empty(0), np.empty(0), np.empty(0))
    mask_e = (layout.energy_eV >= energy_min_eV) & (layout.energy_eV <= energy_max_eV)
    denergy = energy_bin_widths_eV(mode)[mask_e]
    dirs = minpa_directions(mode)
    _, _, omega = angular_geometry(mode)
    velocities, weights, energies, masses = [], [], [], []
    for group in groups:
        mass = float(layout.mass_amu[group])
        energy = layout.energy_eV[mask_e]
        speed = np.sqrt(2.0 * energy * EV_J / (mass * MASS_UNIT_KG)) / 1000.0
        dpf = cube[mask_e, :, group]
        valid = np.isfinite(dpf) & (dpf > 0.0) & (dpf < 1e8)
        dn = np.where(valid, dpf * denergy[:, None] * omega[None, :] / (speed[:, None] * 1e5), 0.0)
        velocities.append((speed[:, None, None] * dirs[None, :, :]).reshape(-1,3))
        weights.append(dn.reshape(-1))
        energies.append(np.repeat(energy, dirs.shape[0]))
        masses.append(np.full(energy.size * dirs.shape[0], mass))
    signal_velocity = np.concatenate(velocities)
    return VelocitySamples(
        signal_velocity,
        np.concatenate(weights),
        np.concatenate(energies),
        np.concatenate(masses),
        coverage_velocity_minpa_km_s=signal_velocity.copy(),
    )


def finite_velocity_cell_samples(
    values: np.ndarray,
    mode: int,
    species: str,
    energy_min_eV: float = 0.0,
    energy_max_eV: float = np.inf,
    *,
    angular_samples: int = 3,
    energy_samples: int = 3,
) -> VelocitySamples:
    """Sub-sample finite MINPA energy-angle cells for VDF projections.

    This follows the mature ``plot_tw1_minpa_h_vdf_xy.py`` workflow.  Each
    parent angular cell is sampled in both pitch and azimuth, each energy cell
    is sampled between its actual logarithmic edges, and the parent density
    contribution is conserved across the sub-samples. Geometric support is returned
    separately from positive signal so an empty channel is distinguishable
    from an unobserved region.
    """
    if angular_samples < 1 or energy_samples < 1:
        raise ValueError("angular_samples and energy_samples must be positive")
    layout = mode_layout(mode)
    if not layout.directional_velocity_available:
        raise ValueError(f"Mode {mode} does not resolve 3-D directions; VDF and vector velocity are unavailable")
    cube = reshape_dpf(values, mode)
    groups = species_mass_indices(mode, species)
    if groups.size == 0:
        return VelocitySamples(np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0))

    energy_mask = (layout.energy_eV >= energy_min_eV) & (layout.energy_eV <= energy_max_eV)
    energy_centers = layout.energy_eV[energy_mask]
    all_energy_edges = energy_bin_edges_eV(layout.energy_eV)
    energy_lower = all_energy_edges[:-1][energy_mask]
    energy_upper = all_energy_edges[1:][energy_mask]
    theta, phi, omega = angular_geometry(mode)
    pitch_width = np.repeat(np.diff(layout.pitch_edges_deg), layout.azimuth_count)
    azimuth_width = np.full(theta.size, 360.0 / layout.azimuth_count)
    angle_offsets = np.linspace(-0.5, 0.5, angular_samples)
    theta_samples = theta[:, None, None] + pitch_width[:, None, None] * angle_offsets[None, :, None]
    phi_samples = phi[:, None, None] + azimuth_width[:, None, None] * angle_offsets[None, None, :]
    sample_shape = (theta.size, angular_samples, angular_samples)
    theta_flat = np.broadcast_to(theta_samples, sample_shape).reshape(-1)
    phi_flat = np.broadcast_to(phi_samples, sample_shape).reshape(-1)
    angle_index = np.repeat(np.arange(theta.size), angular_samples * angular_samples)
    directions = base_to_legacy_direction(theta_flat, phi_flat)
    omega_samples = omega[angle_index] / float(angular_samples * angular_samples)

    energy_fraction = np.linspace(0.0, 1.0, energy_samples)
    velocities: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    coverage: list[np.ndarray] = []
    for group in groups:
        mass = float(layout.mass_amu[group])
        dpf = cube[energy_mask, :, group]
        denergy = energy_upper - energy_lower
        sampled_energy = energy_lower[:, None] + denergy[:, None] * energy_fraction[None, :]
        speed = np.sqrt(2.0 * sampled_energy * EV_J / (mass * MASS_UNIT_KG)) / 1000.0
        sampled_flux = dpf[:, angle_index]
        dn = (
            sampled_flux[:, None, :]
            * (denergy[:, None, None] / float(energy_samples))
            * omega_samples[None, None, :]
            / (speed[:, :, None] * 1.0e5)
        )
        velocity = speed[:, :, None, None] * directions[None, None, :, :]
        support = np.isfinite(speed[:, :, None]) & (speed[:, :, None] > 0.0) & np.all(np.isfinite(velocity), axis=3)
        if np.any(support):
            coverage.append(velocity[support])
        valid = (
            np.isfinite(dn)
            & (dn > 0.0)
            & (sampled_flux[:, None, :] < 1.0e8)
            & np.all(np.isfinite(velocity), axis=3)
        )
        if np.any(valid):
            velocities.append(velocity[valid])
            weights.append(dn[valid])
            energy_grid = np.broadcast_to(sampled_energy[:, :, None], dn.shape)
            energies.append(energy_grid[valid])
            masses.append(np.full(np.count_nonzero(valid), mass))

    coverage_velocity = np.concatenate(coverage) if coverage else np.empty((0, 3))
    if not velocities:
        return VelocitySamples(
            np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0),
            coverage_velocity_minpa_km_s=coverage_velocity,
            angular_samples_per_dimension=angular_samples,
            energy_samples_per_cell=energy_samples,
        )
    return VelocitySamples(
        np.concatenate(velocities),
        np.concatenate(weights),
        np.concatenate(energies),
        np.concatenate(masses),
        coverage_velocity_minpa_km_s=coverage_velocity,
        angular_samples_per_dimension=angular_samples,
        energy_samples_per_cell=energy_samples,
    )


def _density_only(values: np.ndarray, mode: int, species: str, energy_min_eV: float, energy_max_eV: float) -> tuple[float, int]:
    layout = mode_layout(mode)
    cube = reshape_dpf(values, mode)
    groups = species_mass_indices(mode, species)
    _, _, omega = angular_geometry(mode)
    mask_e = (layout.energy_eV >= energy_min_eV) & (layout.energy_eV <= energy_max_eV)
    denergy = energy_bin_widths_eV(mode)[mask_e]
    total, count = 0.0, 0
    for group in groups:
        mass = float(layout.mass_amu[group])
        energy = layout.energy_eV[mask_e]
        speed = np.sqrt(2.0 * energy * EV_J / (mass * MASS_UNIT_KG)) / 1000.0
        dpf = cube[mask_e, :, group]
        valid = np.isfinite(dpf) & (dpf > 0.0) & (dpf < 1e8)
        dn = np.where(valid, dpf * denergy[:,None] * omega[None,:] / (speed[:,None] * 1e5), 0.0)
        total += float(np.sum(dn))
        count += int(np.count_nonzero(valid))
    return total, count


def compute_moments(
    values: np.ndarray,
    mode: int,
    species: str,
    *,
    energy_min_eV: float = 0.0,
    energy_max_eV: float = np.inf,
    attitude_roll_pitch_yaw_deg: tuple[float, float, float] | None = None,
    spacecraft_velocity_mso_km_s: np.ndarray | None = None,
    add_spacecraft_velocity: bool = False,
) -> MomentResult:
    """Integrate one species in DPF space and optionally rotate its velocity.

    Attitude input order is ``(roll, pitch, yaw)``. The returned compatibility
    ``body`` velocity is the spacecraft-body result after the validated MINPA
    payload mapping ``[-V2,+V3,-V1]``. MSO velocity requires a finite attitude; adding
    spacecraft velocity is an explicit opt-in.
    """
    layout = mode_layout(mode)
    groups = species_mass_indices(mode, species)
    if groups.size == 0:
        nan3 = np.full(3, np.nan)
        return MomentResult(mode, species, np.nan, nan3, nan3.copy(), nan3.copy(), np.nan, 0, "species_not_resolved", (energy_min_eV, energy_max_eV), False)
    if not layout.directional_velocity_available:
        density, count = _density_only(values, mode, species, energy_min_eV, energy_max_eV)
        nan3 = np.full(3, np.nan)
        return MomentResult(mode, species, density if density > 0 else np.nan, nan3, nan3.copy(), nan3.copy(), np.nan, count, "direction_not_resolved", (energy_min_eV, energy_max_eV), False)
    samples = velocity_samples(values, mode, species, energy_min_eV, energy_max_eV)
    valid = np.isfinite(samples.density_weight_cm3) & (samples.density_weight_cm3 > 0.0) & np.all(np.isfinite(samples.velocity_minpa_km_s), axis=1)
    if not np.any(valid):
        nan3 = np.full(3, np.nan)
        return MomentResult(mode, species, np.nan, nan3, nan3.copy(), nan3.copy(), np.nan, 0, "no_valid_signal", (energy_min_eV, energy_max_eV), False)
    weights = samples.density_weight_cm3[valid]
    velocity = samples.velocity_minpa_km_s[valid]
    density = float(np.sum(weights))
    bulk_minpa = np.sum(velocity * weights[:,None], axis=0) / density
    body = minpa_to_body_direction(bulk_minpa[None, :])[0]
    mso = np.full(3, np.nan)
    status = "attitude_unavailable"
    added = False
    if attitude_roll_pitch_yaw_deg is not None and np.all(np.isfinite(attitude_roll_pitch_yaw_deg)):
        roll, pitch, yaw = attitude_roll_pitch_yaw_deg
        mso = tw1_orbiter_to_mso_matrix(roll, pitch, yaw) @ body
        status = "ok"
        if add_spacecraft_velocity and spacecraft_velocity_mso_km_s is not None and np.all(np.isfinite(spacecraft_velocity_mso_km_s)):
            mso = mso + np.asarray(spacecraft_velocity_mso_km_s, dtype=float)
            added = True
    mass_kg = np.average(samples.mass_amu[valid], weights=weights) * MASS_UNIT_KG
    variance_m2_s2 = np.sum(weights * np.sum((velocity - bulk_minpa)**2, axis=1)) / density * 1e6
    temperature = float(mass_kg * variance_m2_s2 / (3.0 * EV_J))
    return MomentResult(mode, species, density, bulk_minpa, body, mso, temperature, int(np.count_nonzero(valid)), status, (energy_min_eV, energy_max_eV), added)


def project_vdf(
    samples: VelocitySamples,
    *,
    axes: tuple[int, int] = (0, 1),
    bins: int | np.ndarray = 80,
    frame: str = "MINPA",
    rotation: np.ndarray | None = None,
    velocity_limit_km_s: float | None = None,
) -> VdfProjection:
    """Bin density contributions in a 2-D velocity plane.

    This is not an absolute phase-space density calibration.  It is a
    density-conserving projection of the measured angular/energy cells and is
    deliberately labeled as such in figures and metadata.
    """
    velocity = np.asarray(samples.velocity_minpa_km_s, dtype=float)
    coverage_velocity = np.asarray(samples.coverage_velocity_minpa_km_s, dtype=float)
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3,3):
            raise ValueError("rotation must have shape (3, 3)")
        velocity = velocity @ rotation.T
        if coverage_velocity.size:
            coverage_velocity = coverage_velocity @ rotation.T
    weight = np.asarray(samples.density_weight_cm3, dtype=float)
    valid = np.all(np.isfinite(velocity[:, list(axes)]), axis=1) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        raise ValueError("No positive finite VDF samples")
    if np.isscalar(bins):
        source = coverage_velocity if coverage_velocity.size else velocity[valid]
        finite = np.abs(source[np.isfinite(source)])
        auto_limit = max(100.0, np.ceil(np.nanpercentile(finite, 99.0) / 50.0) * 50.0) if finite.size else 700.0
        limit = float(velocity_limit_km_s if velocity_limit_km_s is not None else auto_limit)
        edges = np.linspace(-limit, limit, int(bins) + 1)
        histogram_bins: int | list[np.ndarray] = [edges, edges]
    else:
        histogram_bins = bins
    hist, xedge, yedge = np.histogram2d(
        velocity[valid, axes[0]], velocity[valid, axes[1]], bins=histogram_bins, weights=weight[valid]
    )
    coverage_mask = np.zeros_like(hist, dtype=bool)
    if coverage_velocity.size:
        finite_coverage = np.all(np.isfinite(coverage_velocity[:, list(axes)]), axis=1)
        coverage_hist, _, _ = np.histogram2d(
            coverage_velocity[finite_coverage, axes[0]],
            coverage_velocity[finite_coverage, axes[1]],
            bins=[xedge, yedge],
        )
        coverage_mask = coverage_hist > 0
    area = np.diff(xedge)[:,None] * np.diff(yedge)[None,:]
    density_area = np.divide(hist, area, out=np.zeros_like(hist), where=area > 0)
    labels = ("Vx", "Vy", "Vz")
    bulk = np.sum(velocity[valid] * weight[valid, None], axis=0) / np.sum(weight[valid])
    remaining = ({0, 1, 2} - set(axes)).pop()
    return VdfProjection(
        xedge,
        yedge,
        density_area,
        float(np.sum(weight[valid])),
        (labels[axes[0]], labels[axes[1]]),
        frame,
        coverage_mask,
        bulk,
        labels[remaining],
    )
