"""Project quality flags for MINPA H+, O+ and O2+ records.

This is a NumPy implementation of the approved MATLAB definition in
``scripts/tw1_minpa_quality_flag_all_species_core.m``.  Native ``Quality`` is
kept separately because its bit meanings are not published in the local label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .minpa_modes import angular_geometry, mode_layout, species_mass_indices

ALGORITHM_VERSION = "2026-07-13-seven-band-all-species-v2-mode12-product-time"
FLAG_SPARSE_CAUTION = np.uint32(1)
FLAG_SPARSE_INVALID = np.uint32(2)
FLAG_UV = np.uint32(4)
FLAG_EVEN_ODD = np.uint32(8)
FLAG_HIGH_CHANNEL = np.uint32(16)
ENERGY_BANDS_EV = np.array([[0,np.inf],[0,30],[30,300],[300,np.inf],[0,50],[50,500],[500,5000]], dtype=float)


@dataclass(frozen=True)
class QualityEvaluation:
    flag: np.uint32
    valid_2d_channel_count: int
    uv_fraction: float
    evenodd_band_pass: np.ndarray
    max_def_2d: float
    algorithm_version: str = ALGORITHM_VERSION

    def bit(self, bit_number: int) -> bool:
        return bool(int(self.flag) & (1 << (bit_number - 1)))


def _alternating_pass(y: np.ndarray) -> bool:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return False
    left, mid, right = y[:-2], y[1:-1], y[2:]
    usable = np.isfinite(left) & np.isfinite(mid) & np.isfinite(right)
    peak = (mid > left) & (mid > right) & (mid >= 1e5)
    valley = (mid < 0.5 * left) & (mid < 0.5 * right)
    extrema = usable & (peak | valley)
    usable_n, extrema_n = np.count_nonzero(usable), np.count_nonzero(extrema)
    if usable_n == 0 or extrema_n == 0:
        return False
    interior = np.arange(2, y.size)  # MATLAB 1-based indices 2..n-1
    odd = interior % 2 == 1
    odd_phase = (peak & odd) | (valley & ~odd)
    even_phase = (peak & ~odd) | (valley & odd)
    phase = max(np.count_nonzero(usable & odd_phase), np.count_nonzero(usable & even_phase)) / extrema_n
    return extrema_n / usable_n >= 0.525 and phase >= 0.55


def evaluate_quality(dpf: np.ndarray, mode: int, species: str) -> QualityEvaluation:
    """Evaluate the five project problem bits for one split science record."""
    layout = mode_layout(mode)
    flat = np.asarray(dpf, dtype=float).reshape(-1)
    if flat.size != layout.values_per_subrecord:
        raise ValueError(f"Mode {mode} split record requires {layout.values_per_subrecord} values, got {flat.size}")
    cube = flat.reshape(layout.shape)
    groups = species_mass_indices(mode, species)
    if groups.size == 0:
        return QualityEvaluation(np.uint32(FLAG_SPARSE_INVALID), 0, np.nan, np.zeros(7, bool), np.nan)
    species_dpf = np.sum(cube[:, :, groups], axis=2)
    energy = layout.energy_eV
    _, _, omega = angular_geometry(mode)
    omega_sum = np.nansum(omega)
    def2d = species_dpf * energy[:, None]
    def1d = np.nansum(def2d * omega[None, :], axis=1) / omega_sum
    valid_count = int(np.count_nonzero(np.isfinite(species_dpf) & (species_dpf > 0.0) & (species_dpf < 1e8)))
    uv_fraction = float(np.mean(np.isfinite(def1d) & (def1d > 1e5)))
    max_def = float(np.nanmax(def2d)) if np.any(np.isfinite(def2d)) else np.nan
    band_pass = np.array([_alternating_pass(def1d[(energy >= lo) & (energy < hi)]) for lo, hi in ENERGY_BANDS_EV])
    flag = np.uint32(0)
    if valid_count < 5:
        flag |= FLAG_SPARSE_INVALID
    elif valid_count <= 10:
        flag |= FLAG_SPARSE_CAUTION
    if uv_fraction > 0.8:
        flag |= FLAG_UV
    if np.any(band_pass):
        flag |= FLAG_EVEN_ODD
    if np.isfinite(max_def) and max_def > 1e10:
        flag |= FLAG_HIGH_CHANNEL
    return QualityEvaluation(flag, valid_count, uv_fraction, band_pass, max_def)


def reject_for_moments(flag: int, *, reject_high_channel: bool = False) -> bool:
    """Recommended production mask: reject bits 2--4; bit 5 is configurable."""
    mask = int(FLAG_SPARSE_INVALID | FLAG_UV | FLAG_EVEN_ODD)
    if reject_high_channel:
        mask |= int(FLAG_HIGH_CHANNEL)
    return bool(int(flag) & mask)


def reject_for_background(flag: int) -> bool:
    """Background-window rule agreed for this project: reject bits 3--4 only."""
    return bool(int(flag) & int(FLAG_UV | FLAG_EVEN_ODD))
