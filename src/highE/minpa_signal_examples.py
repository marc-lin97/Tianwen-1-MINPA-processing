"""Conservative feature and validation helpers for MINPA signal examples."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .minpa_mode1_temporal_review import extract_energy_band_excess_features
from .minpa_modes import species_mass_indices


def contiguous_true_run(mask: np.ndarray) -> int:
    """Return the longest contiguous true run in a one-dimensional mask."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def extract_signal_prefilter_features(spectrum: np.ndarray) -> dict[str, float | int]:
    """Extract dimensionless persistent-band features from a time-energy spectrum.

    Absolute ``day_spe`` amplitudes are deliberately not compared with the
    channel background because the two products use different angular/mass
    aggregation.  The final decision is made from native channel DPF.
    """

    values = np.asarray(spectrum, dtype=float)
    base = extract_energy_band_excess_features(values)
    clean = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    start = int(base["three_bin_start_index"])
    stop = int(base["three_bin_stop_index_exclusive"])
    total = clean.sum(axis=1)
    band = clean[:, start:stop].sum(axis=1)
    fraction = np.divide(band, total, out=np.zeros_like(band), where=total > 0.0)
    persistent_fraction = float(np.mean(fraction >= 0.20))
    peak_ratio = float(np.expm1(base["log1p_peak_to_positive_median"]))
    median_peak_ratio = float(
        np.expm1(base["log1p_time_median_peak_to_positive_median"])
    )
    score = float(
        np.log1p(max(peak_ratio, 0.0))
        + np.log1p(max(median_peak_ratio, 0.0))
        + 2.0 * float(base["three_bin_energy_fraction"])
        + 2.0 * persistent_fraction
    )
    return {
        **base,
        "peak_to_positive_median": peak_ratio,
        "time_median_peak_to_positive_median": median_peak_ratio,
        "three_bin_persistent_record_fraction": persistent_fraction,
        "median_total_spectrum": float(np.median(total)),
        "prefilter_score": score,
    }


def passes_signal_prefilter(
    features: Mapping[str, float | int],
    *,
    minimum_peak_ratio: float = 8.0,
    minimum_time_median_peak_ratio: float = 5.0,
    minimum_three_bin_fraction: float = 0.20,
    minimum_persistent_fraction: float = 0.60,
) -> bool:
    """Return whether a fixed window has a persistent, elevated energy band."""

    return bool(
        float(features["peak_to_positive_median"]) >= minimum_peak_ratio
        and float(features["time_median_peak_to_positive_median"])
        >= minimum_time_median_peak_ratio
        and float(features["three_bin_energy_fraction"]) >= minimum_three_bin_fraction
        and float(features["three_bin_persistent_record_fraction"])
        >= minimum_persistent_fraction
    )


def validate_native_channel_signal(
    raw: np.ndarray,
    corrected: np.ndarray,
    background_dpf: np.ndarray,
    valid_channel_mask: np.ndarray,
    *,
    mode: int,
    species: str,
    raw_spectrum: np.ndarray,
    corrected_spectrum: np.ndarray,
    minimum_signal_to_background: float = 20.0,
    minimum_time_occupancy: float = 0.50,
    minimum_contiguous_energy_channels: int = 2,
    minimum_peak_retention: float = 0.95,
    maximum_peak_shift_channels: int = 1,
) -> dict[str, Any]:
    """Validate a persistent signal against the frozen native-channel model."""

    raw_values = np.asarray(raw, dtype=float)
    corrected_values = np.asarray(corrected, dtype=float)
    if raw_values.shape != corrected_values.shape or raw_values.ndim != 5:
        raise ValueError("raw and corrected must share time x energy x pitch x azimuth x mass")
    masses = species_mass_indices(mode, species)
    if masses.size == 0:
        raise ValueError(f"Mode {mode} has no reviewed {species} mass channel")
    selected_raw = np.maximum(raw_values[..., masses], 0.0)
    selected_corrected = np.maximum(corrected_values[..., masses], 0.0)
    background = np.asarray(background_dpf, dtype=float)[..., masses]
    supported = (
        np.asarray(valid_channel_mask, dtype=bool)[..., masses]
        & np.isfinite(background)
        & (background > 0.0)
    )
    denominator = np.broadcast_to(background, selected_raw.shape)
    usable = np.broadcast_to(supported, selected_raw.shape) & np.isfinite(selected_raw)
    ratio = np.divide(
        selected_raw,
        denominator,
        out=np.full(selected_raw.shape, np.nan),
        where=usable,
    )
    corrected_ratio = np.divide(
        selected_corrected,
        denominator,
        out=np.full(selected_corrected.shape, np.nan),
        where=usable,
    )
    with np.errstate(all="ignore"):
        maximum_by_time_energy = np.nanmax(ratio, axis=(2, 3, 4))
        corrected_maximum_by_time_energy = np.nanmax(
            corrected_ratio, axis=(2, 3, 4)
        )
    finite_by_energy = np.isfinite(maximum_by_time_energy)
    occupancy = np.divide(
        np.count_nonzero(
            finite_by_energy
            & (maximum_by_time_energy >= minimum_signal_to_background), axis=0
        ),
        np.count_nonzero(finite_by_energy, axis=0),
        out=np.zeros(maximum_by_time_energy.shape[1], dtype=float),
        where=np.count_nonzero(finite_by_energy, axis=0) > 0,
    )
    median_ratio = np.nanmedian(maximum_by_time_energy, axis=0)
    corrected_median_ratio = np.nanmedian(corrected_maximum_by_time_energy, axis=0)
    strong_energy = (
        np.isfinite(median_ratio)
        & (median_ratio >= minimum_signal_to_background)
        & (occupancy >= minimum_time_occupancy)
    )
    longest_run = contiguous_true_run(strong_energy)
    peak_index = int(np.nanargmax(median_ratio))
    corrected_peak_index = (
        int(np.nanargmax(corrected_median_ratio))
        if np.any(np.isfinite(corrected_median_ratio))
        else None
    )
    raw_mean = np.nanmean(np.asarray(raw_spectrum, dtype=float), axis=0)
    corrected_mean = np.nanmean(np.asarray(corrected_spectrum, dtype=float), axis=0)
    peak_value = float(raw_mean[peak_index])
    retention = (
        float(corrected_mean[peak_index] / peak_value)
        if np.isfinite(peak_value) and peak_value > 0.0
        else np.nan
    )
    peak_shift = (
        abs(corrected_peak_index - peak_index)
        if corrected_peak_index is not None
        else None
    )
    passed = bool(
        longest_run >= minimum_contiguous_energy_channels
        and retention >= minimum_peak_retention
        and peak_shift is not None
        and peak_shift <= maximum_peak_shift_channels
    )
    return {
        "passed": passed,
        "minimum_signal_to_background": float(minimum_signal_to_background),
        "maximum_median_channel_signal_to_background": float(
            np.nanmax(median_ratio)
        ),
        "signal_peak_energy_index": peak_index,
        "corrected_ratio_peak_energy_index": corrected_peak_index,
        "ratio_peak_shift_channels": peak_shift,
        "peak_energy_time_occupancy": float(occupancy[peak_index]),
        "longest_contiguous_strong_energy_run": longest_run,
        "signal_peak_retention_fraction": retention,
        "strong_energy_indices": np.flatnonzero(strong_energy).astype(int).tolist(),
        "median_channel_ratio_by_energy": median_ratio.tolist(),
        "occupancy_by_energy": occupancy.tolist(),
        "corrected_total_not_above_raw": bool(
            np.nansum(selected_corrected)
            <= np.nansum(selected_raw)
            + max(1.0e-12, float(np.nansum(selected_raw)) * 1.0e-12)
        ),
    }
