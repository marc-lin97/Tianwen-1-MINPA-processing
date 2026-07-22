"""Identify MINPA Mode-1 intervals containing only instrumental background.

The classifier is deliberately small and interpretable.  A weak physical ion
population must produce both a prominent time-mean H+ DEF peak and a
concentration of inferred H+ counts in adjacent energy channels.  Requiring
both signatures prevents an isolated random channel excursion from being
treated as a coherent ion spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .minpa_background import MODE1_ENERGY_EV, MODE1_SHAPE


QUIET_CLASSIFIER_VERSION = "minpa-mode1-quiet-reference-v1"
BOUNDARY_REFINEMENT_VERSION = "minpa-mode1-borderline-boundary-v1"


@dataclass(frozen=True)
class QuietReferenceClassifierConfig:
    """Thresholds calibrated from the 21 manually reviewed December windows."""

    def_peak_to_median_threshold: float = 2.75
    h_count_top3_fraction_threshold: float = 0.133
    adjacent_energy_bins: int = 3
    minimum_valid_def_energy_bins: int = 30
    borderline_relative_margin: float = 0.10


@dataclass(frozen=True)
class QuietBoundaryRefinementConfig:
    """Search grid for trimming borderline candidate boundaries."""

    minimum_duration_fraction: float = 0.50
    boundary_step_s: float = 60.0
    minimum_def_records: int = 12
    minimum_dpf_records: int = 18


def time_mean_h_def_spectrum(h_def: np.ndarray) -> np.ndarray:
    """Average non-negative H+ DEF over time without conditioning on positivity."""

    values = np.asarray(h_def, dtype=float)
    if values.ndim != 2 or values.shape[1] != MODE1_ENERGY_EV.size:
        raise ValueError("h_def must have shape (time, 40)")
    finite = np.isfinite(values)
    clean = np.where(finite, np.maximum(values, 0.0), np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(clean, axis=0)


def hplus_count_energy_spectrum(
    dpf_cube: np.ndarray,
    dpf_quantum: np.ndarray,
) -> np.ndarray:
    """Sum inferred H+ count equivalents over time, pitch, and azimuth."""

    values = np.asarray(dpf_cube, dtype=float)
    quantum = np.asarray(dpf_quantum, dtype=float)
    if values.ndim != 5 or values.shape[1:] != MODE1_SHAPE:
        raise ValueError(f"dpf_cube must have shape (time, {MODE1_SHAPE})")
    if quantum.shape != MODE1_SHAPE:
        raise ValueError(f"dpf_quantum must have shape {MODE1_SHAPE}")
    h_values = values[..., 0]
    h_quantum = quantum[..., 0]
    ratio = np.divide(
        h_values,
        h_quantum[None, ...],
        out=np.full(h_values.shape, np.nan, dtype=float),
        where=(
            np.isfinite(h_values)
            & (h_values >= 0.0)
            & np.isfinite(h_quantum[None, ...])
            & (h_quantum[None, ...] > 0.0)
        ),
    )
    return np.nansum(np.maximum(ratio, 0.0), axis=(0, 2, 3))


def extract_quiet_reference_features(
    h_def: np.ndarray,
    dpf_cube: np.ndarray,
    dpf_quantum: np.ndarray,
    config: QuietReferenceClassifierConfig | None = None,
) -> dict[str, object]:
    """Return the two spectral-structure features and their decision score."""

    cfg = config or QuietReferenceClassifierConfig()
    if cfg.def_peak_to_median_threshold <= 0.0:
        raise ValueError("def_peak_to_median_threshold must be positive")
    if not 1 <= cfg.adjacent_energy_bins <= MODE1_ENERGY_EV.size:
        raise ValueError("adjacent_energy_bins is outside the energy dimension")
    if not 1 <= cfg.minimum_valid_def_energy_bins <= MODE1_ENERGY_EV.size:
        raise ValueError("minimum_valid_def_energy_bins is invalid")
    if not 0.0 < cfg.h_count_top3_fraction_threshold < 1.0:
        raise ValueError("h_count_top3_fraction_threshold must lie between zero and one")
    if not 0.0 <= cfg.borderline_relative_margin < 1.0:
        raise ValueError("borderline_relative_margin must lie in [0, 1)")

    mean_def = time_mean_h_def_spectrum(h_def)
    valid_def = np.isfinite(mean_def) & (mean_def > 0.0)
    if int(np.count_nonzero(valid_def)) < cfg.minimum_valid_def_energy_bins:
        raise ValueError("too few valid H+ DEF energy bins")
    median_def = float(np.nanmedian(mean_def[valid_def]))
    peak_index = int(np.nanargmax(mean_def))
    peak_to_median = float(mean_def[peak_index] / median_def)

    count_energy = hplus_count_energy_spectrum(dpf_cube, dpf_quantum)
    total_count = float(np.nansum(count_energy))
    if not np.isfinite(total_count) or total_count <= 0.0:
        raise ValueError("no finite positive H+ count equivalent")
    width = cfg.adjacent_energy_bins
    rolling = np.convolve(np.nan_to_num(count_energy, nan=0.0), np.ones(width), mode="valid")
    top_start = int(np.argmax(rolling))
    top_fraction = float(rolling[top_start] / total_count)

    def_ratio = peak_to_median / cfg.def_peak_to_median_threshold
    count_ratio = top_fraction / cfg.h_count_top3_fraction_threshold
    decision_score = float(min(def_ratio, count_ratio))
    real_spectrum = bool(decision_score > 1.0)
    borderline = bool(abs(decision_score - 1.0) <= cfg.borderline_relative_margin)
    return {
        "classifier_version": QUIET_CLASSIFIER_VERSION,
        "time_mean_def_peak_to_median": peak_to_median,
        "time_mean_def_peak_energy_index": peak_index,
        "time_mean_def_peak_energy_eV": float(MODE1_ENERGY_EV[peak_index]),
        "h_count_top3_energy_fraction": top_fraction,
        "h_count_top3_start_energy_index": top_start,
        "h_count_top3_stop_energy_index_inclusive": top_start + width - 1,
        "h_count_top3_energy_range_eV": [
            float(MODE1_ENERGY_EV[top_start]),
            float(MODE1_ENERGY_EV[top_start + width - 1]),
        ],
        "decision_score": decision_score,
        "real_spectrum_detected": real_spectrum,
        "pure_background_reference": not real_spectrum,
        "classification_confidence": "borderline" if borderline else "high",
        "manual_review_recommended": borderline,
        "decision_rule": (
            "real spectrum iff DEF peak/median exceeds its threshold AND "
            "the adjacent-three-energy H+ count fraction exceeds its threshold"
        ),
        "thresholds": {
            "time_mean_def_peak_to_median": float(cfg.def_peak_to_median_threshold),
            "h_count_top3_energy_fraction": float(cfg.h_count_top3_fraction_threshold),
            "borderline_relative_margin": float(cfg.borderline_relative_margin),
        },
        "def_record_count": int(np.asarray(h_def).shape[0]),
        "dpf_record_count": int(np.asarray(dpf_cube).shape[0]),
    }


def evaluate_quiet_boundary_refinements(
    candidate_start_s: float,
    candidate_stop_s: float,
    h_time_s: np.ndarray,
    h_def: np.ndarray,
    dpf_time_s: np.ndarray,
    dpf_cube: np.ndarray,
    dpf_quantum: np.ndarray,
    classifier_config: QuietReferenceClassifierConfig | None = None,
    refinement_config: QuietBoundaryRefinementConfig | None = None,
) -> list[dict[str, object]]:
    """Evaluate all grid-aligned subwindows down to half the input duration.

    The caller remains responsible for the independent gross-coherence and
    quality-flag vetoes. Results are ordered by longest duration first and,
    within equal duration, by the lowest classifier decision score.
    """

    cfg = classifier_config or QuietReferenceClassifierConfig()
    refine = refinement_config or QuietBoundaryRefinementConfig()
    start_s, stop_s = float(candidate_start_s), float(candidate_stop_s)
    duration_s = stop_s - start_s
    if duration_s <= 0.0:
        raise ValueError("candidate interval must have positive duration")
    if not 0.0 < refine.minimum_duration_fraction <= 1.0:
        raise ValueError("minimum_duration_fraction must lie in (0, 1]")
    if refine.boundary_step_s <= 0.0:
        raise ValueError("boundary_step_s must be positive")
    if refine.minimum_def_records < 1 or refine.minimum_dpf_records < 1:
        raise ValueError("minimum record counts must be positive")

    h_time = np.asarray(h_time_s, dtype=float).reshape(-1)
    h_values = np.asarray(h_def, dtype=float)
    dpf_time = np.asarray(dpf_time_s, dtype=float).reshape(-1)
    dpf_values = np.asarray(dpf_cube, dtype=float)
    if h_values.shape != (h_time.size, MODE1_ENERGY_EV.size):
        raise ValueError("h_def and h_time_s shapes do not match")
    if dpf_values.shape != (dpf_time.size, *MODE1_SHAPE):
        raise ValueError("dpf_cube and dpf_time_s shapes do not match")

    minimum_duration_s = duration_s * refine.minimum_duration_fraction
    maximum_trim_steps = int(
        np.floor((duration_s - minimum_duration_s) / refine.boundary_step_s + 1.0e-9)
    )
    trials: list[dict[str, object]] = []
    # At least one boundary must move; the original borderline interval is not
    # a refinement candidate.
    for total_trim_steps in range(1, maximum_trim_steps + 1):
        for left_trim_steps in range(total_trim_steps + 1):
            right_trim_steps = total_trim_steps - left_trim_steps
            trial_start = start_s + left_trim_steps * refine.boundary_step_s
            trial_stop = stop_s - right_trim_steps * refine.boundary_step_s
            trial_duration = trial_stop - trial_start
            if trial_duration + 1.0e-9 < minimum_duration_s:
                continue
            h_mask = (h_time >= trial_start) & (h_time < trial_stop)
            dpf_mask = (dpf_time >= trial_start) & (dpf_time < trial_stop)
            h_count = int(np.count_nonzero(h_mask))
            dpf_count = int(np.count_nonzero(dpf_mask))
            if h_count < refine.minimum_def_records or dpf_count < refine.minimum_dpf_records:
                continue
            try:
                features = extract_quiet_reference_features(
                    h_values[h_mask], dpf_values[dpf_mask], dpf_quantum, cfg
                )
            except ValueError:
                continue
            trials.append({
                "boundary_refinement_version": BOUNDARY_REFINEMENT_VERSION,
                "start_unix_s": trial_start,
                "stop_unix_s": trial_stop,
                "duration_s": trial_duration,
                "left_trim_s": trial_start - start_s,
                "right_trim_s": stop_s - trial_stop,
                **features,
            })
    return sorted(
        trials,
        key=lambda item: (-float(item["duration_s"]), float(item["decision_score"])),
    )
