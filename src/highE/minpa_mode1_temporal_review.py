"""Coverage audit and deterministic selection for Mode-1 temporal review."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import math
from typing import Any, Mapping, Sequence

import numpy as np


REVIEW_VERSION = "minpa-mode1-temporal-gap-review-v1"
CONTEXT_VETO_VERSION = "minpa-mode1-context-persistence-veto-v1"
SINGLE_BOUNDARY_VETO_VERSION = "minpa-mode1-single-boundary-persistence-veto-v2"
HPLUS_SIGNAL_VETO_VERSION = "minpa-mode1-hplus-signal-prototype-v3"
JOINT_APPROVED_ENVELOPE_VERSION = "minpa-mode1-joint-approved-envelope-v4"
JOINT_LABELLED_PROTOTYPE_VERSION = "minpa-mode1-joint-labelled-prototype-v5"
JOINT_TEMPORAL_COHERENCE_VERSION = "minpa-mode1-joint-temporal-coherence-v6"
ENERGY_BAND_EXCESS_VERSION = "minpa-mode1-energy-band-excess-v7.2"


def parse_utc(value: str) -> datetime:
    """Parse an ISO UTC timestamp and reject naive values."""

    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"UTC timestamp lacks timezone: {value}")
    return parsed.astimezone(UTC)


def exact_deduplicate(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_field: str = "start_utc",
    stop_field: str = "stop_utc",
) -> list[dict[str, Any]]:
    """Deduplicate identical UTC windows without giving aliases extra weight."""

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for source in rows:
        start = parse_utc(str(source[start_field]))
        stop = parse_utc(str(source[stop_field]))
        if stop <= start:
            raise ValueError(f"Invalid interval: {start.isoformat()} -- {stop.isoformat()}")
        key = (start.isoformat(), stop.isoformat())
        if key not in result:
            result[key] = dict(source)
    return list(result.values())


def intervals_overlap(
    start_unix_s: float,
    stop_unix_s: float,
    reviewed: Sequence[tuple[float, float]],
) -> bool:
    """Return true for a positive-duration overlap with a reviewed interval."""

    return any(
        float(start_unix_s) < float(old_stop)
        and float(old_start) < float(stop_unix_s)
        for old_start, old_stop in reviewed
    )


def select_longest_nonoverlapping_joint(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep longest joint windows; quiet score and UTC break duration ties."""

    ordered = sorted(
        rows,
        key=lambda row: (
            -(float(row["stop_unix_s"]) - float(row["start_unix_s"])),
            float(row.get("joint_decision_score", math.inf)),
            float(row["start_unix_s"]),
            str(row.get("source_candidate_id", "")),
        ),
    )
    kept: list[dict[str, Any]] = []
    for source in ordered:
        start = float(source["start_unix_s"])
        stop = float(source["stop_unix_s"])
        if intervals_overlap(
            start,
            stop,
            [(float(row["start_unix_s"]), float(row["stop_unix_s"])) for row in kept],
        ):
            continue
        kept.append(dict(source))
    return sorted(kept, key=lambda row: (float(row["start_unix_s"]), str(row.get("source_candidate_id", ""))))


def coverage_rows(
    approved: Sequence[Mapping[str, Any]],
    years: Sequence[int],
    *,
    period: str,
    target_intervals: int = 20,
    target_days: int = 5,
    excluded_periods: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Summarize exact-deduplicated approved-window gaps by month or quarter."""

    if period not in {"month", "quarter"}:
        raise ValueError("period must be 'month' or 'quarter'")
    if target_intervals < 1 or target_days < 1:
        raise ValueError("coverage targets must be positive")
    deduplicated = exact_deduplicate(approved)
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    dates: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for row in deduplicated:
        start = parse_utc(str(row["start_utc"]))
        index = start.month if period == "month" else (start.month - 1) // 3 + 1
        key = (start.year, index)
        counts[key] += 1
        dates[key].add(start.date().isoformat())
    result: list[dict[str, Any]] = []
    excluded = {str(value) for value in excluded_periods}
    period_count = 12 if period == "month" else 4
    for year in years:
        for index in range(1, period_count + 1):
            key = (int(year), index)
            label = f"{year:04d}-{index:02d}" if period == "month" else f"{year:04d}-Q{index}"
            if label in excluded:
                continue
            interval_count = int(counts[key])
            distinct_days = len(dates[key])
            eligible = interval_count >= target_intervals and distinct_days >= target_days
            result.append(
                {
                    "period_type": period,
                    "period": label,
                    "year": int(year),
                    "period_index": index,
                    "approved_interval_count": interval_count,
                    "approved_distinct_day_count": distinct_days,
                    "approved_dates_utc": sorted(dates[key]),
                    "target_interval_count": int(target_intervals),
                    "target_distinct_day_count": int(target_days),
                    "interval_gap": max(0, int(target_intervals) - interval_count),
                    "distinct_day_gap": max(0, int(target_days) - distinct_days),
                    "currently_eligible": eligible,
                    "gap_class": "meets_target" if eligible else "no_approved" if interval_count == 0 else "deficient",
                }
            )
    return result


def context_persistence_features(
    spectrum: np.ndarray,
    before_mask: np.ndarray,
    inside_mask: np.ndarray,
    after_mask: np.ndarray,
    *,
    adjacent_energy_bins: int = 3,
    minimum_records_per_region: int = 5,
) -> dict[str, float]:
    """Measure a candidate's narrow-band persistence across both boundaries.

    The peak band is selected from the candidate interval.  Shape similarity,
    amplitude similarity, and nonzero occupancy must then persist in both the
    preceding and following context.  All metrics are dimensionless, so the
    veto does not introduce a cross-epoch DPF-amplitude assumption.
    """

    values = np.asarray(spectrum, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("spectrum must have shape (record, energy)")
    masks = [np.asarray(mask, dtype=bool).reshape(-1) for mask in (before_mask, inside_mask, after_mask)]
    if any(mask.size != values.shape[0] for mask in masks):
        raise ValueError("context masks must match spectrum records")
    if adjacent_energy_bins < 1 or adjacent_energy_bins > values.shape[1]:
        raise ValueError("adjacent_energy_bins is outside the energy layout")
    if any(int(np.count_nonzero(mask)) < minimum_records_per_region for mask in masks):
        raise ValueError("insufficient records in one or more context regions")

    def mean_spectrum(mask: np.ndarray) -> np.ndarray:
        selected = np.where(np.isfinite(values[mask]), np.maximum(values[mask], 0.0), np.nan)
        count = np.sum(np.isfinite(selected), axis=0)
        return np.divide(
            np.nansum(selected, axis=0), count,
            out=np.zeros(values.shape[1], dtype=float), where=count > 0,
        )

    before, inside, after = (mean_spectrum(mask) for mask in masks)
    rolling = np.convolve(inside, np.ones(adjacent_energy_bins), mode="valid")
    band_start = int(np.argmax(rolling))
    band_stop = band_start + adjacent_energy_bins
    inside_total = float(np.sum(inside))
    concentration = float(rolling[band_start] / inside_total) if inside_total > 0.0 else 0.0

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        norm = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.dot(left, right) / norm) if norm > 0.0 else 0.0

    def amplitude_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_total, right_total = float(np.sum(left)), float(np.sum(right))
        largest = max(left_total, right_total)
        return float(min(left_total, right_total) / largest) if largest > 0.0 else 1.0

    def band_occupancy(mask: np.ndarray) -> float:
        band = np.where(
            np.isfinite(values[mask, band_start:band_stop]),
            np.maximum(values[mask, band_start:band_stop], 0.0), 0.0,
        )
        return float(np.mean(np.sum(band, axis=1) > 0.0))

    before_cosine = cosine(inside, before)
    after_cosine = cosine(inside, after)
    before_amplitude = amplitude_similarity(inside, before)
    after_amplitude = amplitude_similarity(inside, after)
    before_continuity = before_cosine * before_amplitude
    after_continuity = after_cosine * after_amplitude
    before_occupancy = band_occupancy(masks[0])
    inside_occupancy = band_occupancy(masks[1])
    after_occupancy = band_occupancy(masks[2])
    return {
        "adjacent_energy_bin_count": int(adjacent_energy_bins),
        "peak_band_start_index": band_start,
        "peak_band_stop_index_exclusive": band_stop,
        "candidate_adjacent_energy_fraction": concentration,
        "minimum_context_continuity": float(min(before_continuity, after_continuity)),
        "minimum_peak_band_occupancy": float(min(before_occupancy, inside_occupancy, after_occupancy)),
        "before_context_continuity": float(before_continuity),
        "after_context_continuity": float(after_continuity),
        "before_peak_band_occupancy": float(before_occupancy),
        "inside_peak_band_occupancy": float(inside_occupancy),
        "after_peak_band_occupancy": float(after_occupancy),
        "before_shape_cosine": before_cosine,
        "after_shape_cosine": after_cosine,
        "before_amplitude_similarity": before_amplitude,
        "after_amplitude_similarity": after_amplitude,
    }


def context_persistence_veto(
    features: Mapping[str, float], threshold: Mapping[str, float]
) -> bool:
    """Veto only when narrow-band, continuity, and occupancy tests all pass."""

    return bool(
        float(features["candidate_adjacent_energy_fraction"])
        >= float(threshold["minimum_adjacent_energy_fraction"])
        and float(features["minimum_context_continuity"])
        >= float(threshold["minimum_context_continuity"])
        and float(features["minimum_peak_band_occupancy"])
        >= float(threshold["minimum_peak_band_occupancy"])
    )


def single_boundary_context_persistence_veto(
    features: Mapping[str, float], threshold: Mapping[str, float]
) -> dict[str, bool]:
    """Veto when the candidate narrow band persists across either boundary."""

    concentration_pass = (
        float(features["candidate_adjacent_energy_fraction"])
        >= float(threshold["minimum_adjacent_energy_fraction"])
    )
    continuity_limit = float(threshold["minimum_context_continuity"])
    occupancy_limit = float(threshold["minimum_peak_band_occupancy"])
    inside_occupancy = float(features["inside_peak_band_occupancy"])
    before = bool(
        concentration_pass
        and float(features["before_context_continuity"]) >= continuity_limit
        and min(inside_occupancy, float(features["before_peak_band_occupancy"]))
        >= occupancy_limit
    )
    after = bool(
        concentration_pass
        and float(features["after_context_continuity"]) >= continuity_limit
        and min(inside_occupancy, float(features["after_peak_band_occupancy"]))
        >= occupancy_limit
    )
    return {
        "before_boundary_veto": before,
        "after_boundary_veto": after,
        "veto": bool(before or after),
    }


HPLUS_SIGNAL_FEATURE_NAMES = (
    "log10_max_time_median_dpf",
    "maximum_energy_occupancy",
    "three_bin_energy_fraction",
    "three_bin_record_occupancy",
    "log10_three_bin_contrast",
    "log10_three_bin_center_energy_eV",
)


def extract_hplus_signal_features(
    spectrum: np.ndarray,
    energy_eV: np.ndarray,
) -> dict[str, float]:
    """Extract robust H+ band features from quality-clean Mode-1 DEF records."""

    values = np.asarray(spectrum, dtype=float)
    energy = np.asarray(energy_eV, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != energy.size:
        raise ValueError("spectrum must have shape (record, energy)")
    if energy.size < 3 or np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("energy_eV must contain at least three positive finite bins")
    clean = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    mean = np.mean(clean, axis=0)
    time_median = np.median(clean, axis=0)
    occupancy = np.mean(clean > 0.0, axis=0)
    rolling = np.convolve(mean, np.ones(3), mode="valid")
    band_start = int(np.argmax(rolling))
    band_stop = band_start + 3
    band = np.sum(clean[:, band_start:band_stop], axis=1)
    outside = np.concatenate((clean[:, :band_start], clean[:, band_stop:]), axis=1)
    outside_positive = outside[outside > 0.0]
    outside_median = float(np.median(outside_positive)) if outside_positive.size else 1.0
    positive_band = band[band > 0.0]
    band_median = float(np.median(positive_band)) if positive_band.size else 0.0
    total = float(np.sum(mean))
    contrast = band_median / (3.0 * outside_median) if outside_median > 0.0 else 0.0
    return {
        "log10_max_time_median_dpf": float(np.log10(max(float(np.max(time_median)), 1.0))),
        "maximum_energy_occupancy": float(np.max(occupancy)),
        "three_bin_energy_fraction": float(rolling[band_start] / total) if total > 0.0 else 0.0,
        "three_bin_record_occupancy": float(np.mean(band > 0.0)),
        "log10_three_bin_contrast": float(np.log10(max(contrast, 1.0e-3))),
        "log10_three_bin_center_energy_eV": float(np.log10(energy[band_start + 1])),
        "three_bin_start_index": band_start,
        "three_bin_stop_index_exclusive": band_stop,
        "three_bin_energy_range_eV": [float(energy[band_start]), float(energy[band_stop - 1])],
    }


def hplus_feature_vector(features: Mapping[str, float]) -> np.ndarray:
    """Return the ordered six-element H+ classifier vector."""

    return np.asarray([float(features[name]) for name in HPLUS_SIGNAL_FEATURE_NAMES], dtype=float)


def robust_feature_scale(approved_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return approved-sample median and nonzero interquartile scale."""

    values = np.asarray(approved_vectors, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("approved_vectors must contain at least three rows")
    center = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    return center, scale


def prototype_signal_score(
    vector: np.ndarray,
    approved_vectors: np.ndarray,
    signal_vectors: np.ndarray,
) -> dict[str, float]:
    """Score positive values as closer to a signal prototype than approval."""

    point = np.asarray(vector, dtype=float).reshape(1, -1)
    approved = np.asarray(approved_vectors, dtype=float)
    signal = np.asarray(signal_vectors, dtype=float)
    if approved.ndim != 2 or signal.ndim != 2 or approved.shape[1:] != point.shape[1:] or signal.shape[1:] != point.shape[1:]:
        raise ValueError("prototype vector dimensions do not match")
    approved_distance = float(np.min(np.sqrt(np.mean((approved - point) ** 2, axis=1))))
    signal_distance = float(np.min(np.sqrt(np.mean((signal - point) ** 2, axis=1))))
    return {
        "nearest_approved_distance": approved_distance,
        "nearest_signal_distance": signal_distance,
        "prototype_signal_score": approved_distance - signal_distance,
    }


def calibrate_prototype_threshold(
    approved_vectors: np.ndarray,
    signal_vectors: np.ndarray,
    *,
    maximum_approved_false_vetoes: int = 2,
) -> dict[str, Any]:
    """Set a threshold between allowed and disallowed approval scores."""

    approved = np.asarray(approved_vectors, dtype=float)
    signal = np.asarray(signal_vectors, dtype=float)
    if maximum_approved_false_vetoes < 0 or maximum_approved_false_vetoes >= approved.shape[0]:
        raise ValueError("maximum_approved_false_vetoes is outside the approved sample")
    distances = np.sqrt(np.mean((approved[:, None, :] - approved[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest_approved = np.min(distances, axis=1)
    nearest_signal = np.min(
        np.sqrt(np.mean((approved[:, None, :] - signal[None, :, :]) ** 2, axis=2)), axis=1
    )
    scores = nearest_approved - nearest_signal
    descending = np.sort(scores)[::-1]
    upper = descending[maximum_approved_false_vetoes - 1] if maximum_approved_false_vetoes else np.inf
    lower = descending[maximum_approved_false_vetoes]
    threshold = float(lower + (upper - lower) / 2.0) if maximum_approved_false_vetoes else float(descending[0] + 1.0e-12)
    return {
        "threshold": threshold,
        "maximum_approved_false_vetoes": int(maximum_approved_false_vetoes),
        "approved_false_veto_count": int(np.count_nonzero(scores >= threshold)),
        "approved_scores": scores.tolist(),
        "highest_nonvetoed_approved_score": float(lower),
        "lowest_vetoed_approved_score": float(upper),
    }


SPECIES_QUIET_FEATURE_NAMES = (
    "log1p_peak_to_positive_median",
    "three_bin_energy_fraction",
    "peak_energy_record_occupancy",
    "positive_energy_bin_fraction",
    "three_bin_record_occupancy",
    "log1p_three_bin_contrast",
    "log1p_record_total_p90_to_median",
)


def extract_species_quiet_features(spectrum: np.ndarray) -> dict[str, float]:
    """Extract dimensionless signal-shape features for one MINPA species.

    These features intentionally omit absolute DPF amplitude.  They describe
    spectral concentration, temporal persistence, and boundary-like bursts,
    so the same approved envelope can be used across epochs whose background
    levels differ.
    """

    values = np.asarray(spectrum, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 3:
        raise ValueError("spectrum must have at least three records and three energies")
    clean = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    mean = np.mean(clean, axis=0)
    positive_mean = mean[mean > 0.0]
    width = 3
    rolling = np.convolve(mean, np.ones(width), mode="valid")
    band_start = int(np.argmax(rolling)) if rolling.size else 0
    band_stop = band_start + width
    peak_index = int(np.argmax(mean))
    peak_median = float(np.median(positive_mean)) if positive_mean.size else 0.0
    peak_ratio = float(mean[peak_index] / peak_median) if peak_median > 0.0 else 0.0
    total = float(np.sum(mean))
    band_fraction = float(rolling[band_start] / total) if total > 0.0 else 0.0
    band_record_total = np.sum(clean[:, band_start:band_stop], axis=1)
    outside = np.concatenate((clean[:, :band_start], clean[:, band_stop:]), axis=1)
    outside_positive = outside[outside > 0.0]
    outside_median = float(np.median(outside_positive)) if outside_positive.size else 0.0
    band_positive = clean[:, band_start:band_stop]
    band_positive = band_positive[band_positive > 0.0]
    band_median = float(np.median(band_positive)) if band_positive.size else 0.0
    contrast = band_median / outside_median if outside_median > 0.0 else 0.0
    record_total = np.sum(clean, axis=1)
    positive_record_total = record_total[record_total > 0.0]
    if positive_record_total.size:
        median_total = float(np.median(positive_record_total))
        burst_ratio = (
            float(np.quantile(positive_record_total, 0.90) / median_total)
            if median_total > 0.0 else 0.0
        )
    else:
        burst_ratio = 0.0
    return {
        "log1p_peak_to_positive_median": float(np.log1p(peak_ratio)),
        "three_bin_energy_fraction": band_fraction,
        "peak_energy_record_occupancy": float(np.mean(clean[:, peak_index] > 0.0)),
        "positive_energy_bin_fraction": float(np.mean(mean > 0.0)),
        "three_bin_record_occupancy": float(np.mean(band_record_total > 0.0)),
        "log1p_three_bin_contrast": float(np.log1p(contrast)),
        "log1p_record_total_p90_to_median": float(np.log1p(burst_ratio)),
        "three_bin_start_index": band_start,
        "three_bin_stop_index_exclusive": band_stop,
    }


def species_quiet_feature_vector(features: Mapping[str, float]) -> np.ndarray:
    """Return the ordered dimensionless approved-envelope feature vector."""

    return np.asarray(
        [float(features[name]) for name in SPECIES_QUIET_FEATURE_NAMES], dtype=float
    )


def nearest_prototype_distance(
    vector: np.ndarray,
    prototypes: np.ndarray,
) -> float:
    """Return RMS Euclidean distance to the nearest approved prototype."""

    point = np.asarray(vector, dtype=float).reshape(1, -1)
    reference = np.asarray(prototypes, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != point.shape[1]:
        raise ValueError("prototype vector dimensions do not match")
    return float(np.min(np.sqrt(np.mean((reference - point) ** 2, axis=1))))


def calibrate_joint_approved_distance(
    species_vectors: Mapping[str, np.ndarray],
    *,
    maximum_joint_false_rejects: int = 4,
) -> dict[str, Any]:
    """Calibrate a joint one-class limit from leave-one-out approvals.

    Each species is robustly scaled independently.  The interval score is the
    largest species-specific nearest-neighbour distance, so one species with a
    real spectrum is sufficient to reject the joint interval.
    """

    if not species_vectors:
        raise ValueError("species_vectors is empty")
    names = tuple(species_vectors)
    row_counts = {np.asarray(species_vectors[name]).shape[0] for name in names}
    if len(row_counts) != 1:
        raise ValueError("all species must contain the same approved intervals")
    count = row_counts.pop()
    if count < 3 or not 0 <= maximum_joint_false_rejects < count:
        raise ValueError("invalid approved count or false-reject allowance")
    payload: dict[str, Any] = {}
    normalized: list[np.ndarray] = []
    for name in names:
        values = np.asarray(species_vectors[name], dtype=float)
        center, scale = robust_feature_scale(values)
        scaled = (values - center) / scale
        distances = np.sqrt(
            np.mean((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2)
        )
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        species_scale = float(np.median(nearest))
        if not np.isfinite(species_scale) or species_scale <= 1.0e-12:
            species_scale = 1.0
        normalized.append(nearest / species_scale)
        payload[name] = {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "nearest_approved_distance_scale": species_scale,
            "scaled_approved_vectors": scaled.tolist(),
            "leave_one_out_normalized_distances": (nearest / species_scale).tolist(),
        }
    joint_scores = np.max(np.stack(normalized, axis=1), axis=1)
    ordered = np.sort(joint_scores)[::-1]
    if maximum_joint_false_rejects:
        upper = float(ordered[maximum_joint_false_rejects - 1])
        lower = float(ordered[maximum_joint_false_rejects])
        threshold = lower + (upper - lower) / 2.0
    else:
        threshold = float(ordered[0] + 1.0e-12)
    return {
        "species": payload,
        "joint_threshold": threshold,
        "approved_interval_count": count,
        "maximum_joint_false_rejects": int(maximum_joint_false_rejects),
        "joint_false_reject_count": int(np.count_nonzero(joint_scores > threshold)),
        "joint_leave_one_out_scores": joint_scores.tolist(),
    }


def score_joint_approved_distance(
    features: Mapping[str, Mapping[str, float]],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a three-species interval against the approved one-class model."""

    threshold = float(calibration["joint_threshold"])
    species_scores: dict[str, Any] = {}
    for name, model in calibration["species"].items():
        center = np.asarray(model["center"], dtype=float)
        scale = np.asarray(model["scale"], dtype=float)
        prototypes = np.asarray(model["scaled_approved_vectors"], dtype=float)
        vector = (species_quiet_feature_vector(features[name]) - center) / scale
        distance = nearest_prototype_distance(vector, prototypes)
        normalized = distance / float(model["nearest_approved_distance_scale"])
        species_scores[name] = {
            "nearest_approved_distance": distance,
            "normalized_approved_distance": normalized,
            "passes_joint_limit": bool(normalized <= threshold),
        }
    joint_score = max(
        float(item["normalized_approved_distance"]) for item in species_scores.values()
    )
    return {
        "species": species_scores,
        "joint_score": joint_score,
        "joint_threshold": threshold,
        "passes": bool(joint_score <= threshold),
        "failing_species": [
            name for name, item in species_scores.items()
            if not bool(item["passes_joint_limit"])
        ],
    }


def _nearest_k_rms_distances(
    points: np.ndarray,
    prototypes: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Return the mean of the k nearest RMS distances for every point."""

    left = np.asarray(points, dtype=float)
    right = np.asarray(prototypes, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("point and prototype dimensions do not match")
    if k < 1 or k > right.shape[0]:
        raise ValueError("k is outside the prototype count")
    distance = np.sqrt(np.mean((left[:, None, :] - right[None, :, :]) ** 2, axis=2))
    nearest = np.partition(distance, kth=k - 1, axis=1)[:, :k]
    return np.mean(nearest, axis=1)


def calibrate_joint_labelled_prototypes(
    approved_vectors: np.ndarray,
    rejected_vectors: np.ndarray,
    *,
    maximum_approved_false_rejects: int = 8,
    nearest_neighbours: int = 3,
    protected_approved_indices: Sequence[int] = (),
    maximum_protected_false_rejects: int = 1,
) -> dict[str, Any]:
    """Calibrate a joint accepted/rejected nearest-prototype classifier.

    Labels remain interval-level.  Rejected windows are not expanded into
    per-species rejection labels, which would incorrectly label a quiet H+
    spectrum when only O+ caused the human rejection.
    """

    approved = np.asarray(approved_vectors, dtype=float)
    rejected = np.asarray(rejected_vectors, dtype=float)
    if approved.ndim != 2 or rejected.ndim != 2 or approved.shape[1] != rejected.shape[1]:
        raise ValueError("approved/rejected vector dimensions do not match")
    if approved.shape[0] < nearest_neighbours + 1 or rejected.shape[0] < nearest_neighbours + 1:
        raise ValueError("too few prototypes for leave-one-out calibration")
    if not 0 <= maximum_approved_false_rejects < approved.shape[0]:
        raise ValueError("maximum_approved_false_rejects is invalid")
    center, scale = robust_feature_scale(approved)
    approved_scaled = (approved - center) / scale
    rejected_scaled = (rejected - center) / scale

    approved_pairwise = np.sqrt(np.mean(
        (approved_scaled[:, None, :] - approved_scaled[None, :, :]) ** 2, axis=2
    ))
    np.fill_diagonal(approved_pairwise, np.inf)
    rejected_pairwise = np.sqrt(np.mean(
        (rejected_scaled[:, None, :] - rejected_scaled[None, :, :]) ** 2, axis=2
    ))
    np.fill_diagonal(rejected_pairwise, np.inf)
    approved_within = np.mean(
        np.partition(approved_pairwise, kth=nearest_neighbours - 1, axis=1)[:, :nearest_neighbours], axis=1
    )
    rejected_within = np.mean(
        np.partition(rejected_pairwise, kth=nearest_neighbours - 1, axis=1)[:, :nearest_neighbours], axis=1
    )
    approved_distance_scale = max(float(np.median(approved_within)), 1.0e-12)
    rejected_distance_scale = max(float(np.median(rejected_within)), 1.0e-12)

    approved_to_rejected = _nearest_k_rms_distances(
        approved_scaled, rejected_scaled, k=nearest_neighbours
    )
    rejected_to_approved = _nearest_k_rms_distances(
        rejected_scaled, approved_scaled, k=nearest_neighbours
    )
    approved_scores = approved_within / approved_distance_scale - approved_to_rejected / rejected_distance_scale
    rejected_scores = rejected_to_approved / approved_distance_scale - rejected_within / rejected_distance_scale

    def threshold_for(scores: np.ndarray, allowed: int) -> float:
        if not 0 <= allowed < scores.size:
            raise ValueError("false-reject allowance is outside the score count")
        ordered = np.sort(scores)[::-1]
        if allowed == 0:
            return float(ordered[0] + 1.0e-12)
        return float((ordered[allowed - 1] + ordered[allowed]) / 2.0)

    threshold = threshold_for(approved_scores, maximum_approved_false_rejects)
    protected = np.asarray(tuple(protected_approved_indices), dtype=int)
    if protected.size:
        if np.any((protected < 0) | (protected >= approved.shape[0])):
            raise ValueError("protected approved index is invalid")
        threshold = max(
            threshold,
            threshold_for(approved_scores[protected], maximum_protected_false_rejects),
        )
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "approved_scaled_vectors": approved_scaled.tolist(),
        "rejected_scaled_vectors": rejected_scaled.tolist(),
        "nearest_neighbours": int(nearest_neighbours),
        "approved_distance_scale": approved_distance_scale,
        "rejected_distance_scale": rejected_distance_scale,
        "threshold": threshold,
        "approved_leave_one_out_scores": approved_scores.tolist(),
        "rejected_leave_one_out_scores": rejected_scores.tolist(),
        "approved_false_reject_count": int(np.count_nonzero(approved_scores > threshold)),
        "rejected_detected_count": int(np.count_nonzero(rejected_scores > threshold)),
        "approved_count": int(approved.shape[0]),
        "rejected_count": int(rejected.shape[0]),
        "protected_approved_count": int(protected.size),
        "protected_approved_false_reject_count": int(
            np.count_nonzero(approved_scores[protected] > threshold) if protected.size else 0
        ),
    }


def score_joint_labelled_prototypes(
    vector: np.ndarray,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a joint interval; positive values are more rejection-like."""

    center = np.asarray(calibration["center"], dtype=float)
    scale = np.asarray(calibration["scale"], dtype=float)
    point = (np.asarray(vector, dtype=float).reshape(1, -1) - center) / scale
    k = int(calibration["nearest_neighbours"])
    approved_distance = float(_nearest_k_rms_distances(
        point, np.asarray(calibration["approved_scaled_vectors"], dtype=float), k=k
    )[0])
    rejected_distance = float(_nearest_k_rms_distances(
        point, np.asarray(calibration["rejected_scaled_vectors"], dtype=float), k=k
    )[0])
    score = (
        approved_distance / float(calibration["approved_distance_scale"])
        - rejected_distance / float(calibration["rejected_distance_scale"])
    )
    threshold = float(calibration["threshold"])
    return {
        "approved_distance": approved_distance,
        "rejected_distance": rejected_distance,
        "labelled_prototype_score": score,
        "threshold": threshold,
        "passes": bool(score <= threshold),
    }


TEMPORAL_COHERENCE_FEATURE_NAMES = (
    "median_spectral_cosine",
    "q10_spectral_cosine",
    "median_three_bin_record_fraction",
    "q90_three_bin_record_fraction",
    "median_spectral_entropy",
    "log_energy_centroid_std",
    "positive_record_fraction",
)


def extract_temporal_coherence_features(
    spectrum: np.ndarray,
    energy_eV: np.ndarray,
) -> dict[str, float]:
    """Describe record-to-record spectral coherence without DPF amplitude."""

    values = np.asarray(spectrum, dtype=float)
    energy = np.asarray(energy_eV, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != energy.size:
        raise ValueError("spectrum and energy dimensions do not match")
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("energy must be positive and finite")
    clean = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    record_total = np.sum(clean, axis=1)
    normalized = np.divide(
        clean, record_total[:, None], out=np.zeros_like(clean),
        where=record_total[:, None] > 0.0,
    )
    median_shape = np.median(normalized, axis=0)
    denominator = np.linalg.norm(normalized, axis=1) * float(np.linalg.norm(median_shape))
    cosine = np.divide(
        normalized @ median_shape, denominator,
        out=np.zeros(values.shape[0], dtype=float), where=denominator > 0.0,
    )
    rolling = np.stack([
        np.sum(normalized[:, start:start + 3], axis=1)
        for start in range(energy.size - 2)
    ], axis=1)
    record_concentration = np.max(rolling, axis=1)
    entropy = -np.sum(
        np.where(normalized > 0.0, normalized * np.log(np.maximum(normalized, 1.0e-12)), 0.0),
        axis=1,
    ) / math.log(float(energy.size))
    centroid = normalized @ np.log10(energy)
    return {
        "median_spectral_cosine": float(np.median(cosine)),
        "q10_spectral_cosine": float(np.quantile(cosine, 0.10)),
        "median_three_bin_record_fraction": float(np.median(record_concentration)),
        "q90_three_bin_record_fraction": float(np.quantile(record_concentration, 0.90)),
        "median_spectral_entropy": float(np.median(entropy)),
        "log_energy_centroid_std": float(np.std(centroid)),
        "positive_record_fraction": float(np.mean(record_total > 0.0)),
    }


def temporal_coherence_feature_vector(features: Mapping[str, float]) -> np.ndarray:
    """Return the ordered seven-element temporal coherence vector."""

    return np.asarray(
        [float(features[name]) for name in TEMPORAL_COHERENCE_FEATURE_NAMES],
        dtype=float,
    )


def calibrate_species_temporal_outliers(
    approved_features: Sequence[Mapping[str, Mapping[str, float]]],
    *,
    maximum_joint_false_rejects: int = 4,
) -> dict[str, Any]:
    """Calibrate interpretable per-species temporal-coherence outlier scores."""

    if len(approved_features) < 5:
        raise ValueError("too few approved intervals")
    species_names = tuple(approved_features[0])
    fallback_scale = np.asarray((0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05))
    payload: dict[str, Any] = {}
    scores: list[np.ndarray] = []
    for species in species_names:
        values = np.stack([
            temporal_coherence_feature_vector(row[species])
            for row in approved_features
        ])
        center = np.median(values, axis=0)
        scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
        scale = np.maximum(scale, fallback_scale)
        standardized = (values - center) / scale
        # Large centroid variability describes sparse noise, not coherent signal.
        standardized[:, 5] *= -1.0
        score = np.max(standardized, axis=1)
        scores.append(score)
        payload[species] = {"center": center.tolist(), "scale": scale.tolist()}
    joint = np.max(np.stack(scores, axis=1), axis=1)
    ordered = np.sort(joint)[::-1]
    if not 0 <= maximum_joint_false_rejects < joint.size:
        raise ValueError("maximum_joint_false_rejects is invalid")
    threshold = (
        float((ordered[maximum_joint_false_rejects - 1] + ordered[maximum_joint_false_rejects]) / 2.0)
        if maximum_joint_false_rejects else float(ordered[0] + 1.0e-12)
    )
    return {
        "species": payload,
        "threshold": threshold,
        "approved_joint_scores": joint.tolist(),
        "approved_false_reject_count": int(np.count_nonzero(joint > threshold)),
        "approved_count": len(approved_features),
    }


def score_species_temporal_outliers(
    features: Mapping[str, Mapping[str, float]],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Return per-species interpretable temporal-coherence outlier scores."""

    threshold = float(calibration["threshold"])
    species_scores: dict[str, float] = {}
    for species, model in calibration["species"].items():
        center = np.asarray(model["center"], dtype=float)
        scale = np.asarray(model["scale"], dtype=float)
        standardized = (temporal_coherence_feature_vector(features[species]) - center) / scale
        standardized[5] *= -1.0
        species_scores[species] = float(np.max(standardized))
    failing = [species for species, score in species_scores.items() if score > threshold]
    return {
        "species_scores": species_scores,
        "threshold": threshold,
        "failing_species": failing,
        "passes": not failing,
    }


ENERGY_BAND_EXCESS_FEATURE_NAMES = (
    "log1p_peak_to_positive_median",
    "log1p_three_bin_to_positive_median",
    "three_bin_energy_fraction",
    "peak_energy_record_occupancy",
    "log1p_time_median_peak_to_positive_median",
    "log1p_record_peak_ratio_q90",
    "record_three_bin_fraction_q90",
)


def extract_energy_band_excess_features(spectrum: np.ndarray) -> dict[str, float]:
    """Measure energy bands whose flux is elevated above the other bands.

    The interval mean and time median retain zeros. Ratios are relative to the
    positive-energy median, so no absolute DPF threshold or epoch assumption is
    introduced. Record-level q90 features retain sensitivity to brief signals
    near a proposed boundary.
    """

    values = np.asarray(spectrum, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 3:
        raise ValueError("spectrum must contain records and at least three energies")
    clean = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    mean = np.mean(clean, axis=0)
    positive_mean = mean[mean > 0.0]
    spectral_median = float(np.median(positive_mean)) if positive_mean.size else 0.0
    peak_index = int(np.argmax(mean))
    peak_ratio = float(mean[peak_index] / spectral_median) if spectral_median > 0.0 else 0.0
    rolling = np.convolve(mean, np.ones(3), mode="valid")
    band_start = int(np.argmax(rolling))
    three_to_median = (
        float(rolling[band_start] / (3.0 * spectral_median))
        if spectral_median > 0.0 else 0.0
    )
    total = float(np.sum(mean))
    three_fraction = float(rolling[band_start] / total) if total > 0.0 else 0.0
    occupancy = float(np.mean(clean[:, peak_index] > 0.0))

    time_median = np.median(clean, axis=0)
    positive_time_median = time_median[time_median > 0.0]
    if positive_time_median.size:
        time_median_ratio = float(
            np.max(time_median) / np.median(positive_time_median)
        )
    else:
        time_median_ratio = 0.0

    record_peak_ratios: list[float] = []
    record_three_fractions: list[float] = []
    for record in clean:
        positive = record[record > 0.0]
        if not positive.size:
            record_peak_ratios.append(0.0)
            record_three_fractions.append(0.0)
            continue
        median = float(np.median(positive))
        record_peak_ratios.append(float(np.max(record) / median) if median > 0.0 else 0.0)
        record_total = float(np.sum(record))
        record_three = float(np.max(np.convolve(record, np.ones(3), mode="valid")))
        record_three_fractions.append(record_three / record_total if record_total > 0.0 else 0.0)
    return {
        "log1p_peak_to_positive_median": float(np.log1p(peak_ratio)),
        "log1p_three_bin_to_positive_median": float(np.log1p(three_to_median)),
        "three_bin_energy_fraction": three_fraction,
        "peak_energy_record_occupancy": occupancy,
        "log1p_time_median_peak_to_positive_median": float(np.log1p(time_median_ratio)),
        "log1p_record_peak_ratio_q90": float(np.log1p(np.quantile(record_peak_ratios, 0.90))),
        "record_three_bin_fraction_q90": float(np.quantile(record_three_fractions, 0.90)),
        "peak_energy_index": peak_index,
        "three_bin_start_index": band_start,
        "three_bin_stop_index_exclusive": band_start + 3,
    }


def energy_band_excess_feature_vector(features: Mapping[str, float]) -> np.ndarray:
    """Return the ordered seven-element energy-band excess vector."""

    return np.asarray(
        [float(features[name]) for name in ENERGY_BAND_EXCESS_FEATURE_NAMES],
        dtype=float,
    )


def calibrate_energy_band_excess_limits(
    approved_features: Sequence[Mapping[str, Mapping[str, float]]],
    *,
    quantile: float = 0.98,
    joint_score_quantile: float = 0.95,
) -> dict[str, Any]:
    """Calibrate per-species upper limits solely from approved intervals."""

    if len(approved_features) < 10:
        raise ValueError("too few approved intervals")
    if not 0.5 <= quantile < 1.0 or not 0.5 <= joint_score_quantile < 1.0:
        raise ValueError("calibration quantiles must lie in [0.5, 1)")
    species_payload: dict[str, Any] = {}
    approved_scores: list[np.ndarray] = []
    for species in approved_features[0]:
        matrix = np.stack([
            energy_band_excess_feature_vector(row[species])
            for row in approved_features
        ])
        limits = np.quantile(matrix, quantile, axis=0)
        floors = np.asarray((0.05, 0.05, 0.02, 0.05, 0.05, 0.05, 0.02))
        limits = np.maximum(limits, floors)
        score = np.max(matrix / limits, axis=1)
        approved_scores.append(score)
        species_payload[species] = {"limits": limits.tolist()}
    joint = np.max(np.stack(approved_scores, axis=1), axis=1)
    threshold = float(np.quantile(joint, joint_score_quantile))
    return {
        "species": species_payload,
        "feature_names": list(ENERGY_BAND_EXCESS_FEATURE_NAMES),
        "feature_limit_quantile": float(quantile),
        "joint_score_quantile": float(joint_score_quantile),
        "joint_threshold": threshold,
        "approved_count": len(approved_features),
        "approved_false_reject_count": int(np.count_nonzero(joint > threshold)),
        "approved_joint_scores": joint.tolist(),
    }


def score_energy_band_excess(
    features: Mapping[str, Mapping[str, float]],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject when any species contains an unusually elevated energy band."""

    threshold = float(calibration["joint_threshold"])
    species_scores: dict[str, Any] = {}
    for species, model in calibration["species"].items():
        vector = energy_band_excess_feature_vector(features[species])
        limits = np.asarray(model["limits"], dtype=float)
        ratios = vector / limits
        score = float(np.max(ratios))
        species_scores[species] = {
            "score": score,
            "feature_ratios": ratios.tolist(),
            "dominant_feature": ENERGY_BAND_EXCESS_FEATURE_NAMES[int(np.argmax(ratios))],
            "passes": bool(score <= threshold),
        }
    joint_score = max(float(item["score"]) for item in species_scores.values())
    failing = [name for name, item in species_scores.items() if not bool(item["passes"])]
    return {
        "species": species_scores,
        "joint_score": joint_score,
        "joint_threshold": threshold,
        "passes": not failing,
        "failing_species": failing,
    }


def generate_quiet_windows_from_minute_blocks(
    minute_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_minutes: int = 3,
    maximum_minutes: int = 10,
) -> list[dict[str, Any]]:
    """Generate safe windows whose every constituent minute passes.

    Candidates are ordered by the lowest maximum minute-level band-excess
    score. Duration is only a tie breaker, so a cleaner three-minute interval
    is preferred over a contaminated-looking long interval.
    """

    if minimum_minutes < 1 or maximum_minutes < minimum_minutes:
        raise ValueError("invalid minute duration limits")
    rows = sorted((dict(row) for row in minute_rows), key=lambda row: float(row["start_unix_s"]))
    if not rows:
        return []
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        start = float(row["start_unix_s"])
        stop = float(row["stop_unix_s"])
        if stop - start != 60.0:
            raise ValueError("minute blocks must have exactly 60-second duration")
        if not bool(row.get("passes", False)):
            if current:
                runs.append(current)
                current = []
            continue
        if current and abs(start - float(current[-1]["stop_unix_s"])) > 1.0e-6:
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)

    candidates: list[dict[str, Any]] = []
    for run in runs:
        maximum = min(maximum_minutes, len(run))
        for width in range(minimum_minutes, maximum + 1):
            for offset in range(0, len(run) - width + 1):
                selected = run[offset:offset + width]
                candidates.append({
                    "start_unix_s": float(selected[0]["start_unix_s"]),
                    "stop_unix_s": float(selected[-1]["stop_unix_s"]),
                    "duration_s": float(width * 60),
                    "constituent_minute_count": width,
                    "maximum_minute_band_excess_score": float(max(
                        float(row["band_excess_score"]) for row in selected
                    )),
                    "mean_minute_band_excess_score": float(np.mean([
                        float(row["band_excess_score"]) for row in selected
                    ])),
                    "constituent_minutes": selected,
                })
    return sorted(candidates, key=lambda row: (
        float(row["maximum_minute_band_excess_score"]),
        float(row["mean_minute_band_excess_score"]),
        -float(row["duration_s"]),
        float(row["start_unix_s"]),
    ))
