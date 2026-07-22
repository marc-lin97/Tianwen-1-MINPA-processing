"""MINPA Mode-1 background estimation and subtraction.

The public/``ori`` science array is treated as differential particle flux
(DPF), not as detector counts.  The module nevertheless preserves the
conditional-nonzero estimator used by Wang et al. (2024) as a sensitivity
product, alongside a zero-inclusive robust estimator suitable for the
released DPF product.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import h5py
import numpy as np
from scipy.io import loadmat


ALGORITHM_VERSION = "minpa-background-dpf-mode1-v1"
MODE1_ENERGY_EV = np.array(
    [
        2.81, 3.548928, 4.482167, 5.660813, 7.149401, 9.029433,
        11.40385, 14.40264, 18.19001, 22.97332, 29.01447, 36.64422,
        46.28032, 58.45036, 73.82067, 93.23282, 117.7497, 148.7135,
        187.8198, 237.2095, 299.587, 378.3675, 477.8644, 603.5253,
        762.2305, 962.6694, 1215.816, 1535.532, 1939.321, 2449.292,
        3093.366, 3906.809, 4934.157, 6231.661, 7870.361, 9939.98,
        12553.83, 15855.03, 20024.33, 25290.0,
    ],
    dtype=float,
)
MODE1_MASS_AMU = np.array([1.0, 2.0, 4.0, 16.0, 38.0, 32.0, 44.0, 65.0])
MODE1_PITCH_EDGES_DEG = np.linspace(0.0, 90.0, 5)
MODE1_PITCH_CENTERS_DEG = (MODE1_PITCH_EDGES_DEG[:-1] + MODE1_PITCH_EDGES_DEG[1:]) / 2.0
MODE1_AZIMUTH_EDGES_DEG = np.linspace(0.0, 360.0, 17)
MODE1_AZIMUTH_CENTERS_DEG = (MODE1_AZIMUTH_EDGES_DEG[:-1] + MODE1_AZIMUTH_EDGES_DEG[1:]) / 2.0
MODE1_SHAPE = (40, 4, 16, 8)  # energy, pitch, azimuth, mass
MODE1_FLAT_SIZE = int(np.prod(MODE1_SHAPE))

# Project quality bits 3 and 4: UV and alternating energy channels.  Bits 1
# and 2 are retained because sparse/low-count records are relevant background
# candidates.  Bit 5 is also retained under the relaxed background policy; its
# rate and amplitude remain available as diagnostics rather than a hard veto.
PROJECT_BACKGROUND_REJECT_MASK = np.uint32(4 | 8)


@dataclass(frozen=True)
class BackgroundInterval:
    """A candidate or approved UTC background interval."""

    start_utc: str
    stop_utc: str
    approved: bool = False
    label: str = ""
    source: str = "manual"

    @property
    def start_s(self) -> float:
        return _parse_utc(self.start_utc)

    @property
    def stop_s(self) -> float:
        return _parse_utc(self.stop_utc)


def approved_intervals_from_manual_review(
    review: Mapping[str, Any],
    original_candidates: Mapping[str, Any],
    expanded_candidates: Mapping[str, Any],
) -> list[BackgroundInterval]:
    """Resolve reviewed original, full, and boundary-adjusted candidates.

    Candidate numbers are deliberately resolved against the saved candidate
    products instead of being treated as dates.  This keeps the human review
    concise while the returned intervals remain explicit and reproducible.
    """

    original_items = list(original_candidates["intervals"])
    expanded_item_list = list(expanded_candidates["intervals"])
    expanded_by_original = {
        int(item["original_candidate_number"]): item
        for item in expanded_item_list
        if item.get("original_candidate_number") is not None
    }
    original_numbers = {
        int(value) for value in review["decisions"]["approve_no_visible_real_spectrum"]
    }
    intervals: list[BackgroundInterval] = []
    for number in sorted(original_numbers):
        if not 1 <= number <= len(original_items):
            raise ValueError(f"Original candidate number out of range: {number}")
        item = original_items[number - 1]
        start_utc = str(item["start_utc"])
        stop_utc = str(item["stop_utc"])
        label = f"original-{number:02d}"
        source = "manual_original_40_review"
        expanded_match = expanded_by_original.get(number)
        if (
            expanded_match is not None
            and expanded_match.get("boundary_refinement_status")
            == "accepted_high_confidence_subwindow"
        ):
            start_utc = str(expanded_match["refined_start_utc"])
            stop_utc = str(expanded_match["refined_stop_utc"])
            label = f"original-{number:02d}-auto-refined"
            source = "manual_original_40_review_plus_automatic_boundary_refinement"
        intervals.append(BackgroundInterval(
            start_utc=start_utc,
            stop_utc=stop_utc,
            approved=True,
            label=label,
            source=source,
        ))

    expanded_items = {
        int(item["expanded_candidate_number"]): item
        for item in expanded_item_list
    }
    expanded_review = review.get("expanded_review", {}).get("decisions", {})
    full_numbers = {int(value) for value in expanded_review.get("approve_full_interval", [])}
    rejected_numbers = {
        int(value)
        for value in expanded_review.get("reject_visible_or_weak_real_spectrum", [])
    }
    adjusted_items = list(expanded_review.get("approve_adjusted_interval", []))
    adjusted_numbers = {
        int(item["expanded_candidate_number"]) for item in adjusted_items
    }
    conflicts = (full_numbers & rejected_numbers) | (adjusted_numbers & rejected_numbers)
    if conflicts:
        raise ValueError(f"Expanded candidates both approved and rejected: {sorted(conflicts)}")
    if full_numbers & adjusted_numbers:
        raise ValueError(
            "Expanded candidates cannot be both full and adjusted approvals: "
            f"{sorted(full_numbers & adjusted_numbers)}"
        )

    for number in sorted(full_numbers):
        if number not in expanded_items:
            raise ValueError(f"Expanded candidate number not found: {number}")
        item = expanded_items[number]
        intervals.append(BackgroundInterval(
            start_utc=str(item["start_utc"]),
            stop_utc=str(item["stop_utc"]),
            approved=True,
            label=f"expanded-{number:02d}-full",
            source="manual_expanded_review",
        ))

    for adjusted in adjusted_items:
        number = int(adjusted["expanded_candidate_number"])
        if number not in expanded_items:
            raise ValueError(f"Adjusted expanded candidate number not found: {number}")
        candidate = expanded_items[number]
        start_utc = str(adjusted["start_utc"])
        stop_utc = str(adjusted["stop_utc"])
        start_s, stop_s = _parse_utc(start_utc), _parse_utc(stop_utc)
        if stop_s <= start_s:
            raise ValueError(f"Adjusted candidate {number} has a non-positive duration")
        candidate_start = _parse_utc(str(candidate["start_utc"]))
        candidate_stop = _parse_utc(str(candidate["stop_utc"]))
        if stop_s <= candidate_start or start_s >= candidate_stop:
            raise ValueError(f"Adjusted candidate {number} does not overlap its source window")
        intervals.append(BackgroundInterval(
            start_utc=start_utc,
            stop_utc=stop_utc,
            approved=True,
            label=f"expanded-{number:02d}-adjusted",
            source="manual_expanded_review_adjusted_boundary",
        ))

    intervals.sort(key=lambda interval: (interval.start_s, interval.stop_s))
    seen: set[tuple[float, float]] = set()
    for interval in intervals:
        key = (interval.start_s, interval.stop_s)
        if key in seen:
            raise ValueError(f"Duplicate approved interval: {interval.start_utc}--{interval.stop_utc}")
        seen.add(key)
    return intervals


@dataclass(frozen=True)
class BackgroundConfig:
    """Configuration for a Mode-1 DPF background model."""

    mode: int = 1
    primary_estimator: Literal[
        "robust_including_zero_median", "paper_nonzero_mean",
        "count_equivalent_energy_pooled",
    ] = "robust_including_zero_median"
    reject_quality_mask: int = int(PROJECT_BACKGROUND_REJECT_MASK)
    require_project_quality: bool = True
    require_native_quality_zero: bool = True
    reject_interval_if_project_flagged: bool = True
    min_approved_intervals: int = 20
    min_total_records: int = 300
    min_records_per_interval: int = 18
    min_channel_samples: int = 20
    uncertainty_mad_scale: float = 1.4826
    quantum_min_nonzero_samples: int = 5
    quantum_residual_tolerance_counts: float = 0.05
    quantum_min_valid_fraction: float = 0.95


@dataclass(frozen=True)
class CandidateConfig:
    """Deterministic screening configuration for provisional intervals."""

    duration_s: float = 8.0 * 60.0
    step_s: float = 4.0 * 60.0
    min_records: int = 24
    max_record_gap_s: float = 25.0
    reject_quality_mask: int = int(PROJECT_BACKGROUND_REJECT_MASK)
    low_signal_quantile: float = 0.35
    requested_intervals: int = 40
    max_intervals_per_day: int = 4
    min_separation_s: float = 2.0 * 3600.0


@dataclass
class MinpaMode1Records:
    """Labeled Mode-1 records with explicit science dimensions."""

    time_unix_s: np.ndarray
    dpf: np.ndarray
    native_quality: np.ndarray
    ion_start_stop_counts: np.ndarray
    high_voltage_monitor_v: np.ndarray
    instrument_solar_angles_deg: np.ndarray
    project_quality_flag: np.ndarray
    project_quality_available: np.ndarray
    source_file_index: np.ndarray
    source_files: list[str] = field(default_factory=list)
    source_sha256: dict[str, str] = field(default_factory=dict)
    mode: int = 1
    units: str = "1/(s cm^2 sr eV)"

    def __post_init__(self) -> None:
        n = np.asarray(self.time_unix_s).size
        if self.mode != 1:
            raise ValueError(f"Mode-1 records required, got mode={self.mode}")
        if np.asarray(self.dpf).shape != (n, *MODE1_SHAPE):
            raise ValueError(
                f"dpf shape must be (time, energy, pitch, azimuth, mass) "
                f"with trailing shape {MODE1_SHAPE}; got {np.asarray(self.dpf).shape}"
            )
        for name in (
            "native_quality", "project_quality_flag", "project_quality_available",
            "source_file_index",
        ):
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        if np.asarray(self.ion_start_stop_counts).shape != (n, 4):
            raise ValueError("ion_start_stop_counts must have shape (time, 4)")
        if np.asarray(self.high_voltage_monitor_v).shape != (n, 5, 4):
            raise ValueError("high_voltage_monitor_v must have shape (time, 5, 4)")
        if np.asarray(self.instrument_solar_angles_deg).shape != (n, 2):
            raise ValueError("instrument_solar_angles_deg must have shape (time, 2)")

    def subset(self, mask: np.ndarray) -> "MinpaMode1Records":
        keep = np.asarray(mask, dtype=bool)
        return MinpaMode1Records(
            time_unix_s=self.time_unix_s[keep],
            dpf=self.dpf[keep],
            native_quality=self.native_quality[keep],
            ion_start_stop_counts=self.ion_start_stop_counts[keep],
            high_voltage_monitor_v=self.high_voltage_monitor_v[keep],
            instrument_solar_angles_deg=self.instrument_solar_angles_deg[keep],
            project_quality_flag=self.project_quality_flag[keep],
            project_quality_available=self.project_quality_available[keep],
            source_file_index=self.source_file_index[keep],
            source_files=list(self.source_files),
            source_sha256=dict(self.source_sha256),
            mode=self.mode,
            units=self.units,
        )


@dataclass
class MinpaBackgroundModel:
    """A channel-resolved Mode-1 DPF background model."""

    background_dpf: np.ndarray
    robust_background_dpf: np.ndarray
    paper_background_dpf: np.ndarray
    count_equivalent_background_dpf: np.ndarray
    dpf_quantum: np.ndarray
    background_lambda_count_equivalent: np.ndarray
    quantum_valid_mask: np.ndarray
    quantum_reconstruction_valid_fraction: np.ndarray
    uncertainty_dpf: np.ndarray
    interval_mad_dpf: np.ndarray
    sample_count: np.ndarray
    nonzero_sample_count: np.ndarray
    valid_channel_mask: np.ndarray
    interval_summaries: list[dict[str, object]]
    approved_intervals: list[dict[str, object]]
    source_files: list[str]
    source_sha256: dict[str, str]
    primary_estimator: str
    valid: bool
    invalid_reasons: list[str]
    mode: int = 1
    units: str = "1/(s cm^2 sr eV)"
    algorithm_version: str = ALGORITHM_VERSION
    energy_eV: np.ndarray = field(default_factory=lambda: MODE1_ENERGY_EV.copy())
    pitch_deg: np.ndarray = field(default_factory=lambda: MODE1_PITCH_CENTERS_DEG.copy())
    azimuth_deg: np.ndarray = field(default_factory=lambda: MODE1_AZIMUTH_CENTERS_DEG.copy())
    mass_amu: np.ndarray = field(default_factory=lambda: MODE1_MASS_AMU.copy())

    def __post_init__(self) -> None:
        for name in (
            "background_dpf", "robust_background_dpf", "paper_background_dpf",
            "count_equivalent_background_dpf", "dpf_quantum",
            "background_lambda_count_equivalent", "quantum_valid_mask",
            "quantum_reconstruction_valid_fraction", "uncertainty_dpf", "interval_mad_dpf", "sample_count",
            "nonzero_sample_count", "valid_channel_mask",
        ):
            if np.asarray(getattr(self, name)).shape != MODE1_SHAPE:
                raise ValueError(f"{name} must have shape {MODE1_SHAPE}")


@dataclass(frozen=True)
class BackgroundCorrectionResult:
    raw_dpf: np.ndarray
    corrected_dpf: np.ndarray
    removed_dpf: np.ndarray
    background_fraction: np.ndarray
    background_dominated_mask: np.ndarray
    model_valid: bool
    model_version: str


@dataclass(frozen=True)
class UvDetectionConfig:
    """Morphology and persistence thresholds for Mode-1 UV contamination."""

    low_energy_bin_count: int = 4
    heavy_mass_index: int = -1
    candidate_sigma: float = 3.0
    isolated_sigma: float = 6.0
    minimum_consecutive_records: int = 2
    maximum_adjacent_azimuth_sectors: int = 4
    minimum_energy_coverage: int = 3
    minimum_pitch_coverage: int = 2
    maximum_record_gap_s: float = 25.0
    solar_support_min_deg: float = 83.0
    solar_support_max_deg: float = 87.0


@dataclass(frozen=True)
class UvDetectionResult:
    """Per-record UV diagnostics; confirmed records are rejected as a whole."""

    candidate_mask: np.ndarray
    confirmed_mask: np.ndarray
    strong_isolated_mask: np.ndarray
    maximum_robust_sigma: np.ndarray
    affected_azimuth_sector_count: np.ndarray
    affected_azimuth_mask: np.ndarray
    energy_coverage_count: np.ndarray
    pitch_coverage_count: np.ndarray
    solar_angle_support_mask: np.ndarray
    algorithm_version: str = ALGORITHM_VERSION


def estimate_background(
    records: MinpaMode1Records,
    approved_intervals: Sequence[BackgroundInterval],
    config: BackgroundConfig | None = None,
) -> MinpaBackgroundModel:
    """Estimate robust and paper-style channel-resolved DPF backgrounds.

    Project quality bits 3--4 reject an interval by default.  Bits 1--2 and 5
    do not affect selection.  Native nonzero quality records are omitted, without
    interpreting the undocumented native bit definitions.
    """

    cfg = config or BackgroundConfig()
    if cfg.mode != 1 or records.mode != 1:
        raise ValueError("The first background model is restricted to MINPA Mode 1")
    intervals = [interval for interval in approved_intervals if interval.approved]
    interval_zero_means: list[np.ndarray] = []
    interval_nonzero_means: list[np.ndarray] = []
    interval_summaries: list[dict[str, object]] = []
    total_sample_count = np.zeros(MODE1_SHAPE, dtype=np.int64)
    total_nonzero_count = np.zeros(MODE1_SHAPE, dtype=np.int64)
    accepted_records = 0
    accepted_record_mask = np.zeros(records.time_unix_s.size, dtype=bool)

    for interval in intervals:
        in_time = (records.time_unix_s >= interval.start_s) & (records.time_unix_s < interval.stop_s)
        n_raw = int(np.count_nonzero(in_time))
        project_available = records.project_quality_available[in_time]
        project_flag = records.project_quality_flag[in_time]
        project_bad = project_available & ((project_flag & np.uint32(cfg.reject_quality_mask)) != 0)
        reasons: list[str] = []
        if cfg.require_project_quality and (n_raw == 0 or not np.all(project_available)):
            reasons.append("project_quality_unavailable")
        if cfg.reject_interval_if_project_flagged and np.any(project_bad):
            reasons.append("project_quality_bits_3_4_present")

        keep = in_time.copy()
        if cfg.require_project_quality:
            keep &= records.project_quality_available
        keep &= (records.project_quality_flag & np.uint32(cfg.reject_quality_mask)) == 0
        if cfg.require_native_quality_zero:
            keep &= records.native_quality == 0
        cube = np.asarray(records.dpf[keep], dtype=float)
        finite = np.isfinite(cube)
        n_kept = int(cube.shape[0])
        if n_kept < cfg.min_records_per_interval:
            reasons.append("too_few_records")
        accepted = not reasons
        summary = {
            **asdict(interval),
            "start_unix_s": interval.start_s,
            "stop_unix_s": interval.stop_s,
            "raw_records": n_raw,
            "kept_records": n_kept,
            "project_quality_bit_record_counts": {
                str(bit): int(np.count_nonzero(project_available & ((project_flag & np.uint32(1 << (bit - 1))) != 0)))
                for bit in range(1, 6)
            },
            "quality_bits_3_4_records": int(np.count_nonzero(project_bad)),
            "native_quality_nonzero_records": int(np.count_nonzero(records.native_quality[in_time] != 0)),
            "accepted": accepted,
            "reasons": reasons,
        }
        interval_summaries.append(summary)
        if not accepted:
            continue

        values_with_zero = np.where(finite, cube, np.nan)
        with np.errstate(invalid="ignore"):
            zero_mean = np.nanmean(values_with_zero, axis=0)
        positive = finite & (cube > 0.0)
        positive_sum = np.where(positive, cube, 0.0).sum(axis=0)
        positive_count = positive.sum(axis=0)
        nonzero_mean = np.divide(
            positive_sum,
            positive_count,
            out=np.full(MODE1_SHAPE, np.nan, dtype=float),
            where=positive_count > 0,
        )
        interval_zero_means.append(zero_mean)
        interval_nonzero_means.append(nonzero_mean)
        total_sample_count += finite.sum(axis=0)
        total_nonzero_count += positive_count
        accepted_records += n_kept
        accepted_record_mask |= keep

    if interval_zero_means:
        zero_stack = np.stack(interval_zero_means)
        nonzero_stack = np.stack(interval_nonzero_means)
        robust = np.nanmedian(zero_stack, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            paper = np.nanmean(nonzero_stack, axis=0)
        interval_center = np.nanmedian(zero_stack, axis=0)
        mad = np.nanmedian(np.abs(zero_stack - interval_center), axis=0)
        uncertainty = cfg.uncertainty_mad_scale * mad / math.sqrt(len(interval_zero_means))
    else:
        robust = np.full(MODE1_SHAPE, np.nan)
        paper = np.full(MODE1_SHAPE, np.nan)
        mad = np.full(MODE1_SHAPE, np.nan)
        uncertainty = np.full(MODE1_SHAPE, np.nan)

    (
        count_background,
        dpf_quantum,
        background_lambda,
        quantum_valid,
        quantum_valid_fraction,
        count_uncertainty,
    ) = _estimate_count_equivalent_background(
        np.asarray(records.dpf[accepted_record_mask], dtype=np.float32), cfg
    )

    if cfg.primary_estimator == "robust_including_zero_median":
        background = robust
    elif cfg.primary_estimator == "paper_nonzero_mean":
        background = paper
    else:
        background = count_background
        uncertainty = count_uncertainty
    valid_channel = (
        np.isfinite(background)
        & (background >= 0.0)
        & (total_sample_count >= cfg.min_channel_samples)
    )
    invalid_reasons: list[str] = []
    if len(interval_zero_means) < cfg.min_approved_intervals:
        invalid_reasons.append("too_few_accepted_intervals")
    if accepted_records < cfg.min_total_records:
        invalid_reasons.append("too_few_accepted_records")
    if not np.all(valid_channel):
        invalid_reasons.append("one_or_more_channels_insufficient")
    valid = not invalid_reasons

    return MinpaBackgroundModel(
        background_dpf=background,
        robust_background_dpf=robust,
        paper_background_dpf=paper,
        count_equivalent_background_dpf=count_background,
        dpf_quantum=dpf_quantum,
        background_lambda_count_equivalent=background_lambda,
        quantum_valid_mask=quantum_valid,
        quantum_reconstruction_valid_fraction=quantum_valid_fraction,
        uncertainty_dpf=uncertainty,
        interval_mad_dpf=mad,
        sample_count=total_sample_count,
        nonzero_sample_count=total_nonzero_count,
        valid_channel_mask=valid_channel,
        interval_summaries=interval_summaries,
        approved_intervals=[asdict(interval) for interval in intervals],
        source_files=list(records.source_files),
        source_sha256=dict(records.source_sha256),
        primary_estimator=cfg.primary_estimator,
        valid=valid,
        invalid_reasons=invalid_reasons,
    )


def apply_background(
    dpf_cube: np.ndarray,
    model: MinpaBackgroundModel,
    *,
    require_valid_model: bool = True,
) -> BackgroundCorrectionResult:
    """Subtract a matching channel background and clamp finite values at zero."""

    raw = np.asarray(dpf_cube, dtype=float)
    if model.mode != 1:
        raise ValueError(f"Mode-1 model required, got mode={model.mode}")
    if raw.shape[-4:] != MODE1_SHAPE:
        raise ValueError(f"DPF trailing shape must be {MODE1_SHAPE}; got {raw.shape}")
    if require_valid_model and not model.valid:
        raise ValueError(f"Refusing invalid background model: {model.invalid_reasons}")
    background = np.broadcast_to(model.background_dpf, raw.shape)
    finite = np.isfinite(raw) & np.isfinite(background)
    corrected = np.full(raw.shape, np.nan, dtype=float)
    corrected[finite] = np.maximum(raw[finite] - background[finite], 0.0)
    removed = np.full(raw.shape, np.nan, dtype=float)
    removed[finite] = raw[finite] - corrected[finite]
    fraction = np.full(raw.shape, np.nan, dtype=float)
    positive_raw = finite & (raw > 0.0)
    fraction[positive_raw] = removed[positive_raw] / raw[positive_raw]
    dominated = finite & (raw <= background + np.broadcast_to(model.uncertainty_dpf, raw.shape))
    return BackgroundCorrectionResult(
        raw_dpf=raw,
        corrected_dpf=corrected,
        removed_dpf=removed,
        background_fraction=fraction,
        background_dominated_mask=dominated,
        model_valid=model.valid,
        model_version=model.algorithm_version,
    )


def aggregate_energy_count_equivalent(
    dpf_cube: np.ndarray,
    dpf_quantum: np.ndarray,
    *,
    residual_tolerance_counts: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer per-record energy counts by summing all pitch/azimuth/mass cells.

    ``dpf_cube`` must have trailing dimensions ``(energy, pitch, azimuth,
    mass)`` and ``dpf_quantum`` must match those trailing dimensions.  The
    result has the leading dimensions plus ``energy``.  Released DPF values
    are accepted only when they reconstruct an integer multiple of the
    channel-specific empirical quantum within the requested tolerance.

    The result is explicitly a count equivalent inferred from released DPF;
    it is not an undocumented detector count product.
    """

    values = np.asarray(dpf_cube, dtype=float)
    quantum = np.asarray(dpf_quantum, dtype=float)
    if values.ndim < 4:
        raise ValueError("dpf_cube must end with (energy, pitch, azimuth, mass)")
    if values.shape[-4:] != quantum.shape:
        raise ValueError(
            "dpf_quantum must match the four trailing DPF dimensions; "
            f"got {quantum.shape} for {values.shape[-4:]}"
        )
    if not np.isfinite(residual_tolerance_counts) or residual_tolerance_counts < 0.0:
        raise ValueError("residual_tolerance_counts must be finite and non-negative")

    expanded_quantum = quantum.reshape((1,) * (values.ndim - 4) + quantum.shape)
    finite = (
        np.isfinite(values)
        & (values >= 0.0)
        & np.isfinite(expanded_quantum)
        & (expanded_quantum > 0.0)
    )
    ratio = np.divide(
        values,
        expanded_quantum,
        out=np.full(values.shape, np.nan, dtype=float),
        where=finite,
    )
    nearest = np.rint(ratio)
    reconstructed = finite & (np.abs(ratio - nearest) <= residual_tolerance_counts)
    spatial_axes = tuple(range(values.ndim - 3, values.ndim))
    counts = np.where(reconstructed, nearest, 0.0).sum(axis=spatial_axes)
    valid_fraction = reconstructed.mean(axis=spatial_axes)
    return counts, valid_fraction


def detect_uv_contamination(
    dpf_cube: np.ndarray,
    time_unix_s: np.ndarray,
    model: MinpaBackgroundModel,
    config: UvDetectionConfig | None = None,
    *,
    instrument_solar_angles_deg: np.ndarray | None = None,
) -> UvDetectionResult:
    """Identify directional, broadband and persistent Mode-1 UV contamination.

    The indicator is the directional sum over the lowest energy bins, all
    pitch bins, and the heaviest mass channel.  A record must exceed a robust
    per-sector baseline, occupy only a few circularly adjacent azimuth sectors,
    and span multiple energy and pitch bins.  Two consecutive candidates are
    confirmed, while an isolated record requires the higher ``isolated_sigma``
    threshold.  Solar incidence near 85 degrees is returned only as supporting
    evidence and never acts as a hard condition.
    """

    cfg = config or UvDetectionConfig()
    cube = np.asarray(dpf_cube, dtype=float)
    time = np.asarray(time_unix_s, dtype=float).reshape(-1)
    if cube.shape != (time.size, *MODE1_SHAPE):
        raise ValueError(
            f"dpf_cube must have shape (time, {', '.join(map(str, MODE1_SHAPE))}); "
            f"got {cube.shape}"
        )
    if model.mode != 1 or not model.valid:
        raise ValueError("UV detection requires a valid MINPA Mode-1 background model")
    if not 1 <= cfg.low_energy_bin_count <= MODE1_SHAPE[0]:
        raise ValueError("low_energy_bin_count is outside the Mode-1 energy dimension")
    mass_index = cfg.heavy_mass_index % MODE1_SHAPE[3]
    low = np.where(
        np.isfinite(cube[:, : cfg.low_energy_bin_count, :, :, mass_index]),
        np.maximum(cube[:, : cfg.low_energy_bin_count, :, :, mass_index], 0.0),
        0.0,
    )
    indicator = low.sum(axis=(1, 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = np.nanmedian(indicator, axis=0)
        empirical_scale = 1.4826 * np.nanmedian(
            np.abs(indicator - center[None, :]), axis=0
        )
    background = np.asarray(
        model.background_dpf[: cfg.low_energy_bin_count, :, :, mass_index], dtype=float
    )
    background_center = np.nansum(np.maximum(background, 0.0), axis=(0, 1))
    channel_quantum = np.asarray(
        model.dpf_quantum[: cfg.low_energy_bin_count, :, :, mass_index], dtype=float
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        quantum_floor = np.nanmedian(
            np.where(
                np.isfinite(channel_quantum) & (channel_quantum > 0.0),
                channel_quantum,
                np.nan,
            ),
            axis=(0, 1),
        )
    center = np.maximum(center, background_center)
    scale = np.maximum(empirical_scale, quantum_floor)
    finite_positive_scale = scale[np.isfinite(scale) & (scale > 0.0)]
    global_floor = float(np.nanmedian(finite_positive_scale)) if finite_positive_scale.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, global_floor)
    robust_sigma = np.maximum(indicator - center[None, :], 0.0) / scale[None, :]
    affected = robust_sigma >= cfg.candidate_sigma

    candidate = np.zeros(time.size, dtype=bool)
    sector_count = affected.sum(axis=1).astype(np.int16)
    energy_coverage = np.zeros(time.size, dtype=np.int16)
    pitch_coverage = np.zeros(time.size, dtype=np.int16)
    background_low = np.where(np.isfinite(background), background, 0.0)
    excess = np.maximum(low - background_low[None, ...], 0.0)
    for record_index in range(time.size):
        sectors = np.flatnonzero(affected[record_index])
        if not (
            1 <= sectors.size <= cfg.maximum_adjacent_azimuth_sectors
            and _circularly_adjacent(sectors, MODE1_SHAPE[2])
        ):
            continue
        selected = np.take(excess[record_index], sectors, axis=2)
        energy_coverage[record_index] = int(
            np.count_nonzero(np.nansum(selected, axis=(1, 2)) > 0.0)
        )
        pitch_coverage[record_index] = int(
            np.count_nonzero(np.nansum(selected, axis=(0, 2)) > 0.0)
        )
        candidate[record_index] = (
            energy_coverage[record_index] >= cfg.minimum_energy_coverage
            and pitch_coverage[record_index] >= cfg.minimum_pitch_coverage
        )

    maximum_sigma = np.nanmax(robust_sigma, axis=1) if time.size else np.empty(0)
    strong_isolated = candidate & (maximum_sigma >= cfg.isolated_sigma)
    confirmed = strong_isolated.copy()
    start = 0
    while start < time.size:
        if not candidate[start]:
            start += 1
            continue
        stop = start + 1
        while (
            stop < time.size
            and candidate[stop]
            and np.isfinite(time[stop] - time[stop - 1])
            and 0.0 < time[stop] - time[stop - 1] <= cfg.maximum_record_gap_s
        ):
            stop += 1
        if stop - start >= cfg.minimum_consecutive_records:
            confirmed[start:stop] = True
        start = stop

    solar_support = np.zeros(time.size, dtype=bool)
    if instrument_solar_angles_deg is not None:
        solar = np.asarray(instrument_solar_angles_deg, dtype=float)
        if solar.shape != (time.size, 2):
            raise ValueError("instrument_solar_angles_deg must have shape (time, 2)")
        solar_support = np.any(
            np.isfinite(solar)
            & (solar >= cfg.solar_support_min_deg)
            & (solar <= cfg.solar_support_max_deg),
            axis=1,
        )
    return UvDetectionResult(
        candidate_mask=candidate,
        confirmed_mask=confirmed,
        strong_isolated_mask=strong_isolated,
        maximum_robust_sigma=maximum_sigma,
        affected_azimuth_sector_count=sector_count,
        affected_azimuth_mask=affected,
        energy_coverage_count=energy_coverage,
        pitch_coverage_count=pitch_coverage,
        solar_angle_support_mask=solar_support,
    )


def _circularly_adjacent(indices: np.ndarray, sector_count: int) -> bool:
    """Return true when circular sector indices form one contiguous group."""

    selected = np.sort(np.unique(np.asarray(indices, dtype=int)))
    if selected.size <= 1:
        return True
    gaps = np.diff(np.concatenate([selected, selected[:1] + sector_count]))
    return int(np.count_nonzero(gaps > 1)) <= 1


def _estimate_count_equivalent_background(
    cube: np.ndarray, config: BackgroundConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infer DPF quanta and an energy-pooled count-equivalent background.

    Released Mode-1 DPF values are strongly quantized.  This routine uses the
    smallest positive background value as an empirical channel quantum, checks
    whether other positive values are near integer multiples, fills sparsely
    sampled quanta using the approximately energy-invariant ``quantum * E``,
    and estimates a Poisson-like mean count equivalent pooled over energy.
    It does not claim access to undocumented raw detector counts.
    """

    empty_float = np.full(MODE1_SHAPE, np.nan, dtype=float)
    empty_bool = np.zeros(MODE1_SHAPE, dtype=bool)
    if cube.shape[0] == 0:
        return (
            empty_float.copy(), empty_float.copy(), empty_float.copy(),
            empty_bool, empty_float.copy(), empty_float.copy(),
        )
    values = np.asarray(cube, dtype=np.float32)
    finite = np.isfinite(values) & (values >= 0.0)
    positive = finite & (values > 0.0)
    positive_count = positive.sum(axis=0)
    quantum_raw = np.min(
        np.where(positive, values, np.float32(np.inf)), axis=0
    ).astype(float)
    quantum_raw[~np.isfinite(quantum_raw)] = np.nan

    energy_shape = MODE1_ENERGY_EV[:, None, None, None]
    prelim_eligible = (
        np.isfinite(quantum_raw)
        & (quantum_raw > 0.0)
        & (positive_count >= config.quantum_min_nonzero_samples)
    )
    quantum_def = np.where(prelim_eligible, quantum_raw * energy_shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_def = np.nanmedian(quantum_def, axis=0)
    # Fill any pitch/azimuth/mass combination absent at all energies from the
    # corresponding mass median, then from the global median.
    for mass_index in range(MODE1_MASS_AMU.size):
        mass_base = base_def[:, :, mass_index]
        fallback = float(np.nanmedian(mass_base)) if np.any(np.isfinite(mass_base)) else math.nan
        if np.isfinite(fallback):
            missing = ~np.isfinite(mass_base)
            mass_base[missing] = fallback
            base_def[:, :, mass_index] = mass_base
    global_fallback = float(np.nanmedian(base_def)) if np.any(np.isfinite(base_def)) else math.nan
    if np.isfinite(global_fallback):
        base_def[~np.isfinite(base_def)] = global_fallback
    quantum_filled = quantum_raw.copy()
    inferred = base_def[None, :, :, :] / energy_shape
    quantum_filled[~prelim_eligible] = inferred[~prelim_eligible]

    ratio = np.divide(
        values,
        quantum_filled[None, ...],
        out=np.full(values.shape, np.nan, dtype=np.float32),
        where=finite & np.isfinite(quantum_filled[None, ...]) & (quantum_filled[None, ...] > 0.0),
    )
    count_equivalent = np.rint(ratio)
    residual = np.abs(ratio - count_equivalent)
    reconstruction_good = positive & (residual <= config.quantum_residual_tolerance_counts)
    valid_fraction = np.divide(
        reconstruction_good.sum(axis=0),
        positive_count,
        out=np.full(MODE1_SHAPE, np.nan, dtype=float),
        where=positive_count > 0,
    )
    quantum_valid = (
        np.isfinite(quantum_filled)
        & (quantum_filled > 0.0)
        & (positive_count >= config.quantum_min_nonzero_samples)
        & (valid_fraction >= config.quantum_min_valid_fraction)
    )
    count_equivalent[~finite] = np.nan
    # Do not use poorly reconstructed positive samples to estimate lambda.
    count_equivalent[positive & ~reconstruction_good] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lambda_by_energy = np.nanmean(count_equivalent, axis=0)
        # Each energy channel has the same dwell time.  Pool the channel means
        # robustly over energy so a narrow real-ion peak does not define the
        # electronic background.  Zero estimates remain valid upper-limit
        # cases rather than being treated as missing.
        lambda_pam = np.nanmedian(lambda_by_energy, axis=0)
    background_lambda = np.broadcast_to(lambda_pam[None, ...], MODE1_SHAPE).copy()
    background_dpf = quantum_filled * background_lambda
    energy_valid_count = np.sum(quantum_valid, axis=0)
    effective_samples = values.shape[0] * energy_valid_count
    lambda_se = np.sqrt(
        np.divide(
            lambda_pam,
            effective_samples,
            out=np.full(lambda_pam.shape, np.nan),
            where=effective_samples > 0,
        )
    )
    uncertainty_dpf = quantum_filled * lambda_se[None, ...]
    return (
        background_dpf,
        quantum_filled,
        background_lambda,
        quantum_valid,
        valid_fraction,
        uncertainty_dpf,
    )


def reshape_mode1_dpf(flat: np.ndarray) -> np.ndarray:
    """Expand mass-fastest ori rows to energy/pitch/azimuth/mass order."""

    values = np.asarray(flat)
    if values.shape[-1] != MODE1_FLAT_SIZE:
        raise ValueError(f"Mode-1 row has {values.shape[-1]} values, expected {MODE1_FLAT_SIZE}")
    return values.reshape((*values.shape[:-1], *MODE1_SHAPE))


def read_mode1_ori_records(
    paths: Sequence[Path],
    start_utc: str,
    stop_utc: str,
    *,
    quality_root: Path | None = None,
    compute_sha256: bool = True,
) -> MinpaMode1Records:
    """Read Mode-1 ori data without modifying the source files."""

    start_s, stop_s = _parse_utc(start_utc), _parse_utc(stop_utc)
    if not start_s < stop_s:
        raise ValueError("start_utc must precede stop_utc")
    rows: list[dict[str, object]] = []
    source_files = [str(Path(path).resolve()) for path in paths]
    source_sha256: dict[str, str] = {}
    for file_index, path_value in enumerate(paths):
        path = Path(path_value)
        if "MINPA-MOD1-" not in path.name:
            raise ValueError(f"Non-Mode-1 file supplied: {path}")
        if compute_sha256:
            source_sha256[str(path.resolve())] = sha256_file(path)
        with h5py.File(path, "r") as handle:
            group = handle["tw1_MINPA_data"]
            field_values = {
                index: _matlab_t_values(handle, handle[group["value"][index, 0]]["t"])
                for index in (0, 21, 22, 23, 24, 25, 31, 32, 36, 37, 38, 39, 40, 41)
            }
            record_count = min(len(values) for values in field_values.values())
            for record_index in range(record_count):
                utc_text = _matlab_char(handle, field_values[0][record_index])
                try:
                    time_s = _parse_utc(utc_text)
                except ValueError:
                    continue
                if not start_s <= time_s < stop_s:
                    continue
                flat = np.asarray(field_values[41][record_index], dtype=np.float32).reshape(-1)
                if flat.size != MODE1_FLAT_SIZE:
                    continue
                hv = np.stack(
                    [_fixed_vector(field_values[index][record_index], 4) for index in range(21, 26)]
                )
                ion_totals = np.array(
                    [_scalar(field_values[index][record_index]) for index in range(37, 41)], dtype=float
                )
                rows.append(
                    {
                        "time": time_s,
                        "dpf": flat,
                        "native_quality": _parse_native_quality(
                            _matlab_char(handle, field_values[36][record_index])
                        ),
                        "ion_totals": ion_totals,
                        "hv": hv,
                        "solar": np.array(
                            [_scalar(field_values[31][record_index]), _scalar(field_values[32][record_index])],
                            dtype=float,
                        ),
                        "source_file_index": file_index,
                    }
                )

    if rows:
        rows.sort(key=lambda row: float(row["time"]))
        unique_rows: list[dict[str, object]] = []
        previous = -math.inf
        for row in rows:
            time_s = float(row["time"])
            if time_s == previous:
                continue
            unique_rows.append(row)
            previous = time_s
        time = np.array([row["time"] for row in unique_rows], dtype=float)
        dpf = reshape_mode1_dpf(np.stack([row["dpf"] for row in unique_rows])).astype(np.float32)
        native = np.array([row["native_quality"] for row in unique_rows], dtype=np.uint32)
        totals = np.stack([row["ion_totals"] for row in unique_rows]).astype(float)
        hv = np.stack([row["hv"] for row in unique_rows]).astype(float)
        solar = np.stack([row["solar"] for row in unique_rows]).astype(float)
        source_index = np.array([row["source_file_index"] for row in unique_rows], dtype=np.int32)
    else:
        time = np.empty(0, dtype=float)
        dpf = np.empty((0, *MODE1_SHAPE), dtype=np.float32)
        native = np.empty(0, dtype=np.uint32)
        totals = np.empty((0, 4), dtype=float)
        hv = np.empty((0, 5, 4), dtype=float)
        solar = np.empty((0, 2), dtype=float)
        source_index = np.empty(0, dtype=np.int32)

    if quality_root is None:
        project_flag = np.zeros(time.size, dtype=np.uint32)
        project_available = np.zeros(time.size, dtype=bool)
    else:
        project_flag, project_available = load_project_quality_for_times(time, quality_root)
    return MinpaMode1Records(
        time_unix_s=time,
        dpf=dpf,
        native_quality=native,
        ion_start_stop_counts=totals,
        high_voltage_monitor_v=hv,
        instrument_solar_angles_deg=solar,
        project_quality_flag=project_flag,
        project_quality_available=project_available,
        source_file_index=source_index,
        source_files=source_files,
        source_sha256=source_sha256,
    )


def concatenate_mode1_records(parts: Sequence[MinpaMode1Records]) -> MinpaMode1Records:
    """Concatenate interval reads while preserving a unique source-file table."""

    if not parts:
        return MinpaMode1Records(
            time_unix_s=np.empty(0),
            dpf=np.empty((0, *MODE1_SHAPE), dtype=np.float32),
            native_quality=np.empty(0, dtype=np.uint32),
            ion_start_stop_counts=np.empty((0, 4)),
            high_voltage_monitor_v=np.empty((0, 5, 4)),
            instrument_solar_angles_deg=np.empty((0, 2)),
            project_quality_flag=np.empty(0, dtype=np.uint32),
            project_quality_available=np.empty(0, dtype=bool),
            source_file_index=np.empty(0, dtype=np.int32),
        )
    files: list[str] = []
    file_lookup: dict[str, int] = {}
    source_indices: list[np.ndarray] = []
    hashes: dict[str, str] = {}
    for part in parts:
        remap = np.empty(len(part.source_files), dtype=np.int32)
        for old_index, file_name in enumerate(part.source_files):
            if file_name not in file_lookup:
                file_lookup[file_name] = len(files)
                files.append(file_name)
            remap[old_index] = file_lookup[file_name]
        source_indices.append(remap[part.source_file_index] if part.source_file_index.size else part.source_file_index)
        hashes.update(part.source_sha256)
    time = np.concatenate([part.time_unix_s for part in parts])
    order = np.argsort(time, kind="stable")
    # Intervals are expected not to overlap, but exact duplicate epochs are
    # removed defensively so one measurement cannot be counted twice.
    sorted_time = time[order]
    unique = np.ones(sorted_time.size, dtype=bool)
    unique[1:] = np.diff(sorted_time) != 0.0
    keep = order[unique]
    return MinpaMode1Records(
        time_unix_s=time[keep],
        dpf=np.concatenate([part.dpf for part in parts], axis=0)[keep],
        native_quality=np.concatenate([part.native_quality for part in parts])[keep],
        ion_start_stop_counts=np.concatenate([part.ion_start_stop_counts for part in parts], axis=0)[keep],
        high_voltage_monitor_v=np.concatenate([part.high_voltage_monitor_v for part in parts], axis=0)[keep],
        instrument_solar_angles_deg=np.concatenate(
            [part.instrument_solar_angles_deg for part in parts], axis=0
        )[keep],
        project_quality_flag=np.concatenate([part.project_quality_flag for part in parts])[keep],
        project_quality_available=np.concatenate([part.project_quality_available for part in parts])[keep],
        source_file_index=np.concatenate(source_indices)[keep],
        source_files=files,
        source_sha256=hashes,
    )


def mode1_ori_files_for_interval(ori_root: Path, start_utc: str, stop_utc: str) -> list[Path]:
    """Return Mode-1 ori files whose filename coverage overlaps an interval."""

    start_s, stop_s = _parse_utc(start_utc), _parse_utc(stop_utc)
    pattern = re.compile(r"MINPA-MOD1-.*?_(\d{8})(\d{6})_(\d{8})(\d{6})_")
    files: list[Path] = []
    for path in Path(ori_root).glob("*MINPA-MOD1-*.mat"):
        match = pattern.search(path.name)
        if not match:
            continue
        file_start = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()
        file_stop = datetime.strptime(match.group(3) + match.group(4), "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()
        if file_start < stop_s and file_stop >= start_s:
            files.append(path)
    return sorted(files)


def read_mode1_selected_dpf_channels(
    paths: Sequence[Path],
    start_utc: str,
    stop_utc: str,
    channel_indices: Sequence[tuple[int, int, int, int]],
    *,
    quality_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream selected ``(energy,pitch,azimuth,mass)`` DPF cells from ori."""

    start_s, stop_s = _parse_utc(start_utc), _parse_utc(stop_utc)
    flat_indices = np.array(
        [np.ravel_multi_index(index, MODE1_SHAPE) for index in channel_indices], dtype=int
    )
    rows: list[tuple[float, np.ndarray]] = []
    for path_value in paths:
        path = Path(path_value)
        if "MINPA-MOD1-" not in path.name:
            raise ValueError(f"Non-Mode-1 file supplied: {path}")
        with h5py.File(path, "r") as handle:
            group = handle["tw1_MINPA_data"]
            utc_values = _matlab_t_values(handle, handle[group["value"][0, 0]]["t"])
            ion_values = _matlab_t_values(handle, handle[group["value"][41, 0]]["t"])
            for utc_value, ion_value in zip(utc_values, ion_values):
                try:
                    time_s = _parse_utc(_matlab_char(handle, utc_value))
                except ValueError:
                    continue
                if not start_s <= time_s < stop_s:
                    continue
                flat = np.asarray(ion_value, dtype=float).reshape(-1)
                if flat.size == MODE1_FLAT_SIZE:
                    rows.append((time_s, flat[flat_indices]))
    rows.sort(key=lambda row: row[0])
    if rows:
        time = np.array([row[0] for row in rows], dtype=float)
        values = np.stack([row[1] for row in rows])
        unique = np.ones(time.size, dtype=bool)
        unique[1:] = np.diff(time) != 0.0
        time, values = time[unique], values[unique]
    else:
        time = np.empty(0)
        values = np.empty((0, len(channel_indices)))
    if quality_root is None:
        flags = np.zeros(time.size, dtype=np.uint32)
        available = np.zeros(time.size, dtype=bool)
    else:
        flags, available = load_project_quality_for_times(time, quality_root)
    return time, values, flags, available


def discover_background_candidates(
    day_spe_paths: Sequence[Path],
    config: CandidateConfig | None = None,
) -> list[dict[str, object]]:
    """Find low-signal, Mode-1 windows for later human approval.

    The absolute signal proxy is the median positive H+ one-dimensional DEF
    summed over energy.  It is used only to rank candidates; the selected
    intervals remain explicitly provisional until reviewed.
    """

    cfg = config or CandidateConfig()
    candidates: list[dict[str, object]] = []
    for path in sorted(Path(value) for value in day_spe_paths):
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        containers = {name: data[f"{name}_spe_num"] for name in ("H", "O", "O2")}
        h_fields = list(getattr(containers["H"], "_fieldnames", []) or [])
        for segment_name in h_fields:
            species_items = []
            for name in ("H", "O", "O2"):
                container = containers[name]
                if segment_name not in (getattr(container, "_fieldnames", []) or []):
                    species_items = []
                    break
                species_items.append(getattr(container, segment_name))
            if len(species_items) != 3:
                continue
            segment_modes = []
            for item in species_items:
                mode_values = np.asarray(getattr(item, "mod", np.nan)).reshape(-1)
                segment_modes.append(
                    int(mode_values[0])
                    if mode_values.size == 1 and np.isfinite(mode_values[0])
                    else -1
                )
            if segment_modes != [1, 1, 1]:
                continue

            h_item = species_items[0]
            time = np.asarray(h_item.t, dtype=float).reshape(-1)
            spectrum = np.asarray(h_item.p, dtype=float)
            energy = np.asarray(h_item.f, dtype=float).reshape(-1)
            if spectrum.ndim != 2 or spectrum.shape != (time.size, energy.size):
                continue
            if energy.size != MODE1_ENERGY_EV.size or not np.allclose(
                energy, MODE1_ENERGY_EV, rtol=1.0e-5, atol=1.0e-5
            ):
                continue
            combined_flag = np.zeros(time.size, dtype=np.uint32)
            combined_available = np.ones(time.size, dtype=bool)
            for item in species_items:
                item_time = np.asarray(item.t, dtype=float).reshape(-1)
                if item_time.shape != time.shape or not np.allclose(
                    item_time, time, atol=1.0e-3, rtol=0.0
                ):
                    combined_available[:] = False
                    break
                combined_flag |= np.asarray(item.quality_flag, dtype=np.uint32).reshape(-1)
                combined_available &= np.asarray(
                    item.quality_flag_available, dtype=bool
                ).reshape(-1)
            finite_positive = np.isfinite(spectrum) & (spectrum > 0.0)
            total_def = np.where(finite_positive, spectrum, 0.0).sum(axis=1)
            dpf = np.divide(
                spectrum,
                energy[None, :],
                out=np.full_like(spectrum, np.nan, dtype=float),
                where=np.isfinite(spectrum) & (energy[None, :] > 0.0),
            )
            if time.size == 0:
                continue
            start = math.ceil(time[0] / cfg.step_s) * cfg.step_s
            last_start = time[-1] - cfg.duration_s
            while start <= last_start:
                stop = start + cfg.duration_s
                mask = (time >= start) & (time < stop)
                indices = np.flatnonzero(mask)
                if indices.size >= cfg.min_records:
                    gaps = np.diff(time[indices])
                    quality_bad = (
                        (not np.all(combined_available[indices]))
                        or np.any(
                            (combined_flag[indices] & np.uint32(cfg.reject_quality_mask))
                            != 0
                        )
                    )
                    if (
                        gaps.size == 0
                        or float(np.nanmax(gaps)) <= cfg.max_record_gap_s
                    ) and not quality_bad:
                        positive_fraction = float(
                            np.mean(np.isfinite(dpf[indices]) & (dpf[indices] > 0.0))
                        )
                        candidates.append(
                            {
                                "start_unix_s": float(start),
                                "stop_unix_s": float(stop),
                                "start_utc": _format_utc(start),
                                "stop_utc": _format_utc(stop),
                                "day": datetime.fromtimestamp(start, UTC).strftime("%Y%m%d"),
                                "records": int(indices.size),
                                "median_total_h_def": float(
                                    np.nanmedian(total_def[indices])
                                ),
                                "mean_positive_channel_fraction": positive_fraction,
                                "quality_reject_mask": int(cfg.reject_quality_mask),
                                "approved": False,
                                "source": "automatic_low_signal_screen",
                                "source_day_spe": str(path.resolve()),
                                "source_day_spe_segment": segment_name,
                                "source_mode": 1,
                            }
                        )
                start += cfg.step_s
    if not candidates:
        return []
    scores = np.array([float(candidate["median_total_h_def"]) for candidate in candidates])
    cutoff = float(np.nanquantile(scores, cfg.low_signal_quantile))
    ranked = sorted(
        (candidate for candidate in candidates if float(candidate["median_total_h_def"]) <= cutoff),
        key=lambda candidate: (float(candidate["median_total_h_def"]), str(candidate["start_utc"])),
    )
    selected: list[dict[str, object]] = []
    day_count: dict[str, int] = {}
    for candidate in ranked:
        day = str(candidate["day"])
        if day_count.get(day, 0) >= cfg.max_intervals_per_day:
            continue
        midpoint = (float(candidate["start_unix_s"]) + float(candidate["stop_unix_s"])) / 2.0
        if any(
            abs(midpoint - (float(item["start_unix_s"]) + float(item["stop_unix_s"])) / 2.0)
            < cfg.min_separation_s
            for item in selected
        ):
            continue
        candidate = dict(candidate)
        candidate["selection_score_cutoff"] = cutoff
        selected.append(candidate)
        day_count[day] = day_count.get(day, 0) + 1
        if len(selected) >= cfg.requested_intervals:
            break
    return sorted(selected, key=lambda candidate: float(candidate["start_unix_s"]))


def load_project_quality_for_times(
    time_unix_s: np.ndarray,
    quality_root: Path,
    *,
    tolerance_s: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the OR-combined H/O/O2 project flags nearest to each epoch."""

    target = np.asarray(time_unix_s, dtype=float)
    flags = np.zeros(target.size, dtype=np.uint32)
    available = np.zeros(target.size, dtype=bool)
    if target.size == 0:
        return flags, available
    days = sorted({datetime.fromtimestamp(float(value), UTC).strftime("%Y%m%d") for value in target})
    source_time: list[np.ndarray] = []
    source_flag: list[np.ndarray] = []
    source_available: list[np.ndarray] = []
    for day in days:
        path = Path(quality_root) / day / f"quality_flags_{day}.mat"
        if not path.exists():
            continue
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        record = data["recordFlags"]
        day_time = np.asarray(record.time_unix_s, dtype=float).reshape(-1)
        combined = np.zeros(day_time.size, dtype=np.uint32)
        day_available = np.ones(day_time.size, dtype=bool)
        for species in ("H", "O", "O2"):
            item = getattr(record, species)
            combined |= np.asarray(item.flag, dtype=np.uint32).reshape(-1)
            day_available &= np.asarray(item.available, dtype=bool).reshape(-1)
        source_time.append(day_time)
        source_flag.append(combined)
        source_available.append(day_available)
    if not source_time:
        return flags, available
    time = np.concatenate(source_time)
    order = np.argsort(time)
    time = time[order]
    combined_flag = np.concatenate(source_flag)[order]
    combined_available = np.concatenate(source_available)[order]
    index = np.searchsorted(time, target)
    for target_index, insertion in enumerate(index):
        candidates = [candidate for candidate in (insertion - 1, insertion) if 0 <= candidate < time.size]
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: abs(time[candidate] - target[target_index]))
        if abs(time[best] - target[target_index]) <= tolerance_s:
            flags[target_index] = combined_flag[best]
            available[target_index] = combined_available[best]
    return flags, available


def save_background_model(path: Path, model: MinpaBackgroundModel) -> None:
    """Save arrays and JSON metadata in one compressed, versioned NPZ file."""

    metadata = {
        "algorithm_version": model.algorithm_version,
        "mode": model.mode,
        "units": model.units,
        "primary_estimator": model.primary_estimator,
        "valid": model.valid,
        "invalid_reasons": model.invalid_reasons,
        "interval_summaries": model.interval_summaries,
        "approved_intervals": model.approved_intervals,
        "source_files": model.source_files,
        "source_sha256": model.source_sha256,
        "dimension_order": ["energy", "pitch", "azimuth", "mass"],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        background_dpf=model.background_dpf,
        robust_background_dpf=model.robust_background_dpf,
        paper_background_dpf=model.paper_background_dpf,
        count_equivalent_background_dpf=model.count_equivalent_background_dpf,
        dpf_quantum=model.dpf_quantum,
        background_lambda_count_equivalent=model.background_lambda_count_equivalent,
        quantum_valid_mask=model.quantum_valid_mask,
        quantum_reconstruction_valid_fraction=model.quantum_reconstruction_valid_fraction,
        uncertainty_dpf=model.uncertainty_dpf,
        interval_mad_dpf=model.interval_mad_dpf,
        sample_count=model.sample_count,
        nonzero_sample_count=model.nonzero_sample_count,
        valid_channel_mask=model.valid_channel_mask,
        energy_eV=model.energy_eV,
        pitch_deg=model.pitch_deg,
        azimuth_deg=model.azimuth_deg,
        mass_amu=model.mass_amu,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )


def load_background_model(path: Path) -> MinpaBackgroundModel:
    """Load a model saved by :func:`save_background_model`."""

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return MinpaBackgroundModel(
            background_dpf=data["background_dpf"],
            robust_background_dpf=data["robust_background_dpf"],
            paper_background_dpf=data["paper_background_dpf"],
            count_equivalent_background_dpf=data["count_equivalent_background_dpf"],
            dpf_quantum=data["dpf_quantum"],
            background_lambda_count_equivalent=data["background_lambda_count_equivalent"],
            quantum_valid_mask=data["quantum_valid_mask"],
            quantum_reconstruction_valid_fraction=data["quantum_reconstruction_valid_fraction"],
            uncertainty_dpf=data["uncertainty_dpf"],
            interval_mad_dpf=data["interval_mad_dpf"],
            sample_count=data["sample_count"],
            nonzero_sample_count=data["nonzero_sample_count"],
            valid_channel_mask=data["valid_channel_mask"],
            interval_summaries=metadata["interval_summaries"],
            approved_intervals=metadata["approved_intervals"],
            source_files=metadata["source_files"],
            source_sha256=metadata["source_sha256"],
            primary_estimator=metadata["primary_estimator"],
            valid=bool(metadata["valid"]),
            invalid_reasons=list(metadata["invalid_reasons"]),
            mode=int(metadata["mode"]),
            units=metadata["units"],
            algorithm_version=metadata["algorithm_version"],
            energy_eV=data["energy_eV"],
            pitch_deg=data["pitch_deg"],
            azimuth_deg=data["azimuth_deg"],
            mass_amu=data["mass_amu"],
        )


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str) -> float:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _format_utc(value: float) -> str:
    return datetime.fromtimestamp(float(value), UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _matlab_t_values(handle: h5py.File, dataset: h5py.Dataset) -> list[np.ndarray]:
    values = np.asarray(dataset)
    if values.dtype == object:
        return [np.asarray(handle[reference]) for reference in values.reshape(-1)]
    matlab_class = dataset.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, np.bytes_):
        matlab_class = bytes(matlab_class)
    if matlab_class == b"char" and values.ndim == 2 and values.shape[1] > 1:
        return [values[:, index] for index in range(values.shape[1])]
    if values.ndim == 2 and values.shape[1] > 1:
        return [values[:, index] for index in range(values.shape[1])]
    return [values]


def _matlab_char(handle: h5py.File, value: np.ndarray | h5py.Reference) -> str:
    values = np.asarray(handle[value] if isinstance(value, h5py.Reference) else value)
    if values.dtype == object:
        return _matlab_char(handle, values.reshape(-1)[0])
    if values.dtype.kind in ("u", "i"):
        return "".join(chr(int(item)) for item in values.reshape(-1) if int(item) != 0)
    return str(values.reshape(-1)[0]) if values.size else ""


def _scalar(value: np.ndarray) -> float:
    values = np.asarray(value, dtype=float).reshape(-1)
    return float(values[0]) if values.size else math.nan


def _fixed_vector(value: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    result = np.full(size, np.nan, dtype=float)
    result[: min(size, values.size)] = values[:size]
    return result


def _parse_native_quality(text: str) -> np.uint32:
    value = str(text).strip().lower()
    if not value:
        return np.uint32(0)
    try:
        return np.uint32(int(value, 16) if value.startswith("0x") else int(value))
    except ValueError:
        return np.uint32(np.iinfo(np.uint32).max)
