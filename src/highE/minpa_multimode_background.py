"""Mode-4/12 species-reviewed MINPA channel-background models.

The Wang et al. paper estimator is generalized here as a project method.  It
is not claimed that the paper validated Mode 4 or Mode 12.  Released science
values remain DPF in ``1/(s cm^2 sr eV)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from .minpa_background import (
    SUPPORT_LOW,
    SUPPORT_SUPPORTED,
    SUPPORT_UNSUPPORTED,
    _bootstrap_channel_mean_ci,
    sha256_file,
)
from .minpa_modes import mode_layout, species_mass_indices


MULTIMODE_REVIEW_VERSION = "minpa-multimode-noise-review-v2.0.0-rc1"
MULTIMODE_DENOISE_VERSION = "minpa-multimode-channel-denoise-v2.0.0"
MULTIMODE_MODEL_VERSION = "minpa-multimode-background-dpf-v2.0.0"
SUPPORTED_MODES = (4, 12)
SUPPORTED_SPECIES = ("H+", "O+", "O2+")


def mode_channel_shape(mode: int) -> tuple[int, int, int, int]:
    """Return ``energy, pitch, azimuth, mass`` for a reviewed mode."""

    layout = mode_layout(mode)
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Multimode review supports modes {SUPPORTED_MODES}, got {mode}")
    return (
        int(layout.energy_eV.size),
        int(layout.pitch_count),
        int(layout.azimuth_count),
        int(layout.mass_amu.size),
    )


def reshape_mode_dpf(values: np.ndarray, mode: int) -> np.ndarray:
    """Reshape mass-fastest rows to the canonical four-dimensional layout."""

    array = np.asarray(values)
    shape = mode_channel_shape(mode)
    expected = int(np.prod(shape))
    if array.shape[-1] != expected:
        raise ValueError(f"Mode {mode} row has {array.shape[-1]} values, expected {expected}")
    return array.reshape((*array.shape[:-1], *shape))


def reviewed_mass_mask(mode: int) -> np.ndarray:
    """Mask only the H+, O+, and O2+ mass bins covered by manual review."""

    mask = np.zeros(mode_layout(mode).mass_amu.size, dtype=bool)
    for species in SUPPORTED_SPECIES:
        mask[species_mass_indices(mode, species)] = True
    return mask


@dataclass(frozen=True)
class MultimodeBackgroundModel:
    mode: int
    background_dpf: np.ndarray
    uncertainty_dpf: np.ndarray
    interval_support_count: np.ndarray
    sample_count: np.ndarray
    nonzero_sample_count: np.ndarray
    zero_fraction: np.ndarray
    positive_fraction: np.ndarray
    case_standard_deviation_dpf: np.ndarray
    case_standard_error_dpf: np.ndarray
    nonzero_median_dpf: np.ndarray
    nonzero_mad_dpf: np.ndarray
    bootstrap_ci_low_dpf: np.ndarray
    bootstrap_ci_high_dpf: np.ndarray
    support_level: np.ndarray
    valid_channel_mask: np.ndarray
    reviewed_mass_mask: np.ndarray
    approved_intervals_by_species: Mapping[str, Sequence[Mapping[str, Any]]]
    provenance: Mapping[str, Any]
    review_status: str
    valid: bool
    invalid_reasons: Sequence[str]
    units: str = "1/(s cm^2 sr eV)"
    primary_estimator: str = "paper_channel_nonzero_mean_by_species"
    algorithm_version: str = MULTIMODE_MODEL_VERSION

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return mode_channel_shape(self.mode)


@dataclass(frozen=True)
class MultimodeCorrectionResult:
    raw_dpf: np.ndarray
    corrected_dpf: np.ndarray
    removed_dpf: np.ndarray
    applied_channel_mask: np.ndarray
    support_level: np.ndarray
    uncertainty_dpf: np.ndarray
    model_version: str
    denoise_policy_version: str


def _empty_arrays(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    floating = np.full(shape, np.nan, dtype=float)
    return {
        "background": floating.copy(),
        "uncertainty": floating.copy(),
        "interval_support": np.zeros(shape, dtype=np.int32),
        "sample_count": np.zeros(shape, dtype=np.int64),
        "nonzero_count": np.zeros(shape, dtype=np.int64),
        "zero_fraction": floating.copy(),
        "positive_fraction": floating.copy(),
        "case_std": floating.copy(),
        "case_sem": floating.copy(),
        "median": floating.copy(),
        "mad": floating.copy(),
        "ci_low": floating.copy(),
        "ci_high": floating.copy(),
        "support": np.full(shape, SUPPORT_UNSUPPORTED, dtype=np.uint8),
        "valid": np.zeros(shape, dtype=bool),
    }


def estimate_multimode_background(
    mode: int,
    interval_cubes_by_species: Mapping[str, Sequence[np.ndarray]],
    *,
    approved_intervals_by_species: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    bootstrap_replicates: int = 5_000,
    bootstrap_seed: int = 2024070801,
    review_status: str = "provisional_review_only",
    provenance: Mapping[str, Any] | None = None,
) -> MultimodeBackgroundModel:
    """Estimate each reviewed species only from that species' approved windows."""

    shape = mode_channel_shape(mode)
    arrays = _empty_arrays(shape)
    approved = approved_intervals_by_species or {}
    for species_index, species in enumerate(SUPPORTED_SPECIES):
        cubes = [np.asarray(value, dtype=float) for value in interval_cubes_by_species.get(species, [])]
        for cube in cubes:
            if cube.ndim != 5 or cube.shape[1:] != shape:
                raise ValueError(
                    f"Mode {mode} {species} interval cube must have shape (time,{shape}), got {cube.shape}"
                )
        mass_indices = species_mass_indices(mode, species)
        if not cubes:
            continue
        interval_means: list[np.ndarray] = []
        pooled_positive: list[np.ndarray] = []
        total = np.zeros(shape[:-1] + (mass_indices.size,), dtype=np.int64)
        nonzero = np.zeros_like(total)
        for cube in cubes:
            selected = cube[..., mass_indices]
            finite = np.isfinite(selected)
            positive = finite & (selected > 0.0)
            count = positive.sum(axis=0)
            mean = np.divide(
                np.where(positive, selected, 0.0).sum(axis=0),
                count,
                out=np.full(selected.shape[1:], np.nan),
                where=count > 0,
            )
            interval_means.append(mean)
            total += finite.sum(axis=0)
            nonzero += count
            pooled_positive.append(np.where(positive, selected, np.nan))
        stack = np.stack(interval_means)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            background = np.nanmean(stack, axis=0)
            case_std = np.nanstd(stack, axis=0, ddof=1)
        interval_support = np.isfinite(stack).sum(axis=0).astype(np.int32)
        case_std[interval_support < 2] = np.nan
        case_sem = np.divide(
            case_std,
            np.sqrt(interval_support),
            out=np.full(background.shape, np.nan),
            where=interval_support >= 2,
        )
        pooled = np.concatenate(pooled_positive, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(pooled, axis=0)
            mad = np.nanmedian(np.abs(pooled - median[None, ...]), axis=0)
        ci_low_flat, ci_high_flat = _bootstrap_channel_mean_ci(
            stack.reshape(stack.shape[0], -1),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + species_index,
        )
        ci_low = ci_low_flat.reshape(background.shape)
        ci_high = ci_high_flat.reshape(background.shape)
        valid = np.isfinite(background) & (background >= 0.0) & (nonzero > 0)
        support = np.full(background.shape, SUPPORT_UNSUPPORTED, dtype=np.uint8)
        low = valid & ((interval_support < 3) | (nonzero < 5))
        support[low] = SUPPORT_LOW
        support[valid & ~low] = SUPPORT_SUPPORTED
        zero_fraction = np.divide(
            total - nonzero,
            total,
            out=np.full(background.shape, np.nan),
            where=total > 0,
        )
        positive_fraction = np.divide(
            nonzero,
            total,
            out=np.full(background.shape, np.nan),
            where=total > 0,
        )
        target = (..., mass_indices)
        arrays["background"][target] = background
        arrays["uncertainty"][target] = case_sem
        arrays["interval_support"][target] = interval_support
        arrays["sample_count"][target] = total
        arrays["nonzero_count"][target] = nonzero
        arrays["zero_fraction"][target] = zero_fraction
        arrays["positive_fraction"][target] = positive_fraction
        arrays["case_std"][target] = case_std
        arrays["case_sem"][target] = case_sem
        arrays["median"][target] = median
        arrays["mad"][target] = mad
        arrays["ci_low"][target] = ci_low
        arrays["ci_high"][target] = ci_high
        arrays["support"][target] = support
        arrays["valid"][target] = valid

    reviewed = reviewed_mass_mask(mode)
    invalid_reasons: list[str] = []
    if review_status != "finalized":
        invalid_reasons.append("manual_review_not_finalized")
    if not np.any(arrays["valid"]):
        invalid_reasons.append("no_reviewed_channels_with_nonzero_support")
    model_provenance = {
        "mode": mode,
        "shape": list(shape),
        "dimension_order": ["energy", "pitch", "azimuth", "mass"],
        "energy_eV": mode_layout(mode).energy_eV.tolist(),
        "mass_amu": mode_layout(mode).mass_amu.tolist(),
        "units_interpretation": "released differential particle flux; not raw detector counts",
        "review_version": MULTIMODE_REVIEW_VERSION,
        "denoise_policy_version": MULTIMODE_DENOISE_VERSION,
        **dict(provenance or {}),
    }
    return MultimodeBackgroundModel(
        mode=mode,
        background_dpf=arrays["background"],
        uncertainty_dpf=arrays["uncertainty"],
        interval_support_count=arrays["interval_support"],
        sample_count=arrays["sample_count"],
        nonzero_sample_count=arrays["nonzero_count"],
        zero_fraction=arrays["zero_fraction"],
        positive_fraction=arrays["positive_fraction"],
        case_standard_deviation_dpf=arrays["case_std"],
        case_standard_error_dpf=arrays["case_sem"],
        nonzero_median_dpf=arrays["median"],
        nonzero_mad_dpf=arrays["mad"],
        bootstrap_ci_low_dpf=arrays["ci_low"],
        bootstrap_ci_high_dpf=arrays["ci_high"],
        support_level=arrays["support"],
        valid_channel_mask=arrays["valid"],
        reviewed_mass_mask=reviewed,
        approved_intervals_by_species=dict(approved),
        provenance=model_provenance,
        review_status=review_status,
        valid=not invalid_reasons,
        invalid_reasons=invalid_reasons,
    )


def assemble_multimode_background_from_summaries(
    mode: int,
    summaries_by_species: Mapping[str, Mapping[str, np.ndarray]],
    *,
    approved_intervals_by_species: Mapping[str, Sequence[Mapping[str, Any]]],
    bootstrap_replicates: int = 5_000,
    bootstrap_seed: int = 2026081001,
    review_status: str = "finalized",
    provenance: Mapping[str, Any] | None = None,
) -> MultimodeBackgroundModel:
    """Assemble a model from streaming sufficient statistics.

    Each species summary contains interval means plus pooled total/nonzero
    counts and exact pooled-positive median/MAD for only that species' reviewed
    mass bins. This keeps large full-mission builds out of RAM without changing
    the paper estimator or interval-equal weighting.
    """

    shape = mode_channel_shape(mode)
    arrays = _empty_arrays(shape)
    for species_index, species in enumerate(SUPPORTED_SPECIES):
        summary = summaries_by_species.get(species)
        if summary is None:
            continue
        mass_indices = species_mass_indices(mode, species)
        selected_shape = shape[:-1] + (mass_indices.size,)
        interval_means = np.asarray(summary["interval_means_dpf"], dtype=float)
        total = np.asarray(summary["sample_count"], dtype=np.int64)
        nonzero = np.asarray(summary["nonzero_sample_count"], dtype=np.int64)
        median = np.asarray(summary["nonzero_median_dpf"], dtype=float)
        mad = np.asarray(summary["nonzero_mad_dpf"], dtype=float)
        if interval_means.ndim != len(selected_shape) + 1 or interval_means.shape[1:] != selected_shape:
            raise ValueError(
                f"Mode {mode} {species} interval means must have shape (interval,{selected_shape})"
            )
        for name, value in {
            "sample_count": total,
            "nonzero_sample_count": nonzero,
            "nonzero_median_dpf": median,
            "nonzero_mad_dpf": mad,
        }.items():
            if value.shape != selected_shape:
                raise ValueError(f"Mode {mode} {species} {name} has shape {value.shape}, expected {selected_shape}")
        if np.any(nonzero > total) or np.any(total < 0) or np.any(nonzero < 0):
            raise ValueError(f"Mode {mode} {species} has inconsistent sample counts")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            background = np.nanmean(interval_means, axis=0)
            case_std = np.nanstd(interval_means, axis=0, ddof=1)
        interval_support = np.isfinite(interval_means).sum(axis=0).astype(np.int32)
        case_std[interval_support < 2] = np.nan
        case_sem = np.divide(
            case_std,
            np.sqrt(interval_support),
            out=np.full(selected_shape, np.nan),
            where=interval_support >= 2,
        )
        ci_low_flat, ci_high_flat = _bootstrap_channel_mean_ci(
            interval_means.reshape(interval_means.shape[0], -1),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + species_index,
        )
        ci_low = ci_low_flat.reshape(selected_shape)
        ci_high = ci_high_flat.reshape(selected_shape)
        valid = np.isfinite(background) & (background >= 0.0) & (nonzero > 0)
        support = np.full(selected_shape, SUPPORT_UNSUPPORTED, dtype=np.uint8)
        low = valid & ((interval_support < 3) | (nonzero < 5))
        support[low] = SUPPORT_LOW
        support[valid & ~low] = SUPPORT_SUPPORTED
        zero_fraction = np.divide(
            total - nonzero,
            total,
            out=np.full(selected_shape, np.nan),
            where=total > 0,
        )
        positive_fraction = np.divide(
            nonzero,
            total,
            out=np.full(selected_shape, np.nan),
            where=total > 0,
        )
        target = (..., mass_indices)
        arrays["background"][target] = background
        arrays["uncertainty"][target] = case_sem
        arrays["interval_support"][target] = interval_support
        arrays["sample_count"][target] = total
        arrays["nonzero_count"][target] = nonzero
        arrays["zero_fraction"][target] = zero_fraction
        arrays["positive_fraction"][target] = positive_fraction
        arrays["case_std"][target] = case_std
        arrays["case_sem"][target] = case_sem
        arrays["median"][target] = median
        arrays["mad"][target] = mad
        arrays["ci_low"][target] = ci_low
        arrays["ci_high"][target] = ci_high
        arrays["support"][target] = support
        arrays["valid"][target] = valid

    reviewed = reviewed_mass_mask(mode)
    invalid_reasons: list[str] = []
    if review_status != "finalized":
        invalid_reasons.append("manual_review_not_finalized")
    if not np.any(arrays["valid"]):
        invalid_reasons.append("no_reviewed_channels_with_nonzero_support")
    model_provenance = {
        "mode": mode,
        "shape": list(shape),
        "dimension_order": ["energy", "pitch", "azimuth", "mass"],
        "energy_eV": mode_layout(mode).energy_eV.tolist(),
        "mass_amu": mode_layout(mode).mass_amu.tolist(),
        "units_interpretation": "released differential particle flux; not raw detector counts",
        "review_version": MULTIMODE_REVIEW_VERSION,
        "denoise_policy_version": MULTIMODE_DENOISE_VERSION,
        "streaming_summary_assembly": True,
        **dict(provenance or {}),
    }
    return MultimodeBackgroundModel(
        mode=mode,
        background_dpf=arrays["background"],
        uncertainty_dpf=arrays["uncertainty"],
        interval_support_count=arrays["interval_support"],
        sample_count=arrays["sample_count"],
        nonzero_sample_count=arrays["nonzero_count"],
        zero_fraction=arrays["zero_fraction"],
        positive_fraction=arrays["positive_fraction"],
        case_standard_deviation_dpf=arrays["case_std"],
        case_standard_error_dpf=arrays["case_sem"],
        nonzero_median_dpf=arrays["median"],
        nonzero_mad_dpf=arrays["mad"],
        bootstrap_ci_low_dpf=arrays["ci_low"],
        bootstrap_ci_high_dpf=arrays["ci_high"],
        support_level=arrays["support"],
        valid_channel_mask=arrays["valid"],
        reviewed_mass_mask=reviewed,
        approved_intervals_by_species=dict(approved_intervals_by_species),
        provenance=model_provenance,
        review_status=review_status,
        valid=not invalid_reasons,
        invalid_reasons=invalid_reasons,
    )


def apply_multimode_background(
    dpf_cube: np.ndarray,
    model: MultimodeBackgroundModel,
    *,
    allow_provisional: bool = False,
) -> MultimodeCorrectionResult:
    """Subtract supported reviewed channels and preserve all other channels."""

    raw = np.asarray(dpf_cube, dtype=float)
    if raw.shape[-4:] != model.shape:
        raise ValueError(f"Mode {model.mode} DPF trailing shape must be {model.shape}; got {raw.shape}")
    if not model.valid and not allow_provisional:
        raise ValueError(f"Refusing invalid/provisional model: {list(model.invalid_reasons)}")
    background = np.broadcast_to(model.background_dpf, raw.shape)
    supported = np.broadcast_to(model.valid_channel_mask, raw.shape)
    finite = np.isfinite(raw) & np.isfinite(background) & supported
    corrected = raw.copy()
    corrected[finite] = np.maximum(raw[finite] - background[finite], 0.0)
    removed = np.full(raw.shape, np.nan, dtype=float)
    removed[np.isfinite(raw)] = 0.0
    removed[finite] = raw[finite] - corrected[finite]
    return MultimodeCorrectionResult(
        raw_dpf=raw,
        corrected_dpf=corrected,
        removed_dpf=removed,
        applied_channel_mask=finite,
        support_level=np.broadcast_to(model.support_level, raw.shape),
        uncertainty_dpf=np.broadcast_to(model.uncertainty_dpf, raw.shape),
        model_version=model.algorithm_version,
        denoise_policy_version=str(
            model.provenance.get("denoise_policy_version", MULTIMODE_DENOISE_VERSION)
        ),
    )


def save_multimode_background_model(path: Path, model: MultimodeBackgroundModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": model.mode,
        "review_status": model.review_status,
        "valid": model.valid,
        "invalid_reasons": list(model.invalid_reasons),
        "approved_intervals_by_species": model.approved_intervals_by_species,
        "provenance": model.provenance,
        "units": model.units,
        "primary_estimator": model.primary_estimator,
        "algorithm_version": model.algorithm_version,
    }
    np.savez_compressed(
        path,
        background_dpf=model.background_dpf,
        uncertainty_dpf=model.uncertainty_dpf,
        interval_support_count=model.interval_support_count,
        sample_count=model.sample_count,
        nonzero_sample_count=model.nonzero_sample_count,
        zero_fraction=model.zero_fraction,
        positive_fraction=model.positive_fraction,
        case_standard_deviation_dpf=model.case_standard_deviation_dpf,
        case_standard_error_dpf=model.case_standard_error_dpf,
        nonzero_median_dpf=model.nonzero_median_dpf,
        nonzero_mad_dpf=model.nonzero_mad_dpf,
        bootstrap_ci_low_dpf=model.bootstrap_ci_low_dpf,
        bootstrap_ci_high_dpf=model.bootstrap_ci_high_dpf,
        support_level=model.support_level,
        valid_channel_mask=model.valid_channel_mask,
        reviewed_mass_mask=model.reviewed_mass_mask,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )


def load_multimode_background_model(path: Path) -> MultimodeBackgroundModel:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return MultimodeBackgroundModel(
            mode=int(metadata["mode"]),
            background_dpf=data["background_dpf"],
            uncertainty_dpf=data["uncertainty_dpf"],
            interval_support_count=data["interval_support_count"],
            sample_count=data["sample_count"],
            nonzero_sample_count=data["nonzero_sample_count"],
            zero_fraction=data["zero_fraction"],
            positive_fraction=data["positive_fraction"],
            case_standard_deviation_dpf=data["case_standard_deviation_dpf"],
            case_standard_error_dpf=data["case_standard_error_dpf"],
            nonzero_median_dpf=data["nonzero_median_dpf"],
            nonzero_mad_dpf=data["nonzero_mad_dpf"],
            bootstrap_ci_low_dpf=data["bootstrap_ci_low_dpf"],
            bootstrap_ci_high_dpf=data["bootstrap_ci_high_dpf"],
            support_level=data["support_level"],
            valid_channel_mask=data["valid_channel_mask"],
            reviewed_mass_mask=data["reviewed_mass_mask"],
            approved_intervals_by_species=metadata["approved_intervals_by_species"],
            provenance=metadata["provenance"],
            review_status=str(metadata["review_status"]),
            valid=bool(metadata["valid"]),
            invalid_reasons=list(metadata["invalid_reasons"]),
            units=str(metadata["units"]),
            primary_estimator=str(metadata["primary_estimator"]),
            algorithm_version=str(metadata["algorithm_version"]),
        )


def load_multimode_model_bundle(
    path: Path,
    *,
    allow_release_candidate: bool = False,
) -> dict[int, MultimodeBackgroundModel]:
    """Load a finalized bundle; rc1 manifests are refused by default."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    status = str(payload.get("status", ""))
    if status != "frozen" and not allow_release_candidate:
        raise ValueError(f"Refusing non-frozen multimode bundle status={status!r}")
    models: dict[int, MultimodeBackgroundModel] = {}
    for mode_text, item in payload.get("models", {}).items():
        mode = int(mode_text)
        model_path = Path(str(item["path"]))
        if not model_path.is_absolute():
            model_path = source.parent / model_path
        if sha256_file(model_path) != str(item["sha256"]):
            raise ValueError(f"Mode {mode} model hash mismatch")
        model = load_multimode_background_model(model_path)
        if model.mode != mode:
            raise ValueError(f"Bundle mode {mode} points to model mode {model.mode}")
        models[mode] = model
    return models
