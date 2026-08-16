"""Mode-4/12 species-independent quiet-candidate discovery and review rules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.io import loadmat

from .minpa_io import ori_files_for_interval, read_ori_native_quality
from .minpa_modes import mode_layout
from .minpa_multimode_background import (
    MULTIMODE_REVIEW_VERSION,
    SUPPORTED_MODES,
    SUPPORTED_SPECIES,
)


PROJECT_REJECT_MASK = np.uint32(12)
MODE_CADENCE_S = {4: 12.3, 12: 2.05}
SPECIES_CONTAINER = {"H+": "H_spe_num", "O+": "O_spe_num", "O2+": "O2_spe_num"}
SPECIES_TOKEN = {"H+": "Hplus", "O+": "Oplus", "O2+": "O2plus"}


@dataclass(frozen=True)
class MinuteWindow:
    mode: int
    species: str
    month: str
    start_unix_s: float
    stop_unix_s: float
    score: float
    record_count: int
    source_day_spe: tuple[str, ...]
    source_segments: tuple[str, ...]


def utc_text(value: float) -> str:
    return datetime.fromtimestamp(float(value), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _segment_fields(container: Any) -> Sequence[str]:
    return list(getattr(container, "_fieldnames", []) or [])


def _as_spectrum(item: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(item.t, dtype=float).reshape(-1)
    energy = np.asarray(item.f, dtype=float).reshape(-1)
    spectrum = np.asarray(item.p, dtype=float)
    if spectrum.shape == (energy.size, time.size):
        spectrum = spectrum.T
    flag = np.asarray(getattr(item, "quality_flag", []), dtype=np.uint32).reshape(-1)
    available = np.asarray(
        getattr(item, "quality_flag_available", []), dtype=bool
    ).reshape(-1)
    return time, energy, spectrum, flag, available


def collect_eligible_minute_windows(
    day_spe_paths: Sequence[Path],
    *,
    modes: Sequence[int] = SUPPORTED_MODES,
    species_names: Sequence[str] = SUPPORTED_SPECIES,
    coverage_fraction: float = 0.8,
    maximum_gap_factor: float = 1.5,
) -> tuple[list[MinuteWindow], list[dict[str, Any]]]:
    """Collect quality-clean, UTC-aligned one-minute windows from day spectra."""

    if not 0.0 < coverage_fraction <= 1.0:
        raise ValueError("coverage_fraction must be in (0,1]")
    rows: list[MinuteWindow] = []
    ineligible: list[dict[str, Any]] = []
    for path in sorted(Path(value) for value in day_spe_paths):
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        for species in species_names:
            container_name = SPECIES_CONTAINER[species]
            if container_name not in payload:
                ineligible.append({"path": str(path.resolve()), "species": species, "reason": "missing_species_container"})
                continue
            container = payload[container_name]
            for segment_name in _segment_fields(container):
                item = getattr(container, segment_name)
                mode_values = np.asarray(getattr(item, "mod", np.nan)).reshape(-1)
                if mode_values.size != 1 or not np.isfinite(mode_values[0]):
                    continue
                mode = int(mode_values[0])
                if mode not in modes:
                    continue
                time, energy, spectrum, flag, available = _as_spectrum(item)
                expected_energy = mode_layout(mode).energy_eV
                if (
                    spectrum.shape != (time.size, energy.size)
                    or flag.size != time.size
                    or available.size != time.size
                    or energy.size != expected_energy.size
                    or not np.allclose(energy, expected_energy, rtol=1e-5, atol=1e-5)
                ):
                    ineligible.append({
                        "path": str(path.resolve()),
                        "segment": segment_name,
                        "mode": mode,
                        "species": species,
                        "reason": "layout_energy_or_quality_shape_mismatch",
                    })
                    continue
                if not time.size:
                    continue
                minute_values = np.floor(time / 60.0).astype(np.int64)
                cadence = MODE_CADENCE_S[mode]
                minimum_records = int(math.ceil(coverage_fraction * 60.0 / cadence))
                for minute in np.unique(minute_values):
                    indices = np.flatnonzero(minute_values == minute)
                    if indices.size < minimum_records:
                        continue
                    if not np.all(available[indices]):
                        continue
                    if np.any((flag[indices] & PROJECT_REJECT_MASK) != 0):
                        continue
                    gaps = np.diff(time[indices])
                    if gaps.size and float(np.nanmax(gaps)) > maximum_gap_factor * cadence:
                        continue
                    positive = np.isfinite(spectrum[indices]) & (spectrum[indices] > 0.0)
                    total_def = np.where(positive, spectrum[indices], 0.0).sum(axis=1)
                    score = float(np.nanmedian(total_def))
                    if not np.isfinite(score):
                        continue
                    start_s = float(minute * 60)
                    rows.append(MinuteWindow(
                        mode=mode,
                        species=species,
                        month=datetime.fromtimestamp(start_s, UTC).strftime("%Y%m"),
                        start_unix_s=start_s,
                        stop_unix_s=start_s + 60.0,
                        score=score,
                        record_count=int(indices.size),
                        source_day_spe=(str(path.resolve()),),
                        source_segments=(segment_name,),
                    ))
    return rows, ineligible


def select_and_merge_low_tail(
    minute_windows: Sequence[MinuteWindow],
    *,
    quantile: float = 0.10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select per-mode/species/month low-tail windows and merge adjacency."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    groups: defaultdict[tuple[int, str, str], list[MinuteWindow]] = defaultdict(list)
    for item in minute_windows:
        groups[(item.mode, item.species, item.month)].append(item)
    candidates: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    for (mode, species, month), values in sorted(groups.items()):
        by_minute: dict[float, MinuteWindow] = {}
        for item in sorted(values, key=lambda row: (row.start_unix_s, row.score)):
            previous = by_minute.get(item.start_unix_s)
            if previous is None or item.score < previous.score:
                by_minute[item.start_unix_s] = item
        unique = sorted(by_minute.values(), key=lambda row: row.start_unix_s)
        scores = np.asarray([item.score for item in unique], dtype=float)
        cutoff = float(np.quantile(scores, quantile))
        selected = [item for item in unique if item.score <= cutoff]
        thresholds.append({
            "mode": mode,
            "species": species,
            "month": month,
            "eligible_minute_count": len(unique),
            "selected_minute_count": len(selected),
            "requested_quantile": quantile,
            "score_cutoff": cutoff,
            "actual_selected_fraction": len(selected) / len(unique),
            "tie_policy": "include_all_scores_at_or_below_quantile_cutoff",
        })
        run: list[MinuteWindow] = []
        for item in selected:
            if run and item.start_unix_s != run[-1].stop_unix_s:
                candidates.append(_merge_run(run, cutoff, quantile))
                run = []
            run.append(item)
        if run:
            candidates.append(_merge_run(run, cutoff, quantile))
    candidates.sort(key=lambda row: (row["start_unix_s"], row["mode"], row["species"]))
    counters: defaultdict[tuple[int, str, str], int] = defaultdict(int)
    for item in candidates:
        key = (int(item["mode"]), str(item["species"]), str(item["month"]))
        counters[key] += 1
        item["candidate_id"] = (
            f"m{key[0]:02d}-{SPECIES_TOKEN[key[1]]}-{key[2]}-{counters[key]:05d}"
        )
        start_token = datetime.fromtimestamp(float(item["start_unix_s"]), UTC).strftime("%Y%m%dT%H%M%S")
        stop_token = datetime.fromtimestamp(float(item["stop_unix_s"]), UTC).strftime("%Y%m%dT%H%M%S")
        item["plot_filename"] = (
            f"mode{key[0]:02d}_{SPECIES_TOKEN[key[1]]}_{item['candidate_id']}_{start_token}_{stop_token}.png"
        )
    return candidates, thresholds


def _merge_run(run: Sequence[MinuteWindow], cutoff: float, quantile: float) -> dict[str, Any]:
    starts = [item.start_unix_s for item in run]
    stops = [item.stop_unix_s for item in run]
    source_paths = sorted({path for item in run for path in item.source_day_spe})
    segments = sorted({segment for item in run for segment in item.source_segments})
    return {
        "mode": run[0].mode,
        "species": run[0].species,
        "month": run[0].month,
        "start_unix_s": float(min(starts)),
        "stop_unix_s": float(max(stops)),
        "start_utc": utc_text(min(starts)),
        "stop_utc": utc_text(max(stops)),
        "duration_s": float(max(stops) - min(starts)),
        "minute_count": len(run),
        "day_spe_record_count": int(sum(item.record_count for item in run)),
        "median_minute_score": float(np.median([item.score for item in run])),
        "maximum_minute_score": float(max(item.score for item in run)),
        "selection_score_cutoff": cutoff,
        "selection_quantile": quantile,
        "source_day_spe": source_paths,
        "source_day_spe_segments": segments,
        "review_status": "pending",
        "analysis_status": "provisional_review_only",
    }


def validate_native_quality(
    candidates: Sequence[Mapping[str, Any]],
    ori_catalog: Sequence[tuple[Path, int, float, float]],
    *,
    coverage_fraction: float = 0.8,
    maximum_gap_factor: float = 1.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Require native Quality==0 and adequate raw/product-time coverage."""

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_item in candidates:
        item = dict(source_item)
        mode = int(item["mode"])
        start_s, stop_s = float(item["start_unix_s"]), float(item["stop_unix_s"])
        paths = ori_files_for_interval(ori_catalog, mode, start_s, stop_s)
        if not paths:
            rejected.append({**item, "preflight_reasons": ["no_overlapping_ori_file"]})
            continue
        time_parts: list[np.ndarray] = []
        quality_parts: list[np.ndarray] = []
        for path in paths:
            key = str(path.resolve())
            if key not in cache:
                cache[key] = read_ori_native_quality(path)
            time, quality = cache[key]
            keep = (time >= start_s) & (time < stop_s)
            time_parts.append(time[keep])
            quality_parts.append(quality[keep])
        time = np.concatenate(time_parts) if time_parts else np.empty(0)
        quality = np.concatenate(quality_parts) if quality_parts else np.empty(0, dtype=np.uint32)
        if time.size:
            order = np.argsort(time, kind="stable")
            time, quality = time[order], quality[order]
            unique = np.ones(time.size, dtype=bool)
            unique[1:] = np.diff(time) != 0.0
            time, quality = time[unique], quality[unique]
        cadence = MODE_CADENCE_S[mode]
        expected = (stop_s - start_s) / cadence
        reasons: list[str] = []
        if time.size < int(math.ceil(coverage_fraction * expected)):
            reasons.append("insufficient_native_record_coverage")
        gaps = np.diff(time)
        if gaps.size and float(np.nanmax(gaps)) > maximum_gap_factor * cadence:
            reasons.append("native_record_gap_too_large")
        if np.any(quality != 0):
            reasons.append("native_quality_nonzero")
        item.update({
            "source_ori": [str(path.resolve()) for path in paths],
            "native_record_count": int(time.size),
            "native_quality_nonzero_record_count": int(np.count_nonzero(quality != 0)),
            "native_max_record_gap_s": float(np.nanmax(gaps)) if gaps.size else None,
        })
        if reasons:
            rejected.append({**item, "preflight_reasons": reasons})
        else:
            accepted.append(item)
    return accepted, rejected


def finalize_species_review(
    candidates: Sequence[Mapping[str, Any]],
    pending_dir: Path,
    *,
    review_version: str = MULTIMODE_REVIEW_VERSION,
) -> dict[str, Any]:
    """Infer independent species decisions from retained/deleted PNG files."""

    root = Path(pending_dir)
    expected = {str(item["plot_filename"]): item for item in candidates}
    actual = {path.name for path in root.glob("*.png")}
    nested = [str(path) for path in root.rglob("*.png") if path.parent != root]
    if nested:
        raise ValueError(f"Nested review PNG files are not allowed: {nested[:5]}")
    unexpected = sorted(actual - set(expected))
    if unexpected:
        raise ValueError(f"Unexpected or renamed review PNG files: {unexpected[:10]}")
    for name in sorted(actual):
        expected_hash = str(expected[name].get("plot_sha256", ""))
        if expected_hash:
            digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError(f"Modified review PNG file: {name}")
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for name, source in expected.items():
        item = dict(source)
        if name in actual:
            approved.append({**item, "manual_decision": "approved", "decision_source": "retained_png"})
        else:
            rejected.append({
                **item,
                "manual_decision": "rejected",
                "decision_source": "deleted_png",
                "interpretation": "Whole species window rejected; no automatic trimming or salvage.",
            })
    return {
        "review_version": str(review_version),
        "review_status": "finalized",
        "decision_policy": "species-independent retained PNG approval",
        "counts": {
            "baseline_total": len(expected),
            "approved": len(approved),
            "rejected": len(rejected),
        },
        "approved_intervals": approved,
        "rejected_intervals": rejected,
    }
