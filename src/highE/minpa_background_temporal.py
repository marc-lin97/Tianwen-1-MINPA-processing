"""Exploratory full-mission MINPA Mode-1 background-noise analysis.

The workflow scans every local daily spectrum segment to establish Mode-1
coverage, then reads the much larger ``ori`` cubes only for quiet-window
candidates.  All post-2021 classifications remain explicitly exploratory
because no cross-year manual audit is available.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import rankdata, spearmanr

from .minpa_background import (
    MODE1_AZIMUTH_EDGES_DEG,
    MODE1_ENERGY_EV,
    MODE1_MASS_AMU,
    MODE1_PITCH_EDGES_DEG,
    MODE1_SHAPE,
    PROJECT_BACKGROUND_REJECT_MASK,
    BackgroundConfig,
    BackgroundInterval,
    CandidateConfig,
    MinpaMode1Records,
    concatenate_mode1_records,
    discover_background_candidates,
    estimate_background,
    read_mode1_ori_records,
    read_mode1_selected_dpf_channels,
    sha256_file,
)
from .minpa_figure5 import screen_hplus_coherent_signal
from .minpa_modes import energy_bin_edges_eV
from .minpa_quiet_classifier import (
    QuietBoundaryRefinementConfig,
    QuietReferenceClassifierConfig,
    evaluate_quiet_boundary_refinements,
    extract_quiet_reference_features,
)


TEMPORAL_ALGORITHM_VERSION = "minpa-mode1-temporal-noise-v2"
EXPLORATORY_STATUS = "exploratory_auto_no_cross_year_manual_audit"
EV_J = 1.602176634e-19
MP_KG = 1.67262192369e-27
ORI_PATTERN = re.compile(
    r"MINPA-MOD1-.*?_(\d{8})(\d{6})_(\d{8})(\d{6})_"
)


def _finite_mean(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """NaN-aware mean without warnings for entirely invalid slices."""

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    total = np.sum(np.where(finite, array, 0.0), axis=axis)
    count = np.sum(finite, axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=float),
        where=count > 0,
    )


def _utc_text(value: float) -> str:
    return datetime.fromtimestamp(float(value), UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_utc(value: str) -> float:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        temp = Path(handle.name)
    os.replace(temp, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, delete=False, suffix=".npz.tmp"
    ) as handle:
        temp = Path(handle.name)
    try:
        np.savez_compressed(temp, **arrays)
        # np.savez appends .npz when the supplied name lacks that suffix.
        actual = temp if temp.exists() and temp.stat().st_size else Path(str(temp) + ".npz")
        os.replace(actual, path)
    finally:
        for candidate in (temp, Path(str(temp) + ".npz")):
            if candidate.exists():
                candidate.unlink()


def _input_signature(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(dict(row), sort_keys=True, default=_json_default).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strict_mode1_segments(path: Path) -> tuple[list[dict[str, Any]], Counter[int]]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    containers = {name: data[f"{name}_spe_num"] for name in ("H", "O", "O2")}
    fields = list(getattr(containers["H"], "_fieldnames", []) or [])
    mode_counts: Counter[int] = Counter()
    segments: list[dict[str, Any]] = []
    for field in fields:
        items = []
        for species in ("H", "O", "O2"):
            container = containers[species]
            if field not in (getattr(container, "_fieldnames", []) or []):
                items = []
                break
            items.append(getattr(container, field))
        if len(items) != 3:
            continue
        modes: list[int] = []
        for item in items:
            raw = np.asarray(getattr(item, "mod", np.nan)).reshape(-1)
            modes.append(int(raw[0]) if raw.size == 1 and np.isfinite(raw[0]) else -1)
        mode_counts[modes[0]] += 1
        if modes != [1, 1, 1]:
            continue
        times = [np.asarray(item.t, dtype=float).reshape(-1) for item in items]
        if any(time.shape != times[0].shape for time in times[1:]):
            continue
        if any(not np.allclose(time, times[0], atol=1.0e-3, rtol=0.0) for time in times[1:]):
            continue
        energy_valid = True
        for item in items:
            energy = np.asarray(item.f, dtype=float).reshape(-1)
            if energy.size != MODE1_ENERGY_EV.size or not np.allclose(
                energy, MODE1_ENERGY_EV, rtol=1.0e-5, atol=1.0e-5
            ):
                energy_valid = False
                break
        if not energy_valid or times[0].size == 0:
            continue
        quality_available = True
        for item in items:
            available = np.asarray(
                getattr(item, "quality_flag_available", np.zeros(times[0].size)),
                dtype=bool,
            ).reshape(-1)
            if available.shape != times[0].shape or not np.all(available):
                quality_available = False
                break
        segments.append({
            "segment": field,
            "mode": 1,
            "records": int(times[0].size),
            "start_unix_s": float(times[0][0]),
            "stop_unix_s": float(times[0][-1]),
            "quality_available_all_records": quality_available,
        })
    return segments, mode_counts


def build_mode1_inventory(
    day_spe_root: Path,
    ori_root: Path,
    quality_root: Path,
) -> dict[str, Any]:
    """Inspect every local daily spectrum and Mode-1 ori file without mutation."""

    day_root, raw_root, flag_root = map(Path, (day_spe_root, ori_root, quality_root))
    daily_rows: list[dict[str, Any]] = []
    mode_segments: Counter[int] = Counter()
    month_records: Counter[str] = Counter()
    month_days: defaultdict[str, set[str]] = defaultdict(set)
    strict_segments = 0
    strict_records = 0
    mixed_mode_days = 0
    errors: list[dict[str, str]] = []
    for path in sorted(day_root.glob("Ion_spe_*.mat")):
        day = path.stem.rsplit("_", 1)[-1]
        stat = path.stat()
        try:
            segments, modes = _strict_mode1_segments(path)
            mode_segments.update(modes)
            mixed = len(modes) > 1
            mixed_mode_days += int(mixed)
            records = sum(int(item["records"]) for item in segments)
            strict_segments += len(segments)
            strict_records += records
            if segments:
                month = day[:6]
                month_days[month].add(day)
                month_records[month] += records
            daily_rows.append({
                "day": day,
                "month": day[:6],
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "mode1_segment_count": len(segments),
                "mode1_record_count": records,
                "mixed_mode_day": mixed,
                "segments": segments,
            })
        except Exception as exc:  # keep the inventory auditable instead of aborting silently
            errors.append({"path": str(path.resolve()), "error": f"{type(exc).__name__}: {exc}"})

    ori_rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.glob("*MINPA-MOD1-*.mat")):
        match = ORI_PATTERN.search(path.name)
        if not match:
            continue
        start = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        stop = datetime.strptime(match.group(3) + match.group(4), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        stat = path.stat()
        ori_rows.append({
            "path": str(path.resolve()),
            "start_unix_s": start.timestamp(),
            "stop_unix_s": stop.timestamp(),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    quality_days = sorted(
        path.name for path in flag_root.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{8}", path.name)
    ) if flag_root.exists() else []
    quality_set = set(quality_days)
    mode1_days = {row["day"] for row in daily_rows if row["mode1_segment_count"] > 0}
    months = [
        {
            "month": month,
            "mode1_days": len(month_days[month]),
            "mode1_records": int(month_records[month]),
        }
        for month in sorted(month_days)
    ]
    signature_rows = [
        {key: row[key] for key in ("path", "size_bytes", "mtime_ns")}
        for row in daily_rows
    ] + [
        {key: row[key] for key in ("path", "size_bytes", "mtime_ns")}
        for row in ori_rows
    ]
    return {
        "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
        "analysis_status": EXPLORATORY_STATUS,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_roots": {
            "day_spe": str(day_root.resolve()),
            "ori": str(raw_root.resolve()),
            "quality": str(flag_root.resolve()),
        },
        "input_signature": _input_signature(signature_rows),
        "day_spe_file_count": len(daily_rows),
        "ori_mode1_file_count": len(ori_rows),
        "ori_total_bytes": int(sum(row["size_bytes"] for row in ori_rows)),
        "strict_mode1_day_count": len(mode1_days),
        "strict_mode1_segment_count": strict_segments,
        "strict_mode1_record_count": strict_records,
        "mixed_mode_day_count": mixed_mode_days,
        "mode_segment_counts": {str(key): int(value) for key, value in sorted(mode_segments.items())},
        "quality_day_count": len(quality_days),
        "mode1_days_missing_quality": sorted(mode1_days - quality_set),
        "coverage_start_utc": _utc_text(min(row["start_unix_s"] for row in ori_rows)) if ori_rows else None,
        "coverage_stop_utc": _utc_text(max(row["stop_unix_s"] for row in ori_rows)) if ori_rows else None,
        "months": months,
        "day_spe_files": daily_rows,
        "ori_files": ori_rows,
        "errors": errors,
    }


def discover_monthly_quiet_candidates(
    inventory: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Apply the same relative quiet-tail screen independently in every month."""

    by_month: defaultdict[str, list[Path]] = defaultdict(list)
    for row in inventory["day_spe_files"]:
        if int(row["mode1_segment_count"]) > 0:
            by_month[str(row["month"])].append(Path(str(row["path"])))
    cfg = CandidateConfig(
        duration_s=float(candidate_config["duration_s"]),
        step_s=float(candidate_config["step_s"]),
        min_records=int(candidate_config["min_records"]),
        max_record_gap_s=float(candidate_config["max_record_gap_s"]),
        reject_quality_mask=int(candidate_config["reject_quality_mask"]),
        low_signal_quantile=float(candidate_config["low_signal_quantile"]),
        requested_intervals=2_147_483_647,
        max_intervals_per_day=int(candidate_config["max_intervals_per_day"]),
        min_separation_s=float(candidate_config["min_separation_s"]),
    )
    output: dict[str, list[dict[str, Any]]] = {}
    for month, paths in sorted(by_month.items()):
        rows = discover_background_candidates(sorted(paths), cfg)
        for number, row in enumerate(rows, start=1):
            row["month"] = month
            row["monthly_candidate_number"] = number
            row["candidate_id"] = f"{month}-{number:04d}"
            row["review_status"] = EXPLORATORY_STATUS
        output[month] = rows
    return output


def _ori_paths_for_interval(
    ori_entries: Sequence[Mapping[str, Any]], start_utc: str, stop_utc: str
) -> list[Path]:
    start_s, stop_s = _parse_utc(start_utc), _parse_utc(stop_utc)
    return [
        Path(str(item["path"]))
        for item in ori_entries
        if float(item["start_unix_s"]) < stop_s and float(item["stop_unix_s"]) >= start_s
    ]


def _manual_decision_map(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("intervals", []):
        manual_class = str(item.get("manual_class", ""))
        if manual_class not in (
            "weak_real_spectrum", "pure_background", "pure_background_adjusted"
        ):
            continue
        start_utc = str(item["start_utc"])
        stop_utc = str(item["stop_utc"])
        final_start, final_stop = start_utc, stop_utc
        if manual_class == "pure_background_adjusted":
            final_start = str(item["manual_approved_start_utc"])
            final_stop = str(item["manual_approved_stop_utc"])
        elif (
            manual_class == "pure_background"
            and item.get("boundary_refinement_status") == "accepted_high_confidence_subwindow"
        ):
            final_start = str(item["refined_start_utc"])
            final_stop = str(item["refined_stop_utc"])
        decisions[(start_utc, stop_utc)] = {
            "manual_class": manual_class,
            "accepted": manual_class != "weak_real_spectrum",
            "final_start_utc": final_start,
            "final_stop_utc": final_stop,
            "source_candidate_id": item.get("expanded_candidate_number"),
        }
    return decisions


def _load_h_segment(path: str, segment: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    item = getattr(payload["H_spe_num"], segment)
    return (
        np.asarray(item.t, dtype=float).reshape(-1),
        np.asarray(item.f, dtype=float).reshape(-1),
        np.asarray(item.p, dtype=float),
    )


def _read_candidate_records(
    candidate: Mapping[str, Any],
    ori_entries: Sequence[Mapping[str, Any]],
    quality_root: Path,
) -> MinpaMode1Records:
    paths = _ori_paths_for_interval(
        ori_entries, str(candidate["start_utc"]), str(candidate["stop_utc"])
    )
    return read_mode1_ori_records(
        paths,
        str(candidate["start_utc"]),
        str(candidate["stop_utc"]),
        quality_root=quality_root,
        compute_sha256=False,
    )


def _provisional_month_quantum(
    candidates: Sequence[Mapping[str, Any]],
    ori_entries: Sequence[Mapping[str, Any]],
    quality_root: Path,
    background_config: BackgroundConfig,
    maximum_intervals: int,
) -> tuple[np.ndarray, dict[tuple[str, str], MinpaMode1Records]]:
    ranked = sorted(
        candidates,
        key=lambda item: (float(item["median_total_h_def"]), str(item["start_utc"])),
    )[:maximum_intervals]
    parts: list[MinpaMode1Records] = []
    intervals: list[BackgroundInterval] = []
    cache: dict[tuple[str, str], MinpaMode1Records] = {}
    for item in ranked:
        key = (str(item["start_utc"]), str(item["stop_utc"]))
        part = _read_candidate_records(item, ori_entries, quality_root)
        cache[key] = part
        if part.time_unix_s.size == 0:
            continue
        parts.append(part)
        intervals.append(BackgroundInterval(
            start_utc=key[0], stop_utc=key[1], approved=True,
            label=f"preliminary-{item['candidate_id']}",
            source="monthly_relative_quiet_preliminary_quantum",
        ))
    if not parts:
        return np.full(MODE1_SHAPE, np.nan), cache
    provisional_cfg = BackgroundConfig(
        **{
            **asdict(background_config),
            "min_approved_intervals": 1,
            "min_total_records": 1,
            "min_channel_samples": 1,
        }
    )
    model = estimate_background(concatenate_mode1_records(parts), intervals, provisional_cfg)
    return model.dpf_quantum, cache


def _classify_candidate(
    candidate: Mapping[str, Any],
    records: MinpaMode1Records,
    dpf_quantum: np.ndarray,
    classifier_config: QuietReferenceClassifierConfig,
    refinement_config: QuietBoundaryRefinementConfig,
    manual_decision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = dict(candidate)
    row.update({
        "accepted": False,
        "predicted_class": "unclassified",
        "final_start_utc": str(candidate["start_utc"]),
        "final_stop_utc": str(candidate["stop_utc"]),
        "manual_override": manual_decision is not None,
    })
    if records.time_unix_s.size == 0 or not np.any(np.isfinite(dpf_quantum)):
        row["predicted_class"] = "rejected_missing_ori_or_quantum"
        return row
    day_time, energy, day_def = _load_h_segment(
        str(candidate["source_day_spe"]),
        str(candidate["source_day_spe_segment"]),
    )
    if energy.size != MODE1_ENERGY_EV.size or not np.allclose(
        energy, MODE1_ENERGY_EV, rtol=1.0e-5, atol=1.0e-5
    ):
        row["predicted_class"] = "rejected_energy_grid_mismatch"
        return row
    candidate_mask = (
        (day_time >= float(candidate["start_unix_s"]))
        & (day_time < float(candidate["stop_unix_s"]))
    )
    try:
        gross = screen_hplus_coherent_signal(records.dpf, dpf_quantum)
        features = extract_quiet_reference_features(
            day_def[candidate_mask], records.dpf, dpf_quantum, classifier_config
        )
    except ValueError as exc:
        row["predicted_class"] = "rejected_feature_error"
        row["rejection_reason"] = str(exc)
        return row
    row.update(gross)
    row.update(features)
    if bool(gross["coherent_signal_rejected"]) or bool(features["real_spectrum_detected"]):
        predicted = "real_spectrum_rejected"
    elif bool(features["manual_review_recommended"]):
        predicted = "pure_background_borderline"
    else:
        predicted = "pure_background_high_confidence"

    if predicted == "pure_background_borderline":
        quality_eligible = (
            records.project_quality_available
            & ((records.project_quality_flag & PROJECT_BACKGROUND_REJECT_MASK) == 0)
            & (records.native_quality == 0)
        )
        trials = evaluate_quiet_boundary_refinements(
            float(candidate["start_unix_s"]),
            float(candidate["stop_unix_s"]),
            day_time,
            day_def,
            records.time_unix_s[quality_eligible],
            records.dpf[quality_eligible],
            dpf_quantum,
            classifier_config,
            refinement_config,
        )
        selected: Mapping[str, Any] | None = None
        for trial in trials:
            start_s, stop_s = float(trial["start_unix_s"]), float(trial["stop_unix_s"])
            all_in = (records.time_unix_s >= start_s) & (records.time_unix_s < stop_s)
            if int(np.count_nonzero(all_in)) < refinement_config.minimum_dpf_records:
                continue
            if not np.all(records.project_quality_available[all_in]):
                continue
            if np.any((records.project_quality_flag[all_in] & PROJECT_BACKGROUND_REJECT_MASK) != 0):
                continue
            kept = all_in & quality_eligible
            if int(np.count_nonzero(kept)) < refinement_config.minimum_dpf_records:
                continue
            trial_gross = screen_hplus_coherent_signal(records.dpf[kept], dpf_quantum)
            if bool(trial_gross["coherent_signal_rejected"]):
                continue
            if bool(trial["real_spectrum_detected"]) or bool(trial["manual_review_recommended"]):
                continue
            selected = trial
            row.update({key: trial[key] for key in features})
            row.update({f"refined_{key}": value for key, value in trial_gross.items()})
            break
        if selected is None:
            predicted = "borderline_discarded_after_refinement"
        else:
            predicted = "pure_background_high_confidence_refined"
            row["final_start_utc"] = _utc_text(float(selected["start_unix_s"]))
            row["final_stop_utc"] = _utc_text(float(selected["stop_unix_s"]))
            row["refined_duration_s"] = float(selected["duration_s"])

    if manual_decision is not None:
        if not bool(manual_decision["accepted"]):
            predicted = "manual_real_spectrum_rejected"
        else:
            predicted = "manual_pure_background_approved"
            row["final_start_utc"] = str(manual_decision["final_start_utc"])
            row["final_stop_utc"] = str(manual_decision["final_stop_utc"])
        row["manual_class"] = str(manual_decision["manual_class"])

    row["predicted_class"] = predicted
    row["accepted"] = predicted in (
        "pure_background_high_confidence",
        "pure_background_high_confidence_refined",
        "manual_pure_background_approved",
    )
    return row


def _background_density_by_mass(background_dpf: np.ndarray) -> np.ndarray:
    values = np.asarray(background_dpf, dtype=float)
    if values.shape != MODE1_SHAPE:
        raise ValueError(f"background_dpf must have shape {MODE1_SHAPE}")
    theta = np.deg2rad(MODE1_PITCH_EDGES_DEG)
    phi = np.deg2rad(MODE1_AZIMUTH_EDGES_DEG)
    pitch_omega = np.cos(theta[:-1]) - np.cos(theta[1:])
    azimuth_width = np.diff(phi)
    omega = pitch_omega[:, None] * azimuth_width[None, :]
    density = np.full(MODE1_MASS_AMU.size, np.nan)
    energy_width_eV = np.diff(energy_bin_edges_eV(MODE1_ENERGY_EV))
    for mass_index, mass in enumerate(MODE1_MASS_AMU):
        speed_cm_s = np.sqrt(2.0 * EV_J * MODE1_ENERGY_EV / (mass * MP_KG)) * 100.0
        contribution = (
            values[..., mass_index]
            * energy_width_eV[:, None, None]
            * omega[None, :, :]
            / speed_cm_s[:, None, None]
        )
        density[mass_index] = float(np.nansum(np.maximum(contribution, 0.0)))
    return density


def estimate_interval_noise(
    records: MinpaMode1Records,
    dpf_quantum: np.ndarray,
    config: BackgroundConfig | None = None,
) -> dict[str, Any]:
    """Compute interval-level count-equivalent and DPF noise diagnostics."""

    cfg = config or BackgroundConfig(primary_estimator="count_equivalent_energy_pooled")
    quantum = np.asarray(dpf_quantum, dtype=float)
    if quantum.shape != MODE1_SHAPE:
        raise ValueError(f"dpf_quantum must have shape {MODE1_SHAPE}")
    keep = records.project_quality_available.copy()
    keep &= (records.project_quality_flag & np.uint32(cfg.reject_quality_mask)) == 0
    if cfg.require_native_quality_zero:
        keep &= records.native_quality == 0
    cube = np.asarray(records.dpf[keep], dtype=float)
    if cube.shape[0] == 0:
        raise ValueError("interval contains no quality-accepted Mode-1 records")
    finite = np.isfinite(cube) & (cube >= 0.0)
    ratio = np.divide(
        cube,
        quantum[None, ...],
        out=np.full(cube.shape, np.nan, dtype=float),
        where=finite & np.isfinite(quantum[None, ...]) & (quantum[None, ...] > 0.0),
    )
    count_equivalent = np.rint(ratio)
    residual = np.abs(ratio - count_equivalent)
    positive = finite & (cube > 0.0)
    reconstruction_good = positive & (residual <= cfg.quantum_residual_tolerance_counts)
    count_equivalent[~finite] = np.nan
    count_equivalent[positive & ~reconstruction_good] = np.nan
    with np.errstate(invalid="ignore"):
        lambda_by_energy = np.nanmean(count_equivalent, axis=0)
        lambda_pam = np.nanmedian(lambda_by_energy, axis=0)
    background_dpf = quantum * lambda_pam[None, ...]
    quantum_def = quantum * MODE1_ENERGY_EV[:, None, None, None]
    # Energy has already been pooled robustly into each pitch/azimuth/mass
    # channel.  Angular pooling must retain the sparse nonzero occurrence rate;
    # an angular median collapses the ion background to exactly zero whenever
    # more than half of the directional cells are empty.
    lambda_mass = _finite_mean(lambda_pam, axis=(0, 1))
    quantum_def_mass = np.nanmedian(quantum_def, axis=(0, 1, 2))
    zero_mass = np.mean(finite & (cube == 0.0), axis=(0, 1, 2, 3))
    positive_mass = np.mean(positive, axis=(0, 1, 2, 3))
    positive_count = np.sum(positive, axis=(0, 1, 2, 3))
    good_count = np.sum(reconstruction_good, axis=(0, 1, 2, 3))
    reconstruction_mass = np.divide(
        good_count,
        positive_count,
        out=np.full(MODE1_MASS_AMU.size, np.nan),
        where=positive_count > 0,
    )
    density_mass = _background_density_by_mass(background_dpf)
    midpoint_s = float(np.nanmedian(records.time_unix_s[keep]))
    return {
        "midpoint_unix_s": midpoint_s,
        "day": datetime.fromtimestamp(midpoint_s, UTC).strftime("%Y%m%d"),
        "raw_records": int(records.time_unix_s.size),
        "kept_records": int(np.count_nonzero(keep)),
        "lambda_global": float(_finite_mean(lambda_pam)),
        "lambda_mass": lambda_mass,
        "quantum_def_mass": quantum_def_mass,
        "zero_fraction_mass": zero_mass,
        "positive_fraction_mass": positive_mass,
        "quantum_reconstruction_fraction_mass": reconstruction_mass,
        "background_density_cm3_mass": density_mass,
        "start_stop_median": np.nanmedian(records.ion_start_stop_counts[keep], axis=0),
        "high_voltage_median_v": np.nanmedian(records.high_voltage_monitor_v[keep], axis=0),
        "solar_angle_median_deg": np.nanmedian(records.instrument_solar_angles_deg[keep], axis=0),
        "lambda_pam": lambda_pam,
    }


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", dir=path.parent,
        delete=False, suffix=".csv.tmp"
    ) as handle:
        temp = Path(handle.name)
    try:
        frame.to_csv(temp, index=False, encoding="utf-8-sig")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _flatten_interval_metric(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        key: value for key, value in row.items()
        if key not in (
            "lambda_mass", "quantum_def_mass", "zero_fraction_mass",
            "positive_fraction_mass", "quantum_reconstruction_fraction_mass",
            "background_density_cm3_mass", "start_stop_median",
            "high_voltage_median_v", "solar_angle_median_deg", "lambda_pam",
        )
    }
    for name in (
        "lambda_mass", "quantum_def_mass", "zero_fraction_mass",
        "positive_fraction_mass", "quantum_reconstruction_fraction_mass",
        "background_density_cm3_mass",
    ):
        values = np.asarray(row[name], dtype=float).reshape(-1)
        for index, value in enumerate(values):
            output[f"{name}_{index}"] = float(value)
    for index, value in enumerate(np.asarray(row["start_stop_median"], dtype=float).reshape(-1)):
        output[f"start_stop_median_{index}"] = float(value)
    for index, value in enumerate(np.asarray(row["high_voltage_median_v"], dtype=float).reshape(-1)):
        output[f"high_voltage_median_v_{index}"] = float(value)
    for index, value in enumerate(np.asarray(row["solar_angle_median_deg"], dtype=float).reshape(-1)):
        output[f"solar_angle_median_deg_{index}"] = float(value)
    return output


def _config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(config), sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()


def upgrade_monthly_lambda_pooling(
    output_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Upgrade completed month checkpoints from angular median to mean pooling.

    The stored per-interval ``lambda_pam`` arrays are authoritative, so this
    migration does not reread raw data or alter candidate classifications,
    background DPF cubes, density contributions, or source hashes.
    """

    root = Path(output_root)
    upgraded: list[str] = []
    unchanged: list[str] = []
    for summary_path in sorted((root / "months").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        month = str(summary["month"])
        arrays_path = summary_path.parent / "noise_arrays.npz"
        with np.load(arrays_path) as arrays:
            interval_lambda = np.asarray(arrays["interval_lambda_pam"], dtype=float)
        metrics = list(summary.get("interval_metrics", []))
        if interval_lambda.shape[0] != len(metrics):
            raise ValueError(
                f"{month}: interval lambda count {interval_lambda.shape[0]} "
                f"does not match metrics {len(metrics)}"
            )
        changed = summary.get("algorithm_version") != TEMPORAL_ALGORITHM_VERSION
        for metric, lambda_pam in zip(metrics, interval_lambda):
            lambda_mass = _finite_mean(lambda_pam, axis=(0, 1))
            lambda_global = float(_finite_mean(lambda_pam))
            if not np.isclose(
                float(metric.get("lambda_global", math.nan)), lambda_global,
                equal_nan=True,
            ):
                changed = True
            metric["lambda_global"] = lambda_global
            metric["lambda_mass"] = lambda_mass
        summary["interval_metrics"] = metrics
        summary["algorithm_version"] = TEMPORAL_ALGORITHM_VERSION
        summary["config_digest"] = _config_digest(config)
        summary["lambda_pooling"] = "energy_median_then_angular_arithmetic_mean"
        summary["lambda_pooling_upgrade_utc"] = datetime.now(UTC).isoformat()
        _atomic_csv(
            summary_path.parent / "interval_metrics.csv",
            [_flatten_interval_metric(item) for item in metrics],
        )
        _atomic_json(summary_path, summary)
        (upgraded if changed else unchanged).append(month)
    report = {
        "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "method": "recompute global and mass lambda from stored interval_lambda_pam",
        "scientific_reason": (
            "angular medians collapsed sparse ion occurrence rates to zero; "
            "energy-pooled directional rates are now combined by arithmetic mean"
        ),
        "raw_data_reread": False,
        "classification_changed": False,
        "upgraded_months": upgraded,
        "unchanged_months": unchanged,
    }
    _atomic_json(root / "logs" / "lambda_pooling_v2_upgrade.json", report)
    return report


def process_month_to_directory(
    month: str,
    candidates: Sequence[Mapping[str, Any]],
    ori_entries: Sequence[Mapping[str, Any]],
    quality_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
    manual_candidates_path: Path | None = None,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Classify, estimate, and atomically checkpoint one calendar month."""

    month_dir = Path(output_root) / "months" / month
    summary_path = month_dir / "summary.json"
    digest = _config_digest(config)
    signature_rows = [
        {key: row[key] for key in ("path", "size_bytes", "mtime_ns")}
        for row in ori_entries
    ] + [{
        "candidate_id": item["candidate_id"],
        "start_utc": item["start_utc"],
        "stop_utc": item["stop_utc"],
        "source_day_spe": item["source_day_spe"],
        "source_day_spe_segment": item["source_day_spe_segment"],
    } for item in candidates]
    month_signature = _input_signature(signature_rows)
    if resume and summary_path.exists():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            prior.get("status") == "complete"
            and prior.get("config_digest") == digest
            and prior.get("month_input_signature") == month_signature
        ):
            return {**prior, "resume_action": "skipped_valid_checkpoint"}

    candidate_cfg = config["candidate_selection"]
    classifier_cfg = QuietReferenceClassifierConfig(**config["classifier"])
    refinement_cfg = QuietBoundaryRefinementConfig(**config["boundary_refinement"])
    background_cfg = BackgroundConfig(**config["background_estimation"])
    manual_map = _manual_decision_map(manual_candidates_path)
    created = datetime.now(UTC).isoformat()
    try:
        quantum, record_cache = _provisional_month_quantum(
            candidates,
            ori_entries,
            Path(quality_root),
            background_cfg,
            int(candidate_cfg["preliminary_quantum_max_intervals"]),
        )
        classified: list[dict[str, Any]] = []
        accepted_parts: list[MinpaMode1Records] = []
        accepted_intervals: list[BackgroundInterval] = []
        accepted_ids: list[str] = []
        for candidate in candidates:
            key = (str(candidate["start_utc"]), str(candidate["stop_utc"]))
            records = record_cache.get(key)
            if records is None:
                records = _read_candidate_records(candidate, ori_entries, Path(quality_root))
            result = _classify_candidate(
                candidate,
                records,
                quantum,
                classifier_cfg,
                refinement_cfg,
                manual_map.get(key),
            )
            if bool(result["accepted"]):
                final_start = str(result["final_start_utc"])
                final_stop = str(result["final_stop_utc"])
                final_start_s, final_stop_s = _parse_utc(final_start), _parse_utc(final_stop)
                source_start = float(candidate["start_unix_s"])
                source_stop = float(candidate["stop_unix_s"])
                if final_start_s >= source_start and final_stop_s <= source_stop:
                    final_part = records.subset(
                        (records.time_unix_s >= final_start_s)
                        & (records.time_unix_s < final_stop_s)
                    )
                else:
                    final_candidate = {
                        **candidate,
                        "start_utc": final_start,
                        "stop_utc": final_stop,
                    }
                    final_part = _read_candidate_records(
                        final_candidate, ori_entries, Path(quality_root)
                    )
                in_quality = final_part.project_quality_available.copy()
                in_quality &= (
                    (final_part.project_quality_flag & np.uint32(background_cfg.reject_quality_mask)) == 0
                )
                if background_cfg.require_native_quality_zero:
                    in_quality &= final_part.native_quality == 0
                project_bad = (
                    final_part.project_quality_available
                    & ((final_part.project_quality_flag & np.uint32(background_cfg.reject_quality_mask)) != 0)
                )
                final_reasons: list[str] = []
                if background_cfg.require_project_quality and (
                    final_part.time_unix_s.size == 0
                    or not np.all(final_part.project_quality_available)
                ):
                    final_reasons.append("project_quality_unavailable")
                if background_cfg.reject_interval_if_project_flagged and np.any(project_bad):
                    final_reasons.append("project_quality_bits_3_4_present")
                if int(np.count_nonzero(in_quality)) < background_cfg.min_records_per_interval:
                    final_reasons.append("too_few_quality_accepted_records")
                if final_reasons:
                    result["accepted"] = False
                    result["predicted_class"] = "rejected_final_interval_quality"
                    result["final_rejection_reasons"] = final_reasons
                else:
                    accepted_parts.append(final_part)
                    accepted_ids.append(str(candidate["candidate_id"]))
                    accepted_intervals.append(BackgroundInterval(
                        start_utc=final_start,
                        stop_utc=final_stop,
                        approved=True,
                        label=str(candidate["candidate_id"]),
                        source=(
                            "manual_2021_reference"
                            if bool(result["manual_override"])
                            else EXPLORATORY_STATUS
                        ),
                    ))
            classified.append(result)

        combined = concatenate_mode1_records(accepted_parts)
        selected_files = sorted({name for part in accepted_parts for name in part.source_files})
        hashes = {name: sha256_file(Path(name)) for name in selected_files}
        combined.source_sha256 = hashes
        model = estimate_background(combined, accepted_intervals, background_cfg)
        interval_metrics: list[dict[str, Any]] = []
        interval_lambda: list[np.ndarray] = []
        for candidate_id, interval, part in zip(accepted_ids, accepted_intervals, accepted_parts):
            metric = estimate_interval_noise(part, model.dpf_quantum, background_cfg)
            metric.update({
                "candidate_id": candidate_id,
                "start_utc": interval.start_utc,
                "stop_utc": interval.stop_utc,
                "selection_source": interval.source,
            })
            interval_lambda.append(np.asarray(metric["lambda_pam"], dtype=float))
            interval_metrics.append(metric)

        accepted_records = int(sum(int(item["kept_records"]) for item in interval_metrics))
        distinct_days = len({str(item["day"]) for item in interval_metrics})
        inference_cfg = config["monthly_inference"]
        inference_eligible = (
            len(interval_metrics) >= int(inference_cfg["min_accepted_intervals"])
            and accepted_records >= int(inference_cfg["min_accepted_records"])
            and distinct_days >= int(inference_cfg["min_distinct_days"])
        )
        class_counts = Counter(str(item["predicted_class"]) for item in classified)
        arrays_path = month_dir / "noise_arrays.npz"
        _atomic_npz(
            arrays_path,
            background_dpf=np.asarray(model.background_dpf, dtype=np.float32),
            dpf_quantum=np.asarray(model.dpf_quantum, dtype=np.float32),
            lambda_pam=np.asarray(model.background_lambda_count_equivalent[0], dtype=np.float32),
            quantum_valid_mask=np.asarray(model.quantum_valid_mask, dtype=np.uint8),
            quantum_reconstruction_valid_fraction=np.asarray(
                model.quantum_reconstruction_valid_fraction, dtype=np.float32
            ),
            interval_lambda_pam=(
                np.stack(interval_lambda).astype(np.float32)
                if interval_lambda else np.empty((0, 4, 16, 8), dtype=np.float32)
            ),
        )
        _atomic_csv(month_dir / "candidates.csv", [
            {
                key: item.get(key)
                for key in (
                    "candidate_id", "start_utc", "stop_utc", "records",
                    "median_total_h_def", "predicted_class", "accepted",
                    "final_start_utc", "final_stop_utc", "manual_override",
                    "decision_score", "classification_confidence",
                    "coherent_signal_rejected",
                )
            }
            for item in classified
        ])
        _atomic_csv(
            month_dir / "interval_metrics.csv",
            [_flatten_interval_metric(item) for item in interval_metrics],
        )
        summary = {
            "status": "complete",
            "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
            "analysis_status": EXPLORATORY_STATUS,
            "month": month,
            "created_utc": created,
            "finished_utc": datetime.now(UTC).isoformat(),
            "config_digest": digest,
            "month_input_signature": month_signature,
            "candidate_count": len(candidates),
            "classification_counts": dict(sorted(class_counts.items())),
            "accepted_interval_count": len(interval_metrics),
            "accepted_record_count": accepted_records,
            "distinct_day_count": distinct_days,
            "inference_eligible": inference_eligible,
            "model_valid_under_production_thresholds": bool(model.valid),
            "model_invalid_reasons": model.invalid_reasons,
            "selected_source_file_count": len(selected_files),
            "selected_source_sha256": hashes,
            "arrays_path": str(arrays_path.resolve()),
            "interval_metrics": interval_metrics,
            "candidates": classified,
            "production_model_written": False,
        }
        _atomic_json(summary_path, summary)
        return summary
    except Exception as exc:
        failure = {
            "status": "failed",
            "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
            "analysis_status": EXPLORATORY_STATUS,
            "month": month,
            "created_utc": created,
            "finished_utc": datetime.now(UTC).isoformat(),
            "config_digest": digest,
            "month_input_signature": month_signature,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(month_dir / "failure.json", failure)
        raise


def run_months_parallel(
    monthly_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    inventory: Mapping[str, Any],
    quality_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
    manual_candidates_path: Path | None,
    *,
    workers: int = 1,
    resume: bool = True,
    months: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run independent month checkpoints sequentially or in deterministic parallel."""

    selected_months = sorted(months or monthly_candidates.keys())
    ori_all = list(inventory["ori_files"])
    tasks: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for month in selected_months:
        start = datetime.strptime(month + "01", "%Y%m%d").replace(tzinfo=UTC)
        if month.endswith("12"):
            next_month = datetime(start.year + 1, 1, 1, tzinfo=UTC)
        else:
            next_month = datetime(start.year, start.month + 1, 1, tzinfo=UTC)
        relevant_ori = [
            row for row in ori_all
            if float(row["start_unix_s"]) < next_month.timestamp()
            and float(row["stop_unix_s"]) >= start.timestamp()
        ]
        tasks.append((month, list(monthly_candidates.get(month, [])), relevant_ori))

    if int(workers) <= 1:
        sequential: list[dict[str, Any]] = []
        for month, candidates, ori in tasks:
            try:
                sequential.append(process_month_to_directory(
                    month, candidates, ori, quality_root, output_root, config,
                    manual_candidates_path, resume=resume,
                ))
            except Exception as exc:
                sequential.append({
                    "month": month,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
        return sequential
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        future_map = {
            pool.submit(
                process_month_to_directory,
                month, candidates, ori, quality_root, output_root, config,
                manual_candidates_path, resume=resume,
            ): month
            for month, candidates, ori in tasks
        }
        for future in as_completed(future_map):
            month = future_map[future]
            try:
                results[month] = future.result()
            except Exception as exc:
                results[month] = {
                    "month": month,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
    return [results[month] for month in selected_months]


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values with NaN preservation."""

    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return result
    order = finite_indices[np.argsort(values[finite_indices], kind="stable")]
    ranked = values[order] * finite_indices.size / np.arange(1, order.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def block_bootstrap_median(
    day_values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Median and 95% CI from resampling independent day-level values."""

    values = np.asarray(day_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    center = float(np.nanmedian(values))
    if values.size == 1 or int(replicates) <= 0:
        return center, center, center
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(int(replicates), values.size), replace=True)
    medians = np.nanmedian(samples, axis=1)
    low, high = np.nanquantile(medians, [0.025, 0.975])
    return center, float(low), float(high)


def permutation_median_pvalue(
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> float:
    """Two-sided day-level permutation test for a rank-location shift.

    Monthly point estimates remain day medians, but the test statistic uses
    the mean pooled rank.  A literal difference-of-medians permutation test
    has almost no power when a calibrated quantity is quantized to only one
    or two repeated values per stable period.
    """

    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return math.nan
    pooled = np.concatenate([a, b])
    ranks = rankdata(pooled, method="average")
    observed = abs(float(np.mean(ranks[a.size:]) - np.mean(ranks[:a.size])))
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(int(replicates)):
        permuted = rng.permutation(ranks)
        difference = abs(float(np.mean(permuted[a.size:]) - np.mean(permuted[:a.size])))
        exceed += int(difference >= observed - 1.0e-15)
    return float((exceed + 1) / (int(replicates) + 1))


def _month_midpoint(month: str) -> datetime:
    start = datetime.strptime(month + "01", "%Y%m%d").replace(tzinfo=UTC)
    if start.month == 12:
        stop = datetime(start.year + 1, 1, 1, tzinfo=UTC)
    else:
        stop = datetime(start.year, start.month + 1, 1, tzinfo=UTC)
    return start + (stop - start) / 2


def _metric_seed(base_seed: int, *tokens: str) -> int:
    digest = hashlib.sha256("|".join(tokens).encode("utf-8")).digest()
    return int((int(base_seed) + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


def _daily_metric_values(summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    flat = [_flatten_interval_metric(item) for item in summary.get("interval_metrics", [])]
    if not flat:
        return {}
    frame = pd.DataFrame(flat)
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    grouped = frame.groupby("day", sort=True)[numeric].median(numeric_only=True)
    return {column: grouped[column].to_numpy(dtype=float) for column in grouped.columns}


def _segment_cost(prefix: np.ndarray, prefix2: np.ndarray, start: int, stop: int) -> float:
    n = stop - start
    if n <= 0:
        return math.inf
    total = prefix[stop] - prefix[start]
    total2 = prefix2[stop] - prefix2[start]
    return float(max(total2 - total * total / n, 0.0))


def _bic_segments(values: Sequence[float], minimum_segment_months: int) -> list[tuple[int, int]]:
    """Piecewise-constant dynamic programming with a BIC-style penalty."""

    raw = np.asarray(values, dtype=float)
    minimum = int(minimum_segment_months)
    if raw.size < minimum:
        return []
    if raw.size < 2 * minimum:
        return [(0, raw.size)]
    positive = raw[np.isfinite(raw) & (raw > 0.0)]
    if positive.size != raw.size:
        transformed = np.log1p(np.maximum(raw, 0.0))
    else:
        transformed = np.log(raw)
    variance = float(np.nanvar(transformed, ddof=1)) if transformed.size > 1 else 0.0
    if not np.isfinite(variance) or variance <= np.finfo(float).eps:
        return [(0, raw.size)]
    penalty = variance * math.log(raw.size) * 2.0
    prefix = np.concatenate([[0.0], np.cumsum(transformed)])
    prefix2 = np.concatenate([[0.0], np.cumsum(transformed**2)])
    n = raw.size
    dp = np.full(n + 1, np.inf)
    previous = np.full(n + 1, -1, dtype=int)
    dp[0] = -penalty
    for stop in range(minimum, n + 1):
        for start in range(0, stop - minimum + 1):
            if start != 0 and start < minimum:
                continue
            if start != 0 and previous[start] < 0:
                continue
            score = dp[start] + _segment_cost(prefix, prefix2, start, stop) + penalty
            if score < dp[stop]:
                dp[stop] = score
                previous[stop] = start
    if previous[n] < 0:
        return [(0, n)]
    segments: list[tuple[int, int]] = []
    stop = n
    while stop > 0:
        start = int(previous[stop])
        segments.append((start, stop))
        stop = start
    return list(reversed(segments))


def _coverage_blocks(months: Sequence[str], gap_days: float) -> list[list[int]]:
    if not months:
        return []
    blocks: list[list[int]] = [[0]]
    for index in range(1, len(months)):
        gap = (_month_midpoint(months[index]) - _month_midpoint(months[index - 1])).days
        if gap > float(gap_days):
            blocks.append([index])
        else:
            blocks[-1].append(index)
    return blocks


def analyze_temporal_noise(
    monthly_summaries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize monthly noise and test persistent changes using day blocks."""

    stats_cfg = config["statistics"]
    bootstrap_n = int(stats_cfg["bootstrap_replicates"])
    permutation_n = int(stats_cfg["permutation_replicates"])
    base_seed = int(stats_cfg["random_seed"])
    practical = float(stats_cfg["practical_relative_change"])
    alpha = float(stats_cfg["fdr_alpha"])
    metrics = [
        "lambda_global",
        *(f"lambda_mass_{index}" for index in range(8)),
        *(f"quantum_def_mass_{index}" for index in range(8)),
        *(f"background_density_cm3_mass_{index}" for index in range(8)),
    ]
    month_rows: list[dict[str, Any]] = []
    daily_by_month: dict[str, dict[str, np.ndarray]] = {}
    for summary in sorted(monthly_summaries, key=lambda item: str(item["month"])):
        month = str(summary["month"])
        daily = _daily_metric_values(summary)
        daily_by_month[month] = daily
        row: dict[str, Any] = {
            "month": month,
            "month_midpoint_utc": _month_midpoint(month).isoformat(),
            "candidate_count": int(summary.get("candidate_count", 0)),
            "accepted_interval_count": int(summary.get("accepted_interval_count", 0)),
            "accepted_record_count": int(summary.get("accepted_record_count", 0)),
            "distinct_day_count": int(summary.get("distinct_day_count", 0)),
            "inference_eligible": bool(summary.get("inference_eligible", False)),
            "analysis_class": (
                "eligible_exploratory_auto"
                if bool(summary.get("inference_eligible", False))
                else "insufficient_for_inference"
            ),
        }
        for metric in metrics:
            values = daily.get(metric, np.empty(0))
            center, low, high = block_bootstrap_median(
                values,
                replicates=bootstrap_n,
                seed=_metric_seed(base_seed, month, metric, "bootstrap"),
            )
            row[metric] = center
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        # Monthly engineering medians are diagnostics only.
        for prefix, count in (("start_stop_median", 4), ("high_voltage_median_v", 20), ("solar_angle_median_deg", 2)):
            for index in range(count):
                name = f"{prefix}_{index}"
                values = daily.get(name, np.empty(0))
                row[name] = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else math.nan
        month_rows.append(row)

    eligible_rows = [row for row in month_rows if bool(row["inference_eligible"])]
    eligible_months = [str(row["month"]) for row in eligible_rows]
    comparisons: list[dict[str, Any]] = []
    stable_segments: list[dict[str, Any]] = []
    change_metrics = [
        "lambda_global", "lambda_mass_0", "lambda_mass_3", "lambda_mass_5",
        "quantum_def_mass_0", "quantum_def_mass_3", "quantum_def_mass_5",
    ]
    for metric in change_metrics:
        values = [float(row[metric]) for row in eligible_rows]
        finite_indices = [index for index, value in enumerate(values) if np.isfinite(value)]
        finite_months = [eligible_months[index] for index in finite_indices]
        finite_values = [values[index] for index in finite_indices]
        for block_number, block in enumerate(
            _coverage_blocks(finite_months, float(stats_cfg["coverage_gap_days"])), start=1
        ):
            block_months = [finite_months[index] for index in block]
            block_values = [finite_values[index] for index in block]
            segments = _bic_segments(block_values, int(stats_cfg["minimum_segment_months"]))
            segment_rows: list[dict[str, Any]] = []
            for segment_number, (start, stop) in enumerate(segments, start=1):
                segment = {
                    "metric": metric,
                    "coverage_block": block_number,
                    "segment": segment_number,
                    "start_month": block_months[start],
                    "stop_month": block_months[stop - 1],
                    "month_count": stop - start,
                    "median": float(np.nanmedian(block_values[start:stop])),
                    "start_index": start,
                    "stop_index": stop,
                }
                segment_rows.append(segment)
                stable_segments.append({key: value for key, value in segment.items() if not key.endswith("_index")})
            for left, right in zip(segment_rows[:-1], segment_rows[1:]):
                left_months = block_months[int(left["start_index"]):int(left["stop_index"])]
                right_months = block_months[int(right["start_index"]):int(right["stop_index"])]
                left_days = np.concatenate([
                    daily_by_month[month].get(metric, np.empty(0)) for month in left_months
                ])
                right_days = np.concatenate([
                    daily_by_month[month].get(metric, np.empty(0)) for month in right_months
                ])
                left_median = float(np.nanmedian(left_days))
                right_median = float(np.nanmedian(right_days))
                if np.isfinite(left_median) and left_median != 0.0:
                    relative = abs(right_median / left_median - 1.0)
                else:
                    relative = math.inf if right_median != left_median else 0.0
                comparisons.append({
                    "metric": metric,
                    "domain": "count_equivalent" if metric.startswith("lambda") else "calibration_scale",
                    "coverage_block": block_number,
                    "left_start_month": left_months[0],
                    "left_stop_month": left_months[-1],
                    "right_start_month": right_months[0],
                    "right_stop_month": right_months[-1],
                    "left_day_count": int(np.count_nonzero(np.isfinite(left_days))),
                    "right_day_count": int(np.count_nonzero(np.isfinite(right_days))),
                    "left_median": left_median,
                    "right_median": right_median,
                    "relative_change": float(relative),
                    "p_value": permutation_median_pvalue(
                        left_days,
                        right_days,
                        replicates=permutation_n,
                        seed=_metric_seed(base_seed, metric, left_months[-1], right_months[0], "permutation"),
                    ),
                })
    q_values = benjamini_hochberg([float(item["p_value"]) for item in comparisons])
    for item, q_value in zip(comparisons, q_values):
        item["q_value"] = float(q_value) if np.isfinite(q_value) else math.nan
        if not np.isfinite(q_value):
            conclusion = "insufficient"
        elif q_value < alpha and float(item["relative_change"]) >= practical:
            conclusion = "practically_significant"
        elif q_value < alpha:
            conclusion = "statistically_small"
        else:
            conclusion = "no_evidence"
        item["conclusion"] = conclusion

    # Match count-equivalent and quantum changes at the same right-segment month.
    by_boundary: defaultdict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for item in comparisons:
        species = "global" if item["metric"] == "lambda_global" else str(item["metric"]).rsplit("_", 1)[-1]
        key = (species, str(item["right_start_month"]))
        if item["conclusion"] == "practically_significant":
            by_boundary[key][str(item["domain"])] = str(item["metric"])
    interpreted_changes = []
    for (species, boundary), domains in sorted(by_boundary.items()):
        if "count_equivalent" in domains and "calibration_scale" in domains:
            interpretation = "mixed_count_and_calibration_change"
        elif "count_equivalent" in domains:
            interpretation = "count_equivalent_noise_change"
        else:
            interpretation = "calibration_scale_change_only"
        interpreted_changes.append({
            "species_or_mass": species,
            "boundary_month": boundary,
            "interpretation": interpretation,
            "supporting_metrics": domains,
        })

    # Engineering correlations use eligible monthly medians and are non-causal diagnostics.
    engineering_names = [
        *(f"start_stop_median_{index}" for index in range(4)),
        *(f"high_voltage_median_v_{index}" for index in range(20)),
        *(f"solar_angle_median_deg_{index}" for index in range(2)),
    ]
    correlations: list[dict[str, Any]] = []
    y = np.asarray([float(row["lambda_global"]) for row in eligible_rows])
    for name in engineering_names:
        x = np.asarray([float(row[name]) for row in eligible_rows])
        valid = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid) < 5 or np.unique(x[valid]).size < 2:
            rho, p_value = math.nan, math.nan
        else:
            rho, p_value = spearmanr(x[valid], y[valid])
        correlations.append({
            "engineering_metric": name,
            "noise_metric": "lambda_global",
            "month_count": int(np.count_nonzero(valid)),
            "spearman_rho": float(rho) if np.isfinite(rho) else math.nan,
            "p_value": float(p_value) if np.isfinite(p_value) else math.nan,
        })
    correlation_q = benjamini_hochberg([float(item["p_value"]) for item in correlations])
    for item, q_value in zip(correlations, correlation_q):
        item["q_value"] = float(q_value) if np.isfinite(q_value) else math.nan
        item["interpretation"] = "diagnostic_not_causal"

    return {
        "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
        "analysis_status": EXPLORATORY_STATUS,
        "created_utc": datetime.now(UTC).isoformat(),
        "month_count": len(month_rows),
        "eligible_month_count": len(eligible_rows),
        "monthly": month_rows,
        "stable_segments": stable_segments,
        "change_comparisons": comparisons,
        "interpreted_changes": interpreted_changes,
        "engineering_correlations": correlations,
        "significance_rule": {
            "fdr_alpha": alpha,
            "practical_relative_change": practical,
            "unit_of_replication": "day",
            "bootstrap_replicates": bootstrap_n,
            "permutation_replicates": permutation_n,
        },
        "causal_warning": (
            "Automatic quiet-window classification outside the 2021 manual reference "
            "has no cross-year human audit; temporal associations must not be interpreted "
            "as instrument aging without independent calibration evidence."
        ),
    }


def _quarter_label(month: str) -> str:
    return f"{month[:4]}Q{(int(month[4:6]) - 1) // 3 + 1}"


def _quarter_midpoint(quarter: str) -> datetime:
    year = int(quarter[:4])
    number = int(quarter[-1])
    start_month = 1 + (number - 1) * 3
    start = datetime(year, start_month, 1, tzinfo=UTC)
    stop = (
        datetime(year + 1, 1, 1, tzinfo=UTC)
        if number == 4 else datetime(year, start_month + 3, 1, tzinfo=UTC)
    )
    return start + (stop - start) / 2


def aggregate_quarterly_summaries(
    monthly_summaries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Pool accepted interval diagnostics by calendar quarter."""

    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for summary in monthly_summaries:
        grouped[_quarter_label(str(summary["month"]))].append(summary)
    inference = config["monthly_inference"]
    rows: list[dict[str, Any]] = []
    for quarter, members in sorted(grouped.items()):
        metrics = [
            metric
            for member in sorted(members, key=lambda item: str(item["month"]))
            for metric in member.get("interval_metrics", [])
        ]
        accepted_records = int(sum(int(item["kept_records"]) for item in metrics))
        distinct_days = len({str(item["day"]) for item in metrics})
        eligible = (
            len(metrics) >= int(inference["min_accepted_intervals"])
            and accepted_records >= int(inference["min_accepted_records"])
            and distinct_days >= int(inference["min_distinct_days"])
        )
        rows.append({
            "quarter": quarter,
            "source_months": [str(item["month"]) for item in members],
            "candidate_count": int(sum(int(item.get("candidate_count", 0)) for item in members)),
            "accepted_interval_count": len(metrics),
            "accepted_record_count": accepted_records,
            "distinct_day_count": distinct_days,
            "inference_eligible": eligible,
            "interval_metrics": metrics,
        })
    return rows


def select_quarter_review_intervals(
    monthly_summaries: Sequence[Mapping[str, Any]],
    quarter: str,
    *,
    count: int = 5,
    quantiles: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> list[dict[str, Any]]:
    """Select deterministic lambda-stratified review intervals on distinct days."""

    targets = tuple(float(value) for value in quantiles)
    if len(targets) != int(count):
        raise ValueError("quantile count must equal requested interval count")
    rows: list[dict[str, Any]] = []
    for summary in monthly_summaries:
        if _quarter_label(str(summary["month"])) != str(quarter):
            continue
        candidates = {
            str(item["candidate_id"]): item
            for item in summary.get("candidates", [])
            if bool(item.get("accepted", False))
        }
        for metric in summary.get("interval_metrics", []):
            candidate_id = str(metric["candidate_id"])
            candidate = candidates.get(candidate_id)
            value = float(metric.get("lambda_global", math.nan))
            if candidate is not None and np.isfinite(value):
                rows.append({**candidate, **metric, "lambda_global": value})
    if len(rows) < int(count):
        raise ValueError(f"{quarter}: only {len(rows)} finite accepted intervals")
    rows.sort(key=lambda item: (float(item["lambda_global"]), str(item["candidate_id"])))
    available_days = {str(item["day"]) for item in rows}
    require_distinct_days = len(available_days) >= int(count)
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_days: set[str] = set()
    for target in targets:
        target_index = target * (len(rows) - 1)
        ranked = sorted(
            enumerate(rows),
            key=lambda pair: (
                str(pair[1]["day"]) in used_days if require_distinct_days else False,
                abs(pair[0] - target_index),
                str(pair[1]["candidate_id"]),
            ),
        )
        choice = next(item for _, item in ranked if str(item["candidate_id"]) not in used_ids)
        selected.append({
            **choice,
            "review_target_quantile": target,
            "review_order": len(selected) + 1,
        })
        used_ids.add(str(choice["candidate_id"]))
        used_days.add(str(choice["day"]))
    return selected


def _quarter_fake_month_map(
    quarterly: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
    gap_days: float,
) -> dict[str, str]:
    """Map quarters to synthetic adjacent months while preserving real gaps."""

    days_by_quarter: defaultdict[str, list[datetime]] = defaultdict(list)
    for item in inventory.get("day_spe_files", []):
        if int(item.get("mode1_segment_count", 0)) <= 0:
            continue
        day = datetime.strptime(str(item["day"]), "%Y%m%d").replace(tzinfo=UTC)
        days_by_quarter[_quarter_label(day.strftime("%Y%m"))].append(day)
    fake_year, fake_month = 2000, 1
    mapping: dict[str, str] = {}
    previous: str | None = None
    for item in quarterly:
        quarter = str(item["quarter"])
        if previous is not None:
            prior_days = sorted(days_by_quarter.get(previous, []))
            current_days = sorted(days_by_quarter.get(quarter, []))
            actual_gap = (
                (current_days[0] - prior_days[-1]).days
                if prior_days and current_days else math.inf
            )
            fake_month += 1 if actual_gap <= float(gap_days) else 3
            while fake_month > 12:
                fake_month -= 12
                fake_year += 1
        mapping[quarter] = f"{fake_year:04d}{fake_month:02d}"
        previous = quarter
    return mapping


def analyze_quarterly_noise(
    monthly_summaries: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run temporal inference on pooled calendar-quarter day samples."""

    quarterly = aggregate_quarterly_summaries(monthly_summaries, config)
    fake_map = _quarter_fake_month_map(
        quarterly, inventory, float(config["statistics"]["coverage_gap_days"])
    )
    reverse = {value: key for key, value in fake_map.items()}
    transformed = [{**item, "month": fake_map[str(item["quarter"])]} for item in quarterly]
    base = analyze_temporal_noise(transformed, config)
    quarter_rows = []
    for source_row in base["monthly"]:
        row = dict(source_row)
        quarter = reverse[str(row.pop("month"))]
        row.pop("month_midpoint_utc", None)
        quarter_rows.append({
            "quarter": quarter,
            "quarter_midpoint_utc": _quarter_midpoint(quarter).isoformat(),
            **row,
        })
    stable = []
    for source in base["stable_segments"]:
        item = dict(source)
        item["start_quarter"] = reverse[str(item.pop("start_month"))]
        item["stop_quarter"] = reverse[str(item.pop("stop_month"))]
        stable.append(item)
    comparisons = []
    for source in base["change_comparisons"]:
        item = dict(source)
        for side in ("left_start", "left_stop", "right_start", "right_stop"):
            item[f"{side}_quarter"] = reverse[str(item.pop(f"{side}_month"))]
        comparisons.append(item)
    interpreted = []
    for source in base["interpreted_changes"]:
        item = dict(source)
        item["boundary_quarter"] = reverse[str(item.pop("boundary_month"))]
        interpreted.append(item)
    eligible_quarters = [item for item in quarterly if bool(item["inference_eligible"])]
    eligible_fake = [fake_map[str(item["quarter"])] for item in eligible_quarters]
    block_by_fake: dict[str, int] = {}
    for block_number, block in enumerate(
        _coverage_blocks(eligible_fake, float(config["statistics"]["coverage_gap_days"])),
        start=1,
    ):
        for index in block:
            block_by_fake[eligible_fake[index]] = block_number
    contrast_metrics = [
        "lambda_global", "lambda_mass_0", "lambda_mass_3", "lambda_mass_5",
        "quantum_def_mass_0", "quantum_def_mass_3", "quantum_def_mass_5",
    ]
    pairwise: list[dict[str, Any]] = []
    for left_index in range(len(eligible_quarters)):
        for right_index in range(left_index + 1, len(eligible_quarters)):
            left = eligible_quarters[left_index]
            right = eligible_quarters[right_index]
            left_daily = _daily_metric_values(left)
            right_daily = _daily_metric_values(right)
            left_quarter, right_quarter = str(left["quarter"]), str(right["quarter"])
            for metric in contrast_metrics:
                a = left_daily.get(metric, np.empty(0))
                b = right_daily.get(metric, np.empty(0))
                left_median = float(np.nanmedian(a))
                right_median = float(np.nanmedian(b))
                if np.isfinite(left_median) and left_median != 0.0:
                    signed_relative = right_median / left_median - 1.0
                    relative = abs(signed_relative)
                elif right_median == left_median:
                    signed_relative, relative = 0.0, 0.0
                else:
                    signed_relative = math.copysign(math.inf, right_median - left_median)
                    relative = math.inf
                pairwise.append({
                    "left_quarter": left_quarter,
                    "right_quarter": right_quarter,
                    "metric": metric,
                    "domain": "count_equivalent" if metric.startswith("lambda") else "calibration_scale",
                    "left_day_count": int(np.count_nonzero(np.isfinite(a))),
                    "right_day_count": int(np.count_nonzero(np.isfinite(b))),
                    "left_median": left_median,
                    "right_median": right_median,
                    "signed_relative_change": float(signed_relative),
                    "relative_change": float(relative),
                    "p_value": permutation_median_pvalue(
                        a, b,
                        replicates=int(config["statistics"]["permutation_replicates"]),
                        seed=_metric_seed(
                            int(config["statistics"]["random_seed"]),
                            "quarter_pairwise", left_quarter, right_quarter, metric,
                        ),
                    ),
                    "same_continuous_coverage_block": (
                        block_by_fake.get(fake_map[left_quarter])
                        == block_by_fake.get(fake_map[right_quarter])
                    ),
                    "temporal_interpretation": "pairwise_period_contrast_not_change_point",
                })
    q_values = benjamini_hochberg([float(item["p_value"]) for item in pairwise])
    alpha = float(config["statistics"]["fdr_alpha"])
    practical = float(config["statistics"]["practical_relative_change"])
    for item, q_value in zip(pairwise, q_values):
        item["q_value"] = float(q_value) if np.isfinite(q_value) else math.nan
        if not np.isfinite(q_value):
            conclusion = "insufficient"
        elif q_value < alpha and float(item["relative_change"]) >= practical:
            conclusion = "practically_significant"
        elif q_value < alpha:
            conclusion = "statistically_small"
        else:
            conclusion = "no_evidence"
        item["conclusion"] = conclusion
    result = {
        **base,
        "analysis_period": "calendar_quarter",
        "quarter_count": len(quarter_rows),
        "eligible_quarter_count": int(sum(bool(row["inference_eligible"]) for row in quarter_rows)),
        "quarterly": quarter_rows,
        "stable_segments": stable,
        "change_comparisons": comparisons,
        "eligible_quarter_pairwise_contrasts": pairwise,
        "interpreted_changes": interpreted,
        "quarter_mapping_for_gap_analysis": fake_map,
        "quarterly_inference_note": (
            "Intervals are pooled within each calendar quarter and reduced to "
            "independent day medians before bootstrap and permutation tests. "
            "Pairwise contrasts across coverage gaps compare period distributions "
            "but are not interpreted as continuous change points."
        ),
    }
    result.pop("monthly", None)
    result.pop("month_count", None)
    result.pop("eligible_month_count", None)
    return result


def write_quarterly_analysis_outputs(
    output_root: Path,
    result: Mapping[str, Any],
) -> dict[str, str]:
    """Write quarterly tables and diagnostic figures."""

    root = Path(output_root)
    data_dir, figure_dir = root / "data", root / "figures"
    path = data_dir / "quarterly_noise_analysis.json"
    _atomic_json(path, result)
    _atomic_csv(data_dir / "quarterly_noise_summary.csv", result["quarterly"])
    _atomic_csv(data_dir / "quarterly_change_comparisons.csv", result["change_comparisons"])
    _atomic_csv(
        data_dir / "quarterly_pairwise_contrasts.csv",
        result["eligible_quarter_pairwise_contrasts"],
    )
    rows = list(result["quarterly"])
    dates = np.asarray([_quarter_midpoint(str(row["quarter"])) for row in rows])
    labels = [str(row["quarter"]) for row in rows]
    eligible = np.asarray([bool(row["inference_eligible"]) for row in rows])

    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for index, label, color in (
        (0, "H+", "#0072b2"), (3, "O+", "#d55e00"), (5, "O2+", "#009e73")
    ):
        values = np.asarray([float(row[f"lambda_mass_{index}"]) for row in rows])
        low = np.asarray([float(row[f"lambda_mass_{index}_ci_low"]) for row in rows])
        high = np.asarray([float(row[f"lambda_mass_{index}_ci_high"]) for row in rows])
        axis.scatter(dates, values, s=26, color=color, alpha=0.38, label=f"{label} descriptive")
        eligible_values = values[eligible]
        eligible_low = low[eligible]
        eligible_high = high[eligible]
        axis.errorbar(
            dates[eligible], eligible_values,
            yerr=np.vstack([
                np.maximum(eligible_values - eligible_low, 0.0),
                np.maximum(eligible_high - eligible_values, 0.0),
            ]),
            fmt="o", ms=6, mfc="white", mec=color, ecolor=color,
            capsize=3, lw=1.2, label=f"{label} inference eligible",
        )
    axis.set_ylabel("quarterly count-equivalent background rate $\\lambda$")
    axis.set_title(
        "Exploratory MINPA Mode 1 quarterly background noise\n"
        "points are not joined across coverage gaps; error bars are day-bootstrap 95% CI"
    )
    axis.set_xticks(dates)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.legend()
    _save_figure(fig, figure_dir / "quarterly_noise_lambda_h_o_o2")

    fig, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    accepted = np.asarray([int(row["accepted_interval_count"]) for row in rows])
    axis.bar(dates, accepted, width=55, color=np.where(eligible, "#009e73", "#aaaaaa"))
    axis.axhline(20, color="black", ls="--", lw=1.0, label="20-interval threshold")
    axis.set_ylabel("accepted intervals")
    axis.set_title("Quarterly inference coverage (green = all thresholds passed)")
    axis.set_xticks(dates)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.legend()
    _save_figure(fig, figure_dir / "quarterly_inference_coverage")
    return {
        "analysis_json": str(path.resolve()),
        "quarterly_csv": str((data_dir / "quarterly_noise_summary.csv").resolve()),
        "figure_directory": str(figure_dir.resolve()),
    }


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_temporal_analysis_outputs(
    output_root: Path,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Write summary tables and all exploratory temporal-noise figures."""

    root = Path(output_root)
    data_dir, figure_dir = root / "data", root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_path = data_dir / "temporal_noise_analysis.json"
    _atomic_json(result_path, result)
    _atomic_csv(data_dir / "monthly_noise_summary.csv", result["monthly"])
    _atomic_csv(data_dir / "change_comparisons.csv", result["change_comparisons"])
    _atomic_csv(data_dir / "stable_segments.csv", result["stable_segments"])
    _atomic_csv(data_dir / "engineering_correlations.csv", result["engineering_correlations"])

    rows = list(result["monthly"])
    dates = [_month_midpoint(str(row["month"])) for row in rows]
    eligible = np.asarray([bool(row["inference_eligible"]) for row in rows])
    gap_days = float(config["statistics"]["coverage_gap_days"])

    fig, axis = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    candidates = np.asarray([int(row["candidate_count"]) for row in rows])
    accepted = np.asarray([int(row["accepted_interval_count"]) for row in rows])
    axis.bar(dates, candidates, width=20, color="#b9d6f2", label="monthly candidates")
    axis.bar(dates, accepted, width=12, color="#1f77b4", label="accepted quiet intervals")
    axis.scatter(
        np.asarray(dates, dtype=object)[~eligible], accepted[~eligible],
        marker="x", color="#d55e00", s=35, label="insufficient for inference", zorder=5,
    )
    axis.set_ylabel("interval count")
    axis.set_title("Exploratory automatic MINPA Mode-1 background coverage")
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", rotation=45)
    axis.legend(ncol=3, fontsize=8)
    _save_figure(fig, figure_dir / "coverage_and_monthly_selection")

    species_metrics = [
        ("lambda_mass_0", "H+", "#0072b2"),
        ("lambda_mass_3", "O+", "#d55e00"),
        ("lambda_mass_5", "O2+", "#009e73"),
    ]
    fig, axis = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    for metric, label, color in species_metrics:
        values = np.asarray([float(row[metric]) for row in rows])
        low = np.asarray([float(row[f"{metric}_ci_low"]) for row in rows])
        high = np.asarray([float(row[f"{metric}_ci_high"]) for row in rows])
        axis.errorbar(
            dates, values, yerr=np.vstack([values - low, high - values]),
            fmt="none", ecolor=color, alpha=0.25, lw=0.8,
        )
        # Draw each coverage block separately so long data gaps are explicit.
        first = True
        for block in _coverage_blocks([str(row["month"]) for row in rows], gap_days):
            axis.plot(
                [dates[index] for index in block], values[block],
                marker="o", ms=3.2, lw=1.2, color=color,
                label=label if first else None,
            )
            first = False
    for change in result["change_comparisons"]:
        if (
            str(change["metric"]).startswith("lambda")
            and change["conclusion"] == "practically_significant"
        ):
            axis.axvline(
                _month_midpoint(str(change["right_start_month"])),
                color="#555555", ls=":", lw=0.8, alpha=0.55,
            )
    axis.set_ylabel("count-equivalent background $\\lambda$")
    axis.set_title(
        "Exploratory automatic MINPA Mode-1 monthly noise (day-block 95% CI)"
    )
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", rotation=45)
    axis.legend(ncol=3)
    _save_figure(fig, figure_dir / "monthly_noise_lambda_h_o_o2")

    fig, axis = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    for index, label, color in ((0, "H+", "#0072b2"), (3, "O+", "#d55e00"), (5, "O2+", "#009e73")):
        metric = f"quantum_def_mass_{index}"
        values = np.asarray([float(row[metric]) for row in rows])
        first = True
        for block in _coverage_blocks([str(row["month"]) for row in rows], gap_days):
            axis.plot(
                [dates[item] for item in block], values[block],
                marker="o", ms=3.2, lw=1.2, color=color,
                label=label if first else None,
            )
            first = False
    axis.set_yscale("log")
    axis.set_ylabel("median quantum × E [DEF units]")
    axis.set_title("Exploratory MINPA Mode-1 quantization/calibration scale")
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", rotation=45)
    axis.legend(ncol=3)
    _save_figure(fig, figure_dir / "monthly_quantum_scale_h_o_o2")

    lambda_grid = np.asarray([
        [float(row[f"lambda_mass_{index}"]) for row in rows]
        for index in range(8)
    ])
    fig, axis = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    finite_positive = lambda_grid[np.isfinite(lambda_grid) & (lambda_grid > 0.0)]
    if finite_positive.size:
        vmin, vmax = np.nanquantile(finite_positive, [0.02, 0.98])
    else:
        vmin, vmax = 1.0e-3, 1.0
    mesh = axis.imshow(
        lambda_grid,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=(-0.5, len(rows) - 0.5, -0.5, 7.5),
        vmin=vmin,
        vmax=max(vmax, vmin * 1.01),
        cmap="viridis",
    )
    axis.set_yticks(range(8), [f"{mass:g} amu" for mass in MODE1_MASS_AMU])
    tick_index = np.arange(0, len(rows), max(1, len(rows) // 10))
    axis.set_xticks(tick_index, [str(rows[index]["month"]) for index in tick_index], rotation=45)
    axis.set_title("Exploratory monthly count-equivalent noise across all 8 mass channels")
    axis.set_xlabel("month")
    fig.colorbar(mesh, ax=axis, label="median $\\lambda$")
    _save_figure(fig, figure_dir / "monthly_mass_channel_noise_heatmap")

    global_segments = [
        item for item in result["stable_segments"] if item["metric"] == "lambda_global"
    ]
    fig, axis = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    global_values = np.asarray([float(row["lambda_global"]) for row in rows])
    axis.scatter(dates, global_values, s=22, color="#333333", label="monthly median")
    for segment in global_segments:
        start = _month_midpoint(str(segment["start_month"]))
        stop = _month_midpoint(str(segment["stop_month"]))
        axis.hlines(
            float(segment["median"]), start, stop,
            colors="#cc79a7", lw=3.0,
        )
    axis.set_ylabel("global count-equivalent $\\lambda$")
    axis.set_title("Exploratory stable periods from BIC change-point analysis")
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    _save_figure(fig, figure_dir / "global_noise_change_points")

    correlations = [
        item for item in result["engineering_correlations"]
        if np.isfinite(float(item["spearman_rho"]))
    ]
    correlations = sorted(correlations, key=lambda item: abs(float(item["spearman_rho"])), reverse=True)[:15]
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if correlations:
        labels = [str(item["engineering_metric"]) for item in correlations][::-1]
        values = [float(item["spearman_rho"]) for item in correlations][::-1]
        colors = ["#d55e00" if float(item["q_value"]) < 0.05 else "#999999" for item in correlations][::-1]
        axis.barh(labels, values, color=colors)
    axis.axvline(0.0, color="black", lw=0.8)
    axis.set_xlabel("Spearman ρ with monthly global λ")
    axis.set_title("Engineering correlations (diagnostic only; not causal)")
    _save_figure(fig, figure_dir / "engineering_noise_correlations")

    return {
        "analysis_json": str(result_path.resolve()),
        "monthly_csv": str((data_dir / "monthly_noise_summary.csv").resolve()),
        "figure_directory": str(figure_dir.resolve()),
    }


def _benchmark_one_ori(
    work: tuple[Mapping[str, Any], str | None],
) -> dict[str, Any]:
    item, quality_root_text = work
    quality_root = Path(quality_root_text) if quality_root_text else None
    time_values, data, flags, available = read_mode1_selected_dpf_channels(
        [Path(str(item["path"]))],
        _utc_text(float(item["start_unix_s"])),
        _utc_text(float(item["stop_unix_s"]) + 1.0),
        [(0, 0, 0, 0), (20, 2, 8, 3), (39, 3, 15, 5)],
        quality_root=quality_root,
    )
    finite = np.isfinite(data)
    rejected = available & (
        (flags & np.uint32(PROJECT_BACKGROUND_REJECT_MASK)) != 0
    )
    return {
        "path": str(item["path"]),
        "source_size_bytes": int(item["size_bytes"]),
        "source_mtime_ns": int(item["mtime_ns"]),
        "record_count": int(time_values.size),
        "finite_count": int(np.count_nonzero(finite)),
        "value_sum": float(np.nansum(data)),
        "first_time": float(time_values[0]) if time_values.size else None,
        "last_time": float(time_values[-1]) if time_values.size else None,
        "flag_count": int(flags.size),
        "quality_available_count": int(np.count_nonzero(available)),
        "quality_rejected_count": int(np.count_nonzero(rejected)),
        "quality_flag_sha256": hashlib.sha256(
            np.asarray(flags, dtype="<u4").tobytes()
        ).hexdigest(),
    }


def compare_month_output_directories(
    reference_root: Path,
    comparison_root: Path,
    months: Sequence[str],
) -> dict[str, Any]:
    """Verify deterministic month outputs from two execution strategies."""

    ignored_summary_keys = {
        "created_utc", "finished_utc", "arrays_path", "lambda_pooling_upgrade_utc",
    }
    month_rows: list[dict[str, Any]] = []
    for month in months:
        left_dir = Path(reference_root) / "months" / str(month)
        right_dir = Path(comparison_root) / "months" / str(month)
        left_summary = json.loads((left_dir / "summary.json").read_text(encoding="utf-8"))
        right_summary = json.loads((right_dir / "summary.json").read_text(encoding="utf-8"))
        left_summary = {
            key: value for key, value in left_summary.items()
            if key not in ignored_summary_keys
        }
        right_summary = {
            key: value for key, value in right_summary.items()
            if key not in ignored_summary_keys
        }
        table_equal = {
            name: (left_dir / name).read_bytes() == (right_dir / name).read_bytes()
            for name in ("candidates.csv", "interval_metrics.csv")
        }
        with np.load(left_dir / "noise_arrays.npz") as left_npz, np.load(
            right_dir / "noise_arrays.npz"
        ) as right_npz:
            arrays_equal = left_npz.files == right_npz.files and all(
                left_npz[name].shape == right_npz[name].shape
                and left_npz[name].dtype == right_npz[name].dtype
                and np.array_equal(left_npz[name], right_npz[name], equal_nan=True)
                for name in left_npz.files
            )
        row = {
            "month": str(month),
            "summary_equal_ignoring_runtime_fields": left_summary == right_summary,
            "candidates_csv_equal": table_equal["candidates.csv"],
            "interval_metrics_csv_equal": table_equal["interval_metrics.csv"],
            "arrays_equal": bool(arrays_equal),
        }
        row["identical"] = all(value for key, value in row.items() if key != "month")
        month_rows.append(row)
    return {
        "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "reference_root": str(Path(reference_root).resolve()),
        "comparison_root": str(Path(comparison_root).resolve()),
        "months": month_rows,
        "identical": all(bool(row["identical"]) for row in month_rows),
    }


def benchmark_ori_files(
    inventory: Mapping[str, Any],
    *,
    file_count: int = 100,
    workers: int = 4,
    quality_root: Path | None = None,
) -> dict[str, Any]:
    """Compare values, quality masks, and metadata for stratified ori files."""

    rows = list(inventory["ori_files"])
    if not rows:
        raise ValueError("inventory contains no Mode-1 ori files")
    count = min(int(file_count), len(rows))
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    selected = [rows[index] for index in indices]
    work = [
        (item, str(Path(quality_root).resolve()) if quality_root is not None else None)
        for item in selected
    ]
    start = time.perf_counter()
    single = [_benchmark_one_ori(item) for item in work]
    single_seconds = time.perf_counter() - start
    start = time.perf_counter()
    if int(workers) <= 1:
        parallel = [_benchmark_one_ori(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            parallel = list(pool.map(_benchmark_one_ori, work))
    parallel_seconds = time.perf_counter() - start
    identical = single == parallel
    return {
        "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "file_count": count,
        "workers": int(workers),
        "quality_root": str(Path(quality_root).resolve()) if quality_root is not None else None,
        "single_seconds": single_seconds,
        "parallel_seconds": parallel_seconds,
        "speedup": single_seconds / parallel_seconds if parallel_seconds > 0 else math.nan,
        "identical": identical,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "single_results": single,
        "parallel_results": parallel,
    }
