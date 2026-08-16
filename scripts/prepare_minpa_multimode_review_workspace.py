"""Discover and render the MINPA Mode-4/12 v2.0.0-rc1 review workspace."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import sha256_file  # noqa: E402
from highE.minpa_io import inspect_ori_science_layout, ori_file_coverage  # noqa: E402
from highE.minpa_modes import mode_layout  # noqa: E402
from highE.minpa_multimode_background import MULTIMODE_REVIEW_VERSION  # noqa: E402
from highE.minpa_multimode_review import (  # noqa: E402
    PROJECT_REJECT_MASK,
    SPECIES_CONTAINER,
    collect_eligible_minute_windows,
    select_and_merge_low_tail,
    validate_native_quality,
)


DEFAULT_CONFIG = ROOT / "config" / "minpa_multimode_noise_review_v2.0.0-rc1.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "minpa_multimode_noise_review_v2.0.0-rc1"
SPECIES_TOKEN = {"H+": "Hplus", "O+": "Oplus", "O2+": "O2plus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day-spe-root", type=Path, required=True)
    parser.add_argument("--ori-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--benchmark-files", type=int, default=100)
    parser.add_argument(
        "--stage",
        choices=("all", "discover", "render"),
        default="all",
        help="Render resumes from the discovery checkpoint and skips existing PNGs.",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_chunk(paths: Sequence[str]) -> tuple[list[Any], list[dict[str, Any]]]:
    return collect_eligible_minute_windows([Path(value) for value in paths])


def _chunks(values: Sequence[Path], count: int) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(max(1, count))]
    for index, value in enumerate(values):
        buckets[index % len(buckets)].append(str(value))
    return [bucket for bucket in buckets if bucket]


def discover_parallel(paths: Sequence[Path], workers: int) -> tuple[list[Any], list[dict[str, Any]]]:
    if workers <= 1:
        return collect_eligible_minute_windows(paths)
    windows: list[Any] = []
    excluded: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for rows, diagnostics in executor.map(_discover_chunk, _chunks(paths, workers)):
            windows.extend(rows)
            excluded.extend(diagnostics)
    windows.sort(key=lambda row: (row.start_unix_s, row.mode, row.species, row.score))
    excluded.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return windows, excluded


def _window_signature(rows: Sequence[Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            row.mode,
            row.species,
            row.month,
            row.start_unix_s,
            row.stop_unix_s,
            row.score,
            row.record_count,
            row.source_day_spe,
            row.source_segments,
        )
        for row in rows
    )


def benchmark_discovery(paths: Sequence[Path], workers: int, count: int) -> dict[str, Any]:
    sample = list(paths[: min(count, len(paths))])
    started = time.perf_counter()
    serial_rows, serial_excluded = discover_parallel(sample, 1)
    serial_s = time.perf_counter() - started
    started = time.perf_counter()
    parallel_rows, parallel_excluded = discover_parallel(sample, workers)
    parallel_s = time.perf_counter() - started
    identical = (
        _window_signature(serial_rows) == _window_signature(parallel_rows)
        and sorted(json.dumps(row, sort_keys=True) for row in serial_excluded)
        == sorted(json.dumps(row, sort_keys=True) for row in parallel_excluded)
    )
    if not identical:
        raise RuntimeError("Serial and parallel candidate discovery are not identical")
    return {
        "file_count": len(sample),
        "workers": workers,
        "serial_seconds": serial_s,
        "parallel_seconds": parallel_s,
        "speedup": serial_s / parallel_s if parallel_s else None,
        "outputs_identical": True,
        "parallel_selected": bool(workers > 1 and parallel_s < serial_s),
    }


def _segment_names(container: Any) -> list[str]:
    return list(getattr(container, "_fieldnames", []) or [])


@lru_cache(maxsize=4)
def _load_day_spe_cached(path_text: str) -> Mapping[str, Any]:
    return loadmat(Path(path_text), squeeze_me=True, struct_as_record=False)


def _segment_arrays(item: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_values = np.asarray(item.t, dtype=float).reshape(-1)
    energy = np.asarray(item.f, dtype=float).reshape(-1)
    spectrum = np.asarray(item.p, dtype=float)
    if spectrum.shape == (energy.size, time_values.size):
        spectrum = spectrum.T
    flag = np.asarray(getattr(item, "quality_flag", []), dtype=np.uint32).reshape(-1)
    available = np.asarray(getattr(item, "quality_flag_available", []), dtype=bool).reshape(-1)
    return time_values, energy, spectrum, flag, available


def load_candidate_context(
    candidate: Mapping[str, Any], context_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = int(candidate["mode"])
    species = str(candidate["species"])
    start_s = float(candidate["start_unix_s"])
    stop_s = float(candidate["stop_unix_s"])
    times: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    interval_spectra: list[np.ndarray] = []
    expected_energy = mode_layout(mode).energy_eV
    selected_segments = {
        str(value) for value in candidate["source_day_spe_segments"]
    }
    for source_text in candidate["source_day_spe"]:
        source = Path(source_text)
        payload = _load_day_spe_cached(str(source.resolve()))
        container = payload[SPECIES_CONTAINER[species]]
        for name in _segment_names(container):
            item = getattr(container, name)
            item_mode = int(np.asarray(getattr(item, "mod", -1)).reshape(-1)[0])
            if item_mode != mode:
                continue
            t, energy, spectrum, flag, available = _segment_arrays(item)
            if (
                spectrum.shape != (t.size, expected_energy.size)
                or flag.size != t.size
                or available.size != t.size
                or not np.allclose(energy, expected_energy, rtol=1e-5, atol=1e-5)
            ):
                if name in selected_segments:
                    raise ValueError(
                        f"Invalid selected day_spe layout in {source} {name} {species}"
                    )
                continue
            good = available & ((flag & PROJECT_REJECT_MASK) == 0)
            context = (t >= start_s - context_seconds) & (t <= stop_s + context_seconds)
            if np.any(context):
                part = spectrum[context].astype(float, copy=True)
                part[~good[context]] = np.nan
                times.append(t[context])
                spectra.append(part)
            inside = (t >= start_s) & (t < stop_s) & good
            if np.any(inside):
                interval_spectra.append(spectrum[inside].astype(float, copy=False))
    if not times or not interval_spectra:
        raise ValueError(f"No quality-clean spectrum for {candidate['candidate_id']}")
    t = np.concatenate(times)
    p = np.concatenate(spectra, axis=0)
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]
    unique = np.ones(t.size, dtype=bool)
    unique[1:] = np.diff(t) != 0.0
    t, p = t[unique], p[unique]
    interval = np.concatenate(interval_spectra, axis=0)
    finite = np.isfinite(interval)
    mean = np.divide(
        np.where(finite, interval, 0.0).sum(axis=0),
        finite.sum(axis=0),
        out=np.full(expected_energy.size, np.nan),
        where=finite.sum(axis=0) > 0,
    )
    return t, expected_energy, p, mean


def color_limits(candidates: Sequence[Mapping[str, Any]], context_seconds: float) -> dict[str, dict[str, Any]]:
    del context_seconds
    samples: dict[tuple[int, str], list[np.ndarray]] = {}
    combinations_by_source: dict[
        str, dict[tuple[int, str], set[str]]
    ] = {}
    for candidate in candidates:
        key = (int(candidate["mode"]), str(candidate["species"]))
        for source in candidate["source_day_spe"]:
            combinations_by_source.setdefault(str(source), {}).setdefault(
                key, set()
            ).update(str(value) for value in candidate["source_day_spe_segments"])
    for source_text, combinations in sorted(combinations_by_source.items()):
        payload = _load_day_spe_cached(str(Path(source_text).resolve()))
        for (mode, species), selected_segments in sorted(combinations.items()):
            container = payload[SPECIES_CONTAINER[species]]
            for name in _segment_names(container):
                if name not in selected_segments:
                    continue
                item = getattr(container, name)
                item_mode = int(np.asarray(getattr(item, "mod", -1)).reshape(-1)[0])
                if item_mode != mode:
                    continue
                _, _, spectrum, flag, available = _segment_arrays(item)
                if flag.size != spectrum.shape[0] or available.size != spectrum.shape[0]:
                    raise ValueError(f"Invalid quality shape in {source_text} {name}")
                good = available & ((flag & PROJECT_REJECT_MASK) == 0)
                positive = spectrum[good]
                positive = positive[np.isfinite(positive) & (positive > 0.0)]
                if positive.size > 20_000:
                    positive = positive[
                        np.linspace(0, positive.size - 1, 20_000, dtype=int)
                    ]
                if positive.size:
                    samples.setdefault((mode, species), []).append(positive)
    result: dict[str, dict[str, Any]] = {}
    for (mode, species), parts in sorted(samples.items()):
        values = np.concatenate(parts)
        low, high = np.quantile(values, [0.01, 0.997])
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= low:
            raise ValueError(f"Cannot establish LogNorm for Mode {mode} {species}")
        result[f"mode{mode:02d}_{SPECIES_TOKEN[species]}"] = {
            "vmin": float(low),
            "vmax": float(high),
            "positive_sample_count": int(values.size),
            "quantiles": [0.01, 0.997],
            "sampling": "quality-clean positive source-segment values represented by selected candidates; deterministic linspace cap 20000 per segment",
        }
    return result


def _render_job(job: tuple[dict[str, Any], str, dict[str, Any], float]) -> str:
    candidate, output_text, scale, context_seconds = job
    output = Path(output_text)
    render_candidate(candidate, output, scale, context_seconds)
    return output.name


def render_candidate(
    candidate: Mapping[str, Any],
    output_path: Path,
    scale: Mapping[str, float],
    context_seconds: float,
) -> None:
    t, energy, spectrum, mean = load_candidate_context(candidate, context_seconds)
    times = [datetime.fromtimestamp(value, UTC) for value in t]
    start = datetime.fromtimestamp(float(candidate["start_unix_s"]), UTC)
    stop = datetime.fromtimestamp(float(candidate["stop_unix_s"]), UTC)
    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [3.35, 1.0]}, constrained_layout=True
    )
    image = axes[0].pcolormesh(
        times,
        energy,
        np.where(spectrum > 0.0, spectrum, np.nan).T,
        shading="nearest",
        norm=LogNorm(vmin=float(scale["vmin"]), vmax=float(scale["vmax"])),
        cmap="viridis",
        rasterized=True,
    )
    for boundary in (start, stop):
        axes[0].axvline(boundary, color="black", linewidth=1.4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].set_xlabel("UTC")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    axes[0].set_title("Raw energy-time spectrum (bad-quality records blank)")
    colorbar = fig.colorbar(image, ax=axes[0], pad=0.015)
    colorbar.set_label("DPF [1/(s cm² sr eV)]")
    positive = np.isfinite(mean) & (mean > 0.0)
    if np.any(positive):
        axes[1].plot(mean[positive], energy[positive], color="#0072B2", marker="o", markersize=2.8)
        x_min = min(float(scale["vmin"]), float(np.nanmin(mean[positive])))
        x_max = max(float(scale["vmax"]), float(np.nanmax(mean[positive])))
    else:
        x_min, x_max = float(scale["vmin"]), float(scale["vmax"])
        axes[1].text(
            0.5,
            0.5,
            "No positive candidate samples",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].set_xlabel("Mean DPF")
    axes[1].set_title("Candidate mean spectrum")
    duration = float(candidate["duration_s"])
    prefilter_text = (
        f"  |  prefilter score={float(candidate['decision_score']):.3f}"
        if "decision_score" in candidate
        else ""
    )
    fig.suptitle(
        f"Mode {int(candidate['mode'])} · {candidate['species']} · {candidate['candidate_id']}\n"
        f"{candidate['start_utc']} — {candidate['stop_utc']}  |  {duration / 60:.1f} min  |  "
        f"day_spe n={int(candidate['day_spe_record_count'])}, native n={int(candidate['native_record_count'])}  |  "
        f"low-tail q={float(candidate['selection_quantile']):.2f}, cutoff={float(candidate['selection_score_cutoff']):.4g}"
        f"{prefilter_text}",
        fontsize=10.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=145, metadata={"Software": MULTIMODE_REVIEW_VERSION})
    plt.close(fig)
    temporary.replace(output_path)


def _file_provenance(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for path_text in sorted(set(paths)):
        path = Path(path_text)
        stat = path.stat()
        rows.append({
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": _sha256(path),
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for key, value in list(row.items()):
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
    temporary.replace(path)


def discover(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    years = tuple(str(value) for value in config["coverage_years"])
    day_paths = sorted(
        path for path in args.day_spe_root.rglob("Ion_spe_*.mat")
        if any(f"Ion_spe_{year}" in path.name for year in years)
    )
    if not day_paths:
        raise FileNotFoundError(f"No day_spe files under {args.day_spe_root}")
    benchmark = benchmark_discovery(day_paths, args.workers, args.benchmark_files)
    workers = args.workers if benchmark["parallel_selected"] else 1
    windows, day_excluded = discover_parallel(day_paths, workers)
    candidates, thresholds = select_and_merge_low_tail(
        windows, quantile=float(config["selection"]["quantile"])
    )
    ori_catalog = []
    for path in sorted(args.ori_root.rglob("*.mat")):
        try:
            mode, start_s, stop_s = ori_file_coverage(path)
        except ValueError:
            continue
        if mode in (4, 12) and any(
            datetime.fromtimestamp(start_s, UTC).strftime("%Y") == year for year in years
        ):
            ori_catalog.append((path, mode, start_s, stop_s))
    accepted, native_rejected = validate_native_quality(candidates, ori_catalog)
    layout_cache: dict[str, dict[str, int]] = {}
    for candidate in accepted:
        for path_text in candidate["source_ori"]:
            if path_text not in layout_cache:
                layout_cache[path_text] = inspect_ori_science_layout(path_text)
            audit = layout_cache[path_text]
            if audit["inspected_raw_record_count"] == 0 or audit["invalid_raw_record_count"]:
                raise RuntimeError(
                    f"Selected candidate {candidate['candidate_id']} failed ori layout audit: {audit}"
                )
        candidate["ori_layout_audit"] = [
            layout_cache[path_text] for path_text in candidate["source_ori"]
        ]
    checkpoint = {
        "review_version": MULTIMODE_REVIEW_VERSION,
        "analysis_status": "provisional_review_only",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "benchmark": benchmark,
        "full_discovery_workers": workers,
        "inventory": {
            "day_spe_file_count": len(day_paths),
            "ori_file_count": len(ori_catalog),
            "eligible_minute_count": len(windows),
            "pre_native_candidate_count": len(candidates),
            "candidate_count": len(accepted),
            "native_rejected_candidate_count": len(native_rejected),
            "day_spe_ineligible_segment_count": len(day_excluded),
        },
        "thresholds": thresholds,
        "candidates": accepted,
        "native_rejected_candidates": native_rejected,
        "day_spe_ineligible_segments": day_excluded,
    }
    _atomic_json(args.output_root / "checkpoints" / "discovery.json", checkpoint)
    return checkpoint


def publish(args: argparse.Namespace, config: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    candidates = [dict(row) for row in checkpoint["candidates"]]
    pending = args.output_root / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    nested = [path for path in pending.iterdir() if path.is_dir()]
    if nested:
        raise RuntimeError(f"Pending directory must remain flat: {nested[:5]}")
    context_seconds = float(config["plot"]["context_minutes"]) * 60.0
    scale_path = args.output_root / "checkpoints" / "color_limits.json"
    if scale_path.exists():
        scales = json.loads(scale_path.read_text(encoding="utf-8"))
    else:
        scales = color_limits(candidates, context_seconds)
        _atomic_json(scale_path, scales)
    expected_names = {str(row["plot_filename"]) for row in candidates}
    for temporary in pending.glob("*.tmp.png"):
        intended = temporary.name.removesuffix(".tmp.png") + ".png"
        if intended not in expected_names:
            raise RuntimeError(f"Unexpected temporary PNG: {temporary.name}")
        temporary.unlink()
    extra = sorted(path.name for path in pending.glob("*.png") if path.name not in expected_names)
    if extra:
        raise RuntimeError(f"Unexpected PNG prevents resumable render: {extra[:10]}")
    missing_jobs = []
    existing_count = 0
    for candidate in candidates:
        output_path = pending / str(candidate["plot_filename"])
        key = f"mode{int(candidate['mode']):02d}_{SPECIES_TOKEN[str(candidate['species'])]}"
        if output_path.exists():
            existing_count += 1
        else:
            missing_jobs.append((candidate, str(output_path), scales[key], context_seconds))
    if existing_count:
        print(f"resume_existing_png={existing_count}", flush=True)
    if args.workers > 1 and missing_jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, _ in enumerate(
                executor.map(_render_job, missing_jobs, chunksize=8), start=1
            ):
                total = existing_count + index
                if total % 100 == 0 or index == len(missing_jobs):
                    print(f"rendered_or_verified={total}/{len(candidates)}", flush=True)
    else:
        for index, job in enumerate(missing_jobs, start=1):
            _render_job(job)
            total = existing_count + index
            if total % 100 == 0 or index == len(missing_jobs):
                print(f"rendered_or_verified={total}/{len(candidates)}", flush=True)
    missing = sorted(expected_names - {path.name for path in pending.glob("*.png")})
    if missing:
        raise RuntimeError(f"Cannot publish incomplete workspace; missing {len(missing)} PNGs")
    for candidate in candidates:
        plot = pending / str(candidate["plot_filename"])
        candidate["plot_path"] = str(plot.resolve())
        candidate["plot_sha256"] = sha256_file(plot)
        candidate["manual_decision"] = ""
        candidate["manual_adjusted_start_utc"] = ""
        candidate["manual_adjusted_stop_utc"] = ""
    source_day = [value for row in candidates for value in row["source_day_spe"]]
    source_ori = [value for row in candidates for value in row["source_ori"]]
    counts = {
        f"mode{mode:02d}_{SPECIES_TOKEN[species]}": sum(
            int(row["mode"]) == mode and row["species"] == species for row in candidates
        )
        for mode in (4, 12) for species in ("H+", "O+", "O2+")
    }
    index_payload = {
        "review_version": MULTIMODE_REVIEW_VERSION,
        "analysis_status": "provisional_review_only",
        "review_status": "pending",
        "approval_rule": "Each PNG is one independent mode/species interval; retained=approved, deleted=rejected.",
        "candidate_count": len(candidates),
        "counts_by_mode_species": counts,
        "dimension_order": ["energy", "pitch", "azimuth", "mass"],
        "units": "1/(s cm^2 sr eV)",
        "candidates": candidates,
    }
    _atomic_json(args.output_root / "candidate_index.json", index_payload)
    _write_csv(args.output_root / "candidate_index.csv", candidates)
    _write_csv(args.output_root / "manual_decisions.csv", candidates)
    manifest = {
        "review_version": MULTIMODE_REVIEW_VERSION,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_status": "provisional_review_only",
        "review_status": "pending",
        "candidate_count": len(candidates),
        "counts_by_mode_species": counts,
        "flat_pending_directory": str(pending.resolve()),
        "selection": config["selection"],
        "quality": config["quality"],
        "plot": {**config["plot"], "color_limits": scales},
        "layouts": config["layouts"],
        "benchmark": checkpoint["benchmark"],
        "source_day_spe": _file_provenance(source_day),
        "source_ori": _file_provenance(source_ori),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
    }
    _atomic_json(args.output_root / "manifest.json", manifest)
    readme = f"""# MINPA Mode 4/12 noise review ({MULTIMODE_REVIEW_VERSION})

This is a provisional, species-independent manual-review workspace. It is not a production background model.

- Pending PNGs: `{pending.resolve()}` (flat; no subfolders)
- Candidate count: {len(candidates)}
- Keep a PNG to approve that one mode/species interval; delete it to reject the whole interval.
- A deletion never triggers automatic trimming or salvage, even when only part of the interval contains a real observation.
- H+, O+, and O2+ are reviewed independently.
- Spectra are raw released DPF in `1/(s cm^2 sr eV)`; there is no smoothing or interpolation.
- Black vertical lines mark the proposed interval; context is ±{config['plot']['context_minutes']} minutes.
- Do not rename PNGs or add other PNGs. The finalizer treats either as an error.

Formal candidates were published only after every plot rendered and every selected ori interval passed native Quality and raw-layout checks. See `manifest.json` and `candidate_index.json` for provenance and hashes.
"""
    temporary = (args.output_root / "README.md.tmp")
    temporary.write_text(readme, encoding="utf-8")
    temporary.replace(args.output_root / "README.md")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["review_version"] != MULTIMODE_REVIEW_VERSION:
        raise ValueError("Config/review version mismatch")
    checkpoint_path = args.output_root / "checkpoints" / "discovery.json"
    if args.stage in ("all", "discover"):
        checkpoint = discover(args, config)
        if args.stage == "discover":
            print(json.dumps(checkpoint["inventory"], indent=2))
            return
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError("Render stage requires checkpoints/discovery.json")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    publish(args, config, checkpoint)
    print(json.dumps({"candidate_count": len(checkpoint["candidates"]), "output": str(args.output_root.resolve())}, indent=2))


if __name__ == "__main__":
    main()
