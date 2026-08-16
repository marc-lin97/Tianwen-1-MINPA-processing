"""Find and render conservative MINPA real-signal candidates for v2.2.0.

The output is review-only.  It pairs the existing nine manually approved
noise examples with three observational strong-signal candidates for every
Mode 1/4/12 and H+/O+/O2+ combination.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from highE.minpa_background import (  # noqa: E402
    PROJECT_BACKGROUND_REJECT_MASK,
    read_mode1_ori_records,
    sha256_file,
)
from highE.minpa_io import (  # noqa: E402
    ori_file_coverage,
    ori_files_for_interval,
    read_ori_records,
)
from highE.minpa_modes import mode_layout, split_raw_record  # noqa: E402
from highE.minpa_multimode_background import apply_multimode_background  # noqa: E402
from highE.minpa_signal_examples import (  # noqa: E402
    extract_signal_prefilter_features,
    validate_native_channel_signal,
)
from highE.tw1_minpa import load_unified_static_background_bundle  # noqa: E402
from scripts.plot_minpa_denoise_examples_3modes_3species import (  # noqa: E402
    SPECIES,
    SPECIES_TOKEN,
    species_dpf_spectrum,
)


VERSION = "minpa-typical-noise-signal-review-v2.2.0"
CADENCE_S = {1: 16.4, 4: 12.3, 12: 2.05}
SPECIES_CONTAINER = {"H+": "H_spe_num", "O+": "O_spe_num", "O2+": "O2_spe_num"}
DEFAULT_BUNDLE = ROOT / "release/minpa_unified_static_channel_denoise_v2.2.0/bundle.json"
DEFAULT_NOISE = ROOT / "outputs/minpa_denoise_examples_3modes_3species_v2.2.0"
DEFAULT_OUTPUT = ROOT / "outputs/minpa_typical_noise_signal_review_v2.2.0"


def _utc(value: float) -> str:
    return datetime.fromtimestamp(float(value), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _segment_names(container: Any) -> list[str]:
    return list(getattr(container, "_fieldnames", []) or [])


def _segment_arrays(item: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(item.t, dtype=float).reshape(-1)
    energy = np.asarray(item.f, dtype=float).reshape(-1)
    spectrum = np.asarray(item.p, dtype=float)
    if spectrum.shape == (energy.size, time.size):
        spectrum = spectrum.T
    flag = np.asarray(getattr(item, "quality_flag", []), dtype=np.uint32).reshape(-1)
    available = np.asarray(getattr(item, "quality_flag_available", []), dtype=bool).reshape(-1)
    return time, energy, spectrum, flag, available


def scan_day_spe_file(path_text: str) -> list[dict[str, Any]]:
    """Return quality-clean fixed five-minute persistent-band candidates."""

    path = Path(path_text)
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    output: list[dict[str, Any]] = []
    for species in SPECIES:
        container_name = SPECIES_CONTAINER[species]
        if container_name not in payload:
            continue
        container = payload[container_name]
        for segment_name in _segment_names(container):
            item = getattr(container, segment_name)
            mode_values = np.asarray(getattr(item, "mod", np.nan)).reshape(-1)
            if mode_values.size != 1 or not np.isfinite(mode_values[0]):
                continue
            mode = int(mode_values[0])
            if mode not in (1, 4, 12):
                continue
            time, energy, spectrum, flag, available = _segment_arrays(item)
            expected_energy = mode_layout(mode).energy_eV
            if (
                spectrum.shape != (time.size, energy.size)
                or flag.size != time.size
                or available.size != time.size
                or energy.size != expected_energy.size
                or not np.allclose(energy, expected_energy, rtol=1e-5, atol=1e-5)
            ):
                continue
            buckets = np.floor(time / 300.0).astype(np.int64)
            minimum_records = int(math.ceil(0.8 * 300.0 / CADENCE_S[mode]))
            for bucket in np.unique(buckets):
                indices = np.flatnonzero(buckets == bucket)
                if indices.size < minimum_records:
                    continue
                if not np.all(available[indices]):
                    continue
                if np.any((flag[indices] & np.uint32(12)) != 0):
                    continue
                gaps = np.diff(time[indices])
                if gaps.size and float(np.nanmax(gaps)) > 1.5 * CADENCE_S[mode]:
                    continue
                values = spectrum[indices]
                try:
                    features = extract_signal_prefilter_features(values)
                except ValueError:
                    continue
                start = float(bucket * 300)
                output.append({
                    "mode": mode,
                    "species": species,
                    "start_unix_s": start,
                    "stop_unix_s": start + 300.0,
                    "start_utc": _utc(start),
                    "stop_utc": _utc(start + 300.0),
                    "duration_s": 300.0,
                    "day_spe_record_count": int(indices.size),
                    "source_day_spe": str(path.resolve()),
                    "source_day_spe_segment": segment_name,
                    "prefilter_features": features,
                })
    return output


def scan_day_spe(paths: Sequence[Path], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        parts = (scan_day_spe_file(str(path)) for path in paths)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        parts = pool.map(scan_day_spe_file, (str(path) for path in paths), chunksize=4)
    rows = [row for part in parts for row in part]
    if workers > 1:
        pool.shutdown()
    deduplicated: dict[tuple[int, str, float, float], dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        key = (
            int(row["mode"]), str(row["species"]),
            float(row["start_unix_s"]), float(row["stop_unix_s"]),
        )
        previous = deduplicated.get(key)
        if previous is None or float(row["prefilter_features"]["prefilter_score"]) > float(
            previous["prefilter_features"]["prefilter_score"]
        ):
            if previous is not None:
                row["alias_day_spe_segments"] = sorted(set(
                    previous.get("alias_day_spe_segments", [])
                    + [str(previous["source_day_spe_segment"])]
                ))
            deduplicated[key] = row
        elif previous is not None:
            previous["alias_day_spe_segments"] = sorted(set(
                previous.get("alias_day_spe_segments", [])
                + [str(row["source_day_spe_segment"])]
            ))
    grouped: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in deduplicated.values():
        grouped[(int(row["mode"]), str(row["species"]))].append(row)
    ranked: list[dict[str, Any]] = []
    for group, values in sorted(grouped.items()):
        log_total = np.log1p([
            float(row["prefilter_features"]["median_total_spectrum"])
            for row in values
        ])
        center = float(np.median(log_total))
        scale = float(np.median(np.abs(log_total - center)))
        scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0
        for row, total in zip(values, log_total):
            row["rank_score"] = float(
                row["prefilter_features"]["prefilter_score"]
                + 0.5 * (float(total) - center) / scale
            )
            row["group"] = f"mode{group[0]:02d}_{SPECIES_TOKEN[group[1]]}"
        ranked.extend(values)
    return sorted(
        ranked,
        key=lambda row: (
            int(row["mode"]), str(row["species"]),
            -float(row["rank_score"]), float(row["start_unix_s"]),
        ),
    )


def _build_ori_catalog(root: Path) -> list[tuple[Path, int, float, float]]:
    rows = []
    for path in sorted(root.rglob("*.mat")):
        try:
            mode, start, stop = ori_file_coverage(path)
        except ValueError:
            continue
        if mode in (1, 4, 12):
            rows.append((path, mode, start, stop))
    return rows


def _load_raw(
    mode: int,
    start: float,
    stop: float,
    catalog: Sequence[tuple[Path, int, float, float]],
    shape: tuple[int, ...],
    quality_root: Path,
    *,
    require_complete: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    paths = ori_files_for_interval(catalog, mode, start, stop)
    if not paths:
        raise FileNotFoundError(f"No Mode {mode} ori for {_utc(start)} -- {_utc(stop)}")
    if mode == 1:
        records = read_mode1_ori_records(
            paths, _utc(start), _utc(stop), quality_root=quality_root,
            compute_sha256=False,
        )
        keep = (
            records.project_quality_available
            & ((records.project_quality_flag & PROJECT_BACKGROUND_REJECT_MASK) == 0)
            & (records.native_quality == 0)
        )
        time = records.time_unix_s[keep]
        cubes = np.asarray(records.dpf[keep], dtype=float)
        sources = [Path(path) for path in records.source_files]
    else:
        by_time: dict[float, np.ndarray] = {}
        sources = []
        for path in paths:
            used = False
            for record in read_ori_records(path, start - 5.0, stop + 5.0):
                if int(record.mode) != mode or int(record.quality_native) != 0:
                    continue
                for product_time, flat in split_raw_record(mode, record.time_unix_s, record.ion_dpf):
                    if not start <= product_time < stop:
                        continue
                    cube = np.asarray(flat, dtype=float).reshape(shape)
                    previous = by_time.get(float(product_time))
                    if previous is not None and not np.array_equal(previous, cube, equal_nan=True):
                        raise ValueError(f"Conflicting duplicate at {product_time}")
                    by_time[float(product_time)] = cube
                    used = True
            if used:
                sources.append(path)
        if not by_time:
            raise ValueError("No native-quality-clean records")
        time = np.asarray(sorted(by_time), dtype=float)
        cubes = np.stack([by_time[value] for value in time])
    if require_complete:
        minimum = int(math.floor(0.8 * (stop - start) / CADENCE_S[mode]))
        if time.size < minimum:
            raise ValueError(f"Native coverage failed: {time.size} < {minimum}")
        if time.size > 1 and float(np.max(np.diff(time))) > 1.5 * CADENCE_S[mode]:
            raise ValueError("Native cadence gap failed")
    return time, cubes, sorted(set(sources))


def _apply(mode: int, raw: np.ndarray, mode1_model: Any, multimode: Mapping[int, Any]) -> Any:
    if mode == 1:
        from highE.minpa_background import apply_background
        return apply_background(raw, mode1_model)
    return apply_multimode_background(raw, multimode[mode])


def validate_candidates(
    ranked: Sequence[Mapping[str, Any]],
    catalog: Sequence[tuple[Path, int, float, float]],
    quality_root: Path,
    mode1_model: Any,
    multimode: Mapping[int, Any],
    *,
    trial_limit_per_group: int,
) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ranked:
        grouped[(int(row["mode"]), str(row["species"]))].append(row)
    passed: dict[tuple[int, str], list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for group in [(mode, species) for mode in (1, 4, 12) for species in SPECIES]:
        mode, species = group
        model = mode1_model if mode == 1 else multimode[mode]
        accepted: list[dict[str, Any]] = []
        separated: list[dict[str, Any]] = []
        print(f"validating Mode {mode} {species}", flush=True)
        for rank, source in enumerate(grouped.get(group, [])[:trial_limit_per_group], start=1):
            row = dict(source)
            try:
                time, raw, paths = _load_raw(
                    mode, float(row["start_unix_s"]), float(row["stop_unix_s"]),
                    catalog, tuple(model.background_dpf.shape), quality_root,
                )
                correction = _apply(mode, raw, mode1_model, multimode)
                raw_spectrum = species_dpf_spectrum(raw, mode, species)
                corrected_spectrum = species_dpf_spectrum(
                    correction.corrected_dpf, mode, species
                )
                result = validate_native_channel_signal(
                    raw, correction.corrected_dpf,
                    model.background_dpf, model.valid_channel_mask,
                    mode=mode, species=species,
                    raw_spectrum=raw_spectrum,
                    corrected_spectrum=corrected_spectrum,
                    minimum_signal_to_background=3.0,
                    minimum_peak_retention=0.65,
                )
                if result["passed"]:
                    ratio = float(result["maximum_median_channel_signal_to_background"])
                    retention = float(result["signal_peak_retention_fraction"])
                    result["strength_tier"] = (
                        "strong_20x" if ratio >= 20.0 and retention >= 0.95
                        else "clear_5x" if ratio >= 5.0 and retention >= 0.80
                        else "moderate_3x"
                    )
                reason = "passed" if result["passed"] else "native_signal_gate_failed"
            except (FileNotFoundError, KeyError, ValueError) as exc:
                result = {"passed": False, "error": str(exc)}
                paths, time = [], np.empty(0)
                reason = "read_or_quality_failed"
            audit.append({
                "mode": mode, "species": species, "prefilter_rank": rank,
                "start_utc": row["start_utc"], "stop_utc": row["stop_utc"],
                "reason": reason, "native_validation": result,
            })
            if not result.get("passed"):
                if rank % 10 == 0:
                    print(
                        f"  Mode {mode} {species}: tried={rank}, "
                        f"passed={len(accepted)}, separated={len(separated)}",
                        flush=True,
                    )
                continue
            row.update({
                "prefilter_rank": rank,
                "native_record_count": int(time.size),
                "source_ori": [str(path.resolve()) for path in paths],
                "native_validation": result,
            })
            accepted.append(row)
            if all(
                abs(float(row["start_unix_s"]) - float(other["start_unix_s"]))
                >= 7.0 * 86400.0
                for other in separated
            ):
                separated.append(row)
            print(
                f"  accepted rank={rank} at {row['start_utc']} "
                f"(separated={len(separated)})",
                flush=True,
            )
            if len(separated) == 3:
                break
        if len(separated) < 3:
            for row in accepted:
                if row not in separated:
                    separated.append(row)
                if len(separated) == 3:
                    break
        passed[group] = separated
    return passed, audit


def _finite_mean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    return np.divide(
        np.where(finite, values, 0.0).sum(axis=0), finite.sum(axis=0),
        out=np.full(values.shape[1], np.nan), where=finite.sum(axis=0) > 0,
    )


def _insert_spectrogram_gap_rows(
    time: np.ndarray, spectrum: np.ndarray, cadence_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN rows so irregular pcolormesh input leaves true gaps blank."""

    if time.size < 2:
        return time, spectrum
    output_time: list[float] = [float(time[0])]
    output_spectrum: list[np.ndarray] = [np.asarray(spectrum[0], dtype=float)]
    blank = np.full(spectrum.shape[1], np.nan, dtype=float)
    for index in range(1, time.size):
        previous, current = float(time[index - 1]), float(time[index])
        if current - previous > 1.5 * cadence_s:
            output_time.append(previous + cadence_s)
            output_spectrum.append(blank.copy())
            if current - cadence_s > previous + cadence_s:
                output_time.append(current - cadence_s)
                output_spectrum.append(blank.copy())
        output_time.append(current)
        output_spectrum.append(np.asarray(spectrum[index], dtype=float))
    return np.asarray(output_time, dtype=float), np.stack(output_spectrum)


def _render_signal(
    row: Mapping[str, Any],
    time: np.ndarray,
    raw: np.ndarray,
    corrected: np.ndarray,
    mode: int,
    species: str,
    output: Path,
    vmin: float,
    vmax: float,
) -> None:
    energy = mode_layout(mode).energy_eV
    raw_spectrum = species_dpf_spectrum(raw, mode, species)
    corrected_spectrum = species_dpf_spectrum(corrected, mode, species)
    plot_time, raw_plot = _insert_spectrogram_gap_rows(
        time, raw_spectrum, CADENCE_S[mode]
    )
    _, corrected_plot = _insert_spectrogram_gap_rows(
        time, corrected_spectrum, CADENCE_S[mode]
    )
    inside = (time >= float(row["start_unix_s"])) & (time < float(row["stop_unix_s"]))
    dates = [datetime.fromtimestamp(value, UTC) for value in plot_time]
    fig, axes = plt.subplots(
        1, 4, figsize=(17.0, 4.9), constrained_layout=True,
        gridspec_kw={"width_ratios": [2.25, 2.25, 1.05, 1.05]},
    )
    image = None
    for axis, spectrum, title in zip(
        axes[:2], (raw_plot, corrected_plot),
        ("Raw released DPF", "v2.2.0 channel-background corrected"),
    ):
        image = axis.pcolormesh(
            dates, energy, np.where(spectrum > 0.0, spectrum, np.nan).T,
            shading="nearest", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax),
            rasterized=True,
        )
        for boundary in (float(row["start_unix_s"]), float(row["stop_unix_s"])):
            axis.axvline(datetime.fromtimestamp(boundary, UTC), color="black", lw=1.3)
        axis.set_yscale("log")
        axis.set_xlabel("UTC")
        axis.set_title(title, fontsize=10)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    axes[0].set_ylabel("Energy (eV)")
    axes[1].tick_params(labelleft=False)
    colorbar = fig.colorbar(image, ax=axes[:2], pad=0.01, aspect=32)
    colorbar.set_label(r"DPF [s$^{-1}$ cm$^{-2}$ sr$^{-1}$ eV$^{-1}$]")
    raw_mean = _finite_mean(raw_spectrum[inside])
    corrected_mean = _finite_mean(corrected_spectrum[inside])
    for values, color, label in (
        (raw_mean, "#555555", "Raw"),
        (corrected_mean, "#0072B2", "Corrected"),
    ):
        valid = np.isfinite(values) & (values > 0.0)
        axes[2].plot(values[valid], energy[valid], color=color, lw=1.5, label=label)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Interval mean DPF")
    axes[2].set_title("Signal-window mean")
    axes[2].grid(True, which="both", alpha=0.22)
    axes[2].legend(fontsize=8)
    validation = row["native_validation"]
    ratio = np.asarray(validation["median_channel_ratio_by_energy"], dtype=float)
    occupancy = np.asarray(validation["occupancy_by_energy"], dtype=float)
    valid_ratio = np.isfinite(ratio) & (ratio > 0.0)
    axes[3].plot(ratio[valid_ratio], energy[valid_ratio], color="#D55E00", lw=1.5)
    axes[3].axvline(
        float(validation["minimum_signal_to_background"]),
        color="black", ls="--", lw=1.0,
    )
    axes[3].set_xscale("log")
    axes[3].xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    axes[3].xaxis.set_minor_formatter(NullFormatter())
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Median max channel / background")
    axes[3].set_title("Native-channel S/B")
    axes[3].grid(True, which="both", alpha=0.22)
    peak = int(validation["signal_peak_energy_index"])
    axes[3].text(
        0.03, 0.03,
        f"peak occupancy={occupancy[peak]:.0%}\ncontiguous E={validation['longest_contiguous_strong_energy_run']}",
        transform=axes[3].transAxes, fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.suptitle(
        f"MINPA Mode {mode} · {species} · observational signal candidate\n"
        f"{row['start_utc']} — {row['stop_utc']} · native n={row['native_record_count']} · "
        f"tier={validation['strength_tier']} · "
        f"median channel S/B max={validation['maximum_median_channel_signal_to_background']:.1f} · "
        f"peak retained={validation['signal_peak_retention_fraction']:.1%} · "
        f"ratio-peak shift={validation['ratio_peak_shift_channels']} channel(s)",
        fontsize=10.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=175, metadata={"Software": VERSION})
    plt.close(fig)
    temporary.replace(output)


def render_selected(
    selected: Mapping[tuple[int, str], Sequence[dict[str, Any]]],
    catalog: Sequence[tuple[Path, int, float, float]],
    quality_root: Path,
    mode1_model: Any,
    multimode: Mapping[int, Any],
    pending: Path,
) -> list[dict[str, Any]]:
    pending.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for mode in (1, 4, 12):
        for species in SPECIES:
            rows = list(selected[(mode, species)])
            contexts = []
            positive_parts = []
            model = mode1_model if mode == 1 else multimode[mode]
            for row in rows:
                time, raw, paths = _load_raw(
                    mode, float(row["start_unix_s"]) - 720.0,
                    float(row["stop_unix_s"]) + 720.0,
                    catalog, tuple(model.background_dpf.shape), quality_root,
                    require_complete=False,
                )
                correction = _apply(mode, raw, mode1_model, multimode)
                raw_spectrum = species_dpf_spectrum(raw, mode, species)
                corrected_spectrum = species_dpf_spectrum(correction.corrected_dpf, mode, species)
                for values in (raw_spectrum, corrected_spectrum):
                    positive = values[np.isfinite(values) & (values > 0.0)]
                    if positive.size:
                        positive_parts.append(positive)
                contexts.append((time, raw, correction.corrected_dpf, paths))
            positive = np.concatenate(positive_parts)
            vmin, vmax = np.quantile(positive, [0.01, 0.997])
            vmax = max(float(vmax), float(vmin) * 10.0)
            for index, (row, context) in enumerate(zip(rows, contexts), start=1):
                time, raw, corrected, context_paths = context
                start_token = datetime.fromtimestamp(float(row["start_unix_s"]), UTC).strftime("%Y%m%dT%H%M%S")
                filename = f"mode{mode:02d}_{SPECIES_TOKEN[species]}_signal{index:02d}_{start_token}.png"
                path = pending / filename
                _render_signal(row, time, raw, corrected, mode, species, path, float(vmin), float(vmax))
                item = dict(row)
                item.update({
                    "candidate_id": f"signal-m{mode:02d}-{SPECIES_TOKEN[species]}-{index:02d}",
                    "plot_filename": filename,
                    "plot_path": str(path.resolve()),
                    "plot_sha256": sha256_file(path),
                    "context_source_ori": [
                        {"path": str(source.resolve()), "sha256": sha256_file(source)}
                        for source in context_paths
                    ],
                    "color_limits": {"vmin": float(vmin), "vmax": float(vmax)},
                    "manual_decision": "",
                })
                output_rows.append(item)
                print(f"rendered Mode {mode} {species} signal candidate {index}", flush=True)
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day-spe-root", type=Path, default=Path(r"D:\Data\TW-1\result\MINPA\day_spe"))
    parser.add_argument("--ori-root", type=Path, default=Path(r"D:\Data\TW-1\result\MINPA\ori"))
    parser.add_argument("--quality-root", type=Path, default=ROOT / "outputs/tw1_minpa_quality_flags_all_species")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--noise-examples", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trial-limit-per-group", type=int, default=160)
    parser.add_argument(
        "--rerender-existing", action="store_true",
        help="Delete only this workspace's generated pending PNGs before rerendering.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.day_spe_root, args.ori_root, args.quality_root, args.bundle, args.noise_examples):
        if not path.exists():
            raise FileNotFoundError(path)
    day_paths = sorted(args.day_spe_root.rglob("Ion_spe_*.mat"))
    checkpoint_path = args.output / "checkpoints" / "day_spe_ranked_top500.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if int(checkpoint["day_spe_file_count"]) != len(day_paths):
            raise RuntimeError("day_spe checkpoint file count does not match current inventory")
        ranked = list(checkpoint["candidates"])
        quality_window_count = int(checkpoint["quality_window_count"])
        print(f"loaded day_spe checkpoint: top candidates={len(ranked)}", flush=True)
    else:
        all_ranked = scan_day_spe(day_paths, args.workers)
        quality_window_count = len(all_ranked)
        grouped_ranked: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in all_ranked:
            grouped_ranked[(int(row["mode"]), str(row["species"]))].append(row)
        ranked = [
            row
            for group in [(mode, species) for mode in (1, 4, 12) for species in SPECIES]
            for row in grouped_ranked[group][:500]
        ]
        _atomic_json(checkpoint_path, {
            "version": VERSION,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "day_spe_file_count": len(day_paths),
            "quality_window_count": quality_window_count,
            "retained_top_per_group": 500,
            "candidates": ranked,
        })
        print(
            f"quality-clean five-minute windows={quality_window_count}; "
            f"retained ranked candidates={len(ranked)}",
            flush=True,
        )
    catalog = _build_ori_catalog(args.ori_root)
    mode1_model, multimode = load_unified_static_background_bundle(args.bundle)
    selected, audit = validate_candidates(
        ranked, catalog, args.quality_root, mode1_model, multimode,
        trial_limit_per_group=args.trial_limit_per_group,
    )
    counts = {f"mode{mode:02d}_{SPECIES_TOKEN[species]}": len(selected[(mode, species)])
              for mode in (1, 4, 12) for species in SPECIES}
    if any(value != 3 for value in counts.values()):
        _atomic_json(args.output / "incomplete_validation_audit.json", {
            "version": VERSION, "counts": counts, "audit": audit,
        })
        raise RuntimeError(f"Could not identify three conservative candidates per group: {counts}")
    pending = args.output / "pending"
    if pending.exists() and any(pending.glob("*.png")):
        if not args.rerender_existing:
            raise RuntimeError("Pending directory already contains PNG files; refuse to overwrite review state")
        for path in pending.glob("*.png"):
            path.unlink()
    rendered = render_selected(selected, catalog, args.quality_root, mode1_model, multimode, pending)
    noise_manifest = json.loads((args.noise_examples / "examples_manifest.json").read_text(encoding="utf-8"))
    noise_lookup = {
        (int(row["mode"]), str(row["species"])): row for row in noise_manifest["examples"]
    }
    groups = []
    for mode in (1, 4, 12):
        for species in SPECIES:
            noise = noise_lookup[(mode, species)]
            groups.append({
                "mode": mode,
                "species": species,
                "noise_reference": {
                    "start_utc": noise["start_utc"], "stop_utc": noise["stop_utc"],
                    "figure": noise["figure"], "figure_sha256": noise["figure_sha256"],
                    "approval": "manually_approved_noise",
                },
                "signal_candidates": [
                    row for row in rendered if int(row["mode"]) == mode and row["species"] == species
                ],
            })
    manifest = {
        "review_version": VERSION,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "pending_manual_signal_review",
        "scope": "Mode 1/4/12 x H+/O+/O2+: one frozen approved-noise reference and three observational signal candidates per combination",
        "manual_rule": "Retain acceptable PNGs and delete rejected PNGs; final paired example requires exactly one retained signal PNG per mode/species group.",
        "signal_prefilter": {
            "window_seconds": 300,
            "day_spe_use": "all quality-clean windows retained for dimensionless persistent-band and total-flux ranking; no spectral hard gate",
            "retained_top_candidates_per_mode_species": 500,
        },
        "native_signal_gate": {
            "minimum_channel_signal_to_background": 3.0,
            "minimum_time_occupancy": 0.50,
            "minimum_contiguous_energy_channels": 2,
            "minimum_peak_retention": 0.65,
            "maximum_ratio_peak_shift_channels": 1,
            "strength_tiers": {
                "strong_20x": "median channel S/B >=20 and peak retention >=95%",
                "clear_5x": "median channel S/B >=5 and peak retention >=80%",
                "moderate_3x": "passes the >=3x, >=65% minimum gate",
            },
        },
        "quality": "project quality available and bits 3-4 clear; native Quality==0; >=80% cadence coverage; max gap <=1.5 cadence",
        "units": "native input and frozen background DPF: 1/(s cm^2 sr eV)",
        "coordinate_scope": "MINPA instrument energy/pitch/azimuth/mass channels; no MSO/MSE transform or orbit-region claim",
        "mars_context_not_used": "altitude, SZA, local time, bow shock/MPB and upstream propagation are intentionally not selection features for this instrument-response example set",
        "smoothing_or_interpolation": "none",
        "bundle": {"path": str(args.bundle.resolve()), "sha256": sha256_file(args.bundle)},
        "day_spe_inventory": {"file_count": len(day_paths)},
        "ori_inventory": {"file_count": len(catalog)},
        "quality_clean_five_minute_window_count": quality_window_count,
        "ranked_candidate_count_used_for_native_validation": len(ranked),
        "native_validation_trial_count": len(audit),
        "pending_png_count": len(rendered),
        "groups": groups,
        "native_validation_audit": audit,
    }
    _atomic_json(args.output / "review_index.json", manifest)
    incomplete = args.output / "incomplete_validation_audit.json"
    if incomplete.exists():
        incomplete.unlink()
    readme = (
        "# MINPA v2.2.0 典型噪声—真实信号配对审核\n\n"
        "`pending/` 中有 27 张真实信号候选图：三种模式、三种物种各 3 张。"
        "保留可接受图片、删除拒绝图片；最终每组应保留 1 张。\n\n"
        "每组典型噪声图沿用已人工批准的 v2.2.0 九组合示例，并在 `review_index.json` 中链接。"
        "拒绝区间没有被自动当作真实信号。day_spe 只用于无量纲谱形与总通量排序，3×背景、"
        "持续性、连续能道、峰值保留和峰位移动均从原始逐通道 DPF 复核。\n\n"
        "图中黑线为 5 分钟候选边界，前后各 12 分钟仅供上下文判断；无平滑、无插值。\n"
    )
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(f"pending={pending}")
    print(f"index={args.output / 'review_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
