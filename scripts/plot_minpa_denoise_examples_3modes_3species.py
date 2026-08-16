"""Render one approved raw/denoised MINPA example per mode and species.

The nine examples cover Mode 1, Mode 4, and Mode 12 crossed with H+, O+,
and O2+.  All modes use the frozen v2.2.0 all-approved, time-invariant
bundle.  Raw ``ori`` files are read-only.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import (  # noqa: E402
    PROJECT_BACKGROUND_REJECT_MASK,
    SUPPORT_LOW,
    SUPPORT_SUPPORTED,
    SUPPORT_UNSUPPORTED,
    apply_background,
    mode1_ori_files_for_interval,
    read_mode1_ori_records,
    sha256_file,
)
from highE.minpa_io import read_ori_records  # noqa: E402
from highE.minpa_modes import (  # noqa: E402
    angular_geometry,
    mode_layout,
    species_mass_indices,
    split_raw_record,
)
from highE.minpa_multimode_background import (  # noqa: E402
    apply_multimode_background,
)
from highE.tw1_minpa import load_unified_static_background_bundle  # noqa: E402


SPECIES = ("H+", "O+", "O2+")
SPECIES_TOKEN = {"H+": "Hplus", "O+": "Oplus", "O2+": "O2plus"}
MODE1_REVIEW = ROOT / "outputs/minpa_background_manual_review_workspace/indexes/final_manual_review_v2.json"
MULTIMODE_REVIEW = ROOT / "outputs/minpa_multimode_noise_review_secondary_prefilter_v2/manual_review_final.json"
UNIFIED_STATIC_BUNDLE = ROOT / "release/minpa_unified_static_channel_denoise_v2.2.0/bundle.json"
DEFAULT_OUTPUT = ROOT / "outputs/minpa_denoise_examples_3modes_3species_v2.2.0"
ARTIFACT_VERSION = "minpa-denoise-examples-v2.2.0"
DPF_UNITS = "1/(s cm^2 sr eV)"


def _parse_utc(value: str) -> float:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _robust_distance(values: np.ndarray) -> np.ndarray:
    """L1 distance from the componentwise median using a MAD scale."""

    median = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - median), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    normalized = np.abs((values - median) / scale)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.sum(axis=1)


def select_mode1_interval(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose a deterministic central interval by duration and mission time."""

    approved = [dict(row) for row in rows]
    if not approved:
        raise ValueError("Mode-1 review has no approved intervals")
    values = []
    for row in approved:
        duration = _parse_utc(str(row["stop_utc"])) - _parse_utc(str(row["start_utc"]))
        row["duration_s"] = duration
        values.append([duration, _parse_utc(str(row["start_utc"]))])
    distance = _robust_distance(np.asarray(values, dtype=float))
    order = sorted(
        range(len(approved)),
        key=lambda index: (distance[index], str(approved[index]["start_utc"]), str(approved[index]["candidate_key"])),
    )
    return approved[order[0]]


def select_multimode_intervals(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    """Choose the feature-space medoid of each approved mode/species group."""

    result: dict[tuple[int, str], dict[str, Any]] = {}
    for mode in (4, 12):
        for species in SPECIES:
            group = [
                dict(row)
                for row in rows
                if int(row["mode"]) == mode
                and str(row["species"]) == species
                and all(Path(str(path)).exists() for path in row["source_ori"])
            ]
            if not group:
                raise ValueError(f"No approved, locally readable Mode {mode} {species} interval")
            features = np.asarray(
                [
                    [
                        float(row["duration_s"]),
                        float(row.get("decision_score", np.nan)),
                        float(row.get("secondary_approved_probability", np.nan)),
                    ]
                    for row in group
                ],
                dtype=float,
            )
            distance = _robust_distance(features)
            order = sorted(
                range(len(group)),
                key=lambda index: (distance[index], str(group[index]["start_utc"]), str(group[index]["candidate_id"])),
            )
            result[(mode, species)] = group[order[0]]
    return result


def _load_mode1(row: Mapping[str, Any], ori_root: Path, quality_root: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    paths = mode1_ori_files_for_interval(ori_root, str(row["start_utc"]), str(row["stop_utc"]))
    if not paths:
        raise FileNotFoundError(f"No Mode-1 ori file for {row['start_utc']} -- {row['stop_utc']}")
    records = read_mode1_ori_records(
        paths,
        str(row["start_utc"]),
        str(row["stop_utc"]),
        quality_root=quality_root,
        compute_sha256=True,
    )
    keep = (
        records.project_quality_available
        & ((records.project_quality_flag & PROJECT_BACKGROUND_REJECT_MASK) == 0)
        & (records.native_quality == 0)
    )
    if not np.any(keep):
        raise ValueError(f"No quality-clean Mode-1 records for {row['candidate_key']}")
    provenance = [
        {"path": path, "sha256": records.source_sha256[path]}
        for path in records.source_files
    ]
    return records.time_unix_s[keep], np.asarray(records.dpf[keep], dtype=float), provenance


def _load_multimode(row: Mapping[str, Any], mode: int, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    start = float(row["start_unix_s"])
    stop = float(row["stop_unix_s"])
    by_time: dict[float, np.ndarray] = {}
    provenance: list[dict[str, Any]] = []
    for source_text in row["source_ori"]:
        source = Path(str(source_text))
        provenance.append({"path": str(source.resolve()), "sha256": sha256_file(source)})
        for record in read_ori_records(source, start, stop):
            if int(record.mode) != mode:
                raise ValueError(f"Expected Mode {mode}, found Mode {record.mode} in {source}")
            if int(record.quality_native) != 0:
                continue
            for product_time, flat in split_raw_record(mode, record.time_unix_s, record.ion_dpf):
                if not start <= product_time < stop:
                    continue
                cube = np.asarray(flat, dtype=float).reshape(shape)
                previous = by_time.get(float(product_time))
                if previous is not None and not np.array_equal(previous, cube, equal_nan=True):
                    raise ValueError(f"Conflicting duplicate Mode {mode} record at {product_time}")
                by_time[float(product_time)] = cube
    if not by_time:
        raise ValueError(f"No quality-clean records for {row['candidate_id']}")
    keys = sorted(by_time)
    return np.asarray(keys, dtype=float), np.stack([by_time[key] for key in keys]), provenance


def species_dpf_spectrum(cubes: np.ndarray, mode: int, species: str) -> np.ndarray:
    """Solid-angle-weighted, reviewed-mass-bin mean DPF, shaped time x energy."""

    values = np.asarray(cubes, dtype=float)
    layout = mode_layout(mode)
    expected = (layout.energy_eV.size, layout.pitch_count, layout.azimuth_count, layout.mass_amu.size)
    if values.ndim != 5 or values.shape[1:] != expected:
        raise ValueError(f"Mode {mode} cubes must have shape (time, {expected}), got {values.shape}")
    masses = species_mass_indices(mode, species)
    selected = values[..., masses].reshape(values.shape[0], values.shape[1], layout.angle_count, masses.size)
    _, _, omega = angular_geometry(mode)
    finite = np.isfinite(selected)
    numerator = np.where(finite, selected * omega[None, None, :, None], 0.0).sum(axis=2)
    denominator = np.where(finite, omega[None, None, :, None], 0.0).sum(axis=2)
    per_mass = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan),
        where=denominator > 0.0,
    )
    finite_mass = np.isfinite(per_mass)
    return np.divide(
        np.where(finite_mass, per_mass, 0.0).sum(axis=2),
        finite_mass.sum(axis=2),
        out=np.full(per_mass.shape[:2], np.nan),
        where=finite_mass.sum(axis=2) > 0,
    )


def _finite_time_mean(spectrum: np.ndarray) -> np.ndarray:
    finite = np.isfinite(spectrum)
    return np.divide(
        np.where(finite, spectrum, 0.0).sum(axis=0),
        finite.sum(axis=0),
        out=np.full(spectrum.shape[1], np.nan),
        where=finite.sum(axis=0) > 0,
    )


def _shared_norm(raw: np.ndarray, corrected: np.ndarray) -> tuple[LogNorm, dict[str, float]]:
    positive = np.concatenate(
        [values[np.isfinite(values) & (values > 0.0)] for values in (raw, corrected)]
    )
    if positive.size == 0:
        raise ValueError("No finite positive DPF for LogNorm")
    low, high = np.quantile(positive, [0.01, 0.997])
    low = max(float(low), np.finfo(float).tiny)
    high = max(float(high), low * 10.0)
    return LogNorm(vmin=low, vmax=high), {"vmin": low, "vmax": high}


def _support_summary(model: Any, mode: int, species: str) -> dict[str, int]:
    masses = species_mass_indices(mode, species)
    support = np.asarray(model.support_level)[..., masses]
    return {
        "unsupported": int(np.count_nonzero(support == SUPPORT_UNSUPPORTED)),
        "low_support": int(np.count_nonzero(support == SUPPORT_LOW)),
        "supported": int(np.count_nonzero(support == SUPPORT_SUPPORTED)),
        "total_reviewed_species_channels": int(support.size),
    }


def _metrics(
    raw_cube: np.ndarray,
    corrected_cube: np.ndarray,
    raw_spectrum: np.ndarray,
    corrected_spectrum: np.ndarray,
    mode: int,
    species: str,
    applied_mask: np.ndarray,
) -> dict[str, Any]:
    masses = species_mass_indices(mode, species)
    selected_raw = raw_cube[..., masses]
    selected_corrected = corrected_cube[..., masses]
    selected_applied = applied_mask[..., masses]
    raw_positive_total = float(np.nansum(np.maximum(selected_raw, 0.0)))
    corrected_positive_total = float(np.nansum(np.maximum(selected_corrected, 0.0)))
    raw_mean = _finite_time_mean(raw_spectrum)
    corrected_mean = _finite_time_mean(corrected_spectrum)
    raw_peak = int(np.nanargmax(raw_mean)) if np.any(np.isfinite(raw_mean)) else None
    corrected_peak = (
        int(np.nanargmax(corrected_mean))
        if np.any(np.isfinite(corrected_mean)) and np.nanmax(corrected_mean) > 0.0
        else None
    )
    finite_selected = np.isfinite(selected_raw)
    return {
        "record_count": int(raw_cube.shape[0]),
        "removed_positive_dpf_fraction": (
            (raw_positive_total - corrected_positive_total) / raw_positive_total
            if raw_positive_total > 0.0
            else None
        ),
        "applied_finite_channel_fraction": (
            float(np.count_nonzero(selected_applied) / np.count_nonzero(finite_selected))
            if np.any(finite_selected)
            else None
        ),
        "raw_mean_spectrum_peak_energy_index": raw_peak,
        "corrected_mean_spectrum_peak_energy_index": corrected_peak,
        "peak_shift_channels": abs(corrected_peak - raw_peak) if raw_peak is not None and corrected_peak is not None else None,
        "corrected_positive_total_not_above_raw": bool(
            corrected_positive_total <= raw_positive_total + max(1.0e-12, raw_positive_total * 1.0e-12)
        ),
    }


def _plot_example(example: Mapping[str, Any], output: Path) -> None:
    time = np.asarray(example["time"], dtype=float)
    energy = np.asarray(example["energy"], dtype=float)
    raw = np.asarray(example["raw_spectrum"], dtype=float)
    corrected = np.asarray(example["corrected_spectrum"], dtype=float)
    norm, _ = _shared_norm(raw, corrected)
    dates = [datetime.fromtimestamp(value, UTC) for value in time]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 4.8),
        gridspec_kw={"width_ratios": [2.2, 2.2, 1.15]},
        constrained_layout=True,
    )
    image = None
    for axis, values, title in zip(axes[:2], (raw, corrected), ("Raw released DPF", "Channel-background corrected")):
        image = axis.pcolormesh(
            dates,
            energy,
            np.where(values > 0.0, values, np.nan).T,
            shading="nearest",
            cmap="viridis",
            norm=norm,
            rasterized=True,
        )
        axis.set_yscale("log")
        axis.set_xlabel("UTC")
        axis.set_title(title, fontsize=10)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    axes[0].set_ylabel("Energy (eV)")
    axes[1].tick_params(labelleft=False)
    colorbar = fig.colorbar(image, ax=axes[:2], pad=0.012, aspect=32)
    colorbar.set_label(r"Angular/mass mean DPF [s$^{-1}$ cm$^{-2}$ sr$^{-1}$ eV$^{-1}$]")
    raw_mean = _finite_time_mean(raw)
    corrected_mean = _finite_time_mean(corrected)
    raw_positive = np.isfinite(raw_mean) & (raw_mean > 0.0)
    corrected_positive = np.isfinite(corrected_mean) & (corrected_mean > 0.0)
    if np.any(raw_positive):
        axes[2].plot(raw_mean[raw_positive], energy[raw_positive], color="#555555", lw=1.6, label="Raw")
    if np.any(corrected_positive):
        axes[2].plot(corrected_mean[corrected_positive], energy[corrected_positive], color="#0072B2", lw=1.6, label="Corrected")
    else:
        axes[2].text(
            0.5,
            0.5,
            "No positive corrected DPF",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
            fontsize=8,
            color="#0072B2",
        )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Interval mean DPF")
    axes[2].grid(True, which="both", alpha=0.22)
    axes[2].legend(loc="best", fontsize=8)
    axes[2].set_title("Interval mean spectrum", fontsize=10)
    metrics = example["metrics"]
    support = example["support"]
    removed = metrics["removed_positive_dpf_fraction"]
    fig.suptitle(
        f"MINPA Mode {example['mode']}  |  {example['species']}  |  manually approved noise interval\n"
        f"{example['start_utc']} to {example['stop_utc']}  |  n={metrics['record_count']}  |  "
        f"positive DPF removed={removed:.1%}  |  support S/L/U="
        f"{support['supported']}/{support['low_support']}/{support['unsupported']}",
        fontsize=10.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=190, metadata={"Software": ARTIFACT_VERSION})
    plt.close(fig)
    temporary.replace(output)


def _plot_overview(examples: Sequence[Mapping[str, Any]], output: Path) -> None:
    lookup = {(int(row["mode"]), str(row["species"])): row for row in examples}
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 10.7), sharey="row", constrained_layout=True)
    for row_index, mode in enumerate((1, 4, 12)):
        for column_index, species in enumerate(SPECIES):
            example = lookup[(mode, species)]
            energy = np.asarray(example["energy"], dtype=float)
            raw = _finite_time_mean(np.asarray(example["raw_spectrum"], dtype=float))
            corrected = _finite_time_mean(np.asarray(example["corrected_spectrum"], dtype=float))
            raw_positive = np.isfinite(raw) & (raw > 0.0)
            corrected_positive = np.isfinite(corrected) & (corrected > 0.0)
            axis = axes[row_index, column_index]
            axis.plot(raw[raw_positive], energy[raw_positive], color="#555555", lw=1.5, label="Raw")
            if np.any(corrected_positive):
                axis.plot(corrected[corrected_positive], energy[corrected_positive], color="#0072B2", lw=1.5, label="Corrected")
            else:
                axis.text(
                    0.97,
                    0.04,
                    "corrected: no positive DPF",
                    ha="right",
                    transform=axis.transAxes,
                    fontsize=7.5,
                    color="#0072B2",
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.2)
            axis.set_title(f"Mode {mode}  {species}", fontsize=10)
            axis.text(
                0.03,
                0.04,
                f"removed {example['metrics']['removed_positive_dpf_fraction']:.1%}",
                transform=axis.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            if row_index == 2:
                axis.set_xlabel("Interval mean DPF")
            if column_index == 0:
                axis.set_ylabel("Energy (eV)")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("MINPA v2.2.0 frozen denoise examples: 3 modes x 3 species", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=190, metadata={"Software": ARTIFACT_VERSION})
    pdf_output = output.with_suffix(".pdf")
    pdf_temporary = pdf_output.with_suffix(".tmp.pdf")
    fig.savefig(pdf_temporary, metadata={"Creator": ARTIFACT_VERSION})
    plt.close(fig)
    temporary.replace(output)
    pdf_temporary.replace(pdf_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ori-root", type=Path, default=Path(r"D:\Data\TW-1\result\MINPA\ori"))
    parser.add_argument("--quality-root", type=Path, default=ROOT / "outputs/tw1_minpa_quality_flags_all_species")
    parser.add_argument("--bundle", type=Path, default=UNIFIED_STATIC_BUNDLE)
    parser.add_argument("--mode1-review", type=Path, default=MODE1_REVIEW)
    parser.add_argument("--multimode-review", type=Path, default=MULTIMODE_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = (args.mode1_review, args.multimode_review, args.bundle)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    mode1_review = _read_json(args.mode1_review)
    multimode_review = _read_json(args.multimode_review)
    bundle_inventory = _read_json(args.bundle)
    if mode1_review.get("review_status") != "finalized" or multimode_review.get("review_status") != "finalized":
        raise ValueError("Both manual-review inventories must be finalized")
    mode1_row = select_mode1_interval(mode1_review["approved_intervals"])
    multimode_rows = select_multimode_intervals(multimode_review["approved_intervals"])
    mode1_model, multimode_models = load_unified_static_background_bundle(args.bundle)
    examples: list[dict[str, Any]] = []
    mode1_time, mode1_raw, mode1_sources = _load_mode1(mode1_row, args.ori_root, args.quality_root)
    mode1_correction = apply_background(mode1_raw, mode1_model)
    for mode in (1, 4, 12):
        for species in SPECIES:
            if mode == 1:
                row = mode1_row
                time = mode1_time
                raw_cube = mode1_raw
                correction = mode1_correction
                model = mode1_model
                sources = mode1_sources
                candidate_id = str(row["candidate_key"])
            else:
                row = multimode_rows[(mode, species)]
                model = multimode_models[mode]
                time, raw_cube, sources = _load_multimode(row, mode, model.shape)
                correction = apply_multimode_background(raw_cube, model)
                candidate_id = str(row["candidate_id"])
            raw_spectrum = species_dpf_spectrum(raw_cube, mode, species)
            corrected_spectrum = species_dpf_spectrum(correction.corrected_dpf, mode, species)
            metrics = _metrics(
                raw_cube,
                correction.corrected_dpf,
                raw_spectrum,
                corrected_spectrum,
                mode,
                species,
                correction.applied_channel_mask,
            )
            if not metrics["corrected_positive_total_not_above_raw"]:
                raise AssertionError(f"Corrected total exceeds raw for Mode {mode} {species}")
            support = _support_summary(model, mode, species)
            _, color_limits = _shared_norm(raw_spectrum, corrected_spectrum)
            filename = f"mode{mode:02d}_{SPECIES_TOKEN[species]}_raw_vs_corrected.png"
            example = {
                "mode": mode,
                "species": species,
                "candidate_id": candidate_id,
                "start_utc": str(row["start_utc"]),
                "stop_utc": str(row["stop_utc"]),
                "time": time,
                "energy": mode_layout(mode).energy_eV,
                "raw_spectrum": raw_spectrum,
                "corrected_spectrum": corrected_spectrum,
                "metrics": metrics,
                "support": support,
                "source_files": sources,
                "color_limits": color_limits,
                "figure": str((args.output / "individual" / filename).resolve()),
            }
            _plot_example(example, args.output / "individual" / filename)
            example["figure_sha256"] = sha256_file(Path(example["figure"]))
            examples.append(example)
            print(f"rendered Mode {mode} {species}: {candidate_id}", flush=True)
    overview = args.output / "minpa_denoise_examples_3modes_3species.png"
    _plot_overview(examples, overview)
    serializable_examples = []
    for example in examples:
        serializable_examples.append({
            key: value
            for key, value in example.items()
            if key not in {"time", "energy", "raw_spectrum", "corrected_spectrum"}
        })
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "complete",
        "scope": "one manually approved example for each of Mode 1/4/12 and H+/O+/O2+",
        "selection": {
            "mode1": "minimum robust L1 distance to approved-inventory medians of duration and mission time; one jointly approved interval reused for all three species",
            "mode4_mode12": "minimum robust L1 distance to each approved mode/species group's medians of duration, Mode-1-envelope decision score, and secondary approved probability; ties by UTC and candidate ID",
            "purpose": "deterministic representative display, not model fitting or validation-case selection",
        },
        "spectrum_definition": "solid-angle-weighted DPF per reviewed mass bin, then arithmetic mean across the species mass bins; no smoothing or interpolation",
        "units": DPF_UNITS,
        "dimension_order": ["time", "energy", "pitch", "azimuth", "mass"],
        "mode12_time_rule": "each native record split at t+1.025 s and t+3.075 s before subtraction",
        "quality_rule": "native Quality==0; Mode-1 also requires project quality available and bits 3-4 absent",
        "correction": "max(raw - channel_background, 0); NaN preserved; unsupported channels unchanged",
        "unified_static_bundle": {
            "path": str(args.bundle.resolve()),
            "sha256": sha256_file(args.bundle),
            "models": bundle_inventory["models"],
            "temporal_policy": bundle_inventory["temporal_policy"],
        },
        "mode1_review": {
            "path": str(args.mode1_review.resolve()),
            "sha256": sha256_file(args.mode1_review),
        },
        "multimode_review": {
            "path": str(args.multimode_review.resolve()),
            "sha256": sha256_file(args.multimode_review),
        },
        "overview_figure": str(overview.resolve()),
        "overview_figure_sha256": sha256_file(overview),
        "overview_pdf": str(overview.with_suffix(".pdf").resolve()),
        "overview_pdf_sha256": sha256_file(overview.with_suffix(".pdf")),
        "examples": serializable_examples,
        "sanity_checks": {
            "all_nine_examples_rendered": len(examples) == 9,
            "all_corrected_positive_totals_not_above_raw": all(
                row["metrics"]["corrected_positive_total_not_above_raw"] for row in examples
            ),
            "coordinate_system": "instrument channel space; no coordinate transform",
            "mode_shapes": {str(mode): list(multimode_models[mode].shape) for mode in (4, 12)},
            "mode1_shape": list(mode1_model.background_dpf.shape),
        },
    }
    _atomic_json(args.output / "examples_manifest.json", manifest)
    readme = args.output / "README.md"
    readme.write_text(
        "# MINPA 三模式、三物种去噪示例\n\n"
        "本目录给出 Mode 1、4、12 与 H+、O+、O2+ 的九种组合各一个人工批准噪声区间示例。"
        "每张单图使用完全相同的原始/校正色标，右侧为区间平均谱；没有平滑或插值。\n\n"
        "所有校正均来自冻结的 v2.2.0 全批准、非时变逐通道 bundle；事件时段与 v2.0.0 示例保持一致，"
        "以便直接比较版本变化。Mode 1 使用新增批准区间重建的 v1.2.0 子模型，Mode 4/12 子模型与 v2.0.0 相同。\n\n"
        "- `minpa_denoise_examples_3modes_3species.png/.pdf`：九组合平均谱总览。\n"
        "- `individual/`：九张原始与去噪能量—时间谱。\n"
        "- `examples_manifest.json`：选择规则、区间、模型/源文件哈希、支持度与数值检查。\n\n"
        "注意：这些都是人工批准的噪声区间，用于展示背景扣除行为，不是强有效信号保真示例。"
        "Mode 4 的低支持通道仍按冻结算法扣除，无支持通道保持原值；Mode 12 为方位积分产品。\n",
        encoding="utf-8",
    )
    print(f"overview={overview}")
    print(f"manifest={args.output / 'examples_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
