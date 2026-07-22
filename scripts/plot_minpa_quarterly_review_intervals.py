"""Plot five lambda-stratified accepted MINPA intervals per eligible quarter."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background_temporal import (  # noqa: E402
    TEMPORAL_ALGORITHM_VERSION,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    select_quarter_review_intervals,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "minpa_background_temporal"
SPECIES = (("H", "H+"), ("O", "O+"), ("O2", "O2+"))
TARGET_COLORS = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quarters", nargs="*", help="Default: all inference-eligible quarters")
    parser.add_argument("--context-minutes", type=float, default=8.0)
    return parser.parse_args()


def _utc_seconds(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _load_month_summaries(output: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "months").glob("*/summary.json"))
    ]


def _load_contexts(
    selected: list[dict[str, Any]], context_s: float,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    cache: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    contexts: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for row in selected:
        source = str(row["source_day_spe"])
        segment = str(row["source_day_spe_segment"])
        if (source, segment, "H") not in cache:
            data = loadmat(source, squeeze_me=True, struct_as_record=False)
            for name, _ in SPECIES:
                item = getattr(data[f"{name}_spe_num"], segment)
                mode = int(np.asarray(item.mod).reshape(-1)[0])
                energy = np.asarray(item.f, dtype=float).reshape(-1)
                spectrum = np.asarray(item.p, dtype=float)
                time = np.asarray(item.t, dtype=float).reshape(-1)
                if mode != 1 or energy.size != 40 or spectrum.shape != (time.size, 40):
                    raise ValueError(f"Non-standard Mode-1 segment: {source} {segment} {name}")
                cache[(source, segment, name)] = {
                    "time": time,
                    "energy": energy,
                    "spectrum": spectrum,
                }
        start_s = _utc_seconds(str(row["final_start_utc"]))
        stop_s = _utc_seconds(str(row["final_stop_utc"]))
        for name, _ in SPECIES:
            values = cache[(source, segment, name)]
            mask = (
                (values["time"] >= start_s - context_s)
                & (values["time"] < stop_s + context_s)
            )
            interval = (
                (values["time"] >= start_s)
                & (values["time"] < stop_s)
            )
            spectrum = np.asarray(values["spectrum"], dtype=float)
            finite = np.isfinite(spectrum[interval])
            mean_spectrum = np.divide(
                np.where(finite, spectrum[interval], 0.0).sum(axis=0),
                finite.sum(axis=0),
                out=np.full(values["energy"].size, np.nan),
                where=finite.sum(axis=0) > 0,
            )
            contexts[(str(row["candidate_id"]), name)] = {
                "time": values["time"][mask],
                "energy": values["energy"],
                "spectrum": spectrum[mask],
                "mean_spectrum": mean_spectrum,
            }
    return contexts


def _species_norms(
    selected: list[dict[str, Any]],
    contexts: dict[tuple[str, str], dict[str, np.ndarray]],
) -> dict[str, LogNorm]:
    norms: dict[str, LogNorm] = {}
    for name, _ in SPECIES:
        positive = np.concatenate([
            contexts[(str(row["candidate_id"]), name)]["spectrum"].reshape(-1)
            for row in selected
        ])
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if positive.size == 0:
            raise ValueError(f"No positive {name} DEF values in selected contexts")
        low, high = np.nanquantile(positive, [0.02, 0.995])
        norms[name] = LogNorm(
            vmin=max(float(low), np.finfo(float).tiny),
            vmax=max(float(high), float(low) * 10.0),
        )
    return norms


def _plot_quarter_montage(
    quarter: str,
    rows: list[dict[str, Any]],
    contexts: dict[tuple[str, str], dict[str, np.ndarray]],
    norms: dict[str, LogNorm],
    output: Path,
) -> list[str]:
    fig, axes = plt.subplots(5, 3, figsize=(15.8, 13.5), sharey=True)
    fig.subplots_adjust(left=0.145, right=0.91, bottom=0.07, top=0.91, wspace=0.14, hspace=0.34)
    meshes: dict[str, Any] = {}
    for row_index, row in enumerate(rows):
        start = datetime.fromisoformat(str(row["final_start_utc"]).replace("Z", "+00:00"))
        stop = datetime.fromisoformat(str(row["final_stop_utc"]).replace("Z", "+00:00"))
        for column, (name, label) in enumerate(SPECIES):
            axis = axes[row_index, column]
            context = contexts[(str(row["candidate_id"]), name)]
            time = [datetime.fromtimestamp(value, UTC) for value in context["time"]]
            plotted = np.ma.masked_invalid(
                np.where(context["spectrum"] > 0.0, context["spectrum"], np.nan)
            )
            meshes[name] = axis.pcolormesh(
                time, context["energy"], plotted.T,
                shading="auto", norm=norms[name], cmap="viridis", rasterized=True,
            )
            axis.axvline(start, color="black", lw=1.5, zorder=6)
            axis.axvline(stop, color="black", lw=1.5, zorder=6)
            axis.set_yscale("log")
            axis.set_ylim(float(context["energy"][0]), float(context["energy"][-1]))
            axis.xaxis.set_major_locator(mdates.MinuteLocator(interval=8, tz=UTC))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
            axis.tick_params(axis="both", labelsize=8)
            if row_index == 0:
                axis.set_title(label, fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel("Energy (eV)")
        center_y = 0.91 - (row_index + 0.5) * (0.84 / 5.0)
        fig.text(
            0.012, center_y,
            f"#{int(row['review_order'])}  {row['candidate_id']}\n"
            f"q={float(row['review_target_quantile']):.1f}  "
            f"lambda={float(row['lambda_global']):.3e}\n"
            f"{start:%Y-%m-%d}\n{start:%H:%M}-{stop:%H:%M} UTC\n"
            f"N={int(row['kept_records'])}",
            ha="left", va="center", fontsize=8.5,
        )
    for column, (name, label) in enumerate(SPECIES):
        colorbar = fig.colorbar(
            meshes[name], ax=axes[:, column].tolist(),
            orientation="horizontal", fraction=0.025, pad=0.045, aspect=35,
        )
        colorbar.set_label(f"{label} DEF [1/(s cm^2 sr)]", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"MINPA Mode 1 accepted background intervals — {quarter}\n"
        "five global-lambda strata; black lines delimit retained interval",
        fontsize=15,
    )
    png = output / f"{quarter}_five_selected_intervals_spectrogram.png"
    pdf = output / f"{quarter}_five_selected_intervals_spectrogram.pdf"
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    return [str(png.resolve()), str(pdf.resolve())]


def _plot_mean_spectra(
    by_quarter: dict[str, list[dict[str, Any]]],
    contexts: dict[tuple[str, str], dict[str, np.ndarray]],
    output: Path,
) -> list[str]:
    quarters = list(by_quarter)
    fig, axes = plt.subplots(len(quarters), 3, figsize=(15, 3.8 * len(quarters)), sharex=True)
    axes = np.atleast_2d(axes)
    handles = []
    for row_index, quarter in enumerate(quarters):
        for column, (name, label) in enumerate(SPECIES):
            axis = axes[row_index, column]
            for item, color in zip(by_quarter[quarter], TARGET_COLORS):
                context = contexts[(str(item["candidate_id"]), name)]
                line, = axis.plot(
                    context["energy"],
                    np.where(context["mean_spectrum"] > 0.0, context["mean_spectrum"], np.nan),
                    color=color, lw=1.35,
                    label=f"q={float(item['review_target_quantile']):.1f}",
                )
                if row_index == 0 and column == 0:
                    handles.append(line)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.18)
            if row_index == 0:
                axis.set_title(label, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{quarter}\nmean DEF [1/(s cm^2 sr)]")
            if row_index == len(quarters) - 1:
                axis.set_xlabel("Energy (eV)")
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        "Time-mean DEF inside the retained interval (zeros included)",
        fontsize=15, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = output / "selected_intervals_time_mean_energy_spectra.png"
    pdf = output / "selected_intervals_time_mean_energy_spectra.pdf"
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    return [str(png.resolve()), str(pdf.resolve())]


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    analysis = json.loads(
        (output / "data" / "quarterly_noise_analysis.json").read_text(encoding="utf-8")
    )
    quarters = args.quarters or [
        str(row["quarter"]) for row in analysis["quarterly"]
        if bool(row["inference_eligible"])
    ]
    summaries = _load_month_summaries(output)
    by_quarter = {
        quarter: select_quarter_review_intervals(summaries, quarter)
        for quarter in quarters
    }
    selected = [item for quarter in quarters for item in by_quarter[quarter]]
    contexts = _load_contexts(selected, float(args.context_minutes) * 60.0)
    norms = _species_norms(selected, contexts)
    figure_dir = output / "figures" / "quarter_interval_review"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []
    for quarter in quarters:
        figure_paths.extend(
            _plot_quarter_montage(quarter, by_quarter[quarter], contexts, norms, figure_dir)
        )
    figure_paths.extend(_plot_mean_spectra(by_quarter, contexts, figure_dir))

    energy = contexts[(str(selected[0]["candidate_id"]), "H")]["energy"]
    mean_arrays = np.stack([
        np.stack([
            contexts[(str(item["candidate_id"]), name)]["mean_spectrum"]
            for name, _ in SPECIES
        ])
        for item in selected
    ])
    _atomic_npz(
        output / "data" / "quarterly_selected_interval_mean_spectra.npz",
        energy_eV=energy,
        mean_def=mean_arrays,
        candidate_id=np.asarray([str(item["candidate_id"]) for item in selected]),
        quarter=np.asarray([
            quarter for quarter in quarters for _ in by_quarter[quarter]
        ]),
    )
    table_rows = []
    for quarter in quarters:
        for item in by_quarter[quarter]:
            table_rows.append({
                "quarter": quarter,
                "review_order": item["review_order"],
                "target_quantile": item["review_target_quantile"],
                "candidate_id": item["candidate_id"],
                "start_utc": item["final_start_utc"],
                "stop_utc": item["final_stop_utc"],
                "kept_records": item["kept_records"],
                "lambda_global": item["lambda_global"],
                "lambda_h": item["lambda_mass"][0],
                "lambda_o": item["lambda_mass"][3],
                "lambda_o2": item["lambda_mass"][5],
                "source_day_spe": item["source_day_spe"],
                "source_day_spe_segment": item["source_day_spe_segment"],
                "selection_source": item["selection_source"],
            })
    csv_path = output / "data" / "quarterly_selected_review_intervals.csv"
    _atomic_csv(csv_path, table_rows)
    provenance = {
        "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "human review of five accepted intervals per inference-eligible quarter",
        "quarters": quarters,
        "selection": (
            "nearest finite global-lambda ranks to 0.1, 0.3, 0.5, 0.7, 0.9; "
            "distinct UTC days preferred"
        ),
        "spectrogram_context_minutes_each_side": float(args.context_minutes),
        "spectrogram_quantity": "published day_spe DEF; positive values on log color scale",
        "mean_spectrum_quantity": "time mean DEF inside retained bounds including finite zeros",
        "black_lines": "final retained interval bounds",
        "color_scale": {
            name: {"vmin": norm.vmin, "vmax": norm.vmax}
            for name, norm in norms.items()
        },
        "selected_intervals_csv": str(csv_path.resolve()),
        "figures": figure_paths,
    }
    json_path = output / "data" / "quarterly_selected_review_intervals.json"
    _atomic_json(json_path, provenance)
    print(json.dumps({
        "quarters": quarters,
        "selected_intervals": len(selected),
        "selection_csv": str(csv_path.resolve()),
        "provenance_json": str(json_path.resolve()),
        "figures": figure_paths,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
