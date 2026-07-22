"""Plot and tabulate every currently accepted MINPA Mode-1 noise interval."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background_temporal import _atomic_csv, _atomic_json, _atomic_npz  # noqa: E402
from highE.minpa_interval_spectra import (  # noqa: E402
    INTERVAL_SPECTRA_VERSION,
    SPECIES,
    extract_interval_mean_spectra,
    interval_quantiles,
    load_accepted_interval_rows,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "minpa_background_temporal"
QUANTILES = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def _species_limits(mean_def: np.ndarray, species_index: int) -> tuple[float, float]:
    positive = mean_def[:, species_index, :]
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return 1.0, 10.0
    low, high = np.nanquantile(positive, [0.005, 0.995])
    return max(float(low) / 1.5, 1.0), max(float(high) * 1.8, float(low) * 10.0)


def _plot_full_mission(
    energy: np.ndarray,
    mean_def: np.ndarray,
    quantiles: np.ndarray,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4), sharex=True)
    cmap = mpl.colormaps["viridis"]
    time_colors = cmap(np.linspace(0.08, 0.92, len(rows)))
    individual_handle = None
    median_handle = None
    outer_handle = None
    inner_handle = None
    for species_index, (_, label, _) in enumerate(SPECIES):
        axis = axes[species_index]
        outer_handle = axis.fill_between(
            energy, _positive_for_log(quantiles[0, species_index]),
            _positive_for_log(quantiles[4, species_index]),
            color="#9ecae1", alpha=0.22, linewidth=0, label="10-90%",
        )
        inner_handle = axis.fill_between(
            energy, _positive_for_log(quantiles[1, species_index]),
            _positive_for_log(quantiles[3, species_index]),
            color="#3182bd", alpha=0.24, linewidth=0, label="25-75%",
        )
        for interval_index, color in enumerate(time_colors):
            line, = axis.plot(
                energy, _positive_for_log(mean_def[interval_index, species_index]),
                color=color, alpha=0.16, lw=0.55, rasterized=True,
            )
            if individual_handle is None:
                individual_handle = line
        median_handle, = axis.plot(
            energy, _positive_for_log(quantiles[2, species_index]),
            color="black", lw=2.7, zorder=10,
            label="Per-energy median across intervals",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(*_species_limits(mean_def, species_index))
        axis.set_title(label, fontsize=13, fontweight="bold")
        axis.set_xlabel("Energy (eV)")
        axis.grid(True, which="both", alpha=0.18)
    axes[0].set_ylabel(r"Time-mean DEF [1/(s cm$^2$ sr)]")
    individual_handle.set_label("Individual accepted interval")
    fig.legend(
        handles=[individual_handle, outer_handle, inner_handle, median_handle],
        loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        f"MINPA Mode 1: all {len(rows)} currently accepted noise candidates\n"
        "finite zeros included in each interval time mean; intervals equally weighted",
        fontsize=15, y=1.015,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    png = figure_dir / "all_accepted_intervals_mean_spectra_with_median.png"
    pdf = figure_dir / "all_accepted_intervals_mean_spectra_with_median.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png.resolve()), str(pdf.resolve())]


def _plot_by_quarter(
    energy: np.ndarray,
    mean_def: np.ndarray,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> list[str]:
    quarters = sorted({str(row["quarter"]) for row in rows})
    fig, axes = plt.subplots(
        len(quarters), 3, figsize=(15.5, 2.55 * len(quarters)), sharex=True,
        squeeze=False,
    )
    species_limits = [_species_limits(mean_def, index) for index in range(3)]
    individual_handle = None
    median_handle = None
    inner_handle = None
    for row_index, quarter in enumerate(quarters):
        indices = np.asarray([
            index for index, item in enumerate(rows) if str(item["quarter"]) == quarter
        ], dtype=int)
        local = mean_def[indices]
        local_quantiles = interval_quantiles(local, QUANTILES)
        colors = mpl.colormaps["viridis"](np.linspace(0.12, 0.88, indices.size))
        for species_index, (_, label, _) in enumerate(SPECIES):
            axis = axes[row_index, species_index]
            inner_handle = axis.fill_between(
                energy, _positive_for_log(local_quantiles[1, species_index]),
                _positive_for_log(local_quantiles[3, species_index]),
                color="#6baed6", alpha=0.22, linewidth=0,
            )
            for local_index, interval_index in enumerate(indices):
                line, = axis.plot(
                    energy, _positive_for_log(mean_def[interval_index, species_index]),
                    color=colors[local_index], alpha=0.34, lw=0.72,
                    rasterized=True,
                )
                if individual_handle is None:
                    individual_handle = line
            median_handle, = axis.plot(
                energy, _positive_for_log(local_quantiles[2, species_index]),
                color="black", lw=2.25, zorder=10,
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_ylim(*species_limits[species_index])
            axis.grid(True, which="both", alpha=0.16)
            if row_index == 0:
                axis.set_title(label, fontsize=12, fontweight="bold")
            if species_index == 0:
                axis.set_ylabel(
                    f"{quarter}  N={indices.size}\nDEF [1/(s cm$^2$ sr)]", fontsize=9,
                )
            if row_index == len(quarters) - 1:
                axis.set_xlabel("Energy (eV)")
            axis.tick_params(labelsize=8)
    individual_handle.set_label("Individual accepted interval")
    inner_handle.set_label("Quarter 25-75%")
    median_handle.set_label("Quarter per-energy median")
    fig.legend(
        handles=[individual_handle, inner_handle, median_handle],
        loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.987),
    )
    fig.suptitle(
        "MINPA Mode 1 accepted noise candidates by quarter\n"
        "finite zeros included in each interval time mean",
        fontsize=15, y=1.006,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    png = figure_dir / "all_accepted_intervals_by_quarter_with_median.png"
    pdf = figure_dir / "all_accepted_intervals_by_quarter_with_median.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png.resolve()), str(pdf.resolve())]


def _median_table(energy: np.ndarray, quantiles: np.ndarray, mean_def: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for energy_index, energy_ev in enumerate(energy):
        row: dict[str, Any] = {"energy_index": energy_index, "energy_eV": float(energy_ev)}
        for species_index, (name, _, _) in enumerate(SPECIES):
            row.update({
                f"{name}_p10_def": float(quantiles[0, species_index, energy_index]),
                f"{name}_p25_def": float(quantiles[1, species_index, energy_index]),
                f"{name}_median_def": float(quantiles[2, species_index, energy_index]),
                f"{name}_p75_def": float(quantiles[3, species_index, energy_index]),
                f"{name}_p90_def": float(quantiles[4, species_index, energy_index]),
                f"{name}_finite_interval_count": int(np.isfinite(mean_def[:, species_index, energy_index]).sum()),
            })
        rows.append(row)
    return rows


def _interval_table(
    rows: list[dict[str, Any]], mean_def: np.ndarray, audits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for interval_index, (row, audit) in enumerate(zip(rows, audits)):
        metric = row["metric"]
        item: dict[str, Any] = {
            "candidate_id": row["candidate_id"],
            "start_utc": row["final_start_utc"],
            "stop_utc": row["final_stop_utc"],
            "year": row["year"],
            "quarter": row["quarter"],
            "selection_source": metric["selection_source"],
            "predicted_class": row.get("predicted_class", ""),
            "manual_override": bool(row.get("manual_override", False)),
            "kept_ori_records": int(metric["kept_records"]),
            "lambda_global": float(metric["lambda_global"]),
            "source_day_spe": row["source_day_spe"],
            "source_day_spe_segment": row["source_day_spe_segment"],
            **audit,
        }
        for species_index, (name, _, mass_index) in enumerate(SPECIES):
            spectrum = mean_def[interval_index, species_index]
            finite = spectrum[np.isfinite(spectrum)]
            positive = finite[finite > 0.0]
            peak_index = int(np.nanargmax(spectrum)) if finite.size else -1
            item.update({
                f"lambda_{name}": float(metric["lambda_mass"][mass_index]),
                f"{name}_energy_median_def_including_zero": float(np.nanmedian(spectrum)),
                f"{name}_positive_energy_median_def": (
                    float(np.nanmedian(positive)) if positive.size else float("nan")
                ),
                f"{name}_peak_def": float(spectrum[peak_index]) if peak_index >= 0 else float("nan"),
                f"{name}_peak_energy_index": peak_index,
                f"{name}_positive_energy_fraction": float(np.mean(spectrum > 0.0)),
            })
        table.append(item)
    return table


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    rows = load_accepted_interval_rows(output / "months")
    energy, mean_def, audits = extract_interval_mean_spectra(rows)
    quantiles = interval_quantiles(mean_def, QUANTILES)

    data_dir = output / "data" / "all_accepted_interval_spectra"
    figure_dir = output / "figures" / "all_accepted_interval_spectra"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    figures.extend(_plot_full_mission(energy, mean_def, quantiles, rows, figure_dir))
    figures.extend(_plot_by_quarter(energy, mean_def, rows, figure_dir))

    median_csv = data_dir / "per_energy_noise_def_quantiles.csv"
    interval_csv = data_dir / "accepted_interval_mean_def_statistics.csv"
    _atomic_csv(median_csv, _median_table(energy, quantiles, mean_def))
    _atomic_csv(interval_csv, _interval_table(rows, mean_def, audits))
    npz_path = data_dir / "accepted_interval_mean_spectra.npz"
    _atomic_npz(
        npz_path,
        energy_eV=energy,
        mean_def=mean_def,
        quantile_levels=QUANTILES,
        per_energy_quantiles=quantiles,
        candidate_id=np.asarray([str(row["candidate_id"]) for row in rows]),
        start_utc=np.asarray([str(row["final_start_utc"]) for row in rows]),
        stop_utc=np.asarray([str(row["final_stop_utc"]) for row in rows]),
        quarter=np.asarray([str(row["quarter"]) for row in rows]),
        species=np.asarray([name for name, _, _ in SPECIES]),
    )

    quarter_counts = dict(sorted(Counter(str(row["quarter"]) for row in rows).items()))
    source_counts = dict(sorted(Counter(str(row["metric"]["selection_source"]) for row in rows).items()))
    summary = {
        "algorithm_version": INTERVAL_SPECTRA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "analysis_status": "exploratory_currently_accepted_noise_candidates",
        "interval_count": len(rows),
        "accepted_ori_record_count": int(sum(int(row["metric"]["kept_records"]) for row in rows)),
        "quarter_count": len(quarter_counts),
        "quarter_interval_counts": quarter_counts,
        "selection_source_counts": source_counts,
        "quantity": "published day_spe solid-angle-weighted one-dimensional DEF",
        "units": "1/(s cm^2 sr)",
        "within_interval_aggregation": (
            "repeated day_spe timestamps are first averaged per energy, then an arithmetic time mean "
            "is taken over unique timestamps; finite zeros included"
        ),
        "cross_interval_aggregation": "equal interval weights; quantiles independently at each of 40 energies",
        "median_curve": "q=0.5 across all accepted interval time-mean spectra at each energy",
        "quality_scope": (
            "intervals already accepted by the temporal pipeline; Mode 1 and 40-energy table revalidated. "
            "No new spectral-structure veto was applied."
        ),
        "scientific_limit": (
            "Only 2021-12 has manual interval decisions. Other accepted intervals are exploratory auto "
            "candidates and can retain real H+/O+/O2+ spectral structure."
        ),
        "median_curve_summary": {
            name: {
                "median_across_40_energy_bins": float(np.nanmedian(quantiles[2, index])),
                "minimum": float(np.nanmin(quantiles[2, index])),
                "maximum": float(np.nanmax(quantiles[2, index])),
            }
            for index, (name, _, _) in enumerate(SPECIES)
        },
        "outputs": {
            "figures": figures,
            "per_energy_quantiles_csv": str(median_csv.resolve()),
            "interval_statistics_csv": str(interval_csv.resolve()),
            "arrays_npz": str(npz_path.resolve()),
        },
    }
    json_path = data_dir / "summary.json"
    _atomic_json(json_path, summary)
    print(json.dumps({**summary, "summary_json": str(json_path.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
