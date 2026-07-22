"""Analyze manually labeled MINPA quiet-window classes and validate a classifier."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import (  # noqa: E402
    MODE1_ENERGY_EV,
    load_background_model,
    mode1_ori_files_for_interval,
    read_mode1_ori_records,
)
from highE.minpa_quiet_classifier import (  # noqa: E402
    QuietReferenceClassifierConfig,
    extract_quiet_reference_features,
    hplus_count_energy_spectrum,
    time_mean_h_def_spectrum,
)


DEFAULT_ORI = Path(r"D:\Data\TW-1\result\MINPA\ori")
DEFAULT_CANDIDATES = ROOT / "outputs/minpa_background_paper_reproduction/data/background_interval_candidates.json"
DEFAULT_REVIEW = ROOT / "config/minpa_background_manual_review.json"
DEFAULT_MODEL = ROOT / "outputs/minpa_background_paper_reproduction/data/minpa_mode1_background_model_provisional.npz"
DEFAULT_OUTPUT = ROOT / "outputs/minpa_background_paper_reproduction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ori-root", type=Path, default=DEFAULT_ORI)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--background-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _fit_and_thresholds(rows: list[dict[str, object]]) -> tuple[float, float]:
    x = np.array([float(row["time_mean_def_peak_to_median"]) for row in rows])
    z = np.array([float(row["h_count_top3_energy_fraction"]) for row in rows])
    y = np.array([bool(row["manual_real_spectrum"]) for row in rows])
    if np.unique(y).size != 2:
        raise ValueError("threshold fitting requires both manual classes")
    x_unique, z_unique = np.unique(x), np.unique(z)
    x_cuts = (x_unique[:-1] + x_unique[1:]) / 2.0
    z_cuts = (z_unique[:-1] + z_unique[1:]) / 2.0
    best: tuple[tuple[float, ...], float, float] | None = None
    for x_cut in x_cuts:
        for z_cut in z_cuts:
            prediction = (x > x_cut) & (z > z_cut)
            recall = float(np.mean(prediction[y]))
            specificity = float(np.mean(~prediction[~y]))
            balanced = 0.5 * (recall + specificity)
            score_value = np.minimum(x / x_cut, z / z_cut)
            margin = float(np.min(np.abs(score_value - 1.0)))
            key = (balanced, recall, specificity, margin)
            if best is None or key > best[0]:
                best = (key, float(x_cut), float(z_cut))
    assert best is not None
    return best[1], best[2]


def _metrics(rows: list[dict[str, object]], x_cut: float, z_cut: float) -> dict[str, object]:
    y = np.array([bool(row["manual_real_spectrum"]) for row in rows])
    prediction = np.array(
        [
            float(row["time_mean_def_peak_to_median"]) > x_cut
            and float(row["h_count_top3_energy_fraction"]) > z_cut
            for row in rows
        ]
    )
    return {
        "true_real_spectrum": int(np.sum(prediction & y)),
        "true_pure_background": int(np.sum(~prediction & ~y)),
        "false_real_spectrum": int(np.sum(prediction & ~y)),
        "false_pure_background": int(np.sum(~prediction & y)),
        "accuracy": float(np.mean(prediction == y)),
        "real_spectrum_recall": float(np.mean(prediction[y])),
        "pure_background_recall": float(np.mean(~prediction[~y])),
    }


def _leave_one_out(rows: list[dict[str, object]]) -> dict[str, object]:
    predictions: list[bool] = []
    fitted: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        training = rows[:index] + rows[index + 1 :]
        x_cut, z_cut = _fit_and_thresholds(training)
        prediction = (
            float(row["time_mean_def_peak_to_median"]) > x_cut
            and float(row["h_count_top3_energy_fraction"]) > z_cut
        )
        predictions.append(bool(prediction))
        fitted.append(
            {
                "candidate_number": int(row["candidate_number"]),
                "def_peak_to_median_threshold": x_cut,
                "h_count_top3_fraction_threshold": z_cut,
                "prediction_real_spectrum": bool(prediction),
            }
        )
    y = np.array([bool(row["manual_real_spectrum"]) for row in rows])
    prediction_array = np.asarray(predictions, dtype=bool)
    return {
        "accuracy": float(np.mean(prediction_array == y)),
        "false_real_spectrum_candidate_numbers": [
            int(rows[index]["candidate_number"])
            for index in np.flatnonzero(prediction_array & ~y)
        ],
        "false_pure_background_candidate_numbers": [
            int(rows[index]["candidate_number"])
            for index in np.flatnonzero(~prediction_array & y)
        ],
        "folds": fitted,
    }


def main() -> None:
    args = parse_args()
    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    review = json.loads(args.manual_review.read_text(encoding="utf-8"))
    model = load_background_model(args.background_model)
    weak = set(review["decisions"]["reject_visible_or_weak_real_spectrum"])
    clean = set(review["decisions"]["approve_no_visible_real_spectrum"])
    labeled = weak | clean
    if weak & clean:
        raise ValueError("manual review classes overlap")

    day_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    mean_spectra: dict[int, np.ndarray] = {}
    count_spectra: dict[int, np.ndarray] = {}
    cfg = QuietReferenceClassifierConfig()
    for number in sorted(labeled):
        candidate = candidate_payload["intervals"][number - 1]
        day_path = str(candidate["source_day_spe"])
        segment_name = str(candidate.get("source_day_spe_segment", "num1"))
        cache_key = f"{day_path}::{segment_name}"
        if cache_key not in day_cache:
            data = loadmat(day_path, squeeze_me=True, struct_as_record=False)
            item = getattr(data["H_spe_num"], segment_name)
            day_cache[cache_key] = (
                np.asarray(item.t, dtype=float).reshape(-1),
                np.asarray(item.p, dtype=float),
            )
        day_time, day_def = day_cache[cache_key]
        day_mask = (
            (day_time >= float(candidate["start_unix_s"]))
            & (day_time < float(candidate["stop_unix_s"]))
        )
        h_def = day_def[day_mask]
        paths = mode1_ori_files_for_interval(
            args.ori_root, str(candidate["start_utc"]), str(candidate["stop_utc"])
        )
        records = read_mode1_ori_records(
            paths,
            str(candidate["start_utc"]),
            str(candidate["stop_utc"]),
            compute_sha256=False,
        )
        features = extract_quiet_reference_features(
            h_def, records.dpf, model.dpf_quantum, cfg
        )
        manual_real = number in weak
        row = {
            "candidate_number": number,
            "start_utc": candidate["start_utc"],
            "stop_utc": candidate["stop_utc"],
            "manual_class": "weak_real_spectrum" if manual_real else "pure_background",
            "manual_real_spectrum": manual_real,
            **features,
            "prediction_matches_manual": bool(
                bool(features["real_spectrum_detected"]) == manual_real
            ),
        }
        rows.append(row)
        mean_spectra[number] = time_mean_h_def_spectrum(h_def)
        count_spectra[number] = hplus_count_energy_spectrum(
            records.dpf, model.dpf_quantum
        )

    fitted_x, fitted_z = _fit_and_thresholds(rows)
    configured_metrics = _metrics(
        rows, cfg.def_peak_to_median_threshold, cfg.h_count_top3_fraction_threshold
    )
    fitted_metrics = _metrics(rows, fitted_x, fitted_z)
    loo = _leave_one_out(rows)

    data_dir = args.output_root / "data"
    figure_dir = args.output_root / "figures"
    log_dir = args.output_root / "logs"
    for directory in (data_dir, figure_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "manual_quiet_window_features.csv"
    flat_keys = [
        "candidate_number",
        "start_utc",
        "stop_utc",
        "manual_class",
        "time_mean_def_peak_to_median",
        "time_mean_def_peak_energy_eV",
        "h_count_top3_energy_fraction",
        "decision_score",
        "real_spectrum_detected",
        "pure_background_reference",
        "classification_confidence",
        "manual_review_recommended",
        "prediction_matches_manual",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in flat_keys})

    colors = {True: "#d55e00", False: "#0072b2"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for row in rows:
        number = int(row["candidate_number"])
        manual_real = bool(row["manual_real_spectrum"])
        normalized = mean_spectra[number] / np.nanmedian(mean_spectra[number])
        axes[0].plot(
            MODE1_ENERGY_EV,
            normalized,
            color=colors[manual_real],
            alpha=0.35,
            lw=0.9,
        )
    for manual_real, label in ((True, "weak real spectrum"), (False, "pure background")):
        group = np.stack(
            [
                mean_spectra[int(row["candidate_number"])]
                / np.nanmedian(mean_spectra[int(row["candidate_number"])])
                for row in rows
                if bool(row["manual_real_spectrum"]) == manual_real
            ]
        )
        axes[0].plot(
            MODE1_ENERGY_EV,
            np.nanmedian(group, axis=0),
            color=colors[manual_real],
            lw=2.5,
            label=label,
        )
    axes[0].set(xscale="log", yscale="log", xlabel="Energy (eV)", ylabel="time-mean H+ DEF / spectral median")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("(a) Time-mean DEF spectral shape")

    for row in rows:
        manual_real = bool(row["manual_real_spectrum"])
        x = float(row["time_mean_def_peak_to_median"])
        y = float(row["h_count_top3_energy_fraction"])
        axes[1].scatter(x, y, s=42, color=colors[manual_real])
        axes[1].annotate(str(row["candidate_number"]), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[1].axvline(cfg.def_peak_to_median_threshold, color="black", ls="--", lw=1)
    axes[1].axhline(cfg.h_count_top3_fraction_threshold, color="black", ls="--", lw=1)
    axes[1].set(xlabel="time-mean DEF peak / median", ylabel="top adjacent-3-energy H+ count fraction")
    axes[1].set_title("(b) Interpretable AND classifier")

    ordered = sorted(rows, key=lambda row: float(row["decision_score"]))
    x_position = np.arange(len(ordered))
    axes[2].bar(
        x_position,
        [float(row["decision_score"]) for row in ordered],
        color=[colors[bool(row["manual_real_spectrum"])] for row in ordered],
    )
    axes[2].axhline(1.0, color="black", ls="--", lw=1)
    axes[2].axhspan(
        1.0 - cfg.borderline_relative_margin,
        1.0 + cfg.borderline_relative_margin,
        color="0.85",
        alpha=0.6,
        zorder=0,
        label="manual-review margin",
    )
    axes[2].set_xticks(x_position, [str(row["candidate_number"]) for row in ordered], rotation=90)
    axes[2].set(xlabel="candidate number", ylabel="decision score (real spectrum > 1)")
    axes[2].set_title("(c) Classification margin")
    png = figure_dir / "manual_quiet_window_feature_analysis.png"
    pdf = figure_dir / "manual_quiet_window_feature_analysis.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "objective": "identify manually defined pure-background MINPA Mode-1 windows",
        "manual_classes": {
            "weak_real_spectrum": sorted(weak),
            "pure_background": sorted(clean),
        },
        "features": {
            "time_mean_def_peak_to_median": "maximum of the time-mean H+ DEF energy spectrum divided by its positive-energy median",
            "h_count_top3_energy_fraction": "largest sum in three adjacent energy channels divided by total H+ count equivalent over energy",
        },
        "configured_thresholds": {
            "time_mean_def_peak_to_median": cfg.def_peak_to_median_threshold,
            "h_count_top3_energy_fraction": cfg.h_count_top3_fraction_threshold,
            "rule": "real spectrum only when both features exceed threshold; otherwise pure background",
        },
        "configured_training_metrics": configured_metrics,
        "configured_borderline_candidate_numbers": [
            int(row["candidate_number"])
            for row in rows
            if bool(row["manual_review_recommended"])
        ],
        "fitted_thresholds": {
            "time_mean_def_peak_to_median": fitted_x,
            "h_count_top3_energy_fraction": fitted_z,
        },
        "fitted_training_metrics": fitted_metrics,
        "nested_leave_one_out": loo,
        "rows": rows,
        "limitations": [
            "Only 21 manually labeled December 2021 windows are available.",
            "Thresholds require validation on additional manually reviewed dates and instrument states.",
            "The count-equivalent feature depends on the released-DPF empirical quantum and is not an undocumented raw detector count.",
            "Borderline scores near one must remain subject to manual review.",
        ],
        "outputs": [str(csv_path.resolve()), str(png.resolve()), str(pdf.resolve())],
    }
    output = log_dir / "manual_quiet_window_feature_analysis.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "configured_training_metrics": configured_metrics,
        "nested_leave_one_out_accuracy": loo["accuracy"],
        "outputs": summary["outputs"] + [str(output.resolve())],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
