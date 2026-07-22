"""Find a broader MINPA pool and classify pure-background reference windows."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import (  # noqa: E402
    CandidateConfig,
    PROJECT_BACKGROUND_REJECT_MASK,
    discover_background_candidates,
    load_background_model,
    mode1_ori_files_for_interval,
    read_mode1_ori_records,
)
from highE.minpa_figure5 import screen_hplus_coherent_signal  # noqa: E402
from highE.minpa_quiet_classifier import (  # noqa: E402
    QuietBoundaryRefinementConfig,
    QuietReferenceClassifierConfig,
    evaluate_quiet_boundary_refinements,
    extract_quiet_reference_features,
)


DEFAULT_DAY_SPE = Path(r"D:\Data\TW-1\result\MINPA\day_spe")
DEFAULT_ORI = Path(r"D:\Data\TW-1\result\MINPA\ori")
DEFAULT_QUALITY = ROOT / "outputs" / "tw1_minpa_quality_flags_all_species"
DEFAULT_CONFIG = ROOT / "config/minpa_background_reproduction.json"
DEFAULT_REVIEW = ROOT / "config/minpa_background_manual_review.json"
DEFAULT_ORIGINAL = ROOT / "outputs/minpa_background_paper_reproduction/data/background_interval_candidates.json"
DEFAULT_MODEL = ROOT / "outputs/minpa_background_paper_reproduction/data/minpa_mode1_background_model_provisional.npz"
DEFAULT_OUTPUT = ROOT / "outputs/minpa_background_paper_reproduction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day-spe-root", type=Path, default=DEFAULT_DAY_SPE)
    parser.add_argument("--ori-root", type=Path, default=DEFAULT_ORI)
    parser.add_argument("--quality-root", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--original-candidates", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--background-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_day(
    path: str,
    segment: str,
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (path, segment)
    if key not in cache:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        item = getattr(data["H_spe_num"], segment)
        cache[key] = (
            np.asarray(item.t, dtype=float).reshape(-1),
            np.asarray(item.f, dtype=float).reshape(-1),
            np.asarray(item.p, dtype=float),
        )
    return cache[key]


def main() -> None:
    args = parse_args()
    config_payload = json.loads(args.config.read_text(encoding="utf-8"))
    candidate_config = CandidateConfig(**{
        key: value
        for key, value in config_payload["expanded_candidate_selection"].items()
        if key != "purpose"
    })
    refinement_config = QuietBoundaryRefinementConfig(**{
        key: value
        for key, value in config_payload["borderline_boundary_refinement"].items()
        if key != "selection_rule"
    })
    day_paths = sorted(args.day_spe_root.glob("Ion_spe_202112*.mat"))
    day_paths += sorted(args.day_spe_root.glob("Ion_spe_20220101.mat"))
    candidates = discover_background_candidates(day_paths, candidate_config)
    model = load_background_model(args.background_model)
    review = json.loads(args.manual_review.read_text(encoding="utf-8"))
    original = json.loads(args.original_candidates.read_text(encoding="utf-8"))
    weak_numbers = set(review["decisions"]["reject_visible_or_weak_real_spectrum"])
    clean_numbers = set(review["decisions"]["approve_no_visible_real_spectrum"])
    original_by_time: dict[tuple[str, str], tuple[int, str]] = {}
    for index, item in enumerate(original["intervals"], start=1):
        manual_class = "unreviewed"
        if index in weak_numbers:
            manual_class = "weak_real_spectrum"
        elif index in clean_numbers:
            manual_class = "pure_background"
        original_by_time[(str(item["start_utc"]), str(item["stop_utc"]))] = (index, manual_class)

    expanded_decisions = review.get("expanded_review", {}).get("decisions", {})
    expanded_full_approved = {
        int(value) for value in expanded_decisions.get("approve_full_interval", [])
    }
    expanded_rejected = {
        int(value)
        for value in expanded_decisions.get("reject_visible_or_weak_real_spectrum", [])
    }
    expanded_adjusted = {
        int(item["expanded_candidate_number"]): item
        for item in expanded_decisions.get("approve_adjusted_interval", [])
    }

    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    classifier_config = QuietReferenceClassifierConfig()
    for expanded_number, candidate in enumerate(candidates, start=1):
        day_time, _, day_def = _load_day(
            str(candidate["source_day_spe"]),
            str(candidate.get("source_day_spe_segment", "num1")),
            cache,
        )
        mask = (
            (day_time >= float(candidate["start_unix_s"]))
            & (day_time < float(candidate["stop_unix_s"]))
        )
        paths = mode1_ori_files_for_interval(
            args.ori_root, str(candidate["start_utc"]), str(candidate["stop_utc"])
        )
        records = read_mode1_ori_records(
            paths,
            str(candidate["start_utc"]),
            str(candidate["stop_utc"]),
            quality_root=args.quality_root,
            compute_sha256=False,
        )
        gross = screen_hplus_coherent_signal(records.dpf, model.dpf_quantum)
        features = extract_quiet_reference_features(
            day_def[mask], records.dpf, model.dpf_quantum, classifier_config
        )
        gross_rejected = bool(gross["coherent_signal_rejected"])
        weak_rejected = bool(features["real_spectrum_detected"])
        if gross_rejected or weak_rejected:
            predicted_class = "real_spectrum_rejected"
        elif bool(features["manual_review_recommended"]):
            predicted_class = "pure_background_borderline"
        else:
            predicted_class = "pure_background_high_confidence"
        initial_predicted_class = predicted_class
        pre_refinement_decision_score = float(features["decision_score"])
        refinement_status = "not_applicable"
        refinement_trials_evaluated = 0
        refined_start_utc = None
        refined_stop_utc = None
        refined_duration_s = None
        if predicted_class == "pure_background_borderline":
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
                model.dpf_quantum,
                classifier_config,
                refinement_config,
            )
            selected = None
            selected_gross = None
            for trial in trials:
                refinement_trials_evaluated += 1
                trial_start = float(trial["start_unix_s"])
                trial_stop = float(trial["stop_unix_s"])
                all_in_trial = (
                    (records.time_unix_s >= trial_start)
                    & (records.time_unix_s < trial_stop)
                )
                if not np.all(records.project_quality_available[all_in_trial]):
                    continue
                if np.any(
                    (records.project_quality_flag[all_in_trial]
                     & PROJECT_BACKGROUND_REJECT_MASK) != 0
                ):
                    continue
                kept_in_trial = all_in_trial & quality_eligible
                if int(np.count_nonzero(kept_in_trial)) < refinement_config.minimum_dpf_records:
                    continue
                trial_gross = screen_hplus_coherent_signal(
                    records.dpf[kept_in_trial], model.dpf_quantum
                )
                if bool(trial_gross["coherent_signal_rejected"]):
                    continue
                if bool(trial["real_spectrum_detected"]):
                    continue
                if bool(trial["manual_review_recommended"]):
                    continue
                selected = trial
                selected_gross = trial_gross
                break
            if selected is None:
                predicted_class = "borderline_discarded_after_refinement"
                refinement_status = "discarded_no_high_confidence_half_or_longer_subwindow"
            else:
                predicted_class = "pure_background_high_confidence_refined"
                refinement_status = "accepted_high_confidence_subwindow"
                refined_start_s = float(selected["start_unix_s"])
                refined_stop_s = float(selected["stop_unix_s"])
                refined_start_utc = datetime.fromtimestamp(
                    refined_start_s, UTC
                ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                refined_stop_utc = datetime.fromtimestamp(
                    refined_stop_s, UTC
                ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                refined_duration_s = float(selected["duration_s"])
                # Replace classifier and gross diagnostics with those of the
                # selected subwindow, while keeping original candidate bounds.
                features = {key: selected[key] for key in features}
                gross = selected_gross
        original_number, manual_class = original_by_time.get(
            (str(candidate["start_utc"]), str(candidate["stop_utc"])),
            (None, "not_in_original_40"),
        )
        manual_approved_start_utc = None
        manual_approved_stop_utc = None
        if expanded_number in expanded_full_approved:
            manual_class = "pure_background"
            manual_approved_start_utc = str(candidate["start_utc"])
            manual_approved_stop_utc = str(candidate["stop_utc"])
        elif expanded_number in expanded_adjusted:
            manual_class = "pure_background_adjusted"
            manual_approved_start_utc = str(expanded_adjusted[expanded_number]["start_utc"])
            manual_approved_stop_utc = str(expanded_adjusted[expanded_number]["stop_utc"])
        elif expanded_number in expanded_rejected:
            manual_class = "weak_real_spectrum"
        prediction_matches_manual = None
        if manual_class in (
            "weak_real_spectrum", "pure_background", "pure_background_adjusted"
        ):
            prediction_matches_manual = bool(
                (predicted_class in (
                    "real_spectrum_rejected",
                    "borderline_discarded_after_refinement",
                ))
                == (manual_class == "weak_real_spectrum")
            )
        candidate.update({
            "expanded_candidate_number": expanded_number,
            "original_candidate_number": original_number,
            "manual_class": manual_class,
            "manual_approved_start_utc": manual_approved_start_utc,
            "manual_approved_stop_utc": manual_approved_stop_utc,
            **gross,
            **features,
            "initial_predicted_class": initial_predicted_class,
            "pre_refinement_decision_score": pre_refinement_decision_score,
            "boundary_refinement_status": refinement_status,
            "boundary_refinement_trials_evaluated": refinement_trials_evaluated,
            "refined_start_utc": refined_start_utc,
            "refined_stop_utc": refined_stop_utc,
            "refined_duration_s": refined_duration_s,
            "predicted_class": predicted_class,
            "prediction_matches_manual": prediction_matches_manual,
            "approved": False,
        })

    counts = {
        name: sum(candidate["predicted_class"] == name for candidate in candidates)
        for name in (
            "pure_background_high_confidence",
            "pure_background_high_confidence_refined",
            "borderline_discarded_after_refinement",
            "real_spectrum_rejected",
        )
    }
    manual_rows = [
        candidate for candidate in candidates
        if candidate["manual_class"] in (
            "weak_real_spectrum", "pure_background", "pure_background_adjusted"
        )
    ]
    manual_agreement = float(np.mean([
        bool(candidate["prediction_matches_manual"]) for candidate in manual_rows
    ]))

    data_dir = args.output_root / "data"
    figure_dir = args.output_root / "figures"
    log_dir = args.output_root / "logs"
    for directory in (data_dir, figure_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "expanded_quiet_window_candidates.json"
    json_path.write_text(json.dumps({
        "status": "provisional_predictions_require_review",
        "candidate_config": config_payload["expanded_candidate_selection"],
        "classifier_config": classifier_config.__dict__,
        "boundary_refinement_config": config_payload["borderline_boundary_refinement"],
        "counts": counts,
        "manual_label_agreement": manual_agreement,
        "intervals": candidates,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = data_dir / "expanded_quiet_window_candidates.csv"
    fields = [
        "expanded_candidate_number", "original_candidate_number", "start_utc", "stop_utc",
        "manual_class", "manual_approved_start_utc", "manual_approved_stop_utc",
        "initial_predicted_class", "predicted_class", "boundary_refinement_status",
        "refined_start_utc", "refined_stop_utc", "refined_duration_s",
        "pre_refinement_decision_score", "time_mean_def_peak_to_median",
        "h_count_top3_energy_fraction", "decision_score", "classification_confidence",
        "maximum_energy_occupancy", "maximum_weak_energy_occupancy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({field: candidate.get(field) for field in fields})

    contexts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for candidate in candidates:
        time, energy, spectrum = _load_day(
            str(candidate["source_day_spe"]),
            str(candidate.get("source_day_spe_segment", "num1")),
            cache,
        )
        mask = (
            (time >= float(candidate["start_unix_s"]) - 720.0)
            & (time < float(candidate["stop_unix_s"]) + 720.0)
        )
        contexts.append((time[mask], energy, spectrum[mask]))
    positive = np.concatenate([
        spectrum[np.isfinite(spectrum) & (spectrum > 0.0)]
        for _, _, spectrum in contexts
    ])
    vmin, vmax = np.nanquantile(positive, [0.02, 0.995])
    ncols = 5
    nrows = int(np.ceil(len(candidates) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 2.65 * nrows), sharey=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(-1)
    class_color = {
        "pure_background_high_confidence": "#009e73",
        "pure_background_high_confidence_refined": "#0072b2",
        "borderline_discarded_after_refinement": "#cc79a7",
        "real_spectrum_rejected": "#d55e00",
    }

    def draw_boundaries(axis: plt.Axes, candidate: dict[str, object]) -> None:
        manual_adjusted_values = (
            candidate.get("manual_approved_start_utc"),
            candidate.get("manual_approved_stop_utc"),
        ) if candidate.get("manual_class") == "pure_background_adjusted" else (None, None)
        refined_values = (
            candidate.get("refined_start_utc"), candidate.get("refined_stop_utc")
        )
        retained_values = (
            manual_adjusted_values if all(manual_adjusted_values) else refined_values
        )
        if all(retained_values):
            for key in ("start_unix_s", "stop_unix_s"):
                axis.axvline(
                    datetime.fromtimestamp(float(candidate[key]), UTC),
                    color="#666666", lw=0.9, ls=":", zorder=5,
                )
            for value in retained_values:
                axis.axvline(
                    datetime.fromisoformat(str(value).replace("Z", "+00:00")),
                    color="black", lw=1.4, zorder=6,
                )
        else:
            for key in ("start_unix_s", "stop_unix_s"):
                axis.axvline(
                    datetime.fromtimestamp(float(candidate[key]), UTC),
                    color="black", lw=1.3, zorder=6,
                )
    mesh = None
    for axis, candidate, context in zip(axes_array, candidates, contexts):
        time, energy, spectrum = context
        mesh = axis.pcolormesh(
            [datetime.fromtimestamp(value, UTC) for value in time],
            energy,
            spectrum.T,
            shading="auto",
            norm=LogNorm(max(vmin, np.finfo(float).tiny), max(vmax, vmin * 10.0)),
            cmap="turbo",
        )
        draw_boundaries(axis, candidate)
        axis.set_yscale("log")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.tick_params(labelsize=6)
        color = class_color[str(candidate["predicted_class"])]
        axis.set_title(
            f"{int(candidate['expanded_candidate_number']):02d} {candidate['start_utc'][5:16].replace('T', ' ')} "
            f"score={float(candidate['decision_score']):.2f}\n{candidate['predicted_class'].replace('_', ' ')}",
            fontsize=7,
            color=color,
            fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.4)
    for axis in axes_array[len(candidates):]:
        axis.set_visible(False)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes_array.tolist(), shrink=0.35, label="H+ DEF [1/(s cm² sr)]")
    fig.suptitle(
        "Expanded MINPA quiet-window classifier: green=direct high-confidence, "
        "blue=high-confidence after trimming, magenta=borderline discarded, red=real spectrum"
    )
    figure_path = figure_dir / "expanded_quiet_window_classifier_montage.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    pure_pairs = [
        (candidate, context)
        for candidate, context in zip(candidates, contexts)
        if str(candidate["predicted_class"]).startswith("pure_background")
    ]
    pure_rows = int(np.ceil(len(pure_pairs) / ncols))
    pure_fig, pure_axes = plt.subplots(
        pure_rows,
        ncols,
        figsize=(18, 2.8 * pure_rows),
        sharey=True,
        constrained_layout=True,
    )
    pure_axes_array = np.atleast_1d(pure_axes).reshape(-1)
    pure_mesh = None
    for axis, (candidate, context) in zip(pure_axes_array, pure_pairs):
        time, energy, spectrum = context
        pure_mesh = axis.pcolormesh(
            [datetime.fromtimestamp(value, UTC) for value in time],
            energy,
            spectrum.T,
            shading="auto",
            norm=LogNorm(max(vmin, np.finfo(float).tiny), max(vmax, vmin * 10.0)),
            cmap="turbo",
        )
        draw_boundaries(axis, candidate)
        axis.set_yscale("log")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.tick_params(labelsize=7)
        color = class_color[str(candidate["predicted_class"])]
        original_number = candidate.get("original_candidate_number")
        original_text = f" old={int(original_number):02d}" if original_number is not None else " new"
        axis.set_title(
            f"expanded={int(candidate['expanded_candidate_number']):02d}{original_text} "
            f"{candidate['start_utc'][5:16].replace('T', ' ')}\n"
            f"score={float(candidate['decision_score']):.2f} "
            f"{str(candidate['classification_confidence'])}",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
    for axis in pure_axes_array[len(pure_pairs):]:
        axis.set_visible(False)
    if pure_mesh is not None:
        pure_fig.colorbar(
            pure_mesh,
            ax=pure_axes_array.tolist(),
            shrink=0.65,
            label="H+ DEF [1/(s cm² sr)]",
        )
    pure_fig.suptitle(
        "Predicted pure-background set: black=retained bounds; gray dotted=original bounds when trimmed"
    )
    pure_figure_path = figure_dir / "pure_background_candidate_review_montage.png"
    pure_fig.savefig(pure_figure_path, dpi=200)
    plt.close(pure_fig)

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "candidate_count": len(candidates),
        "counts": counts,
        "manual_label_count": len(manual_rows),
        "manual_label_agreement": manual_agreement,
        "production_use": False,
        "reason": (
            "Borderline windows are retained only after a half-duration-or-longer "
            "subwindow reaches high confidence; production still requires the configured "
            "approved-interval and record minima."
        ),
        "outputs": [
            str(json_path.resolve()),
            str(csv_path.resolve()),
            str(figure_path.resolve()),
            str(pure_figure_path.resolve()),
        ],
    }
    summary_path = log_dir / "expanded_quiet_window_classifier_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**summary, "summary": str(summary_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
