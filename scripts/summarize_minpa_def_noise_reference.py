"""Derive a review-only MINPA DEF noise reference from approved intervals."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import (  # noqa: E402
    MODE1_AZIMUTH_EDGES_DEG,
    MODE1_ENERGY_EV,
    MODE1_PITCH_EDGES_DEG,
)
from highE.minpa_background_temporal import (  # noqa: E402
    TEMPORAL_ALGORITHM_VERSION,
    _atomic_csv,
    _atomic_json,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "minpa_background_temporal"
SPECIES = ((0, "H+"), (3, "O+"), (5, "O2+"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-month", default="202112")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    month = str(args.reference_month)
    month_dir = output / "months" / month
    summary = json.loads((month_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(month_dir / "noise_arrays.npz") as arrays:
        quantum = np.asarray(arrays["dpf_quantum"], dtype=float)
        interval_lambda = np.asarray(arrays["interval_lambda_pam"], dtype=float)
        pooled_lambda = np.asarray(arrays["lambda_pam"], dtype=float)
    metrics = list(summary["interval_metrics"])
    if interval_lambda.shape[0] != len(metrics):
        raise ValueError("interval lambda arrays and summary metrics do not match")

    theta = np.deg2rad(MODE1_PITCH_EDGES_DEG)
    phi = np.deg2rad(MODE1_AZIMUTH_EDGES_DEG)
    solid_angle = (
        (np.cos(theta[:-1]) - np.cos(theta[1:]))[:, None]
        * np.diff(phi)[None, :]
    )
    solid_angle_sum = float(np.sum(solid_angle))
    quantum_def = quantum * MODE1_ENERGY_EV[:, None, None, None]
    interval_background_def = quantum_def[None, ...] * interval_lambda[:, None, ...]
    interval_spectrum = np.sum(
        interval_background_def * solid_angle[None, None, :, :, None],
        axis=(2, 3),
    ) / solid_angle_sum
    interval_scalar = np.nanmedian(interval_spectrum, axis=1)
    pooled_background_def = quantum_def * pooled_lambda[None, ...]
    pooled_spectrum = np.sum(
        pooled_background_def * solid_angle[None, :, :, None], axis=(1, 2)
    ) / solid_angle_sum
    one_event_1d_def = (
        quantum_def * solid_angle[None, :, :, None] / solid_angle_sum
    )

    species_rows = []
    for mass_index, name in SPECIES:
        values = interval_scalar[:, mass_index]
        one_event = one_event_1d_def[..., mass_index].reshape(-1)
        species_rows.append({
            "species": name,
            "mass_channel_index": mass_index,
            "reference_def_median": float(np.nanmedian(values)),
            "reference_def_p16": float(np.nanpercentile(values, 16)),
            "reference_def_p25": float(np.nanpercentile(values, 25)),
            "reference_def_p75": float(np.nanpercentile(values, 75)),
            "reference_def_p84": float(np.nanpercentile(values, 84)),
            "pooled_record_model_def": float(np.nanmedian(pooled_spectrum[:, mass_index])),
            "one_event_1d_def_p10": float(np.nanpercentile(one_event, 10)),
            "one_event_1d_def_median": float(np.nanmedian(one_event)),
            "one_event_1d_def_p90": float(np.nanpercentile(one_event, 90)),
        })
    species_medians = [row["reference_def_median"] for row in species_rows]
    common_median = float(np.nanmedian(species_medians))
    robust_upper = float(
        10.0 ** np.ceil(np.log10(max(row["reference_def_p75"] for row in species_rows)))
    )
    interval_rows = []
    for index, metric in enumerate(metrics):
        row = {
            "candidate_id": metric["candidate_id"],
            "start_utc": metric["start_utc"],
            "stop_utc": metric["stop_utc"],
            "kept_records": metric["kept_records"],
            "selection_source": metric["selection_source"],
        }
        for mass_index, name in SPECIES:
            row[f"def_{name.replace('+', 'plus')}"] = float(interval_scalar[index, mass_index])
        interval_rows.append(row)

    report = {
        "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "reference_month": month,
        "reference_interval_count": len(metrics),
        "reference_record_count": int(summary["accepted_record_count"]),
        "all_intervals_manual_reference": all(
            item.get("selection_source") == "manual_2021_reference" for item in metrics
        ),
        "model_valid_under_production_thresholds": bool(
            summary["model_valid_under_production_thresholds"]
        ),
        "model_invalid_reasons": summary["model_invalid_reasons"],
        "quantity": "solid-angle-weighted 1D background DEF, energy median",
        "formula": (
            "median_E(sum_pitch,azimuth[E * dpf_quantum * lambda * dOmega] / "
            "sum_pitch,azimuth[dOmega])"
        ),
        "unit": "1/(s cm^2 sr)",
        "species": species_rows,
        "common_species_median_def": common_median,
        "recommended_rounded_typical_def": 5000.0,
        "recommended_robust_upper_def": robust_upper,
        "usage": (
            "Reference for 5-10 minute time/angle-averaged Mode-1 spectra only; "
            "not a constant to subtract from raw four-dimensional cells."
        ),
        "known_limitations": [
            "Only 19 intervals, below the configured 20-interval production threshold.",
            "Quarterly visual audit shows residual structured spectra in some accepted windows.",
            "Individual nonzero cells are quantized near 1e6 DEF; their projected one-event "
            "1D contribution is about 1e4-1e5 DEF and must not be compared directly with "
            "the time-averaged reference.",
        ],
    }
    json_path = output / "data" / "def_noise_reference_202112_manual.json"
    csv_path = output / "data" / "def_noise_reference_202112_intervals.csv"
    _atomic_json(json_path, report)
    _atomic_csv(csv_path, interval_rows)
    print(json.dumps({
        "json": str(json_path.resolve()),
        "interval_csv": str(csv_path.resolve()),
        "typical_def": report["recommended_rounded_typical_def"],
        "robust_upper_def": report["recommended_robust_upper_def"],
        "species": species_rows,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
