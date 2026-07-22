"""Validate provisional MINPA background correction in the high-energy O+/O2+ moment chain."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import load_background_model, sha256_file  # noqa: E402
from highE.tw1_minpa import Tw1Paths, process_tw1_day  # noqa: E402


DEFAULT_MODEL = (
    ROOT
    / "outputs"
    / "minpa_background_paper_reproduction"
    / "data"
    / "minpa_mode1_background_model_provisional.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "minpa_background_paper_reproduction"
EVENTS = {
    "uv": ("20211202", "2021-12-02T14:18:00Z", "2021-12-02T14:26:00Z"),
    "weak": ("20211225", "2021-12-25T23:05:00Z", "2021-12-25T23:15:00Z"),
    "strong": ("20211231", "2021-12-31T02:40:00Z", "2021-12-31T08:40:00Z"),
}
SPECIES = {1: "O+", 2: "O2+"}
COLORS = {1: "#0072B2", 2: "#D55E00"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--project-quality-root",
        type=Path,
        default=ROOT / "outputs" / "tw1_minpa_quality_flags_all_species",
    )
    return parser.parse_args()


def _unix(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _event_mask(result: dict[str, np.ndarray], event: tuple[str, str, str]) -> np.ndarray:
    return (result["epoch_unix_s"] >= _unix(event[1])) & (
        result["epoch_unix_s"] < _unix(event[2])
    )


def _speed(result: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return np.linalg.norm(
        np.column_stack([result[f"{prefix}_v_mso_{axis}_km_s"] for axis in "xyz"]),
        axis=1,
    )


def main() -> int:
    args = _parse_args()
    model = load_background_model(args.background_model)
    paths = Tw1Paths(project_quality_root=args.project_quality_root)
    results: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for name, event in EVENTS.items():
        result, summary = process_tw1_day(
            event[0],
            paths,
            write_output=False,
            background_policy="subtract-and-reject-uv",
            background_model=model,
            background_model_path=args.background_model,
            allow_provisional_background_model=True,
        )
        results[name] = result
        summaries[name] = summary

    default_result, _ = process_tw1_day(EVENTS["strong"][0], paths, write_output=False)
    default_regression = {
        "density_equals_raw_equal_nan": bool(
            np.array_equal(
                default_result["density_cm3"],
                default_result["raw_density_cm3"],
                equal_nan=True,
            )
        ),
        "velocity_mso_equals_raw_equal_nan": bool(
            all(
                np.array_equal(
                    default_result[f"v_mso_{axis}_km_s"],
                    default_result[f"raw_v_mso_{axis}_km_s"],
                    equal_nan=True,
                )
                for axis in "xyz"
            )
        ),
    }

    metrics: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "high-energy O+/O2+ integration validation; not the paper's H+ SWIA comparison",
        "background_model": {
            "path": str(args.background_model.resolve()),
            "sha256": sha256_file(args.background_model),
            "algorithm_version": model.algorithm_version,
            "primary_estimator": model.primary_estimator,
            "provisional": summaries["strong"]["background_model"]["provisional"],
        },
        "default_no_model_regression": default_regression,
        "events": {},
    }

    weak = results["weak"]
    weak_window = _event_mask(weak, EVENTS["weak"])
    weak_metrics: dict[str, object] = {}
    for code, label in SPECIES.items():
        mask = (
            weak_window
            & (weak["species_code"] == code)
            & (weak["uv_rejected_flag"] == 0)
            & np.isfinite(weak["raw_density_cm3"])
            & np.isfinite(weak["corrected_density_cm3"])
            & (weak["raw_density_cm3"] > 0.0)
        )
        ratio = weak["corrected_density_cm3"][mask] / weak["raw_density_cm3"][mask]
        fraction = weak["background_density_fraction"][mask]
        weak_metrics[label] = {
            "comparable_rows": int(np.count_nonzero(mask)),
            "corrected_to_raw_density_median": float(np.nanmedian(ratio)) if ratio.size else None,
            "background_density_fraction_median": float(np.nanmedian(fraction))
            if fraction.size
            else None,
            "background_density_fraction_p95": float(np.nanquantile(fraction, 0.95))
            if fraction.size
            else None,
        }

    strong = results["strong"]
    strong_window = _event_mask(strong, EVENTS["strong"])
    raw_speed = _speed(strong, "raw")
    corrected_speed = _speed(strong, "corrected")
    strong_metrics: dict[str, object] = {}
    for code, label in SPECIES.items():
        mask = (
            strong_window
            & (strong["species_code"] == code)
            & (strong["uv_rejected_flag"] == 0)
            & np.isfinite(raw_speed)
            & np.isfinite(corrected_speed)
            & (raw_speed > 0.0)
        )
        relative = np.abs(corrected_speed[mask] - raw_speed[mask]) / raw_speed[mask]
        strong_metrics[label] = {
            "comparable_rows": int(np.count_nonzero(mask)),
            "relative_speed_change_median": float(np.nanmedian(relative))
            if relative.size
            else None,
            "relative_speed_change_p95": float(np.nanquantile(relative, 0.95))
            if relative.size
            else None,
            "relative_speed_change_above_5pct_fraction": float(np.mean(relative > 0.05))
            if relative.size
            else None,
        }

    uv = results["uv"]
    uv_window = _event_mask(uv, EVENTS["uv"])
    uv_records = uv_window & (uv["species_code"] == 1)
    uv_metrics = {
        "records": int(np.count_nonzero(uv_records)),
        "project_bit3_records": int(
            np.count_nonzero(uv_records & (uv["uv_project_quality_flag"] == 1))
        ),
        "algorithm_confirmed_records": int(
            np.count_nonzero(uv_records & (uv["uv_algorithm_confirmed_flag"] == 1))
        ),
        "final_rejected_records": int(
            np.count_nonzero(uv_records & (uv["uv_rejected_flag"] == 1))
        ),
    }
    metrics["events"] = {
        "2021-12-02_uv": uv_metrics,
        "2021-12-25_weak_background": weak_metrics,
        "2021-12-31_strong_signal": strong_metrics,
    }
    comparable = np.isfinite(strong["raw_density_cm3"]) & np.isfinite(
        strong["corrected_density_cm3"]
    )
    metrics["sanity_checks"] = {
        "corrected_density_not_above_raw_violation_count": int(
            np.count_nonzero(
                comparable
                & (strong["corrected_density_cm3"] > strong["raw_density_cm3"] + 1.0e-12)
            )
        ),
        "mode1_only_background_application": bool(
            np.all(strong["mode"][strong["background_applied"] == 1] == 1)
        ),
    }

    figure_dir = args.output_root / "figures"
    log_dir = args.output_root / "logs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    _plot_validation(results, figure_dir)
    metrics_path = log_dir / "moment_background_validation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "complete", "metrics": str(metrics_path.resolve())}, ensure_ascii=False))
    return 0


def _plot_validation(results: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
    weak = results["weak"]
    weak_window = _event_mask(weak, EVENTS["weak"])
    for code, label in SPECIES.items():
        mask = weak_window & (weak["species_code"] == code)
        time = [datetime.fromtimestamp(value, UTC) for value in weak["epoch_unix_s"][mask]]
        axes[0].plot(time, weak["raw_density_cm3"][mask], color=COLORS[code], alpha=0.45, label=f"{label} raw")
        axes[0].plot(time, weak["corrected_density_cm3"][mask], color=COLORS[code], lw=1.8, label=f"{label} corrected")
        axes[1].plot(time, 100.0 * weak["background_density_fraction"][mask], color=COLORS[code], label=label)
    axes[0].set(ylabel=r"density (cm$^{-3}$)", title="(a) 2021-12-25 23:05–23:15 UTC: high-energy density")
    axes[0].set_yscale("log")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set(ylabel="background contribution (%)", title="(b) Density fraction removed before integration")
    axes[1].legend(ncol=2, fontsize=8)

    strong = results["strong"]
    strong_window = _event_mask(strong, EVENTS["strong"])
    raw_speed = _speed(strong, "raw")
    corrected_speed = _speed(strong, "corrected")
    for code, label in SPECIES.items():
        mask = strong_window & (strong["species_code"] == code) & (raw_speed > 0.0)
        time = [datetime.fromtimestamp(value, UTC) for value in strong["epoch_unix_s"][mask]]
        relative = 100.0 * np.abs(corrected_speed[mask] - raw_speed[mask]) / raw_speed[mask]
        axes[2].plot(time, relative, ".", ms=2.5, color=COLORS[code], label=label)
    axes[2].axhline(5.0, color="black", ls="--", lw=1.0, label="5% acceptance target")
    axes[2].set(ylabel="absolute speed change (%)", title="(c) 2021-12-31 02:40–08:40 UTC: high-energy speed stability")
    axes[2].legend(ncol=3, fontsize=8)

    uv = results["uv"]
    uv_mask = _event_mask(uv, EVENTS["uv"]) & (uv["species_code"] == 1)
    uv_time = [datetime.fromtimestamp(value, UTC) for value in uv["epoch_unix_s"][uv_mask]]
    axes[3].step(uv_time, uv["uv_project_quality_flag"][uv_mask], where="mid", label="project Flag 3")
    axes[3].step(uv_time, 1.4 + uv["uv_algorithm_confirmed_flag"][uv_mask], where="mid", label="morphology algorithm (+1.4)")
    axes[3].step(uv_time, 2.8 + uv["uv_rejected_flag"][uv_mask], where="mid", label="final whole-record rejection (+2.8)")
    axes[3].set(ylabel="flag diagnostic", title="(d) 2021-12-02 14:18–14:26 UTC: UV event closure")
    axes[3].legend(ncol=3, fontsize=8)
    axes[3].set_yticks([])
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle("Provisional MINPA Mode-1 background correction validation in the high-energy O+/O2+ chain")
    fig.savefig(output / "moment_background_validation.png", dpi=200)
    fig.savefig(output / "moment_background_validation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

