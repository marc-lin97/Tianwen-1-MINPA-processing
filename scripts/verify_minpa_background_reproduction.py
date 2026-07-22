"""Verify the MINPA background-quantification evidence generated locally.

SWIA comparisons and downstream H+ Maxwellian fits are intentionally outside
this gate because neither contributes to interval selection or background DPF.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "minpa_background_paper_reproduction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--require-production",
        action="store_true",
        help="Also require manual interval approval and the production benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    reproduction = _read_json(root / "logs" / "reproduction_summary.json")
    moments = _read_json(root / "logs" / "moment_background_validation.json")
    availability = _read_json(root / "logs" / "data_availability_manifest.json")

    poisson_channels = reproduction.get("poisson_test", {}).get("channels", [])
    expected_energy = np.array([3.548928, 11.40385, 9939.98, 25290.0])
    actual_energy = np.array([item.get("energy_eV", np.nan) for item in poisson_channels])
    lambda_relative = np.array(
        [abs(item.get("relative_lambda_difference", np.inf)) for item in poisson_channels]
    )
    quantization = np.array(
        [item.get("quantization_valid_fraction", 0.0) for item in poisson_channels]
    )

    required_artifacts = [
        "data/minpa_mode1_background_model_provisional.npz",
        "data/background_interval_candidates.json",
        "figures/candidate_interval_context_montage.png",
        "figures/figure03_orbit_context_dpf.png",
        "figures/figure04_background_intervals_dpf.png",
        "figures/figure05_hplus_count_equivalent_reproduction.png",
        "figures/figure05_hplus_count_equivalent_reproduction.pdf",
        "figures/figure09_channel_dependence_dpf.png",
        "figures/figure10_count_equivalent_poisson_fit.png",
        "figures/moment_background_validation.png",
    ]
    artifact_checks = {
        relative: (root / relative).exists() and (root / relative).stat().st_size > 0
        for relative in required_artifacts
    }
    acquisition = availability.get("acquisition_policy", {})
    exclusions = reproduction.get("scope_exclusions", {})
    coherence = reproduction.get("coherent_signal_screen", {})
    figure5 = reproduction.get("figure5", {})
    figure5_panels = {
        item.get("panel"): item for item in figure5.get("panels", [])
    }
    quiet_validation = figure5_panels.get("a", {}).get(
        "quiet_reference_validation", {}
    )
    strong_validation = figure5_panels.get("b", {}).get(
        "strong_signal_validation", {}
    )
    method_checks = {
        "model_valid": reproduction.get("model_valid") is True,
        "forty_preliminary_candidate_intervals": reproduction.get("candidate_count") == 40,
        "coherent_hplus_signal_screen_applied": (
            coherence.get("coherent_signal_rejected_count", 0) > 0
            and coherence.get("retained_candidate_count", 0)
            + coherence.get("coherent_signal_rejected_count", 0)
            == reproduction.get("candidate_count")
        ),
        "only_coherence_screened_intervals_enter_model": (
            reproduction.get("accepted_interval_count")
            == coherence.get("retained_candidate_count")
        ),
        "minimum_20_coherence_screened_intervals": reproduction.get(
            "accepted_interval_count", 0
        )
        >= 20,
        "minimum_300_accepted_records": reproduction.get("accepted_record_count", 0) >= 300,
        "quality_retain_bits_1_2_5": reproduction.get("quality_policy", {}).get(
            "retain_bits"
        )
        == [1, 2, 5],
        "quality_reject_bits_3_4": reproduction.get("quality_policy", {}).get(
            "reject_bits"
        )
        == [3, 4],
        "accepted_intervals_have_no_bit3_or_bit4": all(
            reproduction.get("accepted_project_quality_record_counts_by_bit", {}).get(str(bit))
            == 0
            for bit in (3, 4)
        ),
        "figure10_exact_energy_channels": (
            actual_energy.shape == expected_energy.shape
            and np.allclose(actual_energy, expected_energy, rtol=0.0, atol=1.0e-6)
        ),
        "figure10_lambda_within_15_percent_of_paper": (
            lambda_relative.size == 4 and bool(np.all(lambda_relative <= 0.15))
        ),
        "figure10_quantization_valid_above_99_9_percent": (
            quantization.size == 4 and bool(np.all(quantization >= 0.999))
        ),
        "figure5_uses_paper_nonzero_estimator": figure5.get(
            "background_estimator"
        )
        == "paper_nonzero_mean",
        "figure5_quiet_reference_residual_below_5_percent": (
            quiet_validation.get("positive_cell_residual_fraction", np.inf) < 0.05
            and quiet_validation.get("summed_count_residual_fraction", np.inf) < 0.05
        ),
        "figure5_strong_peak_preserved": (
            abs(strong_validation.get("peak_energy_index_shift", 99)) <= 1
            and strong_validation.get("maximum_count_retained_fraction", 0.0) >= 0.90
        ),
        "default_no_model_density_regression": moments.get(
            "default_no_model_regression", {}
        ).get("density_equals_raw_equal_nan")
        is True,
        "default_no_model_velocity_regression": moments.get(
            "default_no_model_regression", {}
        ).get("velocity_mso_equals_raw_equal_nan")
        is True,
        "corrected_density_never_above_raw": moments.get("sanity_checks", {}).get(
            "corrected_density_not_above_raw_violation_count"
        )
        == 0,
        "mode1_only_background_application": moments.get("sanity_checks", {}).get(
            "mode1_only_background_application"
        )
        is True,
        "local_mission_data_read_only": (
            acquisition.get("mission_data_access") == "local_read_only"
            and acquisition.get("network_download_performed") is False
        ),
        "swia_and_maxwellian_excluded": (
            "figures_6_8_swia" in exclusions and "figure_11_maxwellian" in exclusions
        ),
        "all_required_artifacts_nonempty": all(artifact_checks.values()),
    }
    method_ready = all(method_checks.values())

    production_requirements = {
        "method_reproduction_ready": method_ready,
        "intervals_manually_approved": reproduction.get("status")
        == "approved_model_complete",
        "paper_exact_40_interval_times_available": False,
        "single_and_multiprocess_100_file_benchmark_passed": False,
    }
    production_ready = all(production_requirements.values())
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "background_quantification_method_ready": method_ready,
        "method_checks": method_checks,
        "artifact_checks": artifact_checks,
        "production_background_model_ready": production_ready,
        "production_requirements": production_requirements,
        "scope": {
            "included": "MINPA Figures 3-5, 9-10, interval context, DPF/count-equivalent model, UV and moment-impact sanity checks",
            "excluded": "SWIA Figures 6-8 and H+ Maxwellian Figure 11; they do not affect background estimation",
            "momag": "context for interval review only, not a background value input",
        },
        "interpretation": (
            "The background quantification method is independently testable with local MINPA data. "
            "The current model remains provisional until the retained intervals are manually approved "
            "and the deterministic 100-file benchmark is completed."
        ),
    }
    output = root / "logs" / "reproduction_verification.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not method_ready or (args.require_production and not production_ready):
        raise SystemExit(1)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
