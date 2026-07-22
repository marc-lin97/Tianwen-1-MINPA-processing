from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from highE.minpa_background import MODE1_ENERGY_EV, MODE1_SHAPE, MinpaMode1Records
from highE.minpa_background_temporal import (
    _manual_decision_map,
    aggregate_quarterly_summaries,
    analyze_quarterly_noise,
    analyze_temporal_noise,
    benjamini_hochberg,
    block_bootstrap_median,
    build_mode1_inventory,
    compare_month_output_directories,
    discover_monthly_quiet_candidates,
    estimate_interval_noise,
    process_month_to_directory,
    select_quarter_review_intervals,
)


ROOT = Path(__file__).resolve().parents[1]


def _segment(time: np.ndarray, energy: np.ndarray, mode: int, level: float = 1.0) -> dict[str, object]:
    return {
        "t": time,
        "p": np.full((time.size, energy.size), level, dtype=float),
        "f": energy,
        "mod": mode,
        "quality_flag": np.zeros(time.size, dtype=np.uint32),
        "quality_flag_available": np.ones(time.size, dtype=np.uint8),
    }


def _write_day(path: Path, time: np.ndarray, level: float, *, mixed: bool = False) -> None:
    container: dict[str, object] = {}
    if mixed:
        container["num1"] = _segment(time[:10], np.arange(64, dtype=float) + 1.0, 4, level)
    container["num2" if mixed else "num1"] = _segment(time, MODE1_ENERGY_EV, 1, level)
    savemat(path, {
        "H_spe_num": container,
        "O_spe_num": container,
        "O2_spe_num": container,
    })


def test_inventory_strictly_counts_mode1_inside_mixed_day(tmp_path: Path) -> None:
    day_root, ori_root, quality_root = tmp_path / "day", tmp_path / "ori", tmp_path / "quality"
    day_root.mkdir(); ori_root.mkdir(); quality_root.mkdir()
    time = np.arange(30, dtype=float) * 16.0
    _write_day(day_root / "Ion_spe_19700101.mat", time, 1.0, mixed=True)
    (quality_root / "19700101").mkdir()
    (ori_root / "TW1_MINPA-MOD1_x_19700101000000_19700101001000_x.mat").write_bytes(b"x")

    inventory = build_mode1_inventory(day_root, ori_root, quality_root)

    assert inventory["day_spe_file_count"] == 1
    assert inventory["strict_mode1_segment_count"] == 1
    assert inventory["strict_mode1_record_count"] == 30
    assert inventory["mixed_mode_day_count"] == 1
    assert inventory["mode_segment_counts"] == {"1": 1, "4": 1}
    assert inventory["mode1_days_missing_quality"] == []


def test_candidate_quantile_is_computed_independently_per_month(tmp_path: Path) -> None:
    day_root = tmp_path / "day"
    day_root.mkdir()
    first = datetime(2021, 12, 1, tzinfo=UTC).timestamp() + np.arange(40) * 16.0
    second = datetime(2022, 3, 1, tzinfo=UTC).timestamp() + np.arange(40) * 16.0
    _write_day(day_root / "Ion_spe_20211201.mat", first, 1.0)
    _write_day(day_root / "Ion_spe_20220301.mat", second, 100.0)
    inventory = {
        "day_spe_files": [
            {"month": "202112", "path": str(day_root / "Ion_spe_20211201.mat"), "mode1_segment_count": 1},
            {"month": "202203", "path": str(day_root / "Ion_spe_20220301.mat"), "mode1_segment_count": 1},
        ]
    }
    config = {
        "duration_s": 480.0, "step_s": 240.0, "min_records": 24,
        "max_record_gap_s": 25.0, "reject_quality_mask": 12,
        "low_signal_quantile": 0.5, "max_intervals_per_day": 4,
        "min_separation_s": 0.0,
    }

    candidates = discover_monthly_quiet_candidates(inventory, config)

    assert candidates["202112"]
    assert candidates["202203"]
    assert candidates["202203"][0]["median_total_h_def"] > candidates["202112"][0]["median_total_h_def"]


def _records(cube: np.ndarray) -> MinpaMode1Records:
    n = cube.shape[0]
    return MinpaMode1Records(
        time_unix_s=np.arange(n, dtype=float) * 16.0,
        dpf=cube,
        native_quality=np.zeros(n, dtype=np.uint32),
        ion_start_stop_counts=np.tile(np.arange(4, dtype=float), (n, 1)),
        high_voltage_monitor_v=np.ones((n, 5, 4), dtype=float) * 100.0,
        instrument_solar_angles_deg=np.ones((n, 2), dtype=float) * 85.0,
        project_quality_flag=np.zeros(n, dtype=np.uint32),
        project_quality_available=np.ones(n, dtype=bool),
        source_file_index=np.zeros(n, dtype=np.int32),
    )


def test_interval_noise_recovers_known_count_equivalent() -> None:
    quantum = np.broadcast_to(1.0 / MODE1_ENERGY_EV[:, None, None, None], MODE1_SHAPE).copy()
    cube = np.broadcast_to(2.0 * quantum, (24, *MODE1_SHAPE)).copy()

    result = estimate_interval_noise(_records(cube), quantum)

    assert np.isclose(result["lambda_global"], 2.0)
    assert np.allclose(result["lambda_mass"], 2.0)
    assert np.all(result["background_density_cm3_mass"] > 0.0)
    assert np.allclose(result["quantum_reconstruction_fraction_mass"], 1.0)


def test_fdr_and_bootstrap_are_deterministic() -> None:
    assert np.allclose(benjamini_hochberg([0.01, 0.04, 0.2]), [0.03, 0.06, 0.2])
    first = block_bootstrap_median([1, 2, 3, 4], replicates=200, seed=7)
    second = block_bootstrap_median([1, 2, 3, 4], replicates=200, seed=7)
    assert first == second


def _interval_metric(day: str, value: float, quantum: float = 10.0) -> dict[str, object]:
    return {
        "day": day,
        "midpoint_unix_s": 0.0,
        "raw_records": 24,
        "kept_records": 24,
        "lambda_global": value,
        "lambda_mass": np.full(8, value),
        "quantum_def_mass": np.full(8, quantum),
        "zero_fraction_mass": np.full(8, 0.5),
        "positive_fraction_mass": np.full(8, 0.5),
        "quantum_reconstruction_fraction_mass": np.full(8, 1.0),
        "background_density_cm3_mass": np.full(8, value * quantum * 1.0e-6),
        "start_stop_median": np.arange(4, dtype=float),
        "high_voltage_median_v": np.ones((5, 4), dtype=float) * 100.0,
        "solar_angle_median_deg": np.array([85.0, 86.0]),
        "lambda_pam": np.full((4, 16, 8), value),
    }


def _analysis_config() -> dict[str, object]:
    return {
        "statistics": {
            "bootstrap_replicates": 300,
            "permutation_replicates": 300,
            "random_seed": 123,
            "fdr_alpha": 0.05,
            "practical_relative_change": 0.2,
            "coverage_gap_days": 45.0,
            "minimum_segment_months": 2,
        }
    }


def _month_summary(month: str, value: float, *, eligible: bool = True, quantum: float = 10.0) -> dict[str, object]:
    metrics = [
        _interval_metric(f"{month}{day:02d}", value * (1.0 + day * 1.0e-4), quantum)
        for day in range(1, 11)
        for _ in range(2)
    ]
    return {
        "month": month,
        "candidate_count": 40,
        "accepted_interval_count": len(metrics),
        "accepted_record_count": len(metrics) * 24,
        "distinct_day_count": 10,
        "inference_eligible": eligible,
        "interval_metrics": metrics,
    }


def test_temporal_analysis_detects_30_percent_step_but_not_10_percent_as_practical() -> None:
    months = [f"2022{month:02d}" for month in range(1, 9)]
    thirty = [
        _month_summary(month, 1.0 if index < 4 else 1.3)
        for index, month in enumerate(months)
    ]
    ten = [
        _month_summary(month, 1.0 if index < 4 else 1.1)
        for index, month in enumerate(months)
    ]

    detected = analyze_temporal_noise(thirty, _analysis_config())
    small = analyze_temporal_noise(ten, _analysis_config())

    assert any(
        item["metric"] == "lambda_global" and item["conclusion"] == "practically_significant"
        for item in detected["change_comparisons"]
    )
    assert not any(
        item["conclusion"] == "practically_significant"
        for item in small["change_comparisons"]
    )


def test_temporal_analysis_does_not_bridge_long_coverage_gap() -> None:
    summaries = [
        _month_summary("202201", 1.0), _month_summary("202202", 1.0),
        _month_summary("202206", 1.5), _month_summary("202207", 1.5),
    ]

    result = analyze_temporal_noise(summaries, _analysis_config())

    assert not any(
        item["left_stop_month"] == "202202" and item["right_start_month"] == "202206"
        for item in result["change_comparisons"]
    )


def test_quantum_only_step_is_classified_as_calibration_scale_change() -> None:
    months = [f"2022{month:02d}" for month in range(1, 9)]
    summaries = [
        _month_summary(month, 1.0, quantum=10.0 if index < 4 else 13.0)
        for index, month in enumerate(months)
    ]

    result = analyze_temporal_noise(summaries, _analysis_config())

    assert any(
        item["interpretation"] == "calibration_scale_change_only"
        for item in result["interpreted_changes"]
    )


def test_sparse_month_is_descriptive_only() -> None:
    result = analyze_temporal_noise(
        [_month_summary("202201", 1.0, eligible=False)], _analysis_config()
    )
    assert result["monthly"][0]["analysis_class"] == "insufficient_for_inference"
    assert result["eligible_month_count"] == 0


def test_empty_month_checkpoint_is_resumable(tmp_path: Path) -> None:
    config = json.loads((ROOT / "config" / "minpa_background_full_mission.json").read_text(encoding="utf-8"))
    quality = tmp_path / "quality"
    quality.mkdir()

    first = process_month_to_directory(
        "209901", [], [], quality, tmp_path / "out", config, None, resume=True
    )
    second = process_month_to_directory(
        "209901", [], [], quality, tmp_path / "out", config, None, resume=True
    )

    assert first["status"] == "complete"
    assert second["resume_action"] == "skipped_valid_checkpoint"


def test_existing_manual_review_keeps_candidate25_rejected_and_adjusted_bounds() -> None:
    path = ROOT / "outputs" / "minpa_background_paper_reproduction" / "data" / "expanded_quiet_window_candidates.json"
    decisions = _manual_decision_map(path)
    candidate25 = next(
        item for item in json.loads(path.read_text(encoding="utf-8"))["intervals"]
        if int(item["expanded_candidate_number"]) == 25
    )
    candidate1 = next(
        item for item in json.loads(path.read_text(encoding="utf-8"))["intervals"]
        if int(item["expanded_candidate_number"]) == 1
    )

    rejected = decisions[(candidate25["start_utc"], candidate25["stop_utc"])]
    adjusted = decisions[(candidate1["start_utc"], candidate1["stop_utc"])]

    assert rejected["accepted"] is False
    assert adjusted["final_start_utc"].endswith("08:32:00Z")
    assert adjusted["final_stop_utc"].endswith("08:38:00Z")


def test_month_output_comparison_ignores_only_runtime_fields(tmp_path: Path) -> None:
    roots = [tmp_path / "single", tmp_path / "parallel"]
    for index, root in enumerate(roots):
        month = root / "months" / "202112"
        month.mkdir(parents=True)
        (month / "summary.json").write_text(json.dumps({
            "month": "202112",
            "created_utc": f"runtime-{index}",
            "finished_utc": f"runtime-{index}",
            "arrays_path": str(month / "noise_arrays.npz"),
            "accepted_interval_count": 3,
        }), encoding="utf-8")
        (month / "candidates.csv").write_bytes(b"candidate_id,accepted\na,true\n")
        (month / "interval_metrics.csv").write_bytes(b"lambda_global\n1.0\n")
        np.savez_compressed(month / "noise_arrays.npz", values=np.array([1.0, np.nan]))

    result = compare_month_output_directories(roots[0], roots[1], ["202112"])

    assert result["identical"] is True


def test_sparse_directional_lambda_uses_occurrence_mean_not_zero_median() -> None:
    quantum = np.broadcast_to(1.0 / MODE1_ENERGY_EV[:, None, None, None], MODE1_SHAPE).copy()
    cube = np.zeros((24, *MODE1_SHAPE), dtype=float)
    cube[:, :, 0, 0, 0] = quantum[:, 0, 0, 0]

    result = estimate_interval_noise(_records(cube), quantum)

    assert result["lambda_global"] > 0.0
    assert result["lambda_mass"][0] > 0.0
    assert np.isclose(result["lambda_mass"][1:], 0.0).all()


def test_quarterly_pooling_recomputes_thresholds_from_interval_days() -> None:
    config = {
        "monthly_inference": {
            "min_accepted_intervals": 20,
            "min_accepted_records": 300,
            "min_distinct_days": 5,
        }
    }
    monthly = [
        _month_summary("202201", 1.0, eligible=False),
        _month_summary("202202", 1.0, eligible=False),
    ]

    result = aggregate_quarterly_summaries(monthly, config)

    assert result[0]["quarter"] == "2022Q1"
    assert result[0]["accepted_interval_count"] == 40
    assert result[0]["distinct_day_count"] == 20
    assert result[0]["inference_eligible"] is True


def test_quarterly_analysis_detects_persistent_thirty_percent_step() -> None:
    months = ["202201", "202204", "202207", "202210", "202301", "202304", "202307", "202310"]
    summaries = [
        _month_summary(month, 1.0 if index < 4 else 1.3)
        for index, month in enumerate(months)
    ]
    inventory = {
        "day_spe_files": [
            {"day": f"{year}{month:02d}15", "mode1_segment_count": 1}
            for year in (2022, 2023)
            for month in range(1, 13)
        ]
    }

    config = _analysis_config()
    config["monthly_inference"] = {
        "min_accepted_intervals": 20,
        "min_accepted_records": 300,
        "min_distinct_days": 5,
    }
    result = analyze_quarterly_noise(summaries, inventory, config)

    assert result["eligible_quarter_count"] == 8
    assert any(
        item["metric"] == "lambda_global"
        and item["conclusion"] == "practically_significant"
        for item in result["change_comparisons"]
    )
    assert any(
        item["metric"] == "lambda_global"
        and item["conclusion"] == "practically_significant"
        for item in result["eligible_quarter_pairwise_contrasts"]
    )


def test_quarter_review_selection_is_lambda_stratified_and_day_distinct() -> None:
    candidates = [
        {"candidate_id": f"c{index:02d}", "accepted": True, "day": f"202201{index + 1:02d}"}
        for index in range(10)
    ]
    metrics = [
        {"candidate_id": f"c{index:02d}", "lambda_global": float(index), "day": f"202201{index + 1:02d}"}
        for index in range(10)
    ]

    result = select_quarter_review_intervals(
        [{"month": "202201", "candidates": candidates, "interval_metrics": metrics}],
        "2022Q1",
    )

    assert len(result) == 5
    assert len({item["candidate_id"] for item in result}) == 5
    assert len({item["day"] for item in result}) == 5
    assert [item["review_target_quantile"] for item in result] == [0.1, 0.3, 0.5, 0.7, 0.9]
