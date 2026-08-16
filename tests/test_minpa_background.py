from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from highE.minpa_background import (
    DENOISE_POLICY_VERSION,
    MODE1_FLAT_SIZE,
    MODE1_ENERGY_EV,
    MODE1_SHAPE,
    BackgroundConfig,
    BackgroundInterval,
    CandidateConfig,
    MinpaMode1Records,
    aggregate_energy_count_equivalent,
    apply_background,
    approved_intervals_from_manual_review,
    approved_intervals_from_review_inventory,
    detect_uv_contamination,
    discover_background_candidates,
    estimate_background,
    load_background_model,
    reshape_mode1_dpf,
    save_background_model,
)


def test_manual_review_resolves_full_and_adjusted_expanded_intervals() -> None:
    original = {"intervals": [
        {"start_utc": "2021-12-20T00:00:00Z", "stop_utc": "2021-12-20T00:08:00Z"},
    ]}
    expanded = {"intervals": [
        {
            "expanded_candidate_number": 3,
            "start_utc": "2021-12-01T14:56:00Z",
            "stop_utc": "2021-12-01T15:04:00Z",
        },
        {
            "expanded_candidate_number": 7,
            "start_utc": "2021-12-02T07:44:00Z",
            "stop_utc": "2021-12-02T07:52:00Z",
        },
    ]}
    review = {
        "decisions": {"approve_no_visible_real_spectrum": [1]},
        "expanded_review": {"decisions": {
            "approve_full_interval": [3],
            "reject_visible_or_weak_real_spectrum": [],
            "approve_adjusted_interval": [{
                "expanded_candidate_number": 7,
                "start_utc": "2021-12-02T07:46:00Z",
                "stop_utc": "2021-12-02T07:52:00Z",
            }],
        }},
    }

    intervals = approved_intervals_from_manual_review(review, original, expanded)

    assert len(intervals) == 3
    assert intervals[0].label == "expanded-03-full"
    assert intervals[1].label == "expanded-07-adjusted"
    assert intervals[1].start_utc == "2021-12-02T07:46:00Z"
    assert intervals[2].label == "original-01"


def test_manual_review_imports_all_deduplicates_exact_windows_and_rejects_conflicts() -> None:
    original = {"intervals": [
        {"start_utc": "2021-12-20T00:00:00Z", "stop_utc": "2021-12-20T00:08:00Z"},
    ]}
    expanded = {"intervals": [{
        "expanded_candidate_number": 1,
        "start_utc": "2021-12-20T00:00:00Z",
        "stop_utc": "2021-12-20T00:08:00Z",
    }]}
    review = {
        "decisions": {
            "approve_no_visible_real_spectrum": [1],
            "reject_visible_or_weak_real_spectrum": [],
        },
        "expanded_review": {"decisions": {
            "approve_full_interval": [1],
            "reject_visible_or_weak_real_spectrum": [],
            "approve_adjusted_interval": [],
        }},
    }
    intervals = approved_intervals_from_manual_review(review, original, expanded)
    assert len(intervals) == 1
    assert intervals.import_summary == {
        "imported_approved_count": 2,
        "deduplicated_count": 1,
        "unique_approved_count": 1,
    }

    review["decisions"]["reject_visible_or_weak_real_spectrum"] = [1]
    with pytest.raises(ValueError, match="both approved and rejected"):
        approved_intervals_from_manual_review(review, original, expanded)


def test_finalized_review_inventory_loads_all_approvals_and_deduplicates_windows() -> None:
    review = {
        "review_status": "finalized",
        "counts": {"approved": 3},
        "approved_intervals": [
            {
                "label": "later",
                "start_utc": "2022-01-02T00:00:00Z",
                "stop_utc": "2022-01-02T00:08:00Z",
            },
            {
                "label": "first",
                "start_utc": "2022-01-01T00:00:00Z",
                "stop_utc": "2022-01-01T00:08:00Z",
            },
            {
                "label": "duplicate-time",
                "start_utc": "2022-01-01T00:00:00Z",
                "stop_utc": "2022-01-01T00:08:00Z",
            },
        ],
    }

    intervals = approved_intervals_from_review_inventory(review)

    assert [item.label for item in intervals] == ["duplicate-time", "later"]
    assert intervals.import_summary == {
        "imported_approved_count": 3,
        "deduplicated_count": 1,
        "unique_approved_count": 2,
    }
    review["review_status"] = "in_progress"
    with pytest.raises(ValueError, match="not finalized"):
        approved_intervals_from_review_inventory(review)


def test_candidate_discovery_uses_segment_mode_not_num1_name(tmp_path: Path) -> None:
    time_mode4 = np.arange(30, dtype=float) * 16.0
    time_mode1 = 1000.0 + np.arange(30, dtype=float) * 16.0

    def segment(time: np.ndarray, energy: np.ndarray, mode: int) -> dict[str, object]:
        return {
            "t": time,
            "p": np.ones((time.size, energy.size), dtype=float),
            "f": energy,
            "mod": mode,
            "quality_flag": np.zeros(time.size, dtype=np.uint32),
            "quality_flag_available": np.ones(time.size, dtype=np.uint8),
        }

    container = {
        "num1": segment(time_mode4, np.arange(64, dtype=float) + 1.0, 4),
        "num2": segment(time_mode1, MODE1_ENERGY_EV, 1),
    }
    path = tmp_path / "Ion_spe_19700101.mat"
    savemat(
        path,
        {
            "H_spe_num": container,
            "O_spe_num": container,
            "O2_spe_num": container,
        },
    )
    candidates = discover_background_candidates(
        [path],
        CandidateConfig(
            duration_s=400.0,
            step_s=100.0,
            min_records=20,
            low_signal_quantile=1.0,
            requested_intervals=10,
            max_intervals_per_day=10,
            min_separation_s=0.0,
        ),
    )

    assert candidates
    assert all(item["source_mode"] == 1 for item in candidates)
    assert all(item["source_day_spe_segment"] == "num2" for item in candidates)
    assert all(float(item["start_unix_s"]) >= time_mode1[0] for item in candidates)


def _records(values: list[float], flags: list[int] | None = None) -> MinpaMode1Records:
    n = len(values)
    cube = np.stack([np.full(MODE1_SHAPE, value, dtype=np.float32) for value in values])
    return MinpaMode1Records(
        time_unix_s=np.arange(n, dtype=float),
        dpf=cube,
        native_quality=np.zeros(n, dtype=np.uint32),
        ion_start_stop_counts=np.zeros((n, 4)),
        high_voltage_monitor_v=np.zeros((n, 5, 4)),
        instrument_solar_angles_deg=np.zeros((n, 2)),
        project_quality_flag=np.asarray(flags or [0] * n, dtype=np.uint32),
        project_quality_available=np.ones(n, dtype=bool),
        source_file_index=np.zeros(n, dtype=np.int32),
        source_files=["synthetic.mat"],
        source_sha256={"synthetic.mat": "abc"},
    )


def _interval(start: int, stop: int, label: str) -> BackgroundInterval:
    return BackgroundInterval(
        start_utc=f"1970-01-01T00:00:{start:02d}Z",
        stop_utc=f"1970-01-01T00:00:{stop:02d}Z",
        approved=True,
        label=label,
    )


def test_mode1_flat_expansion_order_is_mass_then_azimuth_then_pitch_then_energy() -> None:
    flat = np.arange(MODE1_FLAT_SIZE)
    cube = reshape_mode1_dpf(flat)

    assert cube.shape == MODE1_SHAPE
    assert cube[0, 0, 0, 1] == 1
    assert cube[0, 0, 1, 0] == 8
    assert cube[0, 1, 0, 0] == 16 * 8
    assert cube[1, 0, 0, 0] == 4 * 16 * 8


def test_energy_count_equivalent_sums_all_direction_and_mass_cells() -> None:
    quantum = np.array(
        [
            [[[2.0, 4.0], [5.0, 10.0]]],
            [[[3.0, 6.0], [7.0, 14.0]]],
        ]
    )
    count_cells = np.array(
        [
            [
                [[[1.0, 0.0], [2.0, 1.0]]],
                [[[0.0, 3.0], [1.0, 2.0]]],
            ],
            [
                [[[2.0, 1.0], [0.0, 4.0]]],
                [[[1.0, 1.0], [2.0, 0.0]]],
            ],
        ]
    )
    dpf = count_cells * quantum[None, ...]

    counts, valid_fraction = aggregate_energy_count_equivalent(dpf, quantum)

    np.testing.assert_array_equal(counts, [[4.0, 6.0], [7.0, 4.0]])
    np.testing.assert_array_equal(valid_fraction, np.ones((2, 2)))


def test_energy_count_equivalent_rejects_non_integer_dpf_reconstruction() -> None:
    quantum = np.ones((1, 1, 1, 2))
    dpf = np.array([[[[[2.0, 1.2]]]]])

    counts, valid_fraction = aggregate_energy_count_equivalent(dpf, quantum)

    np.testing.assert_array_equal(counts, [[2.0]])
    np.testing.assert_allclose(valid_fraction, [[0.5]])


def test_bits_one_two_and_five_are_retained_but_bits_three_and_four_reject_interval() -> None:
    records = _records([0.0, 2.0, 4.0, 10.0, 10.0, 10.0], flags=[1, 2, 16, 0, 8, 0])
    model = estimate_background(
        records,
        [_interval(0, 3, "sparse-background"), _interval(3, 6, "bit4-rejected")],
        BackgroundConfig(
            min_approved_intervals=1,
            min_total_records=3,
            min_records_per_interval=3,
            min_channel_samples=1,
        ),
    )

    assert model.interval_summaries[0]["accepted"] is True
    assert model.interval_summaries[0]["project_quality_bit_record_counts"] == {
        "1": 1,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 1,
    }
    assert model.interval_summaries[1]["accepted"] is False
    assert "project_quality_bits_3_4_present" in model.interval_summaries[1]["reasons"]
    assert model.valid is True
    np.testing.assert_allclose(model.robust_background_dpf, 2.0)
    np.testing.assert_allclose(model.paper_background_dpf, 3.0)
    np.testing.assert_allclose(model.background_dpf, 2.0)


def test_paper_estimator_uses_conditional_positive_mean() -> None:
    model = estimate_background(
        _records([0.0, 2.0, 4.0]),
        [_interval(0, 3, "paper")],
        BackgroundConfig(
            primary_estimator="paper_nonzero_mean",
            min_approved_intervals=1,
            min_total_records=3,
            min_records_per_interval=3,
            min_channel_samples=1,
        ),
    )

    np.testing.assert_allclose(model.background_dpf, 3.0)


def test_strict_paper_model_uses_every_interval_and_labels_channel_support() -> None:
    cube = np.ones((6, *MODE1_SHAPE), dtype=float)
    low = (0, 0, 0, 0)
    unsupported = (0, 0, 0, 1)
    supported = (0, 0, 0, 2)
    cube[:, unsupported[0], unsupported[1], unsupported[2], unsupported[3]] = 0.0
    cube[:, low[0], low[1], low[2], low[3]] = [1.0, 3.0, 3.0, 5.0, 0.0, 0.0]
    cube[:, supported[0], supported[1], supported[2], supported[3]] = [1, 1, 3, 3, 5, 5]
    records = _records([1.0] * 6)
    records.dpf = cube
    model = estimate_background(
        records,
        [_interval(0, 2, "a"), _interval(2, 4, "b"), _interval(4, 6, "c")],
        BackgroundConfig(
            primary_estimator="paper_channel_nonzero_mean",
            min_approved_intervals=40,
            min_total_records=10_000,
            min_records_per_interval=2,
            paper_bootstrap_replicates=100,
        ),
    )

    assert model.valid is True
    assert model.background_dpf[low] == pytest.approx(3.0)
    assert model.interval_support_count[low] == 2
    assert model.nonzero_sample_count[low] == 4
    assert model.support_level[low] == 1
    assert model.support_level[unsupported] == 0
    assert np.isnan(model.background_dpf[unsupported])
    assert model.support_level[supported] == 2
    assert np.isfinite(model.bootstrap_ci_low_dpf[supported])
    assert np.isfinite(model.bootstrap_ci_high_dpf[supported])

    raw = np.full(MODE1_SHAPE, 10.0)
    corrected = apply_background(raw, model)
    assert corrected.denoise_policy_version == DENOISE_POLICY_VERSION
    assert corrected.corrected_dpf[low] == pytest.approx(7.0)
    assert corrected.corrected_dpf[unsupported] == pytest.approx(10.0)
    assert corrected.removed_dpf[unsupported] == pytest.approx(0.0)


def test_uv_detector_requires_directional_broadband_morphology() -> None:
    background_records = _records([1.0] * 20)
    model = estimate_background(
        background_records,
        [_interval(0, 20, "uv-baseline")],
        BackgroundConfig(
            min_approved_intervals=1,
            min_total_records=20,
            min_records_per_interval=10,
            min_channel_samples=1,
        ),
    )
    cube = np.ones((10, *MODE1_SHAPE), dtype=float)
    cube[4:6, :4, :, 2:4, -1] = 10.0
    time = np.arange(10, dtype=float) * 16.0
    solar = np.zeros((10, 2), dtype=float)
    solar[4:6, 0] = 85.0

    result = detect_uv_contamination(
        cube,
        time,
        model,
        instrument_solar_angles_deg=solar,
    )

    assert result.candidate_mask.tolist() == [False] * 4 + [True, True] + [False] * 4
    assert result.confirmed_mask.tolist() == [False] * 4 + [True, True] + [False] * 4
    assert result.affected_azimuth_sector_count[4:6].tolist() == [2, 2]
    assert result.energy_coverage_count[4:6].tolist() == [4, 4]
    assert result.pitch_coverage_count[4:6].tolist() == [4, 4]
    assert result.solar_angle_support_mask[4:6].tolist() == [True, True]


def test_uv_detector_rejects_widespread_non_directional_enhancement() -> None:
    background_records = _records([1.0] * 20)
    model = estimate_background(
        background_records,
        [_interval(0, 20, "uv-baseline")],
        BackgroundConfig(
            min_approved_intervals=1,
            min_total_records=20,
            min_records_per_interval=10,
            min_channel_samples=1,
        ),
    )
    cube = np.ones((10, *MODE1_SHAPE), dtype=float)
    cube[4:6, :4, :, :8, -1] = 10.0

    result = detect_uv_contamination(cube, np.arange(10, dtype=float) * 16.0, model)

    assert not np.any(result.candidate_mask)
    assert not np.any(result.confirmed_mask)


def test_subtraction_is_nonnegative_and_preserves_nan() -> None:
    model = estimate_background(
        _records([1.0, 1.0, 1.0]),
        [_interval(0, 3, "valid")],
        BackgroundConfig(
            min_approved_intervals=1,
            min_total_records=3,
            min_records_per_interval=3,
            min_channel_samples=1,
        ),
    )
    raw = np.full((2, *MODE1_SHAPE), 0.5)
    raw[0, 0, 0, 0, 0] = 2.5
    raw[1, 0, 0, 0, 0] = np.nan
    result = apply_background(raw, model)

    assert result.corrected_dpf[0, 0, 0, 0, 0] == pytest.approx(1.5)
    assert np.nanmin(result.corrected_dpf) == 0.0
    assert np.isnan(result.corrected_dpf[1, 0, 0, 0, 0])
    assert np.nanmax(result.corrected_dpf) <= np.nanmax(raw)


def test_count_equivalent_estimator_recovers_quantum_and_mean_rate() -> None:
    values = [0.0, 2.0, 4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0]
    model = estimate_background(
        _records(values),
        [_interval(0, 9, "quantized"), _interval(9, 10, "tail")],
        BackgroundConfig(
            primary_estimator="count_equivalent_energy_pooled",
            min_approved_intervals=1,
            min_total_records=9,
            min_records_per_interval=1,
            min_channel_samples=1,
            quantum_min_nonzero_samples=5,
        ),
    )

    np.testing.assert_allclose(model.dpf_quantum, 2.0)
    np.testing.assert_allclose(model.background_lambda_count_equivalent, 0.6)
    np.testing.assert_allclose(model.background_dpf, 1.2)
    assert model.quantum_valid_mask.all()


def test_invalid_and_mode_mismatched_models_are_refused() -> None:
    invalid = estimate_background(_records([1.0]), [], BackgroundConfig())
    with pytest.raises(ValueError, match="invalid background model"):
        apply_background(np.ones(MODE1_SHAPE), invalid)

    mode12 = replace(invalid, mode=12, valid=True, invalid_reasons=[])
    with pytest.raises(ValueError, match="Mode-1 model"):
        apply_background(np.ones(MODE1_SHAPE), mode12)


def test_model_npz_roundtrip(tmp_path: Path) -> None:
    model = estimate_background(
        _records([1.0, 2.0, 3.0]),
        [_interval(0, 3, "roundtrip")],
        BackgroundConfig(
            min_approved_intervals=1,
            min_total_records=3,
            min_records_per_interval=3,
            min_channel_samples=1,
        ),
    )
    path = tmp_path / "background_model.npz"
    save_background_model(path, model)
    loaded = load_background_model(path)

    assert loaded.algorithm_version == model.algorithm_version
    assert loaded.primary_estimator == model.primary_estimator
    assert loaded.approved_intervals == model.approved_intervals
    np.testing.assert_allclose(loaded.background_dpf, model.background_dpf)
