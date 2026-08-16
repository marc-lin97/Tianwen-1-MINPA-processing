from __future__ import annotations

from highE.minpa_mode1_temporal_review import (
    calibrate_joint_approved_distance,
    calibrate_joint_labelled_prototypes,
    calibrate_species_temporal_outliers,
    calibrate_energy_band_excess_limits,
    context_persistence_features,
    context_persistence_veto,
    calibrate_prototype_threshold,
    extract_hplus_signal_features,
    hplus_feature_vector,
    coverage_rows,
    exact_deduplicate,
    intervals_overlap,
    single_boundary_context_persistence_veto,
    prototype_signal_score,
    extract_species_quiet_features,
    extract_temporal_coherence_features,
    extract_energy_band_excess_features,
    generate_quiet_windows_from_minute_blocks,
    score_joint_approved_distance,
    score_joint_labelled_prototypes,
    score_species_temporal_outliers,
    score_energy_band_excess,
    species_quiet_feature_vector,
    temporal_coherence_feature_vector,
    robust_feature_scale,
    select_longest_nonoverlapping_joint,
)

import numpy as np
import pytest


def test_coverage_rows_deduplicates_and_reports_monthly_gap() -> None:
    approved = [
        {"start_utc": "2023-07-01T00:00:00Z", "stop_utc": "2023-07-01T00:08:00Z"},
        {"start_utc": "2023-07-01T00:00:00Z", "stop_utc": "2023-07-01T00:08:00Z"},
        {"start_utc": "2023-07-02T00:00:00Z", "stop_utc": "2023-07-02T00:08:00Z"},
    ]
    rows = coverage_rows(approved, [2023], period="month", target_intervals=3, target_days=2)
    july = rows[6]
    assert july["approved_interval_count"] == 2
    assert july["approved_distinct_day_count"] == 2
    assert july["approved_dates_utc"] == ["2023-07-01", "2023-07-02"]
    assert july["interval_gap"] == 1
    assert july["distinct_day_gap"] == 0
    assert not july["currently_eligible"]
    assert len(exact_deduplicate(approved)) == 2


def test_quarterly_coverage_and_empty_periods() -> None:
    approved = [
        {"start_utc": "2024-01-01T00:00:00Z", "stop_utc": "2024-01-01T00:08:00Z"},
        {"start_utc": "2024-03-02T00:00:00Z", "stop_utc": "2024-03-02T00:08:00Z"},
    ]
    rows = coverage_rows(approved, [2024], period="quarter", target_intervals=2, target_days=2)
    assert rows[0]["currently_eligible"]
    assert rows[1]["gap_class"] == "no_approved"


def test_coverage_rows_omits_periods_without_local_source_data() -> None:
    rows = coverage_rows(
        [], [2021], period="quarter",
        excluded_periods=["2021-Q1", "2021-Q2", "2021-Q3"],
    )
    assert [row["period"] for row in rows] == ["2021-Q4"]


def test_generate_quiet_windows_supports_two_minute_lower_bound() -> None:
    minutes = [
        {
            "start_unix_s": float(index * 60),
            "stop_unix_s": float((index + 1) * 60),
            "passes": True,
            "band_excess_score": 0.5,
        }
        for index in range(3)
    ]
    windows = generate_quiet_windows_from_minute_blocks(
        minutes, minimum_minutes=2, maximum_minutes=10
    )
    assert sorted(row["duration_s"] for row in windows) == [120.0, 120.0, 180.0]


def test_longest_nonoverlap_uses_score_for_equal_durations() -> None:
    rows = [
        {"start_unix_s": 0.0, "stop_unix_s": 480.0, "joint_decision_score": 0.7, "source_candidate_id": "a"},
        {"start_unix_s": 60.0, "stop_unix_s": 540.0, "joint_decision_score": 0.2, "source_candidate_id": "b"},
        {"start_unix_s": 600.0, "stop_unix_s": 900.0, "joint_decision_score": 0.9, "source_candidate_id": "c"},
    ]
    selected = select_longest_nonoverlapping_joint(rows)
    assert [row["source_candidate_id"] for row in selected] == ["b", "c"]
    assert intervals_overlap(10.0, 20.0, [(0.0, 11.0)])
    assert not intervals_overlap(11.0, 20.0, [(0.0, 11.0)])


def test_context_persistence_veto_detects_same_narrow_band() -> None:
    spectrum = np.zeros((30, 8), dtype=float)
    spectrum[:, 3:6] = 5.0
    features = context_persistence_features(
        spectrum,
        np.arange(30) < 10,
        (np.arange(30) >= 10) & (np.arange(30) < 20),
        np.arange(30) >= 20,
    )
    assert features["candidate_adjacent_energy_fraction"] == 1.0
    assert features["minimum_context_continuity"] == pytest.approx(1.0)
    assert features["minimum_peak_band_occupancy"] == 1.0
    assert context_persistence_veto(features, {
        "minimum_adjacent_energy_fraction": 0.4,
        "minimum_context_continuity": 0.05,
        "minimum_peak_band_occupancy": 0.2,
    })


def test_context_persistence_veto_requires_both_sides() -> None:
    spectrum = np.zeros((30, 8), dtype=float)
    spectrum[10:20, 3:6] = 5.0
    features = context_persistence_features(
        spectrum,
        np.arange(30) < 10,
        (np.arange(30) >= 10) & (np.arange(30) < 20),
        np.arange(30) >= 20,
    )
    assert features["minimum_context_continuity"] == 0.0
    assert not context_persistence_veto(features, {
        "minimum_adjacent_energy_fraction": 0.4,
        "minimum_context_continuity": 0.05,
        "minimum_peak_band_occupancy": 0.2,
    })


def test_single_boundary_veto_accepts_before_only_persistence() -> None:
    spectrum = np.zeros((30, 8), dtype=float)
    spectrum[:20, 3:6] = 5.0
    features = context_persistence_features(
        spectrum,
        np.arange(30) < 10,
        (np.arange(30) >= 10) & (np.arange(30) < 20),
        np.arange(30) >= 20,
    )
    result = single_boundary_context_persistence_veto(features, {
        "minimum_adjacent_energy_fraction": 0.4,
        "minimum_context_continuity": 0.05,
        "minimum_peak_band_occupancy": 0.4,
    })
    assert result == {
        "before_boundary_veto": True,
        "after_boundary_veto": False,
        "veto": True,
    }


def test_hplus_signal_features_find_persistent_three_bin_band() -> None:
    spectrum = np.ones((20, 8), dtype=float)
    spectrum[:, 3:6] = 20.0
    features = extract_hplus_signal_features(spectrum, np.geomspace(3.0, 3000.0, 8))
    assert features["three_bin_start_index"] == 3
    assert features["three_bin_record_occupancy"] == 1.0
    assert features["three_bin_energy_fraction"] > 0.85
    assert features["log10_three_bin_contrast"] == pytest.approx(np.log10(20.0))


def test_prototype_threshold_limits_approval_false_vetoes() -> None:
    approved = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    signal = np.array([[1.0, 1.0], [1.1, 1.0]])
    center, scale = robust_feature_scale(approved)
    approved_scaled = (approved - center) / scale
    signal_scaled = (signal - center) / scale
    calibration = calibrate_prototype_threshold(
        approved_scaled, signal_scaled, maximum_approved_false_vetoes=1
    )
    assert calibration["approved_false_veto_count"] == 1
    score = prototype_signal_score(signal_scaled[0], approved_scaled, signal_scaled)
    assert score["prototype_signal_score"] >= calibration["threshold"]


def test_species_quiet_features_detect_concentrated_persistent_band() -> None:
    quiet = np.ones((30, 40), dtype=float)
    signal = quiet.copy()
    signal[:, 12:15] = 50.0

    quiet_features = extract_species_quiet_features(quiet)
    signal_features = extract_species_quiet_features(signal)

    assert signal_features["three_bin_energy_fraction"] > quiet_features["three_bin_energy_fraction"]
    assert signal_features["log1p_three_bin_contrast"] > quiet_features["log1p_three_bin_contrast"]
    assert species_quiet_feature_vector(signal_features).shape == (7,)


def test_joint_approved_distance_rejects_if_only_one_species_has_signal() -> None:
    rng = np.random.default_rng(123)
    species_vectors = {
        species: rng.normal(0.0, 0.05, size=(20, 7))
        for species in ("H+", "O+", "O2+")
    }
    calibration = calibrate_joint_approved_distance(
        species_vectors, maximum_joint_false_rejects=2
    )
    quiet_features = {
        name: dict(zip((
            "log1p_peak_to_positive_median",
            "three_bin_energy_fraction",
            "peak_energy_record_occupancy",
            "positive_energy_bin_fraction",
            "three_bin_record_occupancy",
            "log1p_three_bin_contrast",
            "log1p_record_total_p90_to_median",
        ), species_vectors[name][0]))
        for name in species_vectors
    }
    quiet_score = score_joint_approved_distance(quiet_features, calibration)
    assert quiet_score["passes"]

    contaminated = {name: dict(values) for name, values in quiet_features.items()}
    contaminated["O+"]["three_bin_energy_fraction"] += 3.0
    contaminated["O+"]["log1p_three_bin_contrast"] += 3.0
    signal_score = score_joint_approved_distance(contaminated, calibration)
    assert not signal_score["passes"]
    assert "O+" in signal_score["failing_species"]


def test_joint_labelled_prototypes_respect_interval_level_labels() -> None:
    rng = np.random.default_rng(77)
    approved = rng.normal(0.0, 0.1, size=(30, 8))
    rejected = rng.normal(2.0, 0.1, size=(40, 8))
    calibration = calibrate_joint_labelled_prototypes(
        approved,
        rejected,
        maximum_approved_false_rejects=2,
        nearest_neighbours=3,
        protected_approved_indices=range(20, 30),
        maximum_protected_false_rejects=1,
    )
    assert calibration["approved_false_reject_count"] <= 2
    assert calibration["protected_approved_false_reject_count"] <= 1
    assert calibration["rejected_detected_count"] >= 38
    assert score_joint_labelled_prototypes(approved[0], calibration)["passes"]
    assert not score_joint_labelled_prototypes(rejected[0], calibration)["passes"]


def test_temporal_coherence_distinguishes_repeated_shape_from_sparse_noise() -> None:
    energy = np.geomspace(3.0, 3.0e4, 40)
    signal = np.zeros((30, 40), dtype=float)
    signal[:, 12:15] = [1.0, 4.0, 2.0]
    noise = np.zeros((30, 40), dtype=float)
    noise[np.arange(30), np.arange(30) % 40] = 1.0

    signal_features = extract_temporal_coherence_features(signal, energy)
    noise_features = extract_temporal_coherence_features(noise, energy)

    assert signal_features["median_spectral_cosine"] > noise_features["median_spectral_cosine"]
    assert signal_features["log_energy_centroid_std"] < noise_features["log_energy_centroid_std"]
    assert temporal_coherence_feature_vector(signal_features).shape == (7,)


def test_species_temporal_outlier_identifies_only_contaminated_species() -> None:
    base = {
        name: 0.1 for name in (
            "median_spectral_cosine", "q10_spectral_cosine",
            "median_three_bin_record_fraction", "q90_three_bin_record_fraction",
            "median_spectral_entropy", "log_energy_centroid_std",
            "positive_record_fraction",
        )
    }
    approved = []
    for index in range(20):
        approved.append({
            species: {key: value + 0.001 * index for key, value in base.items()}
            for species in ("H+", "O+", "O2+")
        })
    calibration = calibrate_species_temporal_outliers(
        approved, maximum_joint_false_rejects=2
    )
    trial = {species: dict(base) for species in ("H+", "O+", "O2+")}
    trial["H+"]["median_spectral_cosine"] = 1.0
    result = score_species_temporal_outliers(trial, calibration)
    assert result["failing_species"] == ["H+"]


def test_energy_band_excess_detects_brief_and_persistent_flux_bands() -> None:
    quiet = np.ones((30, 40), dtype=float)
    persistent = quiet.copy()
    persistent[:, 10:13] = 20.0
    brief = quiet.copy()
    brief[:4, 25:28] = 100.0

    quiet_features = extract_energy_band_excess_features(quiet)
    persistent_features = extract_energy_band_excess_features(persistent)
    brief_features = extract_energy_band_excess_features(brief)

    assert persistent_features["log1p_time_median_peak_to_positive_median"] > quiet_features["log1p_time_median_peak_to_positive_median"]
    assert brief_features["log1p_record_peak_ratio_q90"] > quiet_features["log1p_record_peak_ratio_q90"]


def test_energy_band_excess_joint_rule_fails_one_species() -> None:
    approved = []
    for index in range(30):
        interval = {}
        for species in ("H+", "O+", "O2+"):
            spectrum = np.ones((20, 40), dtype=float) * (1.0 + index * 0.001)
            interval[species] = extract_energy_band_excess_features(spectrum)
        approved.append(interval)
    calibration = calibrate_energy_band_excess_limits(
        approved, quantile=0.98, joint_score_quantile=0.95
    )
    trial = {species: dict(approved[0][species]) for species in ("H+", "O+", "O2+")}
    trial["O+"]["log1p_peak_to_positive_median"] *= 4.0
    result = score_energy_band_excess(trial, calibration)
    assert not result["passes"]
    assert result["failing_species"] == ["O+"]


def test_minute_blocks_make_real_three_minute_boundary_trim() -> None:
    rows = []
    for minute in range(8):
        rows.append({
            "start_unix_s": float(minute * 60),
            "stop_unix_s": float((minute + 1) * 60),
            "passes": minute in {2, 3, 4},
            "band_excess_score": 0.2 + minute * 0.01,
        })
    windows = generate_quiet_windows_from_minute_blocks(
        rows, minimum_minutes=3, maximum_minutes=10
    )
    assert len(windows) == 1
    assert windows[0]["start_unix_s"] == 120.0
    assert windows[0]["stop_unix_s"] == 300.0
    assert windows[0]["duration_s"] == 180.0


def test_minute_window_ranking_prefers_cleaner_shorter_window() -> None:
    rows = []
    for minute, score in enumerate((0.1, 0.1, 0.1, 0.9, 0.9)):
        rows.append({
            "start_unix_s": float(minute * 60),
            "stop_unix_s": float((minute + 1) * 60),
            "passes": True,
            "band_excess_score": score,
        })
    windows = generate_quiet_windows_from_minute_blocks(
        rows, minimum_minutes=3, maximum_minutes=5
    )
    assert windows[0]["duration_s"] == 180.0
    assert windows[0]["start_unix_s"] == 0.0
