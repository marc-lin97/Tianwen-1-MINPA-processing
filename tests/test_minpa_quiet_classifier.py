from __future__ import annotations

import numpy as np

from highE.minpa_background import MODE1_SHAPE
from highE.minpa_quiet_classifier import (
    QuietBoundaryRefinementConfig,
    QuietReferenceClassifierConfig,
    evaluate_quiet_boundary_refinements,
    extract_quiet_reference_features,
    time_mean_h_def_spectrum,
)


def _synthetic_dpf(count_energy: np.ndarray, records: int = 20) -> tuple[np.ndarray, np.ndarray]:
    quantum = np.ones(MODE1_SHAPE, dtype=float)
    cube = np.zeros((records, *MODE1_SHAPE), dtype=float)
    for energy_index, count in enumerate(count_energy):
        cube[:, energy_index, 0, 0, 0] = float(count) / records
    return cube, quantum


def test_time_mean_def_is_invariant_to_repeating_records() -> None:
    spectrum = np.tile(np.linspace(1.0, 2.0, 40), (4, 1))
    assert np.allclose(
        time_mean_h_def_spectrum(spectrum),
        time_mean_h_def_spectrum(np.repeat(spectrum, 3, axis=0)),
    )


def test_flat_background_is_accepted() -> None:
    h_def = np.ones((20, 40), dtype=float)
    cube, quantum = _synthetic_dpf(np.ones(40))
    result = extract_quiet_reference_features(h_def, cube, quantum)
    assert result["pure_background_reference"] is True
    assert result["real_spectrum_detected"] is False
    assert result["classification_confidence"] == "high"


def test_smooth_energy_ridge_is_rejected() -> None:
    h_def = np.ones((20, 40), dtype=float)
    h_def[:, 18:21] = np.array([4.0, 8.0, 5.0])
    count = np.ones(40)
    count[18:21] = np.array([8.0, 12.0, 8.0])
    cube, quantum = _synthetic_dpf(count)
    result = extract_quiet_reference_features(h_def, cube, quantum)
    assert result["real_spectrum_detected"] is True
    assert result["pure_background_reference"] is False


def test_isolated_def_spike_without_count_concentration_is_accepted() -> None:
    h_def = np.ones((20, 40), dtype=float)
    h_def[:, 20] = 5.0
    cube, quantum = _synthetic_dpf(np.ones(40))
    result = extract_quiet_reference_features(
        h_def, cube, quantum, QuietReferenceClassifierConfig()
    )
    assert result["time_mean_def_peak_to_median"] > 2.75
    assert result["h_count_top3_energy_fraction"] < 0.133
    assert result["pure_background_reference"] is True


def test_boundary_refinement_searches_no_shorter_than_half_interval() -> None:
    time = np.arange(48, dtype=float) * 10.0
    h_def = np.ones((48, 40), dtype=float)
    cube, quantum = _synthetic_dpf(np.ones(40), records=48)

    trials = evaluate_quiet_boundary_refinements(
        0.0,
        480.0,
        time,
        h_def,
        time,
        cube,
        quantum,
        refinement_config=QuietBoundaryRefinementConfig(boundary_step_s=60.0),
    )

    assert trials
    assert trials[0]["duration_s"] == 420.0
    assert min(float(item["duration_s"]) for item in trials) == 240.0
    assert all(float(item["start_unix_s"]) >= 0.0 for item in trials)
    assert all(float(item["stop_unix_s"]) <= 480.0 for item in trials)
    assert all(item["classification_confidence"] == "high" for item in trials)
