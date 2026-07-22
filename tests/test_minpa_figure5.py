from __future__ import annotations

import numpy as np

from highE.minpa_background import MODE1_ENERGY_EV, MODE1_SHAPE
from highE.minpa_figure5 import (
    QuietIntervalCoherenceConfig,
    hplus_count_equivalent_spectrum,
    regularize_mode1_spectrogram,
    screen_hplus_coherent_signal,
)


def test_hplus_count_equivalent_sums_only_directional_channels() -> None:
    dpf = np.zeros((2, *MODE1_SHAPE), dtype=float)
    quantum = np.ones(MODE1_SHAPE, dtype=float) * 2.0
    dpf[0, 5, :, :, 0] = 6.0
    dpf[0, 5, :, :, 1] = 1000.0
    dpf[1, 7, 1, 2, 0] = 10.0

    result = hplus_count_equivalent_spectrum(dpf, quantum)

    assert result.shape == (2, MODE1_ENERGY_EV.size)
    assert result[0, 5] == 3.0 * 4.0 * 16.0
    assert result[1, 7] == 5.0
    assert result[0, 7] == 0.0


def test_regularize_mode1_spectrogram_keeps_mode_gap_nan() -> None:
    spectrum = np.arange(3 * MODE1_ENERGY_EV.size, dtype=float).reshape(3, -1)
    time = np.array([100.1, 116.5, 149.3])

    grid_time, grid = regularize_mode1_spectrogram(
        time, spectrum, 100.0, 165.6, cadence_s=16.4
    )

    assert grid_time.shape == (5,)
    np.testing.assert_array_equal(grid[0], spectrum[0])
    np.testing.assert_array_equal(grid[1], spectrum[1])
    assert np.all(np.isnan(grid[2]))
    np.testing.assert_array_equal(grid[3], spectrum[2])


def test_coherent_hplus_ridge_is_rejected_but_sparse_noise_is_retained() -> None:
    quantum = np.ones(MODE1_SHAPE, dtype=float)
    quiet = np.zeros((20, *MODE1_SHAPE), dtype=float)
    quiet[0, 5, :2, :2, 0] = 1.0
    ridge = quiet.copy()
    ridge[:3, 12, :2, :2, 0] = 2.0  # 8 counts in 15% of records.
    config = QuietIntervalCoherenceConfig(
        count_threshold=5.0,
        maximum_energy_occupancy=0.10,
        weak_count_threshold=1.0,
        maximum_weak_energy_occupancy=0.40,
    )
    weak_ridge = quiet.copy()
    weak_ridge[:10, 18, 0, :2, 0] = 1.0  # 2 counts in 50% of records.

    quiet_result = screen_hplus_coherent_signal(quiet, quantum, config)
    ridge_result = screen_hplus_coherent_signal(ridge, quantum, config)
    weak_result = screen_hplus_coherent_signal(weak_ridge, quantum, config)

    assert quiet_result["coherent_signal_rejected"] is False
    assert ridge_result["coherent_signal_rejected"] is True
    assert weak_result["coherent_signal_rejected"] is True
    assert ridge_result["peak_energy_index"] == 12
    assert ridge_result["maximum_energy_occupancy"] == 0.15
    assert weak_result["weak_peak_energy_index"] == 18
    assert weak_result["maximum_weak_energy_occupancy"] == 0.50
