from __future__ import annotations

import numpy as np

from highE.minpa_signal_examples import (
    contiguous_true_run,
    extract_signal_prefilter_features,
    passes_signal_prefilter,
    validate_native_channel_signal,
)
from scripts.prepare_minpa_typical_signal_review_v2_2_0 import (
    _insert_spectrogram_gap_rows,
)


def test_signal_prefilter_detects_persistent_three_energy_band() -> None:
    spectrum = np.ones((20, 40), dtype=float)
    spectrum[:, 12:15] = 100.0
    features = extract_signal_prefilter_features(spectrum)
    assert passes_signal_prefilter(features)
    assert features["three_bin_persistent_record_fraction"] == 1.0


def test_contiguous_true_run() -> None:
    assert contiguous_true_run(np.array([False, True, True, False, True])) == 2


def test_native_validator_requires_persistent_adjacent_twenty_x_signal() -> None:
    shape = (40, 4, 16, 8)
    background = np.ones(shape, dtype=float)
    valid = np.ones(shape, dtype=bool)
    raw = np.zeros((20, *shape), dtype=float)
    raw[:, 10:12, 0, 0, 0] = 100.0
    corrected = np.maximum(raw - background, 0.0)
    raw_spectrum = raw[..., 0].mean(axis=(2, 3))
    corrected_spectrum = corrected[..., 0].mean(axis=(2, 3))
    result = validate_native_channel_signal(
        raw,
        corrected,
        background,
        valid,
        mode=1,
        species="H+",
        raw_spectrum=raw_spectrum,
        corrected_spectrum=corrected_spectrum,
    )
    assert result["passed"]
    assert result["longest_contiguous_strong_energy_run"] == 2
    assert result["signal_peak_retention_fraction"] >= 0.95


def test_native_validator_rejects_single_energy_spike() -> None:
    shape = (40, 4, 16, 8)
    background = np.ones(shape, dtype=float)
    valid = np.ones(shape, dtype=bool)
    raw = np.zeros((20, *shape), dtype=float)
    raw[0, 10, 0, 0, 0] = 1000.0
    corrected = np.maximum(raw - background, 0.0)
    raw_spectrum = raw[..., 0].mean(axis=(2, 3))
    corrected_spectrum = corrected[..., 0].mean(axis=(2, 3))
    result = validate_native_channel_signal(
        raw,
        corrected,
        background,
        valid,
        mode=1,
        species="H+",
        raw_spectrum=raw_spectrum,
        corrected_spectrum=corrected_spectrum,
    )
    assert not result["passed"]
    assert result["longest_contiguous_strong_energy_run"] == 0


def test_spectrogram_gap_rows_are_nan() -> None:
    time = np.array([0.0, 2.0, 20.0])
    spectrum = np.ones((3, 4), dtype=float)
    output_time, output = _insert_spectrogram_gap_rows(time, spectrum, 2.0)
    assert output_time.tolist() == [0.0, 2.0, 4.0, 18.0, 20.0]
    assert np.all(np.isnan(output[2:4]))
