from __future__ import annotations

from dataclasses import replace

import numpy as np

from highE.minpa_background import (
    MODE1_SHAPE,
    BackgroundConfig,
    BackgroundInterval,
    MinpaMode1Records,
    estimate_background,
)
from highE.minpa_modes import species_mass_indices
from highE.minpa_multimode_background import (
    estimate_multimode_background,
    mode_channel_shape,
)
from highE.tw1_minpa import (
    background_model_is_provisional,
    prepare_background_records,
    prepare_multimode_background_records,
    prepare_unified_static_background_records,
)


def _valid_model():
    n = 20
    records = MinpaMode1Records(
        time_unix_s=np.arange(n, dtype=float),
        dpf=np.ones((n, *MODE1_SHAPE), dtype=float),
        native_quality=np.zeros(n, dtype=np.uint32),
        ion_start_stop_counts=np.zeros((n, 4)),
        high_voltage_monitor_v=np.zeros((n, 5, 4)),
        instrument_solar_angles_deg=np.zeros((n, 2)),
        project_quality_flag=np.zeros(n, dtype=np.uint32),
        project_quality_available=np.ones(n, dtype=bool),
        source_file_index=np.zeros(n, dtype=np.int32),
    )
    return estimate_background(
        records,
        [
            BackgroundInterval(
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:20Z",
                approved=True,
                label="approved",
            )
        ],
        BackgroundConfig(
            min_approved_intervals=1,
            min_total_records=n,
            min_records_per_interval=10,
            min_channel_samples=1,
        ),
    )


def test_prepare_background_records_corrects_mode1_only() -> None:
    model = _valid_model()
    mode1 = np.ones(int(np.prod(MODE1_SHAPE)), dtype=float)
    mode12 = np.array([3.0, 4.0, 5.0])

    result = prepare_background_records(
        np.array([1, 12]),
        np.array([0.0, 1.0]),
        [mode1, mode12],
        model,
    )

    np.testing.assert_allclose(result.corrected_counts[0], 0.0)
    np.testing.assert_array_equal(result.corrected_counts[1], mode12)
    assert result.background_applied.tolist() == [True, False]
    assert result.removed_dpf_fraction[0] == 1.0
    assert np.isnan(result.removed_dpf_fraction[1])
    assert not np.any(result.uv_rejected)


def test_provisional_model_is_explicitly_identified() -> None:
    model = _valid_model()
    assert background_model_is_provisional(model) is False

    provisional = replace(
        model,
        approved_intervals=[
            {
                "start_utc": "1970-01-01T00:00:00Z",
                "stop_utc": "1970-01-01T00:00:20Z",
                "approved": True,
                "label": "provisional-01",
                "source": "automatic_low_signal_screen_provisional",
            }
        ],
    )
    assert background_model_is_provisional(provisional) is True


def test_prepare_multimode_records_corrects_only_reviewed_mode_and_mass() -> None:
    shape4 = mode_channel_shape(4)
    h_mass = int(species_mass_indices(4, "H+")[0])
    interval = np.zeros((2, *shape4))
    interval[..., h_mass] = 2.0
    model4 = estimate_multimode_background(
        4,
        {"H+": [interval]},
        bootstrap_replicates=0,
        review_status="finalized",
    )
    mode4 = np.full(np.prod(shape4), 5.0)
    mode7 = np.array([8.0, 9.0])
    mode1 = np.array([11.0, 12.0])
    result = prepare_multimode_background_records(
        np.array([4, 7, 1]),
        np.array([1.0, 2.0, 3.0]),
        [mode4, mode7, mode1],
        {4: model4},
    )
    corrected4 = result.corrected_counts[0].reshape(shape4)
    np.testing.assert_allclose(corrected4[..., h_mass], 3.0)
    np.testing.assert_allclose(corrected4[..., 1], 5.0)
    np.testing.assert_array_equal(result.corrected_counts[1], mode7)
    np.testing.assert_array_equal(result.corrected_counts[2], mode1)
    assert result.background_applied.tolist() == [True, False, False]


def test_prepare_multimode_records_refuses_provisional_model() -> None:
    shape12 = mode_channel_shape(12)
    interval = np.ones((1, *shape12))
    model12 = estimate_multimode_background(
        12, {"H+": [interval]}, bootstrap_replicates=0
    )
    with np.testing.assert_raises_regex(ValueError, "non-finalized"):
        prepare_multimode_background_records(
            np.array([12]),
            np.array([1.025]),
            [np.ones(np.prod(shape12))],
            {12: model12},
        )


def test_prepare_unified_static_records_corrects_mode1_and_mode4() -> None:
    mode1_model = _valid_model()
    shape4 = mode_channel_shape(4)
    h_mass = int(species_mass_indices(4, "H+")[0])
    interval = np.zeros((2, *shape4))
    interval[..., h_mass] = 2.0
    mode4_model = estimate_multimode_background(
        4,
        {"H+": [interval]},
        bootstrap_replicates=0,
        review_status="finalized",
    )
    raw1 = np.full(np.prod(MODE1_SHAPE), 3.0)
    raw4 = np.full(np.prod(shape4), 5.0)
    result = prepare_unified_static_background_records(
        np.array([1, 4]),
        np.array([0.0, 1.0]),
        [raw1, raw4],
        mode1_model,
        {4: mode4_model},
    )
    np.testing.assert_allclose(result.corrected_counts[0], 2.0)
    corrected4 = result.corrected_counts[1].reshape(shape4)
    np.testing.assert_allclose(corrected4[..., h_mass], 3.0)
    np.testing.assert_allclose(corrected4[..., 1], 5.0)
    assert result.background_applied.tolist() == [True, True]
