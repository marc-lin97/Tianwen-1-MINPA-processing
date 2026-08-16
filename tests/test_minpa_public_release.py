from __future__ import annotations

from datetime import UTC, datetime, timedelta

from highE.minpa_public_release import calibration_time_rows, public_example_manifest


def test_calibration_time_rows_contain_only_locator_fields() -> None:
    origin1 = datetime(2024, 1, 1, tzinfo=UTC)
    mode1 = {
        "review_status": "finalized",
        "approved_intervals": [
            {
                "start_utc": (origin1 + timedelta(minutes=i)).isoformat(),
                "stop_utc": (origin1 + timedelta(minutes=i + 1)).isoformat(),
            }
            for i in range(273)
        ],
    }
    origin2 = datetime(2025, 1, 1, tzinfo=UTC)
    multimode = {
        "review_status": "finalized",
        "approved_intervals": [
            {
                "mode": 4 if i % 2 == 0 else 12,
                "species": ("H+", "O+", "O2+")[i % 3],
                "start_utc": (origin2 + timedelta(minutes=i)).isoformat(),
                "stop_utc": (origin2 + timedelta(minutes=i + 1)).isoformat(),
            }
            for i in range(2901)
        ],
    }
    rows = calibration_time_rows(mode1, multimode)
    assert len(rows) == 3174
    assert set(rows[0]) == {"mode", "species", "start_utc", "stop_utc"}


def test_public_example_manifest_omits_rejected_rows_and_sources() -> None:
    noise = [{
        "mode": mode, "species": species, "start_utc": "a", "stop_utc": "b",
        "figure": "noise.png", "figure_sha256": "1", "source_ori": ["private"],
    } for mode in (1, 4, 12) for species in ("H+", "O+", "O2+")]
    signal = [{
        "mode": mode, "species": species, "start_utc": "a", "stop_utc": "b",
        "figure": "signal.png", "figure_sha256": "2", "source_ori": ["private"],
        "strength_tier": "strong_20x",
        "maximum_median_channel_signal_to_background": 20.0,
        "signal_peak_retention_fraction": 0.95,
        "ratio_peak_shift_channels": 0,
    } for mode in (1, 4, 12) for species in ("H+", "O+", "O2+")]
    final = {
        "status": "finalized_manual_review",
        "bundle_sha256": "3",
        "units": "DPF",
        "smoothing_or_interpolation": "none",
        "counts": {
            "mode_species_groups": 9,
            "approved_noise_examples": 9,
            "approved_signal_examples": 9,
            "rejected_signal_candidates": 18,
        },
        "noise_examples": noise,
        "signal_examples": signal,
        "rejected_signal_candidates": [{"private": True}],
    }
    public = public_example_manifest(final)
    assert "rejected_signal_candidates" not in public
    assert "noise_examples" not in public
    assert "source_ori" not in public["signal_examples"][0]
