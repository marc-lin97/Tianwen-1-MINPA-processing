from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from highE.minpa_interval_spectra import (
    interval_quantiles,
    load_accepted_interval_rows,
    unique_timestamp_time_mean,
    zero_inclusive_time_mean,
)


def test_zero_inclusive_time_mean_keeps_finite_zeros() -> None:
    values = np.asarray([
        [0.0, 2.0, np.nan],
        [2.0, 0.0, 4.0],
        [0.0, 4.0, np.nan],
    ])

    result = zero_inclusive_time_mean(values)

    np.testing.assert_allclose(result, [2.0 / 3.0, 2.0, 4.0])


def test_interval_quantiles_are_computed_per_species_and_energy() -> None:
    values = np.asarray([
        [[0.0, 10.0], [2.0, 20.0]],
        [[2.0, 30.0], [4.0, 40.0]],
        [[4.0, 50.0], [6.0, 60.0]],
    ])

    result = interval_quantiles(values, [0.5])

    assert result.shape == (1, 2, 2)
    np.testing.assert_allclose(result[0], [[2.0, 30.0], [4.0, 40.0]])


def test_repeated_timestamp_is_averaged_before_time_mean() -> None:
    time = np.asarray([0.0, 1.0, 1.0, 2.0])
    spectrum = np.asarray([[0.0], [3.0], [9.0], [6.0]])

    result, duplicate_count = unique_timestamp_time_mean(time, spectrum)

    # Unique-time values are 0, mean(3, 9)=6, and 6.
    np.testing.assert_allclose(result, [4.0])
    assert duplicate_count == 1


def test_load_accepted_interval_rows_rejects_candidate_metric_mismatch(tmp_path: Path) -> None:
    month = tmp_path / "202112"
    month.mkdir()
    (month / "summary.json").write_text(json.dumps({
        "status": "complete",
        "candidates": [{"candidate_id": "x", "accepted": True}],
        "interval_metrics": [],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate/metric mismatch"):
        load_accepted_interval_rows(tmp_path)
