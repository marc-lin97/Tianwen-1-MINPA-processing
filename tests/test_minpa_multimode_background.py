from pathlib import Path

import numpy as np
import pytest

from highE.minpa_modes import species_mass_indices
from highE.minpa_multimode_background import (
    SUPPORT_LOW,
    SUPPORT_UNSUPPORTED,
    apply_multimode_background,
    assemble_multimode_background_from_summaries,
    estimate_multimode_background,
    load_multimode_background_model,
    mode_channel_shape,
    reshape_mode_dpf,
    save_multimode_background_model,
)


def test_mode_layouts_and_mass_fastest_reshape() -> None:
    assert mode_channel_shape(4) == (64, 4, 16, 8)
    assert mode_channel_shape(12) == (48, 1, 1, 32)
    flat = np.arange(np.prod(mode_channel_shape(12)), dtype=float)
    cube = reshape_mode_dpf(flat, 12)
    assert cube.shape == mode_channel_shape(12)
    assert cube[0, 0, 0, 1] == 1.0
    with pytest.raises(ValueError, match="supports modes"):
        mode_channel_shape(7)


def test_species_background_is_independent_and_unreviewed_masses_are_preserved() -> None:
    shape = mode_channel_shape(4)
    h_mass = int(species_mass_indices(4, "H+")[0])
    o_mass = int(species_mass_indices(4, "O+")[0])
    other_mass = 1
    h_case1 = np.zeros((2, *shape))
    h_case2 = np.zeros((2, *shape))
    o_case = np.zeros((2, *shape))
    h_case1[0, ..., h_mass] = 1.0
    h_case1[1, ..., h_mass] = 3.0
    h_case2[0, ..., h_mass] = 5.0
    h_case2[1, ..., h_mass] = 0.0
    o_case[..., o_mass] = 7.0

    model = estimate_multimode_background(
        4,
        {"H+": [h_case1, h_case2], "O+": [o_case], "O2+": []},
        bootstrap_replicates=20,
        review_status="finalized",
    )

    np.testing.assert_allclose(model.background_dpf[..., h_mass], 3.5)
    np.testing.assert_allclose(model.background_dpf[..., o_mass], 7.0)
    assert np.all(np.isnan(model.background_dpf[..., other_mass]))
    assert np.all(model.support_level[..., other_mass] == SUPPORT_UNSUPPORTED)
    assert np.all(model.support_level[..., o_mass] == SUPPORT_LOW)
    raw = np.full((1, *shape), 10.0)
    raw[0, 0, 0, 0, h_mass] = np.nan
    corrected = apply_multimode_background(raw, model)
    assert np.isnan(corrected.corrected_dpf[0, 0, 0, 0, h_mass])
    np.testing.assert_allclose(corrected.corrected_dpf[..., other_mass], 10.0)
    np.testing.assert_allclose(corrected.corrected_dpf[0, 1:, ..., h_mass], 6.5)
    assert np.all(corrected.corrected_dpf <= raw, where=np.isfinite(raw))


def test_provisional_model_is_refused_and_roundtrips(tmp_path: Path) -> None:
    shape = mode_channel_shape(12)
    h_mass = species_mass_indices(12, "H+")
    cube = np.zeros((3, *shape))
    cube[..., h_mass] = 2.0
    model = estimate_multimode_background(
        12, {"H+": [cube]}, bootstrap_replicates=0
    )
    assert not model.valid
    with pytest.raises(ValueError, match="provisional"):
        apply_multimode_background(cube, model)
    result = apply_multimode_background(cube, model, allow_provisional=True)
    np.testing.assert_allclose(result.corrected_dpf[..., h_mass], 0.0)
    path = tmp_path / "model.npz"
    save_multimode_background_model(path, model)
    loaded = load_multimode_background_model(path)
    assert loaded.mode == 12
    assert loaded.review_status == "provisional_review_only"
    np.testing.assert_allclose(loaded.background_dpf, model.background_dpf, equal_nan=True)


def test_streaming_summary_assembly_matches_paper_interval_weighting() -> None:
    selected_shape = (48, 1, 1, 2)
    means = np.full((2, *selected_shape), np.nan)
    means[0] = 2.0
    means[1] = 6.0
    summary = {
        "interval_means_dpf": means,
        "sample_count": np.full(selected_shape, 6),
        "nonzero_sample_count": np.full(selected_shape, 4),
        "nonzero_median_dpf": np.full(selected_shape, 3.0),
        "nonzero_mad_dpf": np.full(selected_shape, 1.0),
    }
    model = assemble_multimode_background_from_summaries(
        12,
        {"H+": summary},
        approved_intervals_by_species={"H+": [{"candidate_id": "a"}, {"candidate_id": "b"}]},
        bootstrap_replicates=20,
    )
    masses = species_mass_indices(12, "H+")
    np.testing.assert_allclose(model.background_dpf[..., masses], 4.0)
    assert np.all(model.interval_support_count[..., masses] == 2)
    assert np.all(model.nonzero_sample_count[..., masses] == 4)
    assert model.valid and model.review_status == "finalized"
