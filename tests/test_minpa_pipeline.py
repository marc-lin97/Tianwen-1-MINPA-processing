from __future__ import annotations

import numpy as np
import pytest

from highE.coordinates import tw1_minpa_instrument_to_orbiter_matrix, tw1_orbiter_to_mso_matrix
from highE.minpa_io import MinpaRecord
from highE.minpa_modes import (
    angular_geometry,
    energy_bin_edges_eV,
    energy_bin_widths_eV,
    minpa_body_directions,
    minpa_to_body_direction,
    minpa_directions,
    mode_layout,
    species_mass_indices,
    split_raw_record,
)
from highE.minpa_pipeline import process_records
from highE.minpa_processing import compute_moments, finite_velocity_cell_samples, project_vdf, species_spectrum, velocity_samples
from highE.minpa_quality import FLAG_EVEN_ODD, FLAG_HIGH_CHANNEL, FLAG_SPARSE_INVALID, FLAG_UV, evaluate_quality, reject_for_background


@pytest.mark.parametrize("mode", range(1, 13))
def test_mode_layout_and_direction_norm(mode: int) -> None:
    layout = mode_layout(mode)
    assert layout.values_per_subrecord == layout.energy_eV.size * layout.angle_count * layout.mass_amu.size
    theta, phi, omega = angular_geometry(mode)
    assert theta.shape == phi.shape == omega.shape == (layout.angle_count,)
    assert np.all(omega > 0)
    assert np.allclose(np.linalg.norm(minpa_directions(mode), axis=1), 1.0)


def test_mode_13_rejected_and_mode12_split() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        mode_layout(13)
    layout = mode_layout(12)
    raw = np.arange(layout.values_per_subrecord * 2, dtype=float)
    split = split_raw_record(12, 100.0, raw)
    assert [row[0] for row in split] == pytest.approx([101.025, 103.075])
    assert split[0][1].size == split[1][1].size == 1536
    assert not layout.directional_velocity_available


def test_species_groups() -> None:
    assert species_mass_indices(1, "H+").tolist() == [0]
    assert species_mass_indices(1, "O+").tolist() == [3]
    assert species_mass_indices(1, "O2+").tolist() == [5]
    assert species_mass_indices(12, "H+").tolist() == [0, 1]
    assert species_mass_indices(12, "O+").tolist() == [18, 19]
    assert species_mass_indices(12, "O2+").tolist() == [26, 27]
    assert species_mass_indices(9, "O+").size == 0


def test_uniform_dpf_density_matches_direct_formula_and_vdf_closes() -> None:
    layout = mode_layout(1)
    cube = np.zeros(layout.shape)
    cube[:, :, 0] = 2.0e3
    values = cube.reshape(-1)
    moment = compute_moments(values, 1, "H+", attitude_roll_pitch_yaw_deg=(12.0, -5.0, 22.0))
    _, _, omega = angular_geometry(1)
    speed = np.sqrt(2 * layout.energy_eV * 1.602176634e-19 / 1.67262192369e-27) / 1000
    expected = np.sum(2e3 * energy_bin_widths_eV(1)[:, None] * omega[None, :] / (speed[:, None] * 1e5))
    assert moment.density_cm3 == pytest.approx(expected)
    assert np.linalg.norm(moment.velocity_body_km_s) == pytest.approx(np.linalg.norm(moment.velocity_minpa_km_s))
    samples = velocity_samples(values, 1, "H+")
    projection = project_vdf(samples, bins=25)
    assert projection.density_integral_cm3 == pytest.approx(moment.density_cm3)


def test_mode1_energy_widths_match_nv_mso_2_convention() -> None:
    layout = mode_layout(1)
    widths = energy_bin_widths_eV(1)
    edges = energy_bin_edges_eV(layout.energy_eV)
    np.testing.assert_allclose(widths, np.diff(edges))
    assert np.median(widths / layout.energy_eV) == pytest.approx(0.23399158867724523, rel=2.0e-7)
    assert not np.allclose(widths, 0.15 * layout.energy_eV)


def test_mode12_density_but_no_vector_velocity() -> None:
    layout = mode_layout(12)
    cube = np.zeros(layout.shape)
    cube[:, :, species_mass_indices(12, "O+")] = 100.0
    moment = compute_moments(cube.reshape(-1), 12, "O+", attitude_roll_pitch_yaw_deg=(0, 0, 0))
    assert moment.density_cm3 > 0
    assert moment.status == "direction_not_resolved"
    assert np.all(np.isnan(moment.velocity_mso_km_s))
    with pytest.raises(ValueError, match="does not resolve"):
        velocity_samples(cube.reshape(-1), 12, "O+")


def test_quality_bits_and_background_policy() -> None:
    layout = mode_layout(1)
    zeros = np.zeros(layout.shape)
    q0 = evaluate_quality(zeros.reshape(-1), 1, "H+")
    assert int(q0.flag) & int(FLAG_SPARSE_INVALID)

    uv = np.zeros(layout.shape)
    uv[:, :, 0] = 1e7
    quv = evaluate_quality(uv.reshape(-1), 1, "H+")
    assert int(quv.flag) & int(FLAG_UV)
    assert int(quv.flag) & int(FLAG_HIGH_CHANNEL)
    assert reject_for_background(int(quv.flag))

    alternating = np.zeros(layout.shape)
    target_def = np.where(np.arange(layout.energy_eV.size) % 2 == 0, 3e5, 1e4)
    alternating[:, :, 0] = target_def[:, None] / layout.energy_eV[:, None]
    qalt = evaluate_quality(alternating.reshape(-1), 1, "H+")
    assert int(qalt.flag) & int(FLAG_EVEN_ODD)


def test_series_masks_moments_but_keeps_spectrum() -> None:
    layout = mode_layout(1)
    cube = np.zeros(layout.shape)
    cube[:, :, 0] = 1e7  # UV/high-channel rejected
    record = MinpaRecord(100.0, 1, cube.reshape(-1))
    series = process_records([record], "H+", quality_policy="moments")
    assert not series.accepted[0]
    assert np.any(np.isfinite(series.deflux[0]))
    assert np.isnan(series.density_cm3[0])


def test_rotation_chain_is_proper() -> None:
    install = tw1_minpa_instrument_to_orbiter_matrix()
    attitude = tw1_orbiter_to_mso_matrix(10.0, 20.0, 30.0)
    chain = attitude @ install
    assert np.allclose(chain.T @ chain, np.eye(3), atol=1e-12)
    assert np.linalg.det(chain) == pytest.approx(1.0)


def test_validated_minpa_to_body_mapping_matches_directions() -> None:
    legacy = minpa_directions(1)
    explicit = minpa_to_body_direction(legacy)
    np.testing.assert_allclose(explicit, minpa_body_directions(1), atol=1.0e-15)
    np.testing.assert_allclose(explicit, legacy @ tw1_minpa_instrument_to_orbiter_matrix().T, atol=1.0e-15)


def test_finite_vdf_sampling_separates_coverage_from_signal() -> None:
    layout = mode_layout(1)
    cube = np.zeros(layout.shape)
    cube[10, 5, 0] = 2.0e4
    samples = finite_velocity_cell_samples(cube.reshape(-1), 1, "H+", angular_samples=3, energy_samples=3)
    assert samples.velocity_minpa_km_s.shape[0] == 27
    assert samples.coverage_velocity_minpa_km_s.shape[0] == layout.energy_eV.size * layout.angle_count * 27
    projection = project_vdf(samples, bins=50)
    assert np.count_nonzero(projection.coverage_mask) > np.count_nonzero(projection.density_per_velocity_area)
    assert projection.density_integral_cm3 == pytest.approx(np.sum(samples.density_weight_cm3))
