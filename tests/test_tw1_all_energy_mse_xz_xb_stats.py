from types import SimpleNamespace

import numpy as np

from highE.tw1_all_energy_mse_xz_xb_stats import aggregate_all_energy_classified_chunk
from highE.tw1_mse_xz_fov_stats import (
    _initialize_accumulators,
    _products_from_dense,
    classify_xb_mse,
)
from highE.tw1_mse_grid_records import MseGridSpec
from highE.tw1_mse_xz_stats import ALL_ENERGY_SPECIES


def _species_values(scale: float, bitmask: np.ndarray) -> dict[str, np.ndarray]:
    n = bitmask.size
    return {
        "density_cm3": scale * np.arange(1, n + 1, dtype=float),
        "v_mse_x_km_s": scale * np.arange(1, n + 1, dtype=float),
        "v_mse_y_km_s": np.zeros(n),
        "v_mse_z_km_s": np.zeros(n),
        "processing_status_code": np.ones(n, dtype=np.int16),
        "density_valid_flag": np.ones(n, dtype=np.int8),
        "velocity_valid_flag": np.ones(n, dtype=np.int8),
        "quality_flag_bitmask": bitmask.astype(np.uint32),
        "quality_flag_available_flag": np.ones(n, dtype=np.int8),
        "quality_flag_epoch_matched_flag": np.ones(n, dtype=np.int8),
    }


def test_all_energy_classification_aggregates_three_species_and_quality_bits():
    bitmask = np.array([0, 1, 2, 0], dtype=np.uint32)
    chunk = SimpleNamespace(
        common={"r_matrix_within_tolerance_flag": np.ones(4, dtype=np.int8)},
        flat_bin=np.array([0, 0, 1, 1], dtype=np.int64),
        species={
            "Hplus": _species_values(1.0, bitmask),
            "Oplus": _species_values(2.0, bitmask),
            "O2plus": _species_values(3.0, bitmask),
        },
    )
    classification = classify_xb_mse(
        np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, -1]], dtype=float)
    )

    result = aggregate_all_energy_classified_chunk(chunk, classification)

    assert tuple(result["xb_plus_z_le45"]) == ALL_ENERGY_SPECIES
    plus_flat, plus_h = result["xb_plus_z_le45"]["hplus"]
    minus_flat, minus_o2 = result["xb_minus_z_le45"]["o2plus"]
    np.testing.assert_array_equal(plus_flat, [0])
    np.testing.assert_array_equal(minus_flat, [1])
    assert plus_h[0, 0] == 2
    assert plus_h[1, 0] == 3.0
    assert minus_o2[0, 0] == 1
    assert minus_o2[1, 0] == 12.0


def test_three_species_products_use_exact_half_rm_y_slab():
    grid = MseGridSpec(min_rm=-0.5, max_rm=0.5, step_rm=0.1, mars_radius_km=3397.0)
    category = _initialize_accumulators(
        grid.size**3, species_names=ALL_ENERGY_SPECIES
    )["xb_plus_z_le45"]
    ix, iz = 5, 5
    for iy in range(grid.size):
        flat = np.ravel_multi_index((ix, iy, iz), grid.shape)
        for species in ALL_ENERGY_SPECIES:
            category[species]["density_count"][flat] = 1
            category[species]["density_sum_cm3"][flat] = iy
            category[species]["velocity_count"][flat] = 1
            category[species]["speed_sum_km_s"][flat] = iy * 10

    _, xz = _products_from_dense(
        category,
        grid,
        y_min_rm=-0.5,
        y_max_rm=0.5,
        species_names=ALL_ENERGY_SPECIES,
    )

    for species in ALL_ENERGY_SPECIES:
        assert xz[f"{species}_density_count"][ix, iz] == 10
        np.testing.assert_allclose(xz[f"{species}_mean_density_cm3"][ix, iz], 4.5)
        np.testing.assert_allclose(xz[f"{species}_mean_speed_km_s"][ix, iz], 45.0)
        np.testing.assert_allclose(
            xz[f"{species}_density_speed_flux_cm2_s"][ix, iz],
            4.5 * 45.0 * 1.0e5,
        )
