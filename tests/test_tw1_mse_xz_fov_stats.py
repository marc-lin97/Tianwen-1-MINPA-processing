import numpy as np

from highE.tw1_mse_grid_records import DailyGridChunk
from highE.tw1_mse_xz_fov_stats import (
    _initialize_accumulators,
    _products_from_dense,
    aggregate_classified_daily_chunk,
    classify_xb_mse,
    select_y_grid_centers,
)
from highE.tw1_mse_grid_records import MseGridSpec


def test_classify_xb_mse_uses_inclusive_45_degree_cones_and_nan_is_invalid():
    root_half = np.sqrt(0.5)
    vectors = np.array(
        [
            [0.0, 0.0, 1.0],
            [root_half, 0.0, root_half],
            [0.0, 0.0, -1.0],
            [root_half, 0.0, -root_half],
            [1.0, 0.0, 0.0],
            [np.nan, 0.0, 1.0],
        ]
    )
    result = classify_xb_mse(vectors)

    np.testing.assert_array_equal(result.categories["xb_plus_z_le45"], [1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(result.categories["xb_minus_z_le45"], [0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(result.valid, [1, 1, 1, 1, 1, 0])
    assert not np.any(result.categories["xb_plus_z_le45"] & result.categories["xb_minus_z_le45"])


def _species_values(bitmask: np.ndarray) -> dict[str, np.ndarray]:
    n = bitmask.size
    return {
        "density_cm3": np.arange(1, n + 1, dtype=float),
        "v_mse_x_km_s": np.arange(1, n + 1, dtype=float),
        "v_mse_y_km_s": np.zeros(n),
        "v_mse_z_km_s": np.zeros(n),
        "processing_status_code": np.ones(n, dtype=np.int16),
        "density_valid_flag": np.ones(n, dtype=np.int8),
        "velocity_valid_flag": np.ones(n, dtype=np.int8),
        "quality_flag_bitmask": bitmask.astype(np.uint32),
        "quality_flag_available_flag": np.ones(n, dtype=np.int8),
        "quality_flag_epoch_matched_flag": np.ones(n, dtype=np.int8),
    }


def test_daily_aggregation_retains_bit1_but_rejects_bit2_and_separates_cones():
    bitmask = np.array([0, 1, 2, 0], dtype=np.uint32)
    common = {
        "epoch_unix_s": np.arange(4, dtype=float),
        "r_matrix_within_tolerance_flag": np.ones(4, dtype=np.int8),
    }
    chunk = DailyGridChunk(
        date="20240101",
        source_files={},
        rotation_file="rotation.mat",
        quality_flag_file="NV.mat",
        quality_flag_metadata={},
        common=common,
        species={"Oplus": _species_values(bitmask), "O2plus": _species_values(bitmask)},
        flat_bin=np.array([0, 0, 1, 1], dtype=np.int64),
        validation={},
    )
    classification = classify_xb_mse(
        np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, -1]], dtype=float)
    )

    result = aggregate_classified_daily_chunk(chunk, classification)
    plus_flat, plus_stats = result["xb_plus_z_le45"]["oplus"]
    minus_flat, minus_stats = result["xb_minus_z_le45"]["oplus"]

    np.testing.assert_array_equal(plus_flat, [0])
    assert plus_stats[0, 0] == 2
    assert plus_stats[1, 0] == 3.0
    np.testing.assert_array_equal(minus_flat, [1])
    assert minus_stats[0, 0] == 1
    assert minus_stats[1, 0] == 4.0


def test_y_projection_selects_four_centers_between_minus_and_plus_point_two_rm():
    grid = MseGridSpec(min_rm=-0.3, max_rm=0.3, step_rm=0.1, mars_radius_km=3397.0)
    indices, centers = select_y_grid_centers(grid, y_min_rm=-0.2, y_max_rm=0.2)
    np.testing.assert_array_equal(indices, [1, 2, 3, 4])
    np.testing.assert_allclose(centers, [-0.15, -0.05, 0.05, 0.15], atol=1e-14)

    accumulators = _initialize_accumulators(grid.size**3)["xb_plus_z_le45"]
    ix, iz = 2, 3
    for iy in range(grid.size):
        flat = np.ravel_multi_index((ix, iy, iz), grid.shape)
        for species in ("oplus", "o2plus"):
            accumulators[species]["density_count"][flat] = 1
            accumulators[species]["density_sum_cm3"][flat] = iy + 1
            accumulators[species]["velocity_count"][flat] = 1
            accumulators[species]["speed_sum_km_s"][flat] = (iy + 1) * 10
    _, xz = _products_from_dense(accumulators, grid, y_min_rm=-0.2, y_max_rm=0.2)
    assert xz["oplus_density_count"][ix, iz] == 4
    np.testing.assert_allclose(xz["oplus_mean_density_cm3"][ix, iz], 3.5)
    np.testing.assert_allclose(xz["oplus_mean_speed_km_s"][ix, iz], 35.0)
    np.testing.assert_allclose(xz["oplus_density_speed_flux_cm2_s"][ix, iz], 3.5 * 35.0 * 1.0e5)


def test_requested_half_rm_y_slab_selects_ten_grid_layers():
    grid = MseGridSpec()
    indices, centers = select_y_grid_centers(grid, y_min_rm=-0.5, y_max_rm=0.5)
    np.testing.assert_array_equal(indices, np.arange(45, 55))
    np.testing.assert_allclose(centers, np.arange(-0.45, 0.5, 0.1), atol=1e-12)


def test_requested_one_rm_y_slab_selects_twenty_grid_layers():
    grid = MseGridSpec()
    indices, centers = select_y_grid_centers(grid, y_min_rm=-1.0, y_max_rm=1.0)
    np.testing.assert_array_equal(indices, np.arange(40, 60))
    np.testing.assert_allclose(centers, np.arange(-0.95, 1.0, 0.1), atol=1e-12)
