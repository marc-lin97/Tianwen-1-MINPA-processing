from __future__ import annotations

import numpy as np

from highE.tw1_mse_fov_rebin import (
    project_sparse_statistics_to_coarse_xz,
    rebin_sparse_sufficient_statistics,
)
from highE.tw1_mse_grid_records import MseGridSpec
from highE.tw1_mse_xz_fov_stats import _products_from_dense, select_y_grid_centers
from highE.tw1_mse_xz_stats import SPECIES, STAT_NAMES


def _source_fixture() -> dict[str, np.ndarray]:
    source: dict[str, np.ndarray] = {
        "ix": np.asarray([0, 1, 0, 1], dtype=np.int16),
        "iy": np.asarray([0, 0, 1, 1], dtype=np.int16),
        "iz": np.asarray([0, 0, 1, 1], dtype=np.int16),
    }
    for species in SPECIES:
        for name in STAT_NAMES:
            if name == "density_count" or name == "velocity_count":
                source[f"{species}_{name}"] = np.ones(4, dtype=np.int64)
            elif name == "density_sum_cm3":
                source[f"{species}_{name}"] = np.asarray([1.0, 3.0, 5.0, 7.0])
            elif name == "speed_sum_km_s":
                source[f"{species}_{name}"] = np.asarray([10.0, 30.0, 50.0, 70.0])
            else:
                source[f"{species}_{name}"] = np.asarray([2.0, 4.0, 6.0, 8.0])
    return source


def test_rebin_sums_sufficient_statistics_before_recomputing_means():
    source_grid = MseGridSpec(min_rm=-0.2, max_rm=0.2, step_rm=0.1)
    target_grid = MseGridSpec(min_rm=-0.2, max_rm=0.2, step_rm=0.2)
    rebinned = rebin_sparse_sufficient_statistics(_source_fixture(), source_grid, target_grid)
    flat = np.ravel_multi_index((0, 0, 0), target_grid.shape)
    assert rebinned["oplus"]["density_count"][flat] == 4
    assert rebinned["oplus"]["density_sum_cm3"][flat] == 16.0
    assert rebinned["oplus"]["speed_sum_km_s"][flat] == 160.0

    _, xz = _products_from_dense(rebinned, target_grid, y_min_rm=-0.2, y_max_rm=0.0)
    np.testing.assert_allclose(xz["oplus_mean_density_cm3"][0, 0], 4.0)
    np.testing.assert_allclose(xz["oplus_mean_speed_km_s"][0, 0], 40.0)
    np.testing.assert_allclose(xz["oplus_density_speed_flux_cm2_s"][0, 0], 4.0 * 40.0 * 1.0e5)


def test_coarse_xz_projection_preserves_exact_source_y_slab():
    source_grid = MseGridSpec(min_rm=-0.2, max_rm=0.2, step_rm=0.1)
    target_grid = MseGridSpec(min_rm=-0.2, max_rm=0.2, step_rm=0.2)
    source = _source_fixture()
    xz = project_sparse_statistics_to_coarse_xz(
        source,
        source_grid,
        target_grid,
        y_min_rm=-0.15,
        y_max_rm=-0.1,
    )
    assert xz["oplus_density_count"][0, 0] == 2
    np.testing.assert_allclose(xz["oplus_mean_density_cm3"][0, 0], 2.0)
    np.testing.assert_allclose(xz["oplus_mean_speed_km_s"][0, 0], 20.0)
    np.testing.assert_allclose(
        xz["oplus_density_speed_flux_cm2_s"][0, 0], 2.0 * 20.0 * 1.0e5
    )


def test_half_rm_slab_uses_ten_source_layers_before_coarsening():
    grid = MseGridSpec(step_rm=0.1)
    indices, centers = select_y_grid_centers(grid, y_min_rm=-0.5, y_max_rm=0.5)
    np.testing.assert_array_equal(indices, np.arange(45, 55))
    np.testing.assert_allclose(centers, np.arange(-0.45, 0.5, 0.1), atol=1.0e-12)
