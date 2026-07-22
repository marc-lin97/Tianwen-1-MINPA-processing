import numpy as np

from highE.coordinates import (
    apply_rotation,
    density_from_eflux_cm3,
    mse_plus_z_static_fov_flag,
    quaternion_rotate_scalar_first,
    tw1_minpa_instrument_to_orbiter_matrix,
    tw1_minpa_legacy_to_probe_step_matrix,
    tw1_orbiter_to_mso_matrix,
)


def test_identity_quaternion_rotates_vector_unchanged():
    vectors = np.eye(3)
    out = quaternion_rotate_scalar_first(np.array([1.0, 0.0, 0.0, 0.0]), vectors)
    np.testing.assert_allclose(out, vectors)


def test_rotation_preserves_speed():
    theta = np.deg2rad(20.0)
    r = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ]
        ]
    )
    v = np.array([[10.0, -2.0, 4.0]])
    out = apply_rotation(r, v)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.linalg.norm(v, axis=1), rtol=1e-12)


def test_tw1_orbiter_to_mso_zero_attitude_is_identity():
    matrix = tw1_orbiter_to_mso_matrix(0.0, 0.0, 0.0)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-15)


def test_tw1_orbiter_to_mso_known_attitude():
    matrix = tw1_orbiter_to_mso_matrix(-90.0, 0.0, 90.0)
    expected = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-15)


def test_tw1_orbiter_to_mso_matrices_are_proper_rotations():
    matrix = tw1_orbiter_to_mso_matrix(
        np.array([-90.0, 12.5, 33.0]),
        np.array([0.0, -7.0, 45.0]),
        np.array([90.0, 120.0, -18.0]),
    )
    identity = np.einsum("nij,nkj->nik", matrix, matrix)
    np.testing.assert_allclose(identity, np.broadcast_to(np.eye(3), identity.shape), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(matrix), np.ones(3), atol=1e-14)


def test_tw1_minpa_instrument_to_orbiter_matrix_maps_axes():
    matrix = tw1_minpa_instrument_to_orbiter_matrix()
    np.testing.assert_allclose(matrix @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0])
    np.testing.assert_allclose(matrix @ np.array([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(matrix @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0])


def test_tw1_minpa_instrument_to_orbiter_matrix_is_proper_rotation():
    matrix = tw1_minpa_instrument_to_orbiter_matrix()
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-15)


def test_tw1_minpa_matrix_matches_previous_double_legacy_mapping():
    matrix = tw1_minpa_instrument_to_orbiter_matrix()
    directions = np.array(
        [
            [1.0, 2.0, 3.0],
            [-4.0, 5.0, -6.0],
            [0.25, -0.5, 0.75],
        ]
    )

    def legacy_to_probe(values: np.ndarray) -> np.ndarray:
        return np.column_stack([-values[:, 2], -values[:, 0], values[:, 1]])

    previous = legacy_to_probe(legacy_to_probe(directions))
    explicit = directions @ matrix.T
    np.testing.assert_allclose(explicit, previous, atol=1e-15)


def test_tw1_minpa_direct_nv_mapping_is_algebraically_equal_to_historical_two_steps():
    step = tw1_minpa_legacy_to_probe_step_matrix()
    np.testing.assert_allclose(tw1_minpa_instrument_to_orbiter_matrix(), step @ step, atol=0.0)


def test_density_formula_positive_cell():
    density = density_from_eflux_cm3(
        eflux=np.array([1.0e5]),
        energy_eV=np.array([1000.0]),
        denergy_eV=np.array([100.0]),
        domega_sr=np.array([0.01]),
        speed_km_s=np.array([100.0]),
    )
    np.testing.assert_allclose(density, np.array([1.0e-5]))


def test_mse_plus_z_outside_static_equatorial_fov_for_identity_axes():
    axes_mse = np.eye(3)[None, :, :]
    flag, theta_deg, target_static = mse_plus_z_static_fov_flag(axes_mse, theta_min_deg=-45.866667, theta_max_deg=45.866667)
    np.testing.assert_allclose(flag, [0.0])
    np.testing.assert_allclose(theta_deg, [90.0])
    np.testing.assert_allclose(target_static, [[0.0, 0.0, 1.0]])


def test_mse_plus_z_inside_static_fov_when_instrument_x_axis_points_plus_z():
    axes_mse = np.array(
        [
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ]
    )
    flag, theta_deg, target_static = mse_plus_z_static_fov_flag(axes_mse, theta_min_deg=-45.866667, theta_max_deg=45.866667)
    np.testing.assert_allclose(flag, [1.0])
    np.testing.assert_allclose(theta_deg, [0.0])
    np.testing.assert_allclose(target_static, [[1.0, 0.0, 0.0]])


def test_mse_plus_z_flag_is_nan_when_attitude_is_not_determinable():
    axes_mse = np.full((1, 3, 3), np.nan)
    flag, theta_deg, target_static = mse_plus_z_static_fov_flag(axes_mse, theta_min_deg=-45.866667, theta_max_deg=45.866667)
    assert np.isnan(flag[0])
    assert np.isnan(theta_deg[0])
    assert np.isnan(target_static[0]).all()
