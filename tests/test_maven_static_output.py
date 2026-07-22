import numpy as np
from scipy.io import loadmat

from highE.maven_static import write_maven_output


def test_write_maven_output_splits_species_without_duplicate_times(tmp_path):
    result = {
        "epoch_unix_s": np.array([1.0, 1.0, 2.0, 2.0]),
        "record_index": np.array([0, 0, 1, 1], dtype=np.int32),
        "species_code": np.array([1, 2, 1, 2], dtype=np.int16),
        "species_order": np.array([1, 2, 1, 2], dtype=np.int16),
        "swp_ind": np.zeros(4, dtype=np.int16),
        "quality_flag": np.zeros(4),
        "sc_pot_V": np.zeros(4),
        "mass_window_amu_min": np.zeros(4),
        "mass_window_amu_max": np.zeros(4),
        "density_cm3": np.ones(4),
        "valid_cell_count": np.ones(4, dtype=np.int32),
        "r_matrix_match_dt_s": np.zeros(4),
        "processing_status_code": np.ones(4, dtype=np.int16),
        "mse_plus_z_in_static_fov_flag": np.ones(4),
        "mse_plus_z_static_theta_deg": np.zeros(4),
        "static_fov_theta_min_deg": np.full(4, -45.866667),
        "static_fov_theta_max_deg": np.full(4, 45.866667),
    }
    for axis in "xyz":
        result[f"mse_plus_z_static_{axis}"] = np.zeros(4)
    for prefix in ["v_mso", "v_mse", "pos_mso", "pos_mse"]:
        for axis in "xyz":
            unit = "km_s" if prefix.startswith("v_") else "km"
            result[f"{prefix}_{axis}_{unit}"] = np.zeros(4)
    for frame in ["mso", "mse"]:
        for inst_axis in "xyz":
            for comp in "xyz":
                result[f"inst_{inst_axis}_axis_{frame}_{comp}"] = np.zeros(4)

    summary = {"records": {}, "sanity_checks": {}}
    paths, _ = write_maven_output("20211203", tmp_path, result, summary)

    for species_id, path in paths.items():
        data = loadmat(path)
        assert data["epoch_unix_s"].shape == (2, 1), species_id
        times = data["epoch_unix_s"].reshape(-1)
        assert np.unique(times).size == times.size


def test_write_maven_output_drops_later_duplicate_times_within_species(tmp_path):
    result = {
        "epoch_unix_s": np.array([1.0, 1.0, 2.0]),
        "record_index": np.array([0, 1, 2], dtype=np.int32),
        "species_code": np.array([1, 1, 1], dtype=np.int16),
        "species_order": np.array([1, 1, 1], dtype=np.int16),
        "swp_ind": np.zeros(3, dtype=np.int16),
        "quality_flag": np.zeros(3),
        "sc_pot_V": np.zeros(3),
        "mass_window_amu_min": np.zeros(3),
        "mass_window_amu_max": np.zeros(3),
        "density_cm3": np.array([1.0, 3.0, 2.0]),
        "valid_cell_count": np.ones(3, dtype=np.int32),
        "r_matrix_match_dt_s": np.zeros(3),
        "processing_status_code": np.ones(3, dtype=np.int16),
        "mse_plus_z_in_static_fov_flag": np.ones(3),
        "mse_plus_z_static_theta_deg": np.zeros(3),
        "static_fov_theta_min_deg": np.full(3, -45.866667),
        "static_fov_theta_max_deg": np.full(3, 45.866667),
    }
    for axis in "xyz":
        result[f"mse_plus_z_static_{axis}"] = np.zeros(3)
    for prefix in ["v_mso", "v_mse"]:
        for axis in "xyz":
            result[f"{prefix}_{axis}_km_s"] = np.zeros(3)
    result["v_mso_x_km_s"] = np.array([10.0, 30.0, 50.0])
    result["v_mse_x_km_s"] = np.array([10.0, 30.0, 50.0])
    for prefix in ["pos_mso", "pos_mse"]:
        for axis in "xyz":
            result[f"{prefix}_{axis}_km"] = np.zeros(3)
    for frame in ["mso", "mse"]:
        for inst_axis in "xyz":
            for comp in "xyz":
                result[f"inst_{inst_axis}_axis_{frame}_{comp}"] = np.zeros(3)

    summary = {"records": {}, "sanity_checks": {}}
    paths, _ = write_maven_output("20211204", tmp_path, result, summary)
    data = loadmat(paths["Oplus"])
    times = data["epoch_unix_s"].reshape(-1)
    assert times.tolist() == [1.0, 2.0]
    np.testing.assert_allclose(data["density_cm3"].reshape(-1), [1.0, 2.0])
    np.testing.assert_allclose(data["v_mso_x_km_s"].reshape(-1), [10.0, 50.0])
