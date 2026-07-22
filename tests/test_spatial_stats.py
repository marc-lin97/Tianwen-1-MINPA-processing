import numpy as np
from scipy.io import loadmat, savemat

from highE.spatial_stats import run_spatial_statistics
from highE.spatial_stats import MISSION_CONFIGS, find_species_files


def test_run_spatial_statistics_groups_fov_and_filters_density(tmp_path):
    input_root = tmp_path / "input"
    day_dir = input_root / "MAVEN" / "2024"
    day_dir.mkdir(parents=True)
    path = day_dir / "maven_static_highE_Oplus_20240101.mat"
    savemat(
        path,
        {
            "epoch_unix_s": np.array([1.0, 2.0, 3.0, 4.0])[:, None],
            "density_cm3": np.array([0.01, 0.03, 0.0005, 0.02])[:, None],
            "processing_status_code": np.array([1, 1, 1, 1], dtype=np.int16)[:, None],
            "mse_plus_z_in_static_fov_flag": np.array([1.0, 1.0, 1.0, 0.0])[:, None],
            "pos_mse_x_km": np.array([0.0, 0.0, 0.0, 339.7])[:, None],
            "pos_mse_y_km": np.array([0.0, 0.0, 0.0, 0.0])[:, None],
            "pos_mse_z_km": np.array([0.0, 0.0, 0.0, 0.0])[:, None],
            "v_mse_x_km_s": np.array([1.0, 3.0, 9.0, -2.0])[:, None],
            "v_mse_y_km_s": np.array([2.0, 4.0, 9.0, 0.0])[:, None],
            "v_mse_z_km_s": np.array([2.0, 0.0, 9.0, 0.0])[:, None],
        },
    )

    summary = run_spatial_statistics(
        input_root=input_root,
        output_root=tmp_path / "out",
        missions=["MAVEN"],
        species_ids=["Oplus"],
        workers=1,
        write_plots=False,
    )

    written = [item for item in summary["products"] if item.get("status") == "written"]
    assert len(written) == 2
    in_file = tmp_path / "out" / "data" / "MAVEN_Oplus_plus_z_in_fov_mse_grid_stats.npz"
    out_file = tmp_path / "out" / "data" / "MAVEN_Oplus_plus_z_out_fov_mse_grid_stats.npz"
    assert in_file.exists()
    assert out_file.exists()

    data = np.load(in_file)
    x = data["axis_x_rm"]
    zero = int(np.argmin(np.abs(x)))
    row = np.where((data["ix"] == zero) & (data["iy"] == zero) & (data["iz"] == zero))[0][0]
    assert data["density_count"][row] == 2
    assert data["velocity_count"][row] == 2
    np.testing.assert_allclose(data["mean_density_cm3"][row], 0.02)
    np.testing.assert_allclose(data["median_density_cm3"][row], 0.02)
    np.testing.assert_allclose(data["mean_v_mse_x_km_s"][row], 2.0)
    np.testing.assert_allclose(data["median_v_mse_y_km_s"][row], 3.0)
    np.testing.assert_allclose(data["mean_speed_mse_km_s"][row], 4.0)

    out = np.load(out_file)
    ix = int(np.where(np.isclose(out["axis_x_rm"], 0.1))[0][0])
    row = np.where((out["ix"] == ix) & (out["iy"] == zero) & (out["iz"] == zero))[0][0]
    assert out["density_count"][row] == 1
    np.testing.assert_allclose(out["mean_v_mse_x_km_s"][row], -2.0)


def test_run_spatial_statistics_uses_maven_standalone_flag_fallback(tmp_path):
    input_root = tmp_path / "input"
    day_dir = input_root / "MAVEN" / "2024"
    day_dir.mkdir(parents=True)
    epoch = np.array([1.0, 2.0])[:, None]
    savemat(
        day_dir / "maven_static_highE_O2plus_20240102.mat",
        {
            "epoch_unix_s": epoch,
            "density_cm3": np.array([0.01, 0.02])[:, None],
            "processing_status_code": np.array([1, 1], dtype=np.int16)[:, None],
            "pos_mse_x_km": np.zeros((2, 1)),
            "pos_mse_y_km": np.zeros((2, 1)),
            "pos_mse_z_km": np.zeros((2, 1)),
            "v_mse_x_km_s": np.ones((2, 1)),
            "v_mse_y_km_s": np.zeros((2, 1)),
            "v_mse_z_km_s": np.zeros((2, 1)),
        },
    )
    savemat(
        day_dir / "maven_static_plus_z_fov_flag_20240102.mat",
        {
            "epoch_unix_s": epoch,
            "mse_plus_z_in_static_fov_flag": np.array([1.0, 0.0])[:, None],
        },
    )

    run_spatial_statistics(
        input_root=input_root,
        output_root=tmp_path / "out",
        missions=["MAVEN"],
        species_ids=["O2plus"],
        workers=1,
        write_plots=False,
    )

    data = np.load(tmp_path / "out" / "data" / "MAVEN_O2plus_plus_z_in_fov_mse_grid_stats.npz")
    zero = int(np.argmin(np.abs(data["axis_x_rm"])))
    row = np.where((data["ix"] == zero) & (data["iy"] == zero) & (data["iz"] == zero))[0][0]
    assert data["density_count"][row] == 1


def test_find_species_files_builds_direct_paths_for_dates(tmp_path):
    root = tmp_path / "input"
    day_dir = root / "MAVEN" / "2024"
    day_dir.mkdir(parents=True)
    expected = day_dir / "maven_static_highE_Oplus_20240103.mat"
    expected.write_bytes(b"placeholder")

    files = find_species_files(root, MISSION_CONFIGS["MAVEN"], "Oplus", dates=["20240103", "20240104"])

    assert files == [expected]


def test_grid_records_output_writes_mat_by_grid_and_date(tmp_path):
    input_root = tmp_path / "input"
    day_dir = input_root / "MAVEN" / "2024"
    day_dir.mkdir(parents=True)
    path = day_dir / "maven_static_highE_Oplus_20240105.mat"
    savemat(
        path,
        {
            "epoch_unix_s": np.array([10.0, 20.0])[:, None],
            "density_cm3": np.array([0.01, 0.02])[:, None],
            "processing_status_code": np.array([1, 1], dtype=np.int16)[:, None],
            "mse_plus_z_in_static_fov_flag": np.array([1.0, 1.0])[:, None],
            "pos_mse_x_km": np.zeros((2, 1)),
            "pos_mse_y_km": np.zeros((2, 1)),
            "pos_mse_z_km": np.zeros((2, 1)),
            "v_mse_x_km_s": np.array([1.0, 3.0])[:, None],
            "v_mse_y_km_s": np.array([0.0, 0.0])[:, None],
            "v_mse_z_km_s": np.array([0.0, 4.0])[:, None],
        },
    )

    summary = run_spatial_statistics(
        input_root=input_root,
        output_root=tmp_path / "out",
        missions=["MAVEN"],
        species_ids=["Oplus"],
        workers=1,
        output_mode="grid-records",
    )

    record_products = [item for item in summary["products"] if item.get("status") == "grid_records_written"]
    assert record_products[0]["grid_record_file_count"] == 1
    mat_files = list((tmp_path / "out" / "grid_records").rglob("20240105.mat"))
    assert len(mat_files) == 1
    data = loadmat(mat_files[0], squeeze_me=True)
    assert "utc" not in data
    np.testing.assert_allclose(np.asarray(data["epoch_unix_s"]).reshape(-1), [10.0, 20.0])
    np.testing.assert_allclose(np.asarray(data["density_cm3"]).reshape(-1), [0.01, 0.02])
    np.testing.assert_allclose(np.asarray(data["v_mse_z_km_s"]).reshape(-1), [0.0, 4.0])
