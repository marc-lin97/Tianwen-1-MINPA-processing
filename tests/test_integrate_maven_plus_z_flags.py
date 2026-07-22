import numpy as np
from scipy.io import loadmat, savemat

from scripts.integrate_maven_plus_z_flags_into_oplus import FLAG_FIELDS, integrate_one


def test_integrate_one_adds_flag_fields_and_is_idempotent(tmp_path):
    root = tmp_path / "MAVEN"
    out_dir = root / "2024"
    out_dir.mkdir(parents=True)
    date = "20240102"
    oplus_path = out_dir / f"maven_static_highE_Oplus_{date}.mat"
    flag_path = out_dir / f"maven_static_plus_z_fov_flag_{date}.mat"
    epoch = np.array([1.0, 2.0, 3.0])[:, None]
    savemat(
        oplus_path,
        {
            "epoch_unix_s": epoch,
            "density_cm3": np.ones((3, 1)),
        },
    )
    flag_payload = {
        "epoch_unix_s": epoch,
        "mse_plus_z_in_static_fov_flag": np.array([1.0, 0.0, np.nan])[:, None],
        "mse_plus_z_static_theta_deg": np.array([0.0, 50.0, np.nan])[:, None],
        "mse_plus_z_static_x": np.zeros((3, 1)),
        "mse_plus_z_static_y": np.zeros((3, 1)),
        "mse_plus_z_static_z": np.ones((3, 1)),
        "static_fov_theta_min_deg": np.full((3, 1), -45.866667),
        "static_fov_theta_max_deg": np.full((3, 1), 45.866667),
    }
    savemat(flag_path, flag_payload)

    first = integrate_one(root, date, dry_run=False, overwrite=False, min_source_age_seconds=0)
    assert first["status"] == "written"
    merged = loadmat(oplus_path)
    for field in FLAG_FIELDS:
        assert field in merged
        np.testing.assert_allclose(merged[field], flag_payload[field], equal_nan=True)

    second = integrate_one(root, date, dry_run=False, overwrite=False, min_source_age_seconds=0)
    assert second["status"] == "already_integrated"


def test_integrate_one_rejects_epoch_mismatch(tmp_path):
    root = tmp_path / "MAVEN"
    out_dir = root / "2024"
    out_dir.mkdir(parents=True)
    date = "20240103"
    savemat(out_dir / f"maven_static_highE_Oplus_{date}.mat", {"epoch_unix_s": np.array([1.0, 2.0])[:, None]})
    payload = {field: np.zeros((2, 1)) for field in FLAG_FIELDS}
    payload["epoch_unix_s"] = np.array([1.0, 3.0])[:, None]
    savemat(out_dir / f"maven_static_plus_z_fov_flag_{date}.mat", payload)

    result = integrate_one(root, date, dry_run=False, overwrite=False, min_source_age_seconds=0)
    assert result["status"] == "epoch_mismatch"
