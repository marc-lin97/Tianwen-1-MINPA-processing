import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from highE.tw1_all_energy_mse_grid_audit import audit_tw1_all_energy_mse_grid_records
from highE.tw1_all_energy_mse_grid_records import (
    EXPECTED_RECORD_FIELDS,
    MseGridSpec,
    run_tw1_all_energy_mse_grid_records,
)


def _quality_payload(epoch: np.ndarray, suffix: str, masks: np.ndarray) -> dict[str, np.ndarray]:
    bits = np.column_stack([((masks.astype(np.uint32) >> index) & 1) for index in range(5)])
    return {
        f"quality_flag_{suffix}_TW1": np.column_stack([epoch, masks]),
        f"quality_flag_bit_{suffix}_TW1": np.column_stack([epoch, bits]),
        f"quality_flag_available_{suffix}_TW1": np.column_stack([epoch, np.ones(epoch.size)]),
    }


def _velocity(epoch: np.ndarray, xyz: list[list[float]], time_finite: list[bool]) -> np.ndarray:
    time = np.where(np.asarray(time_finite), epoch, np.nan)
    return np.column_stack([time, np.asarray(xyz, dtype=float)])


def _write_nv(
    path: Path,
    epoch: np.ndarray,
    densities: dict[str, list[float]],
    velocities: dict[str, np.ndarray],
    xb: np.ndarray,
    masks: dict[str, list[int]],
) -> None:
    payload: dict[str, object] = {
        "NH_TW1": np.column_stack([epoch, densities["H"]]),
        "NO_TW1": np.column_stack([epoch, densities["O"]]),
        "NO2_TW1": np.column_stack([epoch, densities["O2"]]),
        "VH_TW1_MSO": velocities["H"],
        "VO_TW1_MSO": velocities["O"],
        "VO2_TW1_MSO": velocities["O2"],
        "Xb_MSO": np.column_stack([epoch, xb]),
        "mode_TW1": np.column_stack([epoch, np.ones(epoch.size)]),
        "NV_MSO_2_info": {
            "coordinate_system": "MSO",
            "density_processing": "synthetic all-energy density",
            "velocity_processing": "synthetic MSO velocity",
            "source_nv_mso": "synthetic",
            "source_day_spe": "synthetic",
            "source_momag": "synthetic",
        },
        "MINPA_quality_flag_info": {
            "algorithm_version": "test-v1",
            "time_match_tolerance_s": 0.001,
            "flag_encoding": "test five-bit species mask",
            "missing_semantics": "available=0 means numeric zero is a placeholder",
        },
    }
    for suffix in ("H", "O", "O2"):
        payload.update(_quality_payload(epoch, suffix, np.asarray(masks[suffix], dtype=np.uint32)))
    savemat(path, payload)


def _write_geometry(root: Path, date: str, epoch: float, position: np.ndarray) -> tuple[Path, Path]:
    momag_root = root / "momag"
    rotation_root = root / "rotation"
    momag_root.mkdir(exist_ok=True)
    rotation_root.mkdir(exist_ok=True)
    times = np.asarray([epoch - 1.0, epoch, epoch + 1.0])
    positions = np.repeat(np.asarray(position, dtype=float)[None, :], times.size, axis=0)
    savemat(
        momag_root / f"Bss{date}.mat",
        {
            "P_TW1": np.column_stack([times, positions]),
            "Pitch_TW1": np.column_stack([times, np.zeros(times.size)]),
        },
    )
    matrices = np.repeat(np.eye(3)[None, :, :], times.size, axis=0)
    savemat(rotation_root / f"R_MSO2MSE_Tianwen-1_{date}.mat", {"data1": times[:, None], "data2": matrices})
    return momag_root, rotation_root


def _synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_root = tmp_path / "nv"
    input_root.mkdir()
    jan1 = 1704110400.0  # 2024-01-01 12:00 UTC
    jan2 = 1704153600.0  # 2024-01-02 00:00 UTC, duplicated in adjacent products
    jan3 = 1704240000.0
    _write_nv(
        input_root / "NV_20240101.mat",
        np.asarray([jan1, jan2]),
        densities={"H": [1.0, 0.0], "O": [2.0, 3.0], "O2": [np.nan, 4.0]},
        velocities={
            "H": _velocity(np.asarray([jan1, jan2]), [[1, 2, 3], [np.nan] * 3], [True, False]),
            "O": _velocity(np.asarray([jan1, jan2]), [[np.nan] * 3, [np.nan] * 3], [True, False]),
            "O2": _velocity(np.asarray([jan1, jan2]), [[7, 8, 9], [np.nan] * 3], [True, False]),
        },
        xb=np.asarray([[0, 0, -1], [np.nan, np.nan, np.nan]]),
        masks={"H": [1, 4], "O": [4, 8], "O2": [16, 2]},
    )
    _write_nv(
        input_root / "NV_20240102.mat",
        np.asarray([jan2, jan3]),
        densities={"H": [0.0, 5.0], "O": [3.0, 6.0], "O2": [4.0, 7.0]},
        velocities={
            "H": _velocity(np.asarray([jan2, jan3]), [[10, 20, 30], [1, 1, 1]], [True, True]),
            "O": _velocity(np.asarray([jan2, jan3]), [[4, 5, 6], [2, 2, 2]], [True, True]),
            "O2": _velocity(np.asarray([jan2, jan3]), [[np.nan] * 3, [3, 3, 3]], [False, True]),
        },
        xb=np.asarray([[0, 0, 1], [0, 0, -1]]),
        masks={"H": [4, 1], "O": [8, 4], "O2": [2, 16]},
    )
    # A selected source day with no geometry exercises the explicit exclusion report.
    savemat(input_root / "NV_20240103.mat", {"NH_TW1": np.asarray([[jan3, 1.0]])})
    momag_root, rotation_root = _write_geometry(tmp_path, "20240101", jan1, np.asarray([-5 * 3397.0, 0, 0]))
    _write_geometry(tmp_path, "20240102", jan2, np.asarray([5 * 3397.0, 0, 0]))
    return input_root, momag_root, rotation_root


def test_all_energy_grid_archive_and_audit(tmp_path: Path, monkeypatch) -> None:
    input_root, momag_root, rotation_root = _synthetic_inputs(tmp_path)
    output_root = tmp_path / "out"
    monkeypatch.setattr(
        "highE.tw1_all_energy_mse_grid_records.minpa_theta_edges_deg", lambda mode: (0.0, 90.0)
    )
    summary = run_tw1_all_energy_mse_grid_records(
        input_root=input_root,
        output_root=output_root,
        momag_root=momag_root,
        r_root=rotation_root,
        min_free_gb=0.0,
    )

    assert summary["requested_days"] == 2
    assert summary["excluded_geometry_days"] == 1
    assert summary["canonical_utc_rows"] == 2
    assert summary["in_grid_rows"] == 2
    assert len(EXPECTED_RECORD_FIELDS) == 59
    coverage = json.loads((output_root / "logs" / "geometry_coverage.json").read_text())
    assert coverage["excluded"][0]["date"] == "20240103"
    assert coverage["excluded_canonical_utc_rows"] == 1

    files = sorted((output_root / "cells").rglob("*.h5"))
    assert len(files) == 2
    first = next(path for path in files if "xm4p95" in path.name)
    last = next(path for path in files if "xp4p95" in path.name)
    with h5py.File(first, "r") as handle:
        records = handle["days/20240101/records"][:]
        assert records.dtype.names == EXPECTED_RECORD_FIELDS
        assert records["hplus_processing_status_code"].tolist() == [1]
        assert records["oplus_processing_status_code"].tolist() == [5]
        assert records["o2plus_processing_status_code"].tolist() == [4]
        assert records["hplus_quality_flag_bit1_sparse_caution"].tolist() == [1]
        assert records["mse_plus_z_in_minpa_fov_flag"].tolist() == [1.0]
    with h5py.File(last, "r") as handle:
        records = handle["days/20240102/records"][:]
        assert records["hplus_processing_status_code"].tolist() == [4]
        assert records["oplus_processing_status_code"].tolist() == [1]
        assert records["o2plus_processing_status_code"].tolist() == [5]
        np.testing.assert_allclose(
            [records["hplus_v_mse_x_km_s"][0], records["hplus_v_mse_y_km_s"][0], records["hplus_v_mse_z_km_s"][0]],
            [10.0, 20.0, 30.0],
        )
        assert records["mse_plus_z_in_minpa_fov_flag"].tolist() == [0.0]

    resumed = run_tw1_all_energy_mse_grid_records(
        input_root=input_root,
        output_root=output_root,
        momag_root=momag_root,
        r_root=rotation_root,
        min_free_gb=0.0,
    )
    assert resumed["processed_days"] == 0
    assert resumed["already_complete_days"] == 2

    audit = audit_tw1_all_energy_mse_grid_records(
        input_root=input_root,
        output_root=output_root,
        momag_root=momag_root,
        r_root=rotation_root,
        expected_days=2,
        expected_excluded_days=1,
        expected_quality_algorithm_version="test-v1",
        source_samples_per_day=1,
    )
    assert audit["status"] == "pass", audit["errors"]
    assert audit["storage"]["records"] == 2
    assert audit["source_reconstruction"]["sample_mismatches"] == 0


def test_grid_boundary_definition_is_project_compatible() -> None:
    grid = MseGridSpec()
    assert grid.shape == (100, 100, 100)
    assert grid.center_rm(0) == pytest.approx(-4.95)
    assert grid.center_rm(99) == pytest.approx(4.95)
    assert grid.mars_radius_km == 3397.0
