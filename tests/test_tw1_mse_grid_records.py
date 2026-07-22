import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from highE.tw1_mse_grid_records import (
    DailyProductPair,
    MseGridSpec,
    bin_mse_positions,
    load_daily_grid_chunk,
    prepare_rotation_time_axis,
    run_tw1_mse_grid_records,
)
from highE.tw1_mse_grid_audit import (
    EXPECTED_RECORD_FIELDS,
    audit_tw1_day_summary_provenance,
    audit_tw1_mse_grid_records,
)


def _write_species(path: Path, epoch: np.ndarray, position: np.ndarray, velocity: np.ndarray, density: np.ndarray, status: np.ndarray) -> None:
    payload = {
        "epoch_unix_s": epoch[:, None],
        "density_cm3": density[:, None],
        "processing_status_code": status[:, None],
        "mse_plus_z_in_minpa_fov_flag": np.array([1.0, np.nan])[:, None],
        "r_matrix_match_dt_s": np.array([1.0, -1.0])[:, None],
    }
    for index, axis in enumerate("xyz"):
        payload[f"pos_mso_{axis}_km"] = position[:, index, None]
        payload[f"pos_mse_{axis}_km"] = position[:, index, None]
        payload[f"v_mso_{axis}_km_s"] = velocity[:, index, None]
        payload[f"v_mse_{axis}_km_s"] = velocity[:, index, None]
    savemat(path, payload)


def _synthetic_inputs(tmp_path: Path, *, mismatch: bool = False) -> tuple[Path, Path, Path, DailyProductPair]:
    input_root = tmp_path / "input"
    day_root = input_root / "2024"
    r_root = tmp_path / "rotation"
    nv_root = tmp_path / "nv_mso_2"
    day_root.mkdir(parents=True)
    r_root.mkdir()
    nv_root.mkdir()
    date = "20240101"
    epoch = np.array([1.0, 2.0])
    radius = 3397.0
    position = np.array([[-5.0 * radius, 0.0, 0.0], [5.0 * radius, 0.0, 0.0]])
    velocity_o = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
    velocity_o2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    oplus = day_root / f"tw1_minpa_highE_Oplus_{date}.mat"
    o2plus = day_root / f"tw1_minpa_highE_O2plus_{date}.mat"
    _write_species(oplus, epoch, position, velocity_o, np.array([0.1, np.nan]), np.array([1, 5], dtype=np.int16))
    _write_species(o2plus, epoch, position, velocity_o2, np.array([0.2, np.nan]), np.array([1, 4], dtype=np.int16))
    if mismatch:
        from scipy.io import loadmat

        for path in (oplus, o2plus):
            changed = loadmat(path)
            changed["pos_mse_x_km"][0, 0] += 1.0
            savemat(path, {key: value for key, value in changed.items() if not key.startswith("__")})
    matrices = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    savemat(r_root / f"R_MSO2MSE_Tianwen-1_{date}.mat", {"data1": np.array([0.0, 3.0])[:, None], "data2": matrices})
    quality = {
        "quality_flag_O_TW1": np.column_stack([epoch, [1, 0]]),
        "quality_flag_bit_O_TW1": np.column_stack([epoch, [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
        "quality_flag_available_O_TW1": np.column_stack([epoch, [1, 0]]),
        "quality_flag_O2_TW1": np.column_stack([epoch, [4, 8]]),
        "quality_flag_bit_O2_TW1": np.column_stack([epoch, [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]]),
        "quality_flag_available_O2_TW1": np.column_stack([epoch, [1, 1]]),
        "MINPA_quality_flag_info": {
            "algorithm_version": "test-v1",
            "time_match_tolerance_s": 0.001,
            "flag_encoding": "test five-bit mask",
            "missing_semantics": "available=false means zero is a placeholder",
        },
    }
    savemat(nv_root / f"NV_{date}.mat", quality)
    return input_root, r_root, nv_root, DailyProductPair(date, oplus, o2plus)


def test_grid_uses_strict_plus_minus_five_edges() -> None:
    grid = MseGridSpec()
    radius = grid.mars_radius_km
    positions = np.array(
        [
            [-5.0 * radius, 0.0, 0.0],
            [5.0 * radius, 0.0, 0.0],
            [(5.0 + 1.0e-6) * radius, 0.0, 0.0],
        ]
    )
    flat, _ = bin_mse_positions(positions, grid)
    first = np.unravel_index(int(flat[0]), grid.shape)
    last = np.unravel_index(int(flat[1]), grid.shape)

    assert grid.shape == (100, 100, 100)
    assert first[0] == 0
    assert last[0] == 99
    assert flat[2] == -1
    assert grid.center_rm(0) == pytest.approx(-4.95)
    assert grid.center_rm(99) == pytest.approx(4.95)


def test_rotation_time_axis_drops_equivalent_duplicate() -> None:
    time = np.array([0.0, 1.0, 1.0, 2.0])
    matrices = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
    matrices[2, 0, 0] += 1.0e-15

    unique_time, unique_matrices, metadata = prepare_rotation_time_axis(time, matrices)

    np.testing.assert_allclose(unique_time, [0.0, 1.0, 2.0])
    assert unique_matrices.shape == (3, 3, 3)
    assert metadata["rotation_time_duplicates_removed"] == 1
    assert metadata["rotation_duplicate_matrix_max_error"] < 1.0e-10


def test_rotation_time_axis_keeps_first_and_records_conflicting_duplicate() -> None:
    time = np.array([0.0, 1.0, 1.0, 2.0])
    matrices = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
    matrices[2, 0, 0] = 2.0

    unique_time, unique_matrices, metadata = prepare_rotation_time_axis(time, matrices)

    np.testing.assert_allclose(unique_time, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(unique_matrices[1], np.eye(3))
    assert metadata["rotation_duplicate_conflicting_complete_groups"] == 1
    assert metadata["rotation_duplicate_matrix_max_error"] == pytest.approx(1.0)


def test_rotation_time_axis_prefers_complete_duplicate() -> None:
    time = np.array([0.0, 1.0, 1.0, 2.0])
    matrices = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
    matrices[1, :, :] = np.nan

    unique_time, unique_matrices, metadata = prepare_rotation_time_axis(time, matrices)

    np.testing.assert_allclose(unique_time, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(unique_matrices[1], np.eye(3))
    assert metadata["rotation_duplicate_groups_preferred_more_finite"] == 1


def test_streaming_writer_combines_species_and_keeps_invalid_moments(tmp_path: Path) -> None:
    input_root, r_root, nv_root, _ = _synthetic_inputs(tmp_path)
    output_root = tmp_path / "out"
    summary = run_tw1_mse_grid_records(
        input_root=input_root,
        output_root=output_root,
        r_root=r_root,
        nv_mso2_root=nv_root,
        dates=["20240101"],
        min_free_gb=0.0,
    )

    files = sorted((output_root / "cells").rglob("*.h5"))
    assert len(files) == 2
    assert summary["processed_days"] == 1
    assert summary["maximum_daily_working_memory_bytes"] > 0
    assert any("xm4p95" in path.name for path in files)
    assert any("xp4p95" in path.name for path in files)

    first_file = next(path for path in files if "xm4p95" in path.name)
    with h5py.File(first_file, "r") as handle:
        assert "_staging" not in handle
        day = handle["days/20240101"]
        records = day["records"][:]
        assert records.dtype.names == EXPECTED_RECORD_FIELDS
        assert len(records.dtype.names) == 43
        assert day.attrs["complete"] == 1
        assert day.attrs["row_count"] == 1
        assert handle.attrs["mars_radius_km"] == pytest.approx(3397.0)
        np.testing.assert_allclose(records["epoch_unix_s"], [1.0])
        np.testing.assert_allclose(records["oplus_density_cm3"], [0.1])
        np.testing.assert_allclose(records["o2plus_v_mse_z_km_s"], [6.0])
        assert records["oplus_processing_status_code"].tolist() == [1]
        assert records["oplus_quality_flag_bitmask"].tolist() == [1]
        assert records["oplus_quality_flag_bit1_sparse_caution"].tolist() == [1]
        assert records["o2plus_quality_flag_bit3_uv_contamination"].tolist() == [1]
        assert "oplus_density_cm3" in json.loads(day["records"].attrs["field_metadata_json"])
        assert day.attrs["quality_flag_algorithm_version"] == "test-v1"

    last_file = next(path for path in files if "xp4p95" in path.name)
    with h5py.File(last_file, "r") as handle:
        day = handle["days/20240101"]
        records = day["records"][:]
        assert np.isnan(records["oplus_density_cm3"]).all()
        assert np.isnan(records["oplus_v_mse_x_km_s"]).all()
        assert records["oplus_processing_status_code"].tolist() == [5]
        assert records["o2plus_processing_status_code"].tolist() == [4]
        assert records["oplus_quality_flag_available_flag"].tolist() == [0]
        assert records["oplus_quality_flag_bitmask"].tolist() == [0]
        assert records["o2plus_quality_flag_bit4_evenodd_error"].tolist() == [1]
        assert records["r_matrix_valid_flag"].tolist() == [1]
        assert records["r_matrix_within_tolerance_flag"].tolist() == [1]

    day_summary = json.loads((output_root / "logs" / "days" / "20240101.json").read_text())
    assert day_summary["rows_finite_position_in_grid"] == 2
    assert day_summary["nonempty_cells"] == 2

    resumed = run_tw1_mse_grid_records(
        input_root=input_root,
        output_root=output_root,
        r_root=r_root,
        nv_mso2_root=nv_root,
        dates=["20240101"],
        min_free_gb=0.0,
    )
    assert resumed["processed_days"] == 0
    assert resumed["already_complete_days"] == 1

    audit = audit_tw1_mse_grid_records(
        input_root,
        output_root,
        expected_days=1,
        expected_quality_algorithm_version="test-v1",
    )
    assert audit["status"] == "pass"
    assert audit["storage"]["records"] == 2
    assert audit["storage"]["cell_day_groups"] == 2

    day_audit = audit_tw1_day_summary_provenance(
        input_root,
        output_root,
        expected_days=1,
        expected_quality_algorithm_version="test-v1",
    )
    assert day_audit["status"] == "pass"
    assert day_audit["unique_daily_source_paths_checked"] == 4
    assert day_audit["rows_total"] == 2


def test_coordinate_recompute_mismatch_fails_day(tmp_path: Path) -> None:
    _, r_root, nv_root, pair = _synthetic_inputs(tmp_path, mismatch=True)

    with pytest.raises(ValueError, match="MSE position maximum absolute error"):
        load_daily_grid_chunk(pair, r_root, MseGridSpec(), nv_mso2_root=nv_root)
