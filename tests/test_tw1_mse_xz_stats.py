from pathlib import Path

import h5py
import numpy as np

from highE.tw1_mse_xz_stats import aggregate_record_blocks, run_tw1_mse_xz_statistics


def _record_dtype() -> np.dtype:
    fields = [("r_matrix_within_tolerance_flag", "i1")]
    for species in ("oplus", "o2plus"):
        fields.extend(
            [
                (f"{species}_density_cm3", "f8"),
                (f"{species}_v_mse_x_km_s", "f8"),
                (f"{species}_v_mse_y_km_s", "f8"),
                (f"{species}_v_mse_z_km_s", "f8"),
                (f"{species}_processing_status_code", "i2"),
                (f"{species}_density_valid_flag", "i1"),
                (f"{species}_velocity_valid_flag", "i1"),
                (f"{species}_quality_flag_bitmask", "u4"),
                (f"{species}_quality_flag_available_flag", "i1"),
                (f"{species}_quality_flag_epoch_matched_flag", "i1"),
            ]
        )
    return np.dtype(fields)


def _records(densities: list[float], vx: list[float], bitmask: list[int]) -> np.ndarray:
    rows = np.zeros(len(densities), dtype=_record_dtype())
    rows["r_matrix_within_tolerance_flag"] = 1
    for species in ("oplus", "o2plus"):
        rows[f"{species}_density_cm3"] = densities
        rows[f"{species}_v_mse_x_km_s"] = vx
        rows[f"{species}_v_mse_y_km_s"] = 0.0
        rows[f"{species}_v_mse_z_km_s"] = 0.0
        rows[f"{species}_processing_status_code"] = 1
        rows[f"{species}_density_valid_flag"] = 1
        rows[f"{species}_velocity_valid_flag"] = 1
        rows[f"{species}_quality_flag_bitmask"] = bitmask
        rows[f"{species}_quality_flag_available_flag"] = 1
        rows[f"{species}_quality_flag_epoch_matched_flag"] = 1
    return rows


def _write_cell(path: Path, ix: int, iy: int, iz: int, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(ix=ix, iy=iy, iz=iz)
        handle.create_group("days").create_group("20240101").create_dataset("records", data=rows)


def test_aggregate_record_blocks_rejects_bits_2_to_5_but_retains_bit1():
    rows = _records([1.0, 3.0, 5.0], [3.0, 4.0, 100.0], [0, 1, 2])
    stats = aggregate_record_blocks([rows], quality_policy="valid")["oplus"]

    assert stats[0] == 2
    assert stats[1] == 4.0
    assert stats[2] == 2
    assert stats[3] == 7.0
    assert stats[6] == 7.0


def test_aggregate_record_blocks_rejects_nan_density_and_velocity():
    rows = _records([1.0, np.nan, 3.0], [2.0, 4.0, np.nan], [0, 0, 0])
    stats = aggregate_record_blocks([rows], quality_policy="valid")["oplus"]

    assert stats[0] == 2
    assert stats[1] == 4.0
    assert stats[2] == 1
    assert stats[3] == 2.0


def test_run_statistics_xz_projection_is_record_weighted_across_y(tmp_path):
    input_root = tmp_path / "input"
    (input_root / "cells").mkdir(parents=True)
    (input_root / "grid_schema.json").write_text(
        '{"grid":{"min_rm":-1.0,"max_rm":1.0,"step_rm":1.0,"mars_radius_km":3397.0,"shape_xyz":[2,2,2]}}',
        encoding="utf-8",
    )
    _write_cell(input_root / "cells" / "a.h5", 0, 0, 1, _records([1.0], [2.0], [0]))
    _write_cell(input_root / "cells" / "b.h5", 0, 1, 1, _records([3.0, 5.0, 7.0], [4.0, 6.0, 8.0], [0, 0, 0]))

    summary = run_tw1_mse_xz_statistics(input_root, tmp_path / "output", workers=1, write_plots=False)

    assert summary["cell_files_processed"] == 2
    product = np.load(tmp_path / "output" / "data" / "tw1_minpa_mse_xz_all_y_means_valid.npz")
    assert product["oplus_density_count"][0, 1] == 4
    np.testing.assert_allclose(product["oplus_mean_density_cm3"][0, 1], 4.0)
    np.testing.assert_allclose(product["oplus_mean_vx_km_s"][0, 1], 5.0)
    np.testing.assert_allclose(product["oplus_mean_speed_km_s"][0, 1], 5.0)
