from pathlib import Path

import h5py
import numpy as np
import pytest

from highE.tw1_mse_xz_stats import (
    aggregate_record_blocks,
    run_tw1_all_energy_mse_xz_statistics,
)


SPECIES = ("hplus", "oplus", "o2plus")


def _record_dtype() -> np.dtype:
    fields = [("r_matrix_within_tolerance_flag", "i1")]
    for species in SPECIES:
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


def _records() -> np.ndarray:
    rows = np.zeros(3, dtype=_record_dtype())
    rows["r_matrix_within_tolerance_flag"] = 1
    for offset, species in enumerate(SPECIES, start=1):
        rows[f"{species}_density_cm3"] = np.array([1.0, 3.0, 100.0]) * offset
        rows[f"{species}_v_mse_x_km_s"] = np.array([3.0, 0.0, 100.0]) * offset
        rows[f"{species}_v_mse_y_km_s"] = np.array([4.0, 0.0, 100.0]) * offset
        rows[f"{species}_v_mse_z_km_s"] = 0.0
        rows[f"{species}_processing_status_code"] = 1
        rows[f"{species}_density_valid_flag"] = 1
        rows[f"{species}_velocity_valid_flag"] = 1
        rows[f"{species}_quality_flag_bitmask"] = [0, 1, 2]
        rows[f"{species}_quality_flag_available_flag"] = 1
        rows[f"{species}_quality_flag_epoch_matched_flag"] = 1
    return rows


def _write_input(root: Path, *, schema_name: str) -> None:
    (root / "cells").mkdir(parents=True)
    (root / "grid_schema.json").write_text(
        '{"schema_name":"' + schema_name + '","grid":{"min_rm":-1.0,"max_rm":1.0,'
        '"step_rm":1.0,"mars_radius_km":3397.0,"shape_xyz":[2,2,2]}}',
        encoding="utf-8",
    )
    with h5py.File(root / "cells" / "cell.h5", "w") as handle:
        handle.attrs.update(ix=0, iy=1, iz=1)
        handle.create_group("days").create_group("20240101").create_dataset(
            "records", data=_records()
        )


def test_all_energy_aggregation_includes_hplus_and_preserves_quality_policy():
    statistics = aggregate_record_blocks(
        [_records()], quality_policy="valid", species_names=SPECIES
    )

    assert tuple(statistics) == SPECIES
    assert statistics["hplus"][0] == 2
    assert statistics["hplus"][1] == 4.0
    assert statistics["hplus"][6] == 5.0
    assert statistics["o2plus"][1] == 12.0
    assert statistics["o2plus"][6] == 15.0


def test_all_energy_run_writes_three_species_with_distinct_namespace(tmp_path):
    input_root = tmp_path / "input"
    _write_input(input_root, schema_name="tw1_minpa_all_energy_mse_grid_records")

    summary = run_tw1_all_energy_mse_xz_statistics(
        input_root, tmp_path / "output", workers=1, write_plots=False
    )

    assert summary["species_names"] == list(SPECIES)
    assert "no >1 keV threshold" in summary["energy_selection"]
    product_path = (
        tmp_path
        / "output"
        / "data"
        / "tw1_minpa_all_energy_mse_xz_all_y_means_valid.npz"
    )
    product = np.load(product_path)
    np.testing.assert_allclose(product["hplus_mean_density_cm3"][0, 1], 2.0)
    np.testing.assert_allclose(product["oplus_mean_speed_km_s"][0, 1], 5.0)
    np.testing.assert_allclose(product["o2plus_mean_vx_km_s"][0, 1], 4.5)


def test_all_energy_run_rejects_wrong_archive_schema(tmp_path):
    input_root = tmp_path / "input"
    _write_input(input_root, schema_name="wrong_product")

    with pytest.raises(ValueError, match="Expected grid schema"):
        run_tw1_all_energy_mse_xz_statistics(
            input_root, tmp_path / "output", workers=1, write_plots=False
        )
