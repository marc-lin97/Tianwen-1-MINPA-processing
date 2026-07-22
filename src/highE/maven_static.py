"""MAVEN STATIC-D1 high-energy O+ and O2+ daily moments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cdflib
import numpy as np
from scipy.io import loadmat, savemat
try:
    import h5py
except ImportError:  # pragma: no cover - exercised only in missing optional dependency environments
    h5py = None

from .constants import (
    D1_BAD_QUALITY_BITS,
    DEFAULT_MAVEN_D1_ROOT,
    DEFAULT_MAVEN_MAG_ROOT,
    DEFAULT_MAVEN_R_ROOT,
    DEFAULT_OUTPUT_ROOT,
    HIGH_ENERGY_MIN_EV,
    MATCH_MAX_GAP_S,
    SPECIES,
    STATIC_FOV_THETA_MAX_DEG,
    STATIC_FOV_THETA_MIN_DEG,
)
from .coordinates import (
    apply_rotation,
    apply_rotation_to_axes,
    bad_quality_mask,
    datetime64_to_unix_s,
    density_from_eflux_cm3,
    energy_to_speed_km_s,
    finite_data,
    finite_difference_velocity_km_s,
    interpolate_matrices_nearest,
    interpolate_vectors,
    mse_plus_z_static_fov_flag,
    quaternion_inverse_rotate_scalar_first,
    quaternion_rotate_scalar_first,
    static_look_direction,
    unix_s_to_utc_text,
)


@dataclass(frozen=True)
class MavEnPaths:
    d1_root: Path = DEFAULT_MAVEN_D1_ROOT
    mag_root: Path = DEFAULT_MAVEN_MAG_ROOT
    r_root: Path = DEFAULT_MAVEN_R_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT


class SkipDay(RuntimeError):
    """Raised for known data-format gaps that should be skipped in batch mode."""


@dataclass(frozen=True)
class SpeciesGeometry:
    species_id: str
    species_code: int
    species_label: str
    mass_window_amu: tuple[float, float]
    swp_ind: int
    indices: tuple[np.ndarray, np.ndarray, np.ndarray]
    mass_amu: np.ndarray
    energy_eV: np.ndarray
    denergy_eV: np.ndarray
    domega_sr: np.ndarray
    direction_static: np.ndarray
    speed_km_s: np.ndarray


def find_maven_d1_file(date: str, root: Path = DEFAULT_MAVEN_D1_ROOT) -> Path:
    matches = sorted(Path(root).rglob(f"mvn_sta_l2_d1-32e4d16a8m_{date}_v*_r*.cdf"))
    if not matches:
        raise FileNotFoundError(f"No MAVEN STATIC-D1 file for {date} under {root}")
    return matches[-1]


def maven_mag_file(date: str, root: Path = DEFAULT_MAVEN_MAG_ROOT) -> Path:
    path = Path(root) / f"Bss{date}.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def maven_rotation_file(date: str, root: Path = DEFAULT_MAVEN_R_ROOT) -> Path:
    path = Path(root) / f"R_MSO2MSE_MAVEN_{date}.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_maven_mag_position(path: Path) -> dict[str, np.ndarray]:
    data, mat_format = _load_mat_arrays(path, ("Bss", "Spss"))
    bss = np.asarray(data["Bss"], dtype=float)
    spss = np.asarray(data["Spss"], dtype=float)
    time_s = bss[:, 0].astype(float)
    if spss.ndim != 2 or spss.shape[0] != bss.shape[0]:
        raise SkipDay(f"Unsupported Spss shape in {path}: Spss={spss.shape}, Bss={bss.shape}")
    if spss.shape[1] >= 4:
        position_km = spss[:, 1:4].astype(float)
        position_columns = "Spss[:,1:4]"
        position_time_note = "Spss has 4 columns; column 0 is ignored and Bss[:,0] is the authoritative time axis."
    elif spss.shape[1] == 3:
        position_km = spss[:, 0:3].astype(float)
        position_columns = "Spss[:,0:3]"
        position_time_note = "Spss has 3 columns and no timestamp; Bss[:,0] is the authoritative time axis."
    else:
        raise SkipDay(f"Unsupported Spss column count in {path}: {spss.shape[1]}")
    return {
        "time_s": time_s,
        "b_mso_nT": bss[:, 1:4].astype(float),
        "position_mso_km": position_km,
        "spacecraft_velocity_mso_km_s": finite_difference_velocity_km_s(time_s, position_km),
        "position_columns": np.asarray(position_columns),
        "position_time_note": np.asarray(position_time_note),
        "mat_format": np.asarray(mat_format),
    }


def load_maven_mso2mse(path: Path) -> dict[str, np.ndarray]:
    data, _ = _load_mat_arrays(path, ("t_R_MSO2MSE", "R_MSO2MSE"))
    return {
        "time_s": np.asarray(data["t_R_MSO2MSE"], dtype=float).reshape(-1),
        "R_mso_to_mse": np.asarray(data["R_MSO2MSE"], dtype=float),
    }


def _load_mat_arrays(path: Path, names: tuple[str, ...]) -> tuple[dict[str, np.ndarray], str]:
    try:
        data = loadmat(path, variable_names=list(names))
        return {name: np.asarray(data[name]) for name in names}, "matlab_pre_v7.3"
    except NotImplementedError:
        if h5py is None:
            raise SkipDay(f"MATLAB v7.3 file requires h5py but h5py is unavailable: {path}")
        with h5py.File(path, "r") as handle:
            out = {}
            for name in names:
                if name not in handle:
                    raise KeyError(f"Variable {name!r} not found in {path}")
                arr = np.asarray(handle[name])
                out[name] = _normalize_hdf_mat_array(arr)
        return out, "matlab_v7.3_hdf5"


def _normalize_hdf_mat_array(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 2 and data.shape[0] in (3, 4) and data.shape[1] > data.shape[0]:
        return data.T
    if data.ndim == 3 and data.shape[0] == 3 and data.shape[1] == 3:
        return np.moveaxis(data, -1, 0)
    return data


def read_d1_support(path: Path) -> dict[str, np.ndarray]:
    cdf = cdflib.CDF(str(path))
    epochs = np.asarray(cdflib.cdfepoch.to_datetime(cdf.varget("epoch")), dtype="datetime64[ns]")
    return {
        "epochs": epochs,
        "epoch_unix_s": datetime64_to_unix_s(epochs),
        "energy": finite_data(cdf.varget("energy")),
        "denergy": finite_data(cdf.varget("denergy")),
        "theta": finite_data(cdf.varget("theta")),
        "dtheta": finite_data(cdf.varget("dtheta")),
        "phi": finite_data(cdf.varget("phi")),
        "dphi": finite_data(cdf.varget("dphi")),
        "domega": finite_data(cdf.varget("domega")),
        "mass_arr": finite_data(cdf.varget("mass_arr")),
        "eflux": finite_data(cdf.varget("eflux")),
        "quality_flag": finite_data(cdf.varget("quality_flag")),
        "swp_ind": np.asarray(cdf.varget("swp_ind"), dtype=int),
        "sc_pot": finite_data(cdf.varget("sc_pot")),
        "quat_mso": finite_data(cdf.varget("quat_mso")),
    }


def build_species_geometries(support: dict[str, np.ndarray], high_energy_min_eV: float) -> dict[tuple[int, str], SpeciesGeometry]:
    out: dict[tuple[int, str], SpeciesGeometry] = {}
    for swp_ind in range(support["energy"].shape[3]):
        mass_grid = support["mass_arr"][:, :, :, swp_ind]
        energy_grid = support["energy"][:, :, :, swp_ind]
        denergy_grid = support["denergy"][:, :, :, swp_ind]
        domega_grid = support["domega"][:, :, :, swp_ind]
        theta_grid = support["theta"][:, :, :, swp_ind]
        phi_grid = support["phi"][:, :, :, swp_ind]
        common = (
            np.isfinite(mass_grid)
            & np.isfinite(energy_grid)
            & np.isfinite(denergy_grid)
            & np.isfinite(domega_grid)
            & np.isfinite(theta_grid)
            & np.isfinite(phi_grid)
            & (energy_grid > high_energy_min_eV)
            & (denergy_grid > 0.0)
            & (domega_grid > 0.0)
        )
        for species_id, spec in SPECIES.items():
            lo, hi = spec["mass_window_amu"]
            valid = common & (mass_grid >= lo) & (mass_grid <= hi)
            indices = np.where(valid)
            mass = mass_grid[indices]
            energy = energy_grid[indices]
            speed = energy_to_speed_km_s(energy, mass)
            out[(swp_ind, species_id)] = SpeciesGeometry(
                species_id=species_id,
                species_code=int(spec["code"]),
                species_label=str(spec["label"]),
                mass_window_amu=(float(lo), float(hi)),
                swp_ind=swp_ind,
                indices=indices,
                mass_amu=mass,
                energy_eV=energy,
                denergy_eV=denergy_grid[indices],
                domega_sr=domega_grid[indices],
                direction_static=static_look_direction(theta_grid[indices], phi_grid[indices]),
                speed_km_s=speed,
            )
    return out


def process_maven_day(
    date: str,
    paths: MavEnPaths | None = None,
    high_energy_min_eV: float = HIGH_ENERGY_MIN_EV,
    add_spacecraft_velocity: bool = True,
    write_output: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    paths = paths or MavEnPaths()
    d1_path = find_maven_d1_file(date, paths.d1_root)
    mag_path = maven_mag_file(date, paths.mag_root)
    r_path = maven_rotation_file(date, paths.r_root)

    support = read_d1_support(d1_path)
    geometries = build_species_geometries(support, high_energy_min_eV)
    mag = load_maven_mag_position(mag_path)
    rot = load_maven_mso2mse(r_path)

    times = support["epoch_unix_s"]
    pos_mso = interpolate_vectors(mag["time_s"], mag["position_mso_km"], times)
    v_sc_mso = interpolate_vectors(mag["time_s"], mag["spacecraft_velocity_mso_km_s"], times)
    r_mso_to_mse, r_dt_s = interpolate_matrices_nearest(rot["time_s"], rot["R_mso_to_mse"], times)
    pos_mse = apply_rotation(r_mso_to_mse, pos_mso)

    quality_bad = bad_quality_mask(support["quality_flag"], D1_BAD_QUALITY_BITS)
    n_records = times.size
    n_species = len(SPECIES)
    n_rows = n_records * n_species

    result = _empty_result(n_rows)
    row = 0
    identity_axes = np.eye(3)

    for record_index in range(n_records):
        swp_ind = int(support["swp_ind"][record_index])
        quat = support["quat_mso"][record_index]
        axes_mso = quaternion_rotate_scalar_first(quat, identity_axes)
        axes_mse = apply_rotation_to_axes(r_mso_to_mse[record_index : record_index + 1], axes_mso[None, :, :])[0]
        plus_z_flag, plus_z_theta_deg, plus_z_static = mse_plus_z_static_fov_flag(
            axes_mse[None, :, :],
            STATIC_FOV_THETA_MIN_DEG,
            STATIC_FOV_THETA_MAX_DEG,
        )
        for species_id, spec in SPECIES.items():
            geom = geometries.get((swp_ind, species_id))
            _fill_common_row(
                result,
                row,
                record_index,
                times[record_index],
                species_id,
                spec,
                swp_ind,
                support["quality_flag"][record_index],
                support["sc_pot"][record_index],
                pos_mso[record_index],
                pos_mse[record_index],
                r_dt_s[record_index],
                axes_mso,
                axes_mse,
                float(plus_z_flag[0]),
                float(plus_z_theta_deg[0]),
                plus_z_static[0],
                STATIC_FOV_THETA_MIN_DEG,
                STATIC_FOV_THETA_MAX_DEG,
            )
            if geom is None or geom.energy_eV.size == 0:
                result["processing_status_code"][row] = 2
                row += 1
                continue
            if quality_bad[record_index]:
                result["processing_status_code"][row] = 3
                row += 1
                continue
            _fill_moment_row(
                result,
                row,
                support["eflux"][record_index],
                geom,
                quat,
                v_sc_mso[record_index],
                r_mso_to_mse[record_index],
                add_spacecraft_velocity=add_spacecraft_velocity,
            )
            row += 1

    summary = _summary(
        date,
        d1_path,
        mag_path,
        r_path,
        support,
        result,
        high_energy_min_eV,
        add_spacecraft_velocity,
        str(mag["position_columns"].item()),
        str(mag["position_time_note"].item()),
        str(mag["mat_format"].item()),
    )
    summary["runtime"] = {
        "processing_seconds_before_write": time.perf_counter() - started,
    }
    if write_output:
        write_maven_output(date, paths.output_root, result, summary)
    return result, summary


def write_maven_output(date: str, output_root: Path, result: dict[str, np.ndarray], summary: dict[str, Any]) -> tuple[dict[str, Path], Path]:
    out_dir = Path(output_root) / "MAVEN" / date[:4]
    out_dir.mkdir(parents=True, exist_ok=True)
    species_paths: dict[str, Path] = {}
    species_outputs: dict[str, dict[str, Any]] = {}
    for species_id, spec in SPECIES.items():
        mask = result["species_code"] == int(spec["code"])
        species_result = {key: np.asarray(value)[mask] for key, value in result.items() if key != "species_order"}
        species_result, duplicate_stats = _drop_duplicate_times(species_result)
        times = species_result["epoch_unix_s"]
        unique_time_count = int(np.unique(times[np.isfinite(times)]).size)
        if unique_time_count != int(times.size):
            raise ValueError(f"{species_id} output would contain duplicate timestamps for {date}")
        mat_payload = {key: np.asarray(value)[:, None] for key, value in species_result.items()}
        mat_payload["utc"] = unix_s_to_utc_text(species_result["epoch_unix_s"])[:, None]
        mat_payload["species_id"] = np.asarray([species_id] * int(times.size), dtype=object)[:, None]
        mat_payload["species_label"] = np.asarray([spec["label"]] * int(times.size), dtype=object)[:, None]
        mat_path = out_dir / f"maven_static_highE_{species_id}_{date}.mat"
        savemat(mat_path, mat_payload)
        species_paths[species_id] = mat_path
        species_outputs[species_id] = {
            "file": str(mat_path),
            "rows": int(times.size),
            "unique_time_count": unique_time_count,
            "pass_rows": int(np.count_nonzero(species_result["processing_status_code"] == 1)),
            **duplicate_stats,
        }
    json_path = out_dir / f"maven_static_highE_{date}_summary.json"
    summary = {**summary, "species_outputs": species_outputs}
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return species_paths, json_path


def _drop_duplicate_times(species_result: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    times = np.asarray(species_result["epoch_unix_s"], dtype=float)
    _, first_indices, counts = np.unique(times, return_index=True, return_counts=True)
    keep = np.sort(first_indices)
    duplicate_groups = int(np.count_nonzero(counts > 1))
    duplicate_extra_records = int(np.sum(counts[counts > 1] - 1))
    if duplicate_groups == 0:
        return species_result, {"duplicate_time_groups_dropped": 0, "duplicate_extra_records_dropped": 0}
    dropped = {key: np.asarray(value)[keep] for key, value in species_result.items()}
    return dropped, {
        "duplicate_time_groups_dropped": duplicate_groups,
        "duplicate_extra_records_dropped": duplicate_extra_records,
    }


def _empty_result(n_rows: int) -> dict[str, np.ndarray]:
    numeric = {
        "epoch_unix_s": np.full(n_rows, np.nan),
        "record_index": np.full(n_rows, -1, dtype=np.int32),
        "species_code": np.full(n_rows, -1, dtype=np.int16),
        "species_order": np.full(n_rows, -1, dtype=np.int16),
        "swp_ind": np.full(n_rows, -1, dtype=np.int16),
        "quality_flag": np.full(n_rows, np.nan),
        "sc_pot_V": np.full(n_rows, np.nan),
        "mass_window_amu_min": np.full(n_rows, np.nan),
        "mass_window_amu_max": np.full(n_rows, np.nan),
        "density_cm3": np.full(n_rows, np.nan),
        "valid_cell_count": np.zeros(n_rows, dtype=np.int32),
        "r_matrix_match_dt_s": np.full(n_rows, np.nan),
        "processing_status_code": np.zeros(n_rows, dtype=np.int16),
        "mse_plus_z_in_static_fov_flag": np.full(n_rows, np.nan),
        "mse_plus_z_static_theta_deg": np.full(n_rows, np.nan),
        "static_fov_theta_min_deg": np.full(n_rows, np.nan),
        "static_fov_theta_max_deg": np.full(n_rows, np.nan),
    }
    for axis in "xyz":
        numeric[f"mse_plus_z_static_{axis}"] = np.full(n_rows, np.nan)
    for prefix in ["v_mso", "v_mse", "pos_mso", "pos_mse"]:
        for axis in "xyz":
            unit = "km_s" if prefix.startswith("v_") else "km"
            numeric[f"{prefix}_{axis}_{unit}"] = np.full(n_rows, np.nan)
    for frame in ["mso", "mse"]:
        for inst_axis in "xyz":
            for comp in "xyz":
                numeric[f"inst_{inst_axis}_axis_{frame}_{comp}"] = np.full(n_rows, np.nan)
    return numeric


def _fill_common_row(
    result: dict[str, np.ndarray],
    row: int,
    record_index: int,
    epoch_unix_s: float,
    species_id: str,
    spec: dict[str, Any],
    swp_ind: int,
    quality_flag: float,
    sc_pot: float,
    pos_mso: np.ndarray,
    pos_mse: np.ndarray,
    r_dt_s: float,
    axes_mso: np.ndarray,
    axes_mse: np.ndarray,
    mse_plus_z_in_static_fov_flag: float,
    mse_plus_z_static_theta_deg: float,
    mse_plus_z_static: np.ndarray,
    static_fov_theta_min_deg: float,
    static_fov_theta_max_deg: float,
) -> None:
    result["epoch_unix_s"][row] = epoch_unix_s
    result["record_index"][row] = record_index
    result["species_code"][row] = int(spec["code"])
    result["species_order"][row] = int(spec["code"])
    result["swp_ind"][row] = swp_ind
    result["quality_flag"][row] = quality_flag
    result["sc_pot_V"][row] = sc_pot
    result["mass_window_amu_min"][row] = float(spec["mass_window_amu"][0])
    result["mass_window_amu_max"][row] = float(spec["mass_window_amu"][1])
    result["r_matrix_match_dt_s"][row] = r_dt_s
    result["mse_plus_z_in_static_fov_flag"][row] = mse_plus_z_in_static_fov_flag
    result["mse_plus_z_static_theta_deg"][row] = mse_plus_z_static_theta_deg
    result["static_fov_theta_min_deg"][row] = static_fov_theta_min_deg
    result["static_fov_theta_max_deg"][row] = static_fov_theta_max_deg
    for i, axis in enumerate("xyz"):
        result[f"mse_plus_z_static_{axis}"][row] = mse_plus_z_static[i]
    for i, axis in enumerate("xyz"):
        result[f"pos_mso_{axis}_km"][row] = pos_mso[i]
        result[f"pos_mse_{axis}_km"][row] = pos_mse[i]
    for axis_index, axis_name in enumerate("xyz"):
        for comp_index, comp_name in enumerate("xyz"):
            result[f"inst_{axis_name}_axis_mso_{comp_name}"][row] = axes_mso[axis_index, comp_index]
            result[f"inst_{axis_name}_axis_mse_{comp_name}"][row] = axes_mse[axis_index, comp_index]


def _fill_moment_row(
    result: dict[str, np.ndarray],
    row: int,
    eflux_record: np.ndarray,
    geom: SpeciesGeometry,
    quat: np.ndarray,
    spacecraft_velocity_mso_km_s: np.ndarray,
    r_mso_to_mse: np.ndarray,
    *,
    add_spacecraft_velocity: bool,
) -> None:
    flux = eflux_record[geom.indices]
    density = density_from_eflux_cm3(flux, geom.energy_eV, geom.denergy_eV, geom.domega_sr, geom.speed_km_s)
    valid = np.isfinite(density) & (density > 0.0)
    if not np.any(valid):
        result["processing_status_code"][row] = 4
        return
    velocity_static = geom.direction_static[valid] * geom.speed_km_s[valid, None]
    if add_spacecraft_velocity and np.all(np.isfinite(spacecraft_velocity_mso_km_s)):
        velocity_static = velocity_static + quaternion_inverse_rotate_scalar_first(quat, spacecraft_velocity_mso_km_s)[None, :]
    velocity_mso = quaternion_rotate_scalar_first(quat, velocity_static)
    weights = density[valid]
    density_sum = float(np.nansum(weights))
    if not np.isfinite(density_sum) or density_sum <= 0.0:
        result["processing_status_code"][row] = 4
        return
    v_mso = np.nansum(velocity_mso * weights[:, None], axis=0) / density_sum
    v_mse = r_mso_to_mse @ v_mso
    result["density_cm3"][row] = density_sum
    result["valid_cell_count"][row] = int(np.count_nonzero(valid))
    for i, axis in enumerate("xyz"):
        result[f"v_mso_{axis}_km_s"][row] = v_mso[i]
        result[f"v_mse_{axis}_km_s"][row] = v_mse[i]
    result["processing_status_code"][row] = 1


def _summary(
    date: str,
    d1_path: Path,
    mag_path: Path,
    r_path: Path,
    support: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    high_energy_min_eV: float,
    add_spacecraft_velocity: bool,
    position_columns: str,
    position_time_note: str,
    mag_mat_format: str,
) -> dict[str, Any]:
    status, counts = np.unique(result["processing_status_code"], return_counts=True)
    finite_density = result["density_cm3"][np.isfinite(result["density_cm3"])]
    finite_speed_mso = np.linalg.norm(
        np.column_stack([result["v_mso_x_km_s"], result["v_mso_y_km_s"], result["v_mso_z_km_s"]]),
        axis=1,
    )
    finite_speed_mse = np.linalg.norm(
        np.column_stack([result["v_mse_x_km_s"], result["v_mse_y_km_s"], result["v_mse_z_km_s"]]),
        axis=1,
    )
    speed_diff = np.abs(finite_speed_mso - finite_speed_mse)
    speed_diff = speed_diff[np.isfinite(speed_diff)]
    return {
        "date": date,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "static_d1": str(d1_path),
            "mag_spss": str(mag_path),
            "r_mso2mse": str(r_path),
        },
        "assumptions": {
            "high_energy_min_eV": high_energy_min_eV,
            "energy_for_threshold_and_moment": "raw STATIC energy, no sc_pot correction",
            "spacecraft_velocity_added": add_spacecraft_velocity,
            "position_source": f"{position_columns} from mag/ss1s BssYYYYMMDD.mat",
            "position_time_axis": position_time_note,
            "mag_mat_format": mag_mat_format,
            "rotation_source": "nearest R_MSO2MSE sample by Unix second",
            "fov_output": "instrument coordinate axes plus MSE +Z in/out STATIC FOV flag",
            "mse_plus_z_fov_flag": "STATIC FOV is fixed in STATIC coordinates; flag=1 when transformed MSE +Z static theta falls inside fixed theta edges, 0 outside, NaN invalid or not determinable",
        },
        "records": {
            "static_records": int(support["epoch_unix_s"].size),
            "internal_rows_before_species_split": int(result["epoch_unix_s"].size),
            "per_species_rows": int(support["epoch_unix_s"].size),
            "status_counts": {str(int(k)): int(v) for k, v in zip(status, counts)},
            "status_code_meaning": {
                "0": "not_processed",
                "1": "pass",
                "2": "no_geometry_cells_for_species",
                "3": "quality_flag_bad_bits_6_or_7",
                "4": "no_positive_density_cells",
            },
        },
        "sanity_checks": {
            "density_cm3_median": float(np.nanmedian(finite_density)) if finite_density.size else None,
            "density_cm3_max": float(np.nanmax(finite_density)) if finite_density.size else None,
            "max_abs_speed_mso_mse_difference_km_s": float(np.nanmax(speed_diff)) if speed_diff.size else None,
            "max_abs_r_matrix_match_dt_s": float(np.nanmax(np.abs(result["r_matrix_match_dt_s"]))),
            "mse_plus_z_in_static_fov_fraction": float(np.nanmean(result["mse_plus_z_in_static_fov_flag"]))
            if np.any(np.isfinite(result["mse_plus_z_in_static_fov_flag"]))
            else None,
        },
    }


def result_with_utc(result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(result)
    out["utc"] = unix_s_to_utc_text(result["epoch_unix_s"])
    return out
