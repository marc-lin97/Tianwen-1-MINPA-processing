"""MSE spatial grid statistics for high-energy Mars ion products."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat, savemat

GRID_MARS_RADIUS_KM = 3397.0
DEFAULT_GRID_STEP_RM = 0.1
DEFAULT_DENSITY_MIN_CM3 = 0.001

SPECIES_IDS = ("Oplus", "O2plus")
FOV_GROUPS = {"plus_z_in_fov": 1.0, "plus_z_out_fov": 0.0}


@dataclass(frozen=True)
class MissionGridConfig:
    mission_id: str
    mission_label: str
    input_subdir: str
    file_template: str
    fov_flag_field: str
    max_abs_rm: float


MISSION_CONFIGS: dict[str, MissionGridConfig] = {
    "MAVEN": MissionGridConfig(
        mission_id="MAVEN",
        mission_label="MAVEN STATIC-D1",
        input_subdir="MAVEN",
        file_template="maven_static_highE_{species}_*.mat",
        fov_flag_field="mse_plus_z_in_static_fov_flag",
        max_abs_rm=3.0,
    ),
    "Tianwen-1": MissionGridConfig(
        mission_id="Tianwen-1",
        mission_label="Tianwen-1 MINPA",
        input_subdir="Tianwen-1",
        file_template="tw1_minpa_highE_{species}_*.mat",
        fov_flag_field="mse_plus_z_in_minpa_fov_flag",
        max_abs_rm=7.0,
    ),
}

GRID_VARIABLES = (
    "density_cm3",
    "v_mse_x_km_s",
    "v_mse_y_km_s",
    "v_mse_z_km_s",
    "speed_mse_km_s",
)


def grid_centers(max_abs_rm: float, step_rm: float = DEFAULT_GRID_STEP_RM) -> np.ndarray:
    """Return bin centers spanning -max_abs_rm to +max_abs_rm, including zero."""

    count_each_side = int(round(max_abs_rm / step_rm))
    return np.arange(-count_each_side, count_each_side + 1, dtype=float) * step_rm


def _flatten_mat_field(data: dict[str, Any], name: str, path: Path) -> np.ndarray:
    if name not in data:
        raise KeyError(f"{name!r} not found in {path}")
    return np.asarray(data[name], dtype=float).reshape(-1)


def _load_fov_flag(data: dict[str, Any], name: str, path: Path) -> tuple[np.ndarray, str]:
    if name in data:
        return np.asarray(data[name], dtype=float).reshape(-1), "species_product"
    if name != "mse_plus_z_in_static_fov_flag":
        raise KeyError(f"{name!r} not found in {path}")

    match = re.search(r"_(\d{8})\.mat$", path.name)
    if not match:
        raise KeyError(f"{name!r} not found in {path} and date could not be inferred for standalone flag fallback")
    flag_path = path.with_name(f"maven_static_plus_z_fov_flag_{match.group(1)}.mat")
    if not flag_path.exists():
        raise KeyError(f"{name!r} not found in {path}; fallback flag file is missing: {flag_path}")

    flag_data = loadmat(flag_path, variable_names=("epoch_unix_s", name), squeeze_me=True, struct_as_record=False)
    product_epoch = np.asarray(data["epoch_unix_s"], dtype=float).reshape(-1)
    flag_epoch = np.asarray(flag_data["epoch_unix_s"], dtype=float).reshape(-1)
    if product_epoch.shape != flag_epoch.shape or not np.allclose(product_epoch, flag_epoch, rtol=0.0, atol=0.0):
        raise ValueError(f"Epoch mismatch between {path} and fallback flag file {flag_path}")
    return np.asarray(flag_data[name], dtype=float).reshape(-1), str(flag_path)


def _grid_indices(position_rm: np.ndarray, centers: np.ndarray, step_rm: float) -> tuple[np.ndarray, np.ndarray]:
    max_abs = float(np.nanmax(np.abs(centers)))
    rel = np.rint((position_rm + max_abs) / step_rm).astype(np.int64)
    shape = np.array([centers.size, centers.size, centers.size], dtype=np.int64)
    inside_index = np.all((rel >= 0) & (rel < shape), axis=1)
    nearest = np.full_like(position_rm, np.nan, dtype=float)
    nearest[inside_index] = centers[rel[inside_index]]
    inside_cell = np.all(np.abs(position_rm - nearest) <= (0.5 * step_rm + 1.0e-9), axis=1)
    inside = inside_index & inside_cell
    flat = np.ravel_multi_index(rel[inside].T, tuple(shape))
    return flat.astype(np.int64), inside


def _date_from_product_path(path: Path) -> str:
    match = re.search(r"_(\d{8})\.mat$", path.name)
    if not match:
        raise ValueError(f"Cannot infer YYYYMMDD date from {path}")
    return match.group(1)


def _format_rm_for_path(value: float) -> str:
    return f"{value:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")


def grid_record_folder(flat_bin: int, shape: tuple[int, int, int], centers: np.ndarray) -> str:
    ix, iy, iz = np.unravel_index(int(flat_bin), shape)
    return (
        f"ix{int(ix):03d}_iy{int(iy):03d}_iz{int(iz):03d}"
        f"__x{_format_rm_for_path(float(centers[ix]))}"
        f"_y{_format_rm_for_path(float(centers[iy]))}"
        f"_z{_format_rm_for_path(float(centers[iz]))}Rm"
    )


def find_species_files(
    input_root: Path,
    config: MissionGridConfig,
    species: str,
    limit: int | None = None,
    years: Iterable[str] | None = None,
    dates: Iterable[str] | None = None,
) -> list[Path]:
    mission_root = Path(input_root) / config.input_subdir
    if not mission_root.exists():
        return []
    prefix, suffix = config.file_template.format(species=species).split("*", 1)
    if dates is not None:
        out = []
        for date in sorted(set(dates)):
            path = mission_root / date[:4] / f"{prefix}{date}{suffix}"
            if path.exists():
                out.append(path)
                if limit is not None and len(out) >= limit:
                    return out
        return out

    out: list[Path] = []
    if years is not None:
        year_dirs = [str(mission_root / year) for year in sorted(set(years))]
    else:
        with os.scandir(mission_root) as year_entries:
            year_dirs = sorted(entry.path for entry in year_entries if entry.is_dir() and Path(entry.path).name.isdigit())
    for year_dir in year_dirs:
        if not Path(year_dir).exists():
            continue
        with os.scandir(year_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                name = entry.name
                if name.startswith(prefix) and name.endswith(suffix):
                    out.append(Path(entry.path))
                    if limit is not None and len(out) >= limit:
                        return sorted(out)
    return sorted(out)


def read_spatial_chunk(
    path: str,
    fov_flag_field: str,
    centers: np.ndarray,
    step_rm: float,
    density_min_cm3: float,
) -> dict[str, Any]:
    mat_path = Path(path)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    epoch_unix_s = _flatten_mat_field(data, "epoch_unix_s", mat_path)
    density = _flatten_mat_field(data, "density_cm3", mat_path)
    status = _flatten_mat_field(data, "processing_status_code", mat_path)
    fov_flag, fov_flag_source = _load_fov_flag(data, fov_flag_field, mat_path)
    pos_km = np.column_stack([_flatten_mat_field(data, f"pos_mse_{axis}_km", mat_path) for axis in "xyz"])
    vel = np.column_stack([_flatten_mat_field(data, f"v_mse_{axis}_km_s", mat_path) for axis in "xyz"])

    n_rows = int(density.size)
    if not (epoch_unix_s.size == status.size == fov_flag.size == pos_km.shape[0] == vel.shape[0] == n_rows):
        raise ValueError(f"Inconsistent row counts in {mat_path}")

    pos_rm = pos_km / GRID_MARS_RADIUS_KM
    finite_pos = np.all(np.isfinite(pos_rm), axis=1)
    finite_density = np.isfinite(density)
    density_ok = finite_density & (density >= density_min_cm3)
    pass_status = status == 1
    finite_flag = np.isfinite(fov_flag)
    base = pass_status & finite_flag & finite_pos & density_ok

    flat_bins_all = np.full(n_rows, -1, dtype=np.int64)
    if np.any(base):
        bins, inside = _grid_indices(pos_rm[base], centers, step_rm)
        base_indices = np.flatnonzero(base)
        flat_bins_all[base_indices[inside]] = bins
    in_grid = flat_bins_all >= 0

    finite_vel = np.all(np.isfinite(vel), axis=1)
    speed = np.linalg.norm(vel, axis=1)
    finite_speed = np.isfinite(speed)

    groups: dict[str, dict[str, np.ndarray]] = {}
    for label, flag_value in FOV_GROUPS.items():
        density_mask = in_grid & (fov_flag == flag_value)
        velocity_mask = density_mask & finite_vel & finite_speed
        record_speed = np.where(finite_vel & finite_speed, speed, np.nan)
        groups[label] = {
            "density_bins": flat_bins_all[density_mask],
            "density_cm3": density[density_mask],
            "epoch_unix_s": epoch_unix_s[density_mask],
            "pos_mse_x_rm": pos_rm[density_mask, 0],
            "pos_mse_y_rm": pos_rm[density_mask, 1],
            "pos_mse_z_rm": pos_rm[density_mask, 2],
            "pos_mse_x_km": pos_km[density_mask, 0],
            "pos_mse_y_km": pos_km[density_mask, 1],
            "pos_mse_z_km": pos_km[density_mask, 2],
            "v_mse_x_km_s_record": vel[density_mask, 0],
            "v_mse_y_km_s_record": vel[density_mask, 1],
            "v_mse_z_km_s_record": vel[density_mask, 2],
            "speed_mse_km_s_record": record_speed[density_mask],
            "velocity_valid": (finite_vel & finite_speed)[density_mask].astype(np.int8),
            "velocity_bins": flat_bins_all[velocity_mask],
            "v_mse_x_km_s": vel[velocity_mask, 0],
            "v_mse_y_km_s": vel[velocity_mask, 1],
            "v_mse_z_km_s": vel[velocity_mask, 2],
            "speed_mse_km_s": speed[velocity_mask],
        }

    return {
        "path": str(mat_path),
        "date": _date_from_product_path(mat_path),
        "fov_flag_source": fov_flag_source,
        "rows_total": n_rows,
        "rows_processing_pass": int(np.count_nonzero(pass_status)),
        "rows_fov_nan": int(np.count_nonzero(~finite_flag)),
        "rows_density_below_threshold": int(np.count_nonzero(pass_status & finite_density & (density < density_min_cm3))),
        "rows_density_valid": int(np.count_nonzero(base)),
        "rows_density_in_grid": int(np.count_nonzero(in_grid)),
        "rows_velocity_in_grid": int(np.count_nonzero(in_grid & finite_vel & finite_speed)),
        "groups": groups,
    }


def write_grid_record_files(
    chunk: dict[str, Any],
    centers: np.ndarray,
    shape: tuple[int, int, int],
    output_root: Path,
    mission: str,
    species: str,
    *,
    step_rm: float,
    density_min_cm3: float,
) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    date = str(chunk["date"])
    source_path = str(chunk["path"])
    for fov_group, group in chunk["groups"].items():
        bins = np.asarray(group["density_bins"], dtype=np.int64)
        if bins.size == 0:
            continue
        for flat_bin in np.unique(bins):
            mask = bins == flat_bin
            folder = (
                output_root
                / "grid_records"
                / mission
                / species
                / fov_group
                / grid_record_folder(int(flat_bin), shape, centers)
            )
            folder.mkdir(parents=True, exist_ok=True)
            ix, iy, iz = np.unravel_index(int(flat_bin), shape)
            path = folder / f"{date}.mat"
            payload = {
                "epoch_unix_s": np.asarray(group["epoch_unix_s"])[mask],
                "density_cm3": np.asarray(group["density_cm3"])[mask],
                "pos_mse_x_rm": np.asarray(group["pos_mse_x_rm"])[mask],
                "pos_mse_y_rm": np.asarray(group["pos_mse_y_rm"])[mask],
                "pos_mse_z_rm": np.asarray(group["pos_mse_z_rm"])[mask],
                "pos_mse_x_km": np.asarray(group["pos_mse_x_km"])[mask],
                "pos_mse_y_km": np.asarray(group["pos_mse_y_km"])[mask],
                "pos_mse_z_km": np.asarray(group["pos_mse_z_km"])[mask],
                "v_mse_x_km_s": np.asarray(group["v_mse_x_km_s_record"])[mask],
                "v_mse_y_km_s": np.asarray(group["v_mse_y_km_s_record"])[mask],
                "v_mse_z_km_s": np.asarray(group["v_mse_z_km_s_record"])[mask],
                "speed_mse_km_s": np.asarray(group["speed_mse_km_s_record"])[mask],
                "velocity_valid": np.asarray(group["velocity_valid"])[mask],
                "source_file": np.asarray([source_path], dtype=object),
                "date": np.asarray([date], dtype=object),
                "mission": np.asarray([mission], dtype=object),
                "species": np.asarray([species], dtype=object),
                "fov_group": np.asarray([fov_group], dtype=object),
                "flat_bin": np.asarray(int(flat_bin), dtype=np.int64),
                "ix": np.asarray(int(ix), dtype=np.int32),
                "iy": np.asarray(int(iy), dtype=np.int32),
                "iz": np.asarray(int(iz), dtype=np.int32),
                "x_center_rm": np.asarray(float(centers[ix])),
                "y_center_rm": np.asarray(float(centers[iy])),
                "z_center_rm": np.asarray(float(centers[iz])),
                "grid_step_rm": np.asarray(float(step_rm)),
                "mars_radius_km": np.asarray(float(GRID_MARS_RADIUS_KM)),
                "density_min_cm3": np.asarray(float(density_min_cm3)),
            }
            savemat(path, payload, do_compression=True)
            written.append(
                {
                    "mission": mission,
                    "species": species,
                    "fov_group": fov_group,
                    "date": date,
                    "flat_bin": int(flat_bin),
                    "ix": int(ix),
                    "iy": int(iy),
                    "iz": int(iz),
                    "rows": int(np.count_nonzero(mask)),
                    "file": str(path),
                }
            )
    return written


def _empty_grid(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"flat_bin": np.asarray([], dtype=np.int64)}
    for axis in ("ix", "iy", "iz"):
        out[axis] = np.asarray([], dtype=np.int32)
    for axis in ("x_rm", "y_rm", "z_rm"):
        out[axis] = np.asarray([], dtype=np.float64)
    out["density_count"] = np.asarray([], dtype=np.int64)
    out["velocity_count"] = np.asarray([], dtype=np.int64)
    for var in GRID_VARIABLES:
        out[f"mean_{var}"] = np.asarray([], dtype=np.float64)
        out[f"median_{var}"] = np.asarray([], dtype=np.float64)
    return out


def _sparse_mean_median(bins: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bins.size == 0:
        empty_i = np.asarray([], dtype=np.int64)
        empty_f = np.asarray([], dtype=np.float64)
        return empty_i, empty_i, empty_f, empty_f
    order = np.argsort(bins, kind="mergesort")
    sorted_bins = bins[order]
    sorted_values = values[order]
    unique_bins, starts, counts = np.unique(sorted_bins, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_values, starts)
    mean = sums / counts
    med = np.full(unique_bins.size, np.nan, dtype=np.float64)
    for bin_id, start, count in zip(unique_bins, starts, counts):
        med[np.searchsorted(unique_bins, bin_id)] = float(np.nanmedian(sorted_values[start : start + count]))
    return unique_bins.astype(np.int64), counts.astype(np.int64), mean.astype(np.float64), med


def compute_group_grid(
    chunks: Iterable[dict[str, np.ndarray]],
    shape: tuple[int, int, int],
    centers: np.ndarray,
) -> dict[str, np.ndarray]:
    chunk_list = list(chunks)
    if not chunk_list:
        return _empty_grid(shape)

    def concat(name: str, dtype: np.dtype | type = np.float64) -> np.ndarray:
        arrays = [np.asarray(chunk[name]) for chunk in chunk_list if np.asarray(chunk[name]).size]
        if not arrays:
            return np.asarray([], dtype=dtype)
        return np.concatenate(arrays)

    density_bins = concat("density_bins", np.int64)
    density_values = concat("density_cm3")
    velocity_bins = concat("velocity_bins", np.int64)

    density_unique, density_count, mean_density, median_density = _sparse_mean_median(density_bins, density_values)
    velocity_unique, velocity_count, _, _ = _sparse_mean_median(velocity_bins, np.ones(velocity_bins.size, dtype=np.float64))
    all_bins = np.union1d(density_unique, velocity_unique).astype(np.int64)
    if all_bins.size == 0:
        return _empty_grid(shape)

    ix, iy, iz = np.unravel_index(all_bins, shape)
    out = _empty_grid(shape)
    out["flat_bin"] = all_bins
    out["ix"] = ix.astype(np.int32)
    out["iy"] = iy.astype(np.int32)
    out["iz"] = iz.astype(np.int32)
    out["x_rm"] = centers[ix]
    out["y_rm"] = centers[iy]
    out["z_rm"] = centers[iz]
    out["density_count"] = np.zeros(all_bins.size, dtype=np.int64)
    out["velocity_count"] = np.zeros(all_bins.size, dtype=np.int64)
    for var in GRID_VARIABLES:
        out[f"mean_{var}"] = np.full(all_bins.size, np.nan, dtype=np.float64)
        out[f"median_{var}"] = np.full(all_bins.size, np.nan, dtype=np.float64)

    if density_unique.size:
        idx = np.searchsorted(all_bins, density_unique)
        out["density_count"][idx] = density_count
        out["mean_density_cm3"][idx] = mean_density
        out["median_density_cm3"][idx] = median_density
    if velocity_unique.size:
        idx = np.searchsorted(all_bins, velocity_unique)
        out["velocity_count"][idx] = velocity_count
    for var in ("v_mse_x_km_s", "v_mse_y_km_s", "v_mse_z_km_s", "speed_mse_km_s"):
        values = np.concatenate([chunk[var] for chunk in chunk_list if chunk[var].size])
        unique_bins, _, mean, median = _sparse_mean_median(velocity_bins, values)
        if unique_bins.size:
            idx = np.searchsorted(all_bins, unique_bins)
            out[f"mean_{var}"][idx] = mean
            out[f"median_{var}"][idx] = median
    return out


def save_sparse_grid_npz(
    path: Path,
    grid: dict[str, np.ndarray],
    centers: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(grid)
    payload["axis_x_rm"] = centers
    payload["axis_y_rm"] = centers
    payload["axis_z_rm"] = centers
    payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, indent=2))
    np.savez(path, **payload)


def _sparse_slice_data(
    grid: dict[str, np.ndarray],
    key: str,
    centers: np.ndarray,
    plane: str,
) -> tuple[np.ndarray, tuple[float, float, float, float], str, str]:
    center_index = int(np.argmin(np.abs(centers)))
    panel = np.full((centers.size, centers.size), np.nan, dtype=np.float64)
    values = np.asarray(grid[key], dtype=float)
    ix = np.asarray(grid["ix"], dtype=np.int64)
    iy = np.asarray(grid["iy"], dtype=np.int64)
    iz = np.asarray(grid["iz"], dtype=np.int64)
    lo = float(centers[0])
    hi = float(centers[-1])
    if plane == "XY":
        mask = iz == center_index
        panel[iy[mask], ix[mask]] = values[mask]
        return panel, (lo, hi, lo, hi), "X_MSE (Rm)", "Y_MSE (Rm)"
    if plane == "XZ":
        mask = iy == center_index
        panel[iz[mask], ix[mask]] = values[mask]
        return panel, (lo, hi, lo, hi), "X_MSE (Rm)", "Z_MSE (Rm)"
    if plane == "YZ":
        mask = ix == center_index
        panel[iz[mask], iy[mask]] = values[mask]
        return panel, (lo, hi, lo, hi), "Y_MSE (Rm)", "Z_MSE (Rm)"
    raise ValueError(f"Unknown plane {plane!r}")


def _plot_one_slice(
    grid: dict[str, np.ndarray],
    key: str,
    centers: np.ndarray,
    plane: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
    *,
    density_log10: bool,
    diverging: bool,
) -> None:
    import matplotlib.pyplot as plt

    panel, extent, xlabel, ylabel = _sparse_slice_data(grid, key, centers, plane)
    panel = np.asarray(panel, dtype=float)
    if density_log10:
        panel = np.where(panel > 0.0, np.log10(panel), np.nan)
        colorbar_label = f"log10 {colorbar_label}"
        cmap = "magma"
        vmin = vmax = None
    elif diverging:
        finite = panel[np.isfinite(panel)]
        scale = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0
        cmap = "RdBu_r"
        vmin, vmax = -scale, scale
    else:
        cmap = "viridis"
        vmin = vmax = None

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = ax.imshow(panel, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="white", lw=0.6, alpha=0.45)
    ax.axvline(0.0, color="white", lw=0.6, alpha=0.45)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_slice_figures(
    grid: dict[str, np.ndarray],
    centers: np.ndarray,
    output_dir: Path,
    mission: str,
    species: str,
    fov_group: str,
) -> list[Path]:
    paths: list[Path] = []
    figure_specs = [
        ("density_cm3", "density (cm^-3)", True, False),
        ("v_mse_x_km_s", "Vx_MSE (km/s)", False, True),
        ("v_mse_y_km_s", "Vy_MSE (km/s)", False, True),
        ("v_mse_z_km_s", "Vz_MSE (km/s)", False, True),
        ("speed_mse_km_s", "|V_MSE| (km/s)", False, False),
    ]
    for stat in ("mean", "median"):
        for var, label, density_log10, diverging in figure_specs:
            key = f"{stat}_{var}"
            for plane in ("XY", "XZ", "YZ"):
                path = output_dir / f"{mission}_{species}_{fov_group}_{stat}_{var}_{plane}.png"
                title = f"{mission} {species} {fov_group} {stat} {var} {plane} center slice"
                _plot_one_slice(
                    grid,
                    key,
                    centers,
                    plane,
                    title,
                    label,
                    path,
                    density_log10=density_log10,
                    diverging=diverging,
                )
                paths.append(path)
    return paths


def _write_summary_files(summary_rows: list[dict[str, Any]], output_root: Path, run_summary: dict[str, Any]) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "spatial_stats_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not summary_rows:
        return
    csv_path = log_dir / "spatial_stats_summary.csv"
    fields = list(summary_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def run_spatial_statistics(
    input_root: Path,
    output_root: Path,
    missions: Iterable[str] = ("MAVEN", "Tianwen-1"),
    species_ids: Iterable[str] = SPECIES_IDS,
    workers: int = 2,
    step_rm: float = DEFAULT_GRID_STEP_RM,
    density_min_cm3: float = DEFAULT_DENSITY_MIN_CM3,
    write_plots: bool = True,
    max_files_per_species: int | None = None,
    years: Iterable[str] | None = None,
    dates: Iterable[str] | None = None,
    output_mode: str = "stats",
) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    write_records = output_mode in ("grid-records", "both")
    write_stats = output_mode in ("stats", "both")
    if output_mode not in ("grid-records", "stats", "both"):
        raise ValueError(f"Unsupported output_mode: {output_mode}")

    for mission in missions:
        config = MISSION_CONFIGS[mission]
        centers = grid_centers(config.max_abs_rm, step_rm)
        shape = (centers.size, centers.size, centers.size)
        for species in species_ids:
            files = find_species_files(input_root, config, species, limit=max_files_per_species, years=years, dates=dates)
            if not files:
                products.append({"mission": mission, "species": species, "status": "no_input_files"})
                continue

            chunks_by_group: dict[str, list[dict[str, np.ndarray]]] = {label: [] for label in FOV_GROUPS}
            file_summaries: list[dict[str, Any]] = []
            grid_record_rows: list[dict[str, Any]] = []
            if int(workers) <= 1:
                for path in files:
                    chunk = read_spatial_chunk(str(path), config.fov_flag_field, centers, step_rm, density_min_cm3)
                    file_summaries.append({key: value for key, value in chunk.items() if key != "groups"})
                    if write_records:
                        grid_record_rows.extend(
                            write_grid_record_files(
                                chunk,
                                centers,
                                shape,
                                output_root,
                                mission,
                                species,
                                step_rm=step_rm,
                                density_min_cm3=density_min_cm3,
                            )
                        )
                    if write_stats:
                        for label in FOV_GROUPS:
                            chunks_by_group[label].append(chunk["groups"][label])
            else:
                with ProcessPoolExecutor(max_workers=int(workers)) as pool:
                    futures = [
                        pool.submit(read_spatial_chunk, str(path), config.fov_flag_field, centers, step_rm, density_min_cm3)
                        for path in files
                    ]
                    for future in as_completed(futures):
                        chunk = future.result()
                        file_summaries.append({key: value for key, value in chunk.items() if key != "groups"})
                        if write_records:
                            grid_record_rows.extend(
                                write_grid_record_files(
                                    chunk,
                                    centers,
                                    shape,
                                    output_root,
                                    mission,
                                    species,
                                    step_rm=step_rm,
                                    density_min_cm3=density_min_cm3,
                                )
                            )
                        if write_stats:
                            for label in FOV_GROUPS:
                                chunks_by_group[label].append(chunk["groups"][label])

            if write_records:
                record_summary = {
                    "mission": mission,
                    "species": species,
                    "status": "grid_records_written",
                    "input_file_count": len(files),
                    "grid_record_file_count": len(grid_record_rows),
                    "grid_record_rows": int(sum(item["rows"] for item in grid_record_rows)),
                    "grid_record_root": str(output_root / "grid_records" / mission / species),
                }
                products.append(record_summary)
                summary_rows.append(
                    {
                        "mission": mission,
                        "species": species,
                        "fov_group": "all",
                        "input_files": len(files),
                        "grid_shape": "x".join(str(item) for item in shape),
                        "density_samples": record_summary["grid_record_rows"],
                        "velocity_samples": "",
                        "nonempty_density_bins": "",
                        "nonempty_velocity_bins": "",
                        "data_file": record_summary["grid_record_root"],
                        "figure_count": 0,
                    }
                )

            if not write_stats:
                products.append(
                    {
                        "mission": mission,
                        "species": species,
                        "status": "input_files_processed",
                        "input_file_count": len(files),
                        "file_summaries": sorted(file_summaries, key=lambda item: item["path"]),
                    }
                )
                continue

            for fov_group, group_chunks in chunks_by_group.items():
                grid = compute_group_grid(group_chunks, shape, centers)
                finite_density_bins = int(np.count_nonzero(grid["density_count"]))
                finite_velocity_bins = int(np.count_nonzero(grid["velocity_count"]))
                density_samples = int(np.sum(grid["density_count"]))
                velocity_samples = int(np.sum(grid["velocity_count"]))
                metadata = {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "mission": mission,
                    "mission_label": config.mission_label,
                    "species": species,
                    "fov_group": fov_group,
                    "fov_flag_field": config.fov_flag_field,
                    "fov_flag_value": FOV_GROUPS[fov_group],
                    "mars_radius_km": GRID_MARS_RADIUS_KM,
                    "grid_step_rm": step_rm,
                    "grid_center_min_rm": float(centers[0]),
                    "grid_center_max_rm": float(centers[-1]),
                    "grid_shape_xyz": list(shape),
                    "density_min_cm3": density_min_cm3,
                    "input_file_count": len(files),
                    "input_files": [str(path) for path in files],
                    "statistics": "mean and exact median per 3-D MSE grid cell; velocity medians are component-wise, speed is |V_MSE|",
                    "storage": "sparse NPZ rows contain non-empty grid cells only; omitted cells have count=0 and statistics=NaN over the full axis ranges",
                }
                product_stem = f"{mission}_{species}_{fov_group}_mse_grid_stats"
                data_path = output_root / "data" / f"{product_stem}.npz"
                save_sparse_grid_npz(data_path, grid, centers, metadata)
                figure_paths: list[Path] = []
                if write_plots:
                    figure_paths = write_slice_figures(grid, centers, output_root / "figures", mission, species, fov_group)
                row = {
                    "mission": mission,
                    "species": species,
                    "fov_group": fov_group,
                    "input_files": len(files),
                    "grid_shape": "x".join(str(item) for item in shape),
                    "density_samples": density_samples,
                    "velocity_samples": velocity_samples,
                    "nonempty_density_bins": finite_density_bins,
                    "nonempty_velocity_bins": finite_velocity_bins,
                    "data_file": str(data_path),
                    "figure_count": len(figure_paths),
                }
                summary_rows.append(row)
                products.append({**row, "status": "written", "figures": [str(path) for path in figure_paths]})

            products.append(
                {
                    "mission": mission,
                    "species": species,
                    "status": "input_files_processed",
                    "input_file_count": len(files),
                    "file_summaries": sorted(file_summaries, key=lambda item: item["path"]),
                }
            )

    run_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "mars_radius_km": GRID_MARS_RADIUS_KM,
        "grid_step_rm": step_rm,
        "density_min_cm3": density_min_cm3,
        "storage": "sparse_npz_nonempty_cells",
        "output_mode": output_mode,
        "missions": list(missions),
        "species": list(species_ids),
        "years": list(years) if years is not None else None,
        "dates": list(dates) if dates is not None else None,
        "products": products,
    }
    _write_summary_files(summary_rows, output_root, run_summary)
    return run_summary
