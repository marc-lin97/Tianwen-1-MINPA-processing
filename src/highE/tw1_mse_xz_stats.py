"""Aggregate Tianwen-1 MINPA per-cell records and plot MSE X-Z means."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle


SPECIES = ("oplus", "o2plus")
SPECIES_LABELS = {"oplus": r"O$^+$", "o2plus": r"O$_2^+$"}
ALL_ENERGY_SPECIES = ("hplus", "oplus", "o2plus")
ALL_ENERGY_SPECIES_LABELS = {
    "hplus": r"H$^+$",
    "oplus": r"O$^+$",
    "o2plus": r"O$_2^+$",
}
QUALITY_POLICIES = ("valid", "strict-clean", "all-valid")
STAT_NAMES = (
    "density_count",
    "density_sum_cm3",
    "velocity_count",
    "vx_sum_km_s",
    "vy_sum_km_s",
    "vz_sum_km_s",
    "speed_sum_km_s",
)


@dataclass(frozen=True)
class CellStatistics:
    """Sufficient statistics for one three-dimensional MSE cell."""

    path: str
    ix: int
    iy: int
    iz: int
    species: dict[str, tuple[float, ...]]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _quality_mask(records: np.ndarray, prefix: str, policy: str) -> np.ndarray:
    base = (
        (records["r_matrix_within_tolerance_flag"] == 1)
        & (records[f"{prefix}_processing_status_code"] == 1)
    )
    if policy == "valid":
        bitmask = records[f"{prefix}_quality_flag_bitmask"].astype(np.uint32, copy=False)
        base &= (
            (records[f"{prefix}_quality_flag_available_flag"] == 1)
            & (records[f"{prefix}_quality_flag_epoch_matched_flag"] == 1)
            & ((bitmask & np.uint32(0b11110)) == 0)
        )
    elif policy == "strict-clean":
        base &= (
            (records[f"{prefix}_quality_flag_available_flag"] == 1)
            & (records[f"{prefix}_quality_flag_epoch_matched_flag"] == 1)
            & (records[f"{prefix}_quality_flag_bitmask"] == 0)
        )
    elif policy != "all-valid":
        raise ValueError(f"Unsupported quality policy: {policy}")
    return base


def _empty_stats() -> tuple[float, ...]:
    return (0.0,) * len(STAT_NAMES)


def aggregate_record_blocks(
    blocks: Iterable[np.ndarray],
    *,
    quality_policy: str = "valid",
    species_names: tuple[str, ...] = SPECIES,
) -> dict[str, tuple[float, ...]]:
    """Aggregate record blocks without retaining cross-block record arrays."""
    if quality_policy not in QUALITY_POLICIES:
        raise ValueError(f"quality_policy must be one of {QUALITY_POLICIES}")
    accum = {species: np.zeros(len(STAT_NAMES), dtype=np.float64) for species in species_names}
    for records in blocks:
        if records.size == 0:
            continue
        for species in species_names:
            density = np.asarray(records[f"{species}_density_cm3"], dtype=np.float64)
            velocity = np.column_stack(
                [np.asarray(records[f"{species}_v_mse_{axis}_km_s"], dtype=np.float64) for axis in "xyz"]
            )
            base = _quality_mask(records, species, quality_policy)
            density_ok = (
                base
                & (records[f"{species}_density_valid_flag"] == 1)
                & np.isfinite(density)
                & (density > 0.0)
            )
            velocity_ok = (
                density_ok
                & (records[f"{species}_velocity_valid_flag"] == 1)
                & np.all(np.isfinite(velocity), axis=1)
            )
            accum[species][0] += int(np.count_nonzero(density_ok))
            accum[species][1] += float(np.sum(density[density_ok], dtype=np.float64))
            accum[species][2] += int(np.count_nonzero(velocity_ok))
            if np.any(velocity_ok):
                selected = velocity[velocity_ok]
                accum[species][3:6] += np.sum(selected, axis=0, dtype=np.float64)
                accum[species][6] += float(np.sum(np.linalg.norm(selected, axis=1), dtype=np.float64))
    return {species: tuple(values.tolist()) for species, values in accum.items()}


def _read_cell_file(arguments: tuple[str, str, tuple[str, ...]]) -> CellStatistics:
    path_text, quality_policy, species_names = arguments
    path = Path(path_text)
    with h5py.File(path, "r") as handle:
        ix = int(handle.attrs["ix"])
        iy = int(handle.attrs["iy"])
        iz = int(handle.attrs["iz"])
        blocks = (handle["days"][date]["records"][:] for date in sorted(handle["days"]))
        statistics = aggregate_record_blocks(
            blocks,
            quality_policy=quality_policy,
            species_names=species_names,
        )
    return CellStatistics(str(path), ix, iy, iz, statistics)


def _safe_mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _projection_arrays(
    cells: list[CellStatistics],
    *,
    size: int,
    species_names: tuple[str, ...] = SPECIES,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sparse: dict[str, np.ndarray] = {
        "ix": np.asarray([cell.ix for cell in cells], dtype=np.int16),
        "iy": np.asarray([cell.iy for cell in cells], dtype=np.int16),
        "iz": np.asarray([cell.iz for cell in cells], dtype=np.int16),
    }
    xz: dict[str, np.ndarray] = {}
    ix = sparse["ix"].astype(np.int64)
    iz = sparse["iz"].astype(np.int64)
    for species in species_names:
        values = np.asarray([cell.species[species] for cell in cells], dtype=np.float64)
        for column, name in enumerate(STAT_NAMES):
            sparse[f"{species}_{name}"] = values[:, column]
        sparse[f"{species}_mean_density_cm3"] = _safe_mean(values[:, 1], values[:, 0])
        for output, numerator in (
            ("mean_vx_km_s", 3),
            ("mean_vy_km_s", 4),
            ("mean_vz_km_s", 5),
            ("mean_speed_km_s", 6),
        ):
            sparse[f"{species}_{output}"] = _safe_mean(values[:, numerator], values[:, 2])

        projected = {name: np.zeros((size, size), dtype=np.float64) for name in STAT_NAMES}
        for column, name in enumerate(STAT_NAMES):
            np.add.at(projected[name], (ix, iz), values[:, column])
            xz[f"{species}_{name}"] = projected[name]
        xz[f"{species}_mean_density_cm3"] = _safe_mean(
            projected["density_sum_cm3"], projected["density_count"]
        )
        for output, numerator in (
            ("mean_vx_km_s", "vx_sum_km_s"),
            ("mean_vy_km_s", "vy_sum_km_s"),
            ("mean_vz_km_s", "vz_sum_km_s"),
            ("mean_speed_km_s", "speed_sum_km_s"),
        ):
            xz[f"{species}_{output}"] = _safe_mean(projected[numerator], projected["velocity_count"])
    return sparse, xz


def _finite_percentile(values: np.ndarray, percentile: float, *, positive: bool = False) -> float:
    selected = values[np.isfinite(values)]
    if positive:
        selected = selected[selected > 0.0]
    if selected.size == 0:
        return math.nan
    return float(np.percentile(selected, percentile))


def _add_mars(ax: plt.Axes) -> None:
    ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor="white", edgecolor="black", linewidth=1.0, zorder=4))
    ax.axhline(0.0, color="0.65", linewidth=0.5, zorder=1)
    ax.axvline(0.0, color="0.65", linewidth=0.5, zorder=1)


def _plot_map(
    values: np.ndarray,
    edges: np.ndarray,
    path_stem: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    symmetric: bool = False,
    color_limits: tuple[float, float] | None = None,
) -> dict[str, float | str]:
    plotted = np.asarray(values, dtype=np.float64).T
    finite = plotted[np.isfinite(plotted)]
    if finite.size == 0:
        raise ValueError(f"No finite values available for {title}")
    norm = None
    vmin: float | None = None
    vmax: float | None = None
    if color_limits is not None:
        vmin, vmax = map(float, color_limits)
        if not np.isfinite(vmin + vmax) or not vmax > vmin:
            raise ValueError(f"Invalid color limits for {title}: {color_limits}")
        if symmetric:
            if not (vmin < 0.0 < vmax):
                raise ValueError(f"Symmetric map must straddle zero: {color_limits}")
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    elif symmetric:
        limit = float(np.percentile(np.abs(finite), 98.0))
        if not np.isfinite(limit) or limit <= 0.0:
            limit = float(np.max(np.abs(finite))) or 1.0
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        vmin, vmax = -limit, limit
    else:
        vmin = float(np.percentile(finite, 2.0))
        vmax = float(np.percentile(finite, 98.0))
        if not vmax > vmin:
            vmax = vmin + max(abs(vmin) * 1.0e-6, 1.0e-12)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    mesh = ax.pcolormesh(edges, edges, plotted, cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
    _add_mars(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_ylim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel(r"$X_{MSE}$ ($R_M$)")
    ax.set_ylabel(r"$Z_{MSE}$ ($R_M$)")
    ax.set_title(title)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=220)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "png": str(path_stem.with_suffix(".png")),
        "pdf": str(path_stem.with_suffix(".pdf")),
        "color_min": float(vmin),
        "color_max": float(vmax),
    }


def create_xz_plots(
    xz: dict[str, np.ndarray],
    edges: np.ndarray,
    output_dir: Path,
    *,
    quality_policy: str,
    species_names: tuple[str, ...] = SPECIES,
    species_labels: dict[str, str] | None = None,
    product_slug: str = "tw1_minpa",
    energy_title: str = "",
) -> list[dict[str, Any]]:
    """Write five separate X-Z maps for each species."""
    products: list[dict[str, Any]] = []
    labels = SPECIES_LABELS if species_labels is None else species_labels
    title_suffix = f", {energy_title}" if energy_title else ""
    for species in species_names:
        label = labels[species]
        density = xz[f"{species}_mean_density_cm3"]
        speed = xz[f"{species}_mean_speed_km_s"]
        log_density = np.where(density > 0.0, np.log10(density), np.nan)
        log_speed = np.where(speed > 0.0, np.log10(speed), np.nan)
        definitions = (
            ("density_log10", log_density, rf"{label} mean density, all-Y X-Z projection{title_suffix}", r"$\log_{10}\langle n\rangle$ (cm$^{-3}$)", "viridis", False),
            ("speed_log10", log_speed, rf"{label} mean speed, all-Y X-Z projection{title_suffix}", r"$\log_{10}\langle |V|\rangle$ (km s$^{-1}$)", "magma", False),
            ("vx", xz[f"{species}_mean_vx_km_s"], rf"{label} mean $V_x$, all-Y X-Z projection{title_suffix}", r"$\langle V_x\rangle$ (km s$^{-1}$)", "RdBu_r", True),
            ("vy", xz[f"{species}_mean_vy_km_s"], rf"{label} mean $V_y$, all-Y X-Z projection{title_suffix}", r"$\langle V_y\rangle$ (km s$^{-1}$)", "RdBu_r", True),
            ("vz", xz[f"{species}_mean_vz_km_s"], rf"{label} mean $V_z$, all-Y X-Z projection{title_suffix}", r"$\langle V_z\rangle$ (km s$^{-1}$)", "RdBu_r", True),
        )
        for variable, values, title, colorbar, cmap, symmetric in definitions:
            stem = output_dir / f"{product_slug}_{species}_{variable}_mse_xz_{quality_policy}"
            plot = _plot_map(
                values,
                edges,
                stem,
                title=title,
                colorbar_label=colorbar,
                cmap=cmap,
                symmetric=symmetric,
            )
            products.append({"species": species, "variable": variable, **plot})
    return products


def _run_tw1_mse_xz_statistics(
    input_root: Path,
    output_root: Path,
    *,
    quality_policy: str = "valid",
    workers: int = 1,
    max_files: int | None = None,
    write_plots: bool = True,
    progress_every: int = 5000,
    species_names: tuple[str, ...] = SPECIES,
    species_labels: dict[str, str] | None = None,
    product_slug: str = "tw1_minpa",
    energy_selection: str = "raw energy > 1000 eV",
    energy_title: str = "",
    expected_schema_name: str | None = None,
) -> dict[str, Any]:
    """Stream all cell files, save sparse 3-D means, and form all-Y X-Z maps."""
    if quality_policy not in QUALITY_POLICIES:
        raise ValueError(f"quality_policy must be one of {QUALITY_POLICIES}")
    input_root = Path(input_root)
    output_root = Path(output_root)
    schema = json.loads((input_root / "grid_schema.json").read_text(encoding="utf-8"))
    if expected_schema_name is not None and schema.get("schema_name") != expected_schema_name:
        raise ValueError(
            f"Expected grid schema {expected_schema_name!r}, got {schema.get('schema_name')!r}"
        )
    grid = schema["grid"]
    grid_min = float(grid["min_rm"])
    grid_max = float(grid["max_rm"])
    grid_step = float(grid["step_rm"])
    size = int(round((grid_max - grid_min) / grid_step))
    if list(grid["shape_xyz"]) != [size, size, size]:
        raise ValueError("Grid schema shape is inconsistent with its edges and spacing")
    files = sorted((input_root / "cells").rglob("*.h5"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No HDF5 cell files found under {input_root / 'cells'}")

    arguments = [(str(path), quality_policy, species_names) for path in files]
    cells: list[CellStatistics] = []
    if workers <= 1:
        iterator = map(_read_cell_file, arguments)
        for completed, cell in enumerate(iterator, start=1):
            cells.append(cell)
            if progress_every > 0 and (completed % progress_every == 0 or completed == len(files)):
                print(f"cell files reduced: {completed}/{len(files)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_read_cell_file, arguments, chunksize=64)
            for completed, cell in enumerate(iterator, start=1):
                cells.append(cell)
                if progress_every > 0 and (completed % progress_every == 0 or completed == len(files)):
                    print(f"cell files reduced: {completed}/{len(files)}", flush=True)
    indices = [(cell.ix, cell.iy, cell.iz) for cell in cells]
    if len(indices) != len(set(indices)):
        raise ValueError("Multiple HDF5 files claim the same three-dimensional grid cell")
    if any(not (0 <= index < size) for triplet in indices for index in triplet):
        raise ValueError("A cell index lies outside the declared grid")

    sparse, xz = _projection_arrays(cells, size=size, species_names=species_names)
    centers = grid_min + (np.arange(size, dtype=np.float64) + 0.5) * grid_step
    edges = grid_min + np.arange(size + 1, dtype=np.float64) * grid_step
    common = {
        "axis_x_rm": centers,
        "axis_y_rm": centers,
        "axis_z_rm": centers,
        "grid_edges_rm": edges,
        "mars_radius_km": np.asarray(float(grid["mars_radius_km"])),
        "quality_policy": np.asarray(quality_policy),
        "projection_method": np.asarray("record-weighted aggregation over all Y cells"),
    }
    data_dir = output_root / "data"
    figure_dir = output_root / "figures"
    logs_dir = output_root / "logs"
    sparse_path = data_dir / f"{product_slug}_mse_3d_cell_means_{quality_policy}.npz"
    xz_path = data_dir / f"{product_slug}_mse_xz_all_y_means_{quality_policy}.npz"
    _atomic_npz(sparse_path, **common, **sparse)
    _atomic_npz(xz_path, **common, **xz)
    plots = (
        create_xz_plots(
            xz,
            edges,
            figure_dir,
            quality_policy=quality_policy,
            species_names=species_names,
            species_labels=species_labels,
            product_slug=product_slug,
            energy_title=energy_title,
        )
        if write_plots
        else []
    )

    species_summary: dict[str, Any] = {}
    for species in species_names:
        density_count = int(np.sum(xz[f"{species}_density_count"], dtype=np.float64))
        velocity_count = int(np.sum(xz[f"{species}_velocity_count"], dtype=np.float64))
        mean_density = xz[f"{species}_mean_density_cm3"]
        mean_speed = xz[f"{species}_mean_speed_km_s"]
        species_summary[species] = {
            "density_records": density_count,
            "velocity_records": velocity_count,
            "xz_pixels_with_density": int(np.count_nonzero(np.isfinite(mean_density))),
            "xz_pixels_with_velocity": int(np.count_nonzero(np.isfinite(mean_speed))),
            "mean_density_positive_min_cm3": _finite_percentile(mean_density, 0.0, positive=True),
            "mean_density_positive_max_cm3": _finite_percentile(mean_density, 100.0, positive=True),
            "mean_speed_positive_min_km_s": _finite_percentile(mean_speed, 0.0, positive=True),
            "mean_speed_positive_max_km_s": _finite_percentile(mean_speed, 100.0, positive=True),
        }
    summary: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "cell_files_processed": len(cells),
        "workers": int(workers),
        "quality_policy": quality_policy,
        "energy_selection": energy_selection,
        "species_names": list(species_names),
        "filters": {
            "common": "r_matrix_within_tolerance_flag == 1 and species processing_status_code == 1",
            "density": "density_valid_flag == 1, finite density, density > 0",
            "velocity": "density filter plus velocity_valid_flag == 1 and all MSE components finite",
            "valid_quality_extra": "quality flag available and epoch matched; NV_MSO_2 bits 2-5 all zero; bit 1 is retained as caution",
            "strict_clean_extra": "optional stricter policy: quality flag available and epoch matched and all five bits zero",
            "fov": "not filtered",
        },
        "aggregation": {
            "three_dimensional": "per-record arithmetic mean inside each 0.1 Rm cell",
            "xz_projection": "record-weighted aggregation over every Y cell for each X-Z pixel",
            "density": "mean of per-record densities; never a cross-record sum interpreted as density",
            "speed": "mean of per-record Euclidean MSE speed magnitudes",
        },
        "grid": grid,
        "data_products": [str(sparse_path), str(xz_path)],
        "plot_products": plots,
        "species": species_summary,
    }
    _atomic_json(logs_dir / f"{product_slug}_mse_xz_summary_{quality_policy}.json", summary)
    return summary


def run_tw1_mse_xz_statistics(
    input_root: Path,
    output_root: Path,
    *,
    quality_policy: str = "valid",
    workers: int = 1,
    max_files: int | None = None,
    write_plots: bool = True,
    progress_every: int = 5000,
) -> dict[str, Any]:
    """Build the established >1 keV O+/O2+ MSE X-Z products."""
    return _run_tw1_mse_xz_statistics(
        input_root,
        output_root,
        quality_policy=quality_policy,
        workers=workers,
        max_files=max_files,
        write_plots=write_plots,
        progress_every=progress_every,
    )


def run_tw1_all_energy_mse_xz_statistics(
    input_root: Path,
    output_root: Path,
    *,
    quality_policy: str = "valid",
    workers: int = 1,
    max_files: int | None = None,
    write_plots: bool = True,
    progress_every: int = 5000,
) -> dict[str, Any]:
    """Build full-energy H+/O+/O2+ MSE X-Z products from the audited archive."""
    return _run_tw1_mse_xz_statistics(
        input_root,
        output_root,
        quality_policy=quality_policy,
        workers=workers,
        max_files=max_files,
        write_plots=write_plots,
        progress_every=progress_every,
        species_names=ALL_ENERGY_SPECIES,
        species_labels=ALL_ENERGY_SPECIES_LABELS,
        product_slug="tw1_minpa_all_energy",
        energy_selection="all NV_MSO_2 energy bins; no >1 keV threshold",
        energy_title="all energy bins",
        expected_schema_name="tw1_minpa_all_energy_mse_grid_records",
    )
