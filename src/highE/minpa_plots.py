"""Reproducible plots for MINPA spectra, moments and finite-cell VDFs."""

from __future__ import annotations

from datetime import UTC, datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

from .minpa_pipeline import MinpaProductSeries
from .minpa_processing import VdfProjection


def _dates(seconds: np.ndarray) -> np.ndarray:
    return np.array([datetime.fromtimestamp(float(t), UTC) for t in seconds], dtype=object)


def plot_spectrogram(series: MinpaProductSeries, path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    data = np.where(series.deflux > 0, series.deflux, np.nan)
    finite = data[np.isfinite(data)]
    norm = LogNorm(vmin=np.nanpercentile(finite, 5), vmax=np.nanpercentile(finite, 99.5)) if finite.size else None
    mesh = ax.pcolormesh(_dates(series.time_unix_s), series.energy_eV, data.T, shading="auto", cmap="turbo", norm=norm)
    ax.set_yscale("log")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"MINPA Mode {series.mode} {series.species} differential energy flux")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    ax.set_xlabel("UTC")
    fig.colorbar(mesh, ax=ax, label=r"DEF [1/(s cm$^2$ sr)]")
    if path:
        fig.savefig(path, dpi=180)
    return fig


def plot_moment_timeseries(series: MinpaProductSeries, path: str | None = None):
    time = _dates(series.time_unix_s)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, series.density_cm3, color="black", lw=1)
    axes[0].set_ylabel(r"n (cm$^{-3}$)")
    axes[0].set_yscale("log")
    for index, (name, color) in enumerate(zip(("Vx", "Vy", "Vz"), ("#d62728", "#2ca02c", "#1f77b4"))):
        axes[1].plot(time, series.velocity_mso_km_s[:, index], label=name, color=color, lw=0.9)
    axes[1].axhline(0, color="0.6", lw=0.5)
    axes[1].set_ylabel("V MSO (km/s)")
    axes[1].legend(ncol=3, loc="upper right")
    axes[2].step(time, series.project_quality, where="mid", color="black", lw=0.8, label="project flag")
    axes[2].scatter(time[~series.accepted], series.project_quality[~series.accepted], s=10, color="#d62728", label="rejected")
    axes[2].set_ylabel("quality")
    axes[2].set_xlabel("UTC")
    axes[2].legend(ncol=2, loc="upper right")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    fig.suptitle(f"MINPA Mode {series.mode} {series.species} moments")
    if path:
        fig.savefig(path, dpi=180)
    return fig


def _smoothed_log_grid(grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    log_grid = np.full(grid.shape, np.nan, dtype=float)
    positive = np.isfinite(grid) & (grid > 0)
    log_grid[positive] = np.log10(grid[positive])
    valid = np.isfinite(log_grid)
    if not np.any(valid):
        return log_grid
    numerator = gaussian_filter(np.where(valid, log_grid, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1.0e-6)


def plot_vdf_projection(projection: VdfProjection, path: str | None = None):
    """Plot one mature-workflow reduced VDF plane with FOV coverage."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    data = np.where(projection.density_per_velocity_area > 0, projection.density_per_velocity_area, np.nan)
    finite = data[np.isfinite(data)]
    norm = LogNorm(vmin=np.nanpercentile(finite, 5), vmax=np.nanmax(finite)) if finite.size else None
    coverage = np.where(projection.coverage_mask, 1.0, np.nan)
    ax.pcolormesh(projection.x_edges_km_s, projection.y_edges_km_s, coverage.T, shading="auto", cmap=ListedColormap(["0.84"]), vmin=0, vmax=1, zorder=0)
    mesh = ax.pcolormesh(projection.x_edges_km_s, projection.y_edges_km_s, data.T, shading="auto", cmap="magma", norm=norm, zorder=2)
    axis_index = {"Vx": 0, "Vy": 1, "Vz": 2}
    ix, iy = axis_index[projection.axes[0]], axis_index[projection.axes[1]]
    ax.scatter(projection.bulk_velocity_km_s[ix], projection.bulk_velocity_km_s[iy], marker="+", s=55, c="#00B8D9", lw=1.6, zorder=5)
    ax.axhline(0, color="0.72", lw=0.5)
    ax.axvline(0, color="0.72", lw=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{projection.axes[0]} {projection.frame} (km/s)")
    ax.set_ylabel(f"{projection.axes[1]} {projection.frame} (km/s)")
    ax.set_title(f"{projection.quantity_label}; {projection.integrated_axis}-integrated\nMean sampled density = {projection.density_integral_cm3:.3g} cm$^{{-3}}$")
    ax.legend(handles=[Patch(facecolor="0.84", label="MINPA geometric coverage"), Line2D([0], [0], marker="+", color="#00B8D9", linestyle="None", label="density-weighted bulk")], loc="upper right", fontsize=8)
    fig.colorbar(mesh, ax=ax, label=r"cm$^{-3}$ (km s$^{-1}$)$^{-2}$")
    if path:
        fig.savefig(path, dpi=180)
    return fig


def plot_vdf_projections(projections: dict[str, VdfProjection], path: str | None = None, *, title: str = "Tianwen-1 MINPA finite-cell VDF projections"):
    """Plot XY, XZ and YZ reductions using the mature MINPA layout."""
    selected = [projections[key] for key in ("xy", "xz", "yz")]
    positive_parts = [p.density_per_velocity_area[np.isfinite(p.density_per_velocity_area) & (p.density_per_velocity_area > 0)] for p in selected]
    finite = np.concatenate([part for part in positive_parts if part.size]) if any(part.size for part in positive_parts) else np.empty(0)
    norm = LogNorm(vmin=max(float(np.nanpercentile(finite, 3)), 1.0e-12), vmax=float(np.nanpercentile(finite, 99.8))) if finite.size else None
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.55), constrained_layout=True)
    mesh = None
    for ax, projection in zip(axes, selected, strict=True):
        coverage = np.where(projection.coverage_mask, 1.0, np.nan)
        ax.pcolormesh(projection.x_edges_km_s, projection.y_edges_km_s, coverage.T, shading="auto", cmap=ListedColormap(["0.84"]), vmin=0, vmax=1, zorder=0)
        data = np.where(projection.density_per_velocity_area > 0, projection.density_per_velocity_area, np.nan)
        mesh = ax.pcolormesh(projection.x_edges_km_s, projection.y_edges_km_s, data.T, shading="auto", cmap="magma", norm=norm, zorder=2)
        positive = data[np.isfinite(data)]
        if positive.size >= 16:
            center_x = 0.5 * (projection.x_edges_km_s[:-1] + projection.x_edges_km_s[1:])
            center_y = 0.5 * (projection.y_edges_km_s[:-1] + projection.y_edges_km_s[1:])
            lo = float(np.nanpercentile(np.log10(positive), 35))
            hi = float(np.nanpercentile(np.log10(positive), 96))
            if hi > lo:
                ax.contour(center_x, center_y, _smoothed_log_grid(data).T, levels=np.linspace(lo, hi, 4), colors="black", linewidths=0.65, zorder=3)
        axis_index = {"Vx": 0, "Vy": 1, "Vz": 2}
        ix, iy = axis_index[projection.axes[0]], axis_index[projection.axes[1]]
        ax.scatter(projection.bulk_velocity_km_s[ix], projection.bulk_velocity_km_s[iy], marker="+", s=48, c="#00B8D9", lw=1.5, zorder=5)
        ax.axhline(0, color="0.75", lw=0.45)
        ax.axvline(0, color="0.75", lw=0.45)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(rf"${projection.axes[0]}_{{,{projection.frame}}}$ (km s$^{{-1}}$)")
        ax.set_ylabel(rf"${projection.axes[1]}_{{,{projection.frame}}}$ (km s$^{{-1}}$)")
        ax.set_title(f"{projection.axes[0][1:]}{projection.axes[1][1:]}; {projection.integrated_axis}-integrated", loc="left", fontsize=9)
    axes[0].legend(handles=[Patch(facecolor="0.84", edgecolor="0.5", label="MINPA coverage"), Line2D([0], [0], marker="+", color="#00B8D9", linestyle="None", label="bulk")], loc="upper right", fontsize=7)
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes, location="top", orientation="horizontal", pad=0.08, fraction=0.075)
        colorbar.set_label(r"MINPA reduced density proxy (cm$^{-3}$ (km s$^{-1}$)$^{-2}$)")
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    if path:
        fig.savefig(path, dpi=300, facecolor="white")
    return fig


def plot_coordinate_transform(rotation_minpa_to_mso: np.ndarray, path: str | None = None):
    rotation = np.asarray(rotation_minpa_to_mso, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_minpa_to_mso must be 3x3")
    fig = plt.figure(figsize=(11, 5.8), constrained_layout=True)
    ax_note = fig.add_subplot(121)
    ax_note.axis("off")
    ax_note.text(
        0.02,
        0.96,
        "Mature MINPA coordinate chain",
        transform=ax_note.transAxes,
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax_note.text(
        0.02,
        0.84,
        "1. Construct MINPA payload look direction from\n"
        "   calibrated pitch and azimuth bins\n\n"
        "2. Map MINPA payload coordinates to spacecraft body:\n"
        "   [Xb, Yb, Zb] = [-V2, +V3, -V1]\n\n"
        "                    [ 0 -1  0]\n"
        "   C_body<-MINPA = [ 0  0  1]\n"
        "                    [-1  0  0]\n\n"
        "3. Rotate spacecraft body to MSO with MOMAG attitude:\n"
        "   C_MSO<-body = Rz(-Yaw) Ry(-Pitch) Rx(-Roll)\n\n"
        "4. V_MSO = C_MSO<-body C_body<-MINPA V_MINPA",
        transform=ax_note.transAxes,
        va="top",
        family="monospace",
        fontsize=10.5,
        linespacing=1.25,
    )
    ax = fig.add_subplot(122, projection="3d")
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    for index, (label, color) in enumerate(zip(("MINPA V1", "MINPA V2", "MINPA V3"), colors)):
        vector = rotation[:, index]
        ax.quiver(0, 0, 0, *vector, color=color, linewidth=2)
        ax.text(*(1.08 * vector), label, color=color)
    for index, label in enumerate(("MSO X", "MSO Y", "MSO Z")):
        vector = np.eye(3)[:, index]
        ax.quiver(0, 0, 0, *vector, color="0.2", linestyle="--", alpha=0.55)
        ax.text(*(1.22 * vector), label, color="0.2")
    ax.set(
        xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), zlim=(-1.3, 1.3),
        xlabel="MSO X", ylabel="MSO Y", zlabel="MSO Z",
    )
    ax.set_title("MINPA payload axes expressed in MSO", y=0.94, pad=4)
    if path:
        fig.savefig(path, dpi=180)
    return fig
