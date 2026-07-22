"""Paper Figure 5 reproduction from released MINPA Mode-1 DPF.

The paper plots detector counts summed over all H+ directional channels.  The
released local product contains DPF instead of undocumented raw counts, so the
figure below uses the empirical per-channel DPF quantum stored in the
background model to form an explicitly labelled count equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from .minpa_background import (
    MODE1_ENERGY_EV,
    MinpaBackgroundModel,
    mode1_ori_files_for_interval,
    read_mode1_ori_records,
)


@dataclass(frozen=True)
class Figure5Panel:
    panel: str
    start_utc: str
    stop_utc: str
    signal_description: str
    interface_utc: tuple[str, ...]
    interface_source: str


@dataclass(frozen=True)
class QuietIntervalCoherenceConfig:
    """Reject a quiet candidate when one energy bin stays coherently strong."""

    count_threshold: float = 5.0
    maximum_energy_occupancy: float = 0.10
    weak_count_threshold: float = 1.0
    maximum_weak_energy_occupancy: float = 0.40


FIGURE5_PANELS = (
    Figure5Panel(
        panel="a",
        start_utc="2021-12-25T20:02:00Z",
        stop_utc="2021-12-26T01:20:00Z",
        signal_description="weak solar-wind signal",
        interface_utc=(
            "2021-12-25T20:23:00Z",
            "2021-12-25T21:45:00Z",
            "2021-12-25T22:33:00Z",
            "2021-12-25T23:10:00Z",
        ),
        interface_source="Figure 3 orbit/interface annotations reproduced by the project",
    ),
    Figure5Panel(
        panel="b",
        start_utc="2021-12-31T02:40:00Z",
        stop_utc="2021-12-31T08:40:00Z",
        signal_description="clear solar-wind signal",
        interface_utc=(
            "2021-12-31T03:57:00Z",
            "2021-12-31T05:24:00Z",
            "2021-12-31T05:53:00Z",
            "2021-12-31T06:33:00Z",
        ),
        interface_source="inferred from Figure 5 dashed-line pixel positions; approximately +/-1 minute",
    ),
)

REGION_NAMES = (
    "Interplanetary space",
    "Magnetosheath",
    "Induced magnetosphere",
    "Magnetosheath",
    "Interplanetary space",
)
REGION_COLORS = ("#8bcf73", "#e8a067", "#78bce2", "#e8a067", "#8bcf73")


def hplus_count_equivalent_spectrum(
    dpf_cube: np.ndarray,
    dpf_quantum: np.ndarray,
) -> np.ndarray:
    """Sum H+ count equivalents over all 4x16 directional channels."""

    values = np.asarray(dpf_cube, dtype=float)
    quantum = np.asarray(dpf_quantum, dtype=float)
    expected = (MODE1_ENERGY_EV.size, 4, 16, 8)
    if values.ndim != 5 or values.shape[1:] != expected:
        raise ValueError(f"dpf_cube must have shape (time, {expected}); got {values.shape}")
    if quantum.shape != expected:
        raise ValueError(f"dpf_quantum must have shape {expected}; got {quantum.shape}")
    h_values = values[..., 0]
    h_quantum = quantum[..., 0]
    ratio = np.divide(
        h_values,
        h_quantum[None, ...],
        out=np.full(h_values.shape, np.nan, dtype=float),
        where=(
            np.isfinite(h_values)
            & (h_values >= 0.0)
            & np.isfinite(h_quantum[None, ...])
            & (h_quantum[None, ...] > 0.0)
        ),
    )
    finite_count = np.isfinite(ratio).sum(axis=(2, 3))
    summed = np.nansum(np.maximum(ratio, 0.0), axis=(2, 3))
    summed[finite_count == 0] = np.nan
    return summed


def screen_hplus_coherent_signal(
    dpf_cube: np.ndarray,
    dpf_quantum: np.ndarray,
    config: QuietIntervalCoherenceConfig | None = None,
) -> dict[str, object]:
    """Quantify whether an interval contains a persistent H+ spectral ridge.

    Paper quiet intervals are selected because no solar-wind signal is
    visible.  In count-equivalent space, sparse electronic background is
    usually 0--3 counts per energy/record, whereas a real ion ridge repeatedly
    exceeds five summed H+ directional counts in the same energy channel.
    """

    cfg = config or QuietIntervalCoherenceConfig()
    for name, value in (
        ("count_threshold", cfg.count_threshold),
        ("weak_count_threshold", cfg.weak_count_threshold),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("maximum_energy_occupancy", cfg.maximum_energy_occupancy),
        ("maximum_weak_energy_occupancy", cfg.maximum_weak_energy_occupancy),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie between zero and one")
    count = hplus_count_equivalent_spectrum(dpf_cube, dpf_quantum)
    finite = np.isfinite(count)
    occupancy = np.divide(
        np.sum(finite & (count > cfg.count_threshold), axis=0),
        np.sum(finite, axis=0),
        out=np.zeros(MODE1_ENERGY_EV.size, dtype=float),
        where=np.sum(finite, axis=0) > 0,
    )
    peak_index = int(np.argmax(occupancy))
    peak_occupancy = float(occupancy[peak_index])
    weak_occupancy = np.divide(
        np.sum(finite & (count > cfg.weak_count_threshold), axis=0),
        np.sum(finite, axis=0),
        out=np.zeros(MODE1_ENERGY_EV.size, dtype=float),
        where=np.sum(finite, axis=0) > 0,
    )
    weak_peak_index = int(np.argmax(weak_occupancy))
    weak_peak_occupancy = float(weak_occupancy[weak_peak_index])
    strong_rejected = peak_occupancy >= cfg.maximum_energy_occupancy
    weak_rejected = weak_peak_occupancy >= cfg.maximum_weak_energy_occupancy
    rejected = strong_rejected or weak_rejected
    reasons = []
    if strong_rejected:
        reasons.append("persistent_above_five_count_equivalents")
    if weak_rejected:
        reasons.append("persistent_weak_ridge_above_one_count_equivalent")
    return {
        "coherent_signal_rejected": bool(rejected),
        "coherent_signal_reasons": reasons,
        "count_threshold": float(cfg.count_threshold),
        "maximum_allowed_energy_occupancy": float(cfg.maximum_energy_occupancy),
        "maximum_energy_occupancy": peak_occupancy,
        "peak_energy_index": peak_index,
        "peak_energy_eV": float(MODE1_ENERGY_EV[peak_index]),
        "weak_count_threshold": float(cfg.weak_count_threshold),
        "maximum_allowed_weak_energy_occupancy": float(
            cfg.maximum_weak_energy_occupancy
        ),
        "maximum_weak_energy_occupancy": weak_peak_occupancy,
        "weak_peak_energy_index": weak_peak_index,
        "weak_peak_energy_eV": float(MODE1_ENERGY_EV[weak_peak_index]),
        "record_count": int(count.shape[0]),
        "criterion": "reject a strong or weak persistent H+ energy ridge using thresholds calibrated against this project's 23:10-23:20 window following the paper's around-23:10 quiet example",
    }


def regularize_mode1_spectrogram(
    time_unix_s: np.ndarray,
    spectrum: np.ndarray,
    start_s: float,
    stop_s: float,
    *,
    cadence_s: float = 16.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Place Mode-1 records on a fixed cadence and leave mode gaps as NaN."""

    time = np.asarray(time_unix_s, dtype=float).reshape(-1)
    values = np.asarray(spectrum, dtype=float)
    if values.shape != (time.size, MODE1_ENERGY_EV.size):
        raise ValueError("spectrum must have shape (time, 40)")
    if not (np.isfinite(start_s) and np.isfinite(stop_s) and stop_s > start_s):
        raise ValueError("invalid plotting interval")
    if not np.isfinite(cadence_s) or cadence_s <= 0.0:
        raise ValueError("cadence_s must be positive")
    count = int(np.floor((stop_s - start_s) / cadence_s)) + 1
    grid_time = start_s + np.arange(count, dtype=float) * cadence_s
    grid = np.full((count, MODE1_ENERGY_EV.size), np.nan, dtype=float)
    index = np.rint((time - start_s) / cadence_s).astype(int)
    valid = (
        np.isfinite(time)
        & (index >= 0)
        & (index < count)
        & (np.abs(grid_time[np.clip(index, 0, count - 1)] - time) <= cadence_s / 2.0)
    )
    grid[index[valid]] = values[valid]
    return grid_time, grid


def plot_figure5(
    ori_root: Path,
    quality_root: Path | None,
    model: MinpaBackgroundModel,
    output: Path,
    panels: Sequence[Figure5Panel] = FIGURE5_PANELS,
) -> dict[str, object]:
    """Create standalone PNG/PDF products matching the paper Figure 5 layout."""

    if len(panels) != 2:
        raise ValueError("Figure 5 requires exactly two panels")
    if model.mode != 1 or not model.valid:
        raise ValueError("Figure 5 requires a valid Mode-1 background model")
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.91, bottom=0.20, top=0.84, wspace=0.16)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    norm = LogNorm(vmin=1.0, vmax=3000.0)
    diagnostics: list[dict[str, object]] = []
    mesh = None

    for axis, panel in zip(axes, panels):
        start_s, stop_s = _parse_utc(panel.start_utc), _parse_utc(panel.stop_utc)
        files = mode1_ori_files_for_interval(ori_root, panel.start_utc, panel.stop_utc)
        records = read_mode1_ori_records(
            files,
            panel.start_utc,
            panel.stop_utc,
            quality_root=quality_root,
            compute_sha256=False,
        )
        # Figure 5 must use the estimator stated in the paper: the mean of
        # each channel's non-zero counts in each quiet interval, averaged over
        # intervals.  The production-candidate model may use a different,
        # zero-inclusive estimator, but substituting it here leaves the sparse
        # one-count background almost unchanged and is not a paper reproduction.
        paper_background = np.asarray(model.paper_background_dpf, dtype=float)
        finite = np.isfinite(records.dpf) & np.isfinite(paper_background[None, ...])
        corrected_dpf = np.full(records.dpf.shape, np.nan, dtype=float)
        corrected_dpf[finite] = np.maximum(
            records.dpf[finite]
            - np.broadcast_to(paper_background, records.dpf.shape)[finite],
            0.0,
        )
        raw_count = hplus_count_equivalent_spectrum(records.dpf, model.dpf_quantum)
        corrected_count = hplus_count_equivalent_spectrum(
            corrected_dpf, model.dpf_quantum
        )
        grid_time, grid_count = regularize_mode1_spectrogram(
            records.time_unix_s, corrected_count, start_s, stop_s
        )
        plot_values = np.ma.masked_invalid(np.where(grid_count > 0.0, grid_count, np.nan))
        plot_time = [datetime.fromtimestamp(value, UTC) for value in grid_time]
        mesh = axis.pcolormesh(
            plot_time,
            MODE1_ENERGY_EV,
            plot_values.T,
            shading="nearest",
            norm=norm,
            cmap=cmap,
            rasterized=True,
        )
        axis.set_yscale("log")
        axis.set_ylim(MODE1_ENERGY_EV[0], MODE1_ENERGY_EV[-1])
        axis.set_xlim(
            datetime.fromtimestamp(start_s, UTC), datetime.fromtimestamp(stop_s, UTC)
        )
        axis.set_xlabel("UTC time")
        axis.set_ylabel("Energy (eV)")
        axis.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=UTC))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H", tz=UTC))
        axis.tick_params(axis="x", labelrotation=0, labelsize=8)
        axis.text(
            0.01,
            0.96,
            f"({panel.panel})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )

        interfaces_s = [_parse_utc(value) for value in panel.interface_utc]
        region_edges = [start_s, *interfaces_s, stop_s]
        for left, right, color in zip(region_edges[:-1], region_edges[1:], REGION_COLORS):
            axis.axvspan(
                datetime.fromtimestamp(left, UTC),
                datetime.fromtimestamp(right, UTC),
                ymin=0.985,
                ymax=1.0,
                color=color,
                linewidth=0,
                zorder=5,
            )
        for value in interfaces_s:
            axis.axvline(
                datetime.fromtimestamp(value, UTC),
                color="#ff4b3e",
                linestyle=(0, (3, 3)),
                linewidth=1.15,
                zorder=6,
            )

        positive = corrected_count[np.isfinite(corrected_count) & (corrected_count > 0.0)]
        raw_positive = raw_count[np.isfinite(raw_count) & (raw_count > 0.0)]
        removed = raw_count - corrected_count
        panel_diagnostic: dict[str, object] = {
                "panel": panel.panel,
                "interval_utc": [panel.start_utc, panel.stop_utc],
                "signal_description": panel.signal_description,
                "interface_utc": list(panel.interface_utc),
                "interface_source": panel.interface_source,
                "region_sequence": list(REGION_NAMES),
                "record_count": int(records.time_unix_s.size),
                "native_quality_nonzero_records": int(
                    np.count_nonzero(records.native_quality != 0)
                ),
                "project_quality_bits_3_4_records": int(
                    np.count_nonzero(
                        records.project_quality_available
                        & ((records.project_quality_flag & np.uint32(12)) != 0)
                    )
                ),
                "quality_filtering": "none; Figure 5 applies channel background subtraction to all available Mode-1 records",
                "maximum_corrected_count_equivalent": (
                    float(np.max(positive)) if positive.size else None
                ),
                "median_positive_raw_count_equivalent": (
                    float(np.median(raw_positive)) if raw_positive.size else None
                ),
                "median_removed_count_equivalent_per_energy_record": float(
                    np.nanmedian(removed)
                ),
                "source_files": list(records.source_files),
            }
        if panel.panel == "a":
            quiet_start = _parse_utc("2021-12-25T23:10:00Z")
            quiet_stop = _parse_utc("2021-12-25T23:20:00Z")
            quiet = (records.time_unix_s >= quiet_start) & (
                records.time_unix_s < quiet_stop
            )
            quiet_raw = raw_count[quiet]
            quiet_corrected = corrected_count[quiet]
            raw_positive_mask = np.isfinite(quiet_raw) & (quiet_raw > 0.0)
            corrected_positive_mask = np.isfinite(quiet_corrected) & (
                quiet_corrected > 0.0
            )
            panel_diagnostic["quiet_reference_validation"] = {
                "interval_utc": [
                    "2021-12-25T23:10:00Z",
                    "2021-12-25T23:20:00Z",
                ],
                "record_count": int(np.count_nonzero(quiet)),
                "positive_cell_residual_fraction": float(
                    np.count_nonzero(corrected_positive_mask)
                    / max(np.count_nonzero(raw_positive_mask), 1)
                ),
                "summed_count_residual_fraction": float(
                    np.nansum(quiet_corrected) / max(np.nansum(quiet_raw), 1.0)
                ),
            }
        if panel.panel == "b":
            raw_energy_max = np.nanmax(raw_count, axis=0)
            corrected_energy_max = np.nanmax(corrected_count, axis=0)
            raw_peak_index = int(np.nanargmax(raw_energy_max))
            corrected_peak_index = int(np.nanargmax(corrected_energy_max))
            panel_diagnostic["strong_signal_validation"] = {
                "raw_peak_energy_index": raw_peak_index,
                "corrected_peak_energy_index": corrected_peak_index,
                "peak_energy_index_shift": corrected_peak_index - raw_peak_index,
                "raw_peak_energy_eV": float(MODE1_ENERGY_EV[raw_peak_index]),
                "corrected_peak_energy_eV": float(
                    MODE1_ENERGY_EV[corrected_peak_index]
                ),
                "maximum_count_retained_fraction": float(
                    np.nanmax(corrected_count) / np.nanmax(raw_count)
                ),
            }
        diagnostics.append(panel_diagnostic)

    if mesh is None:
        raise RuntimeError("Figure 5 plotting produced no mesh")
    color_axis = fig.add_axes([0.925, 0.20, 0.014, 0.64])
    colorbar = fig.colorbar(mesh, cax=color_axis)
    colorbar.set_label("H+ count equivalent per accumulation")
    fig.suptitle(r"H$^+$ Energy Spectrum after Background Reduction", fontsize=15, y=0.96)
    fig.text(
        0.5,
        0.055,
        "Released DPF reconstruction: values are count equivalents, not undocumented raw detector counts.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    png = output / "figure05_hplus_count_equivalent_reproduction.png"
    pdf = output / "figure05_hplus_count_equivalent_reproduction.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    return {
        "performed": True,
        "paper_figure": 5,
        "data_space": "H+ count equivalent reconstructed from released Mode-1 DPF quantum",
        "raw_count_claim": False,
        "background_estimator": "paper_nonzero_mean",
        "background_formula": "max(DPF_raw - paper_background_dpf, 0), then divide by channel quantum and sum 4x16 directions",
        "energy_eV": MODE1_ENERGY_EV.tolist(),
        "color_scale": {"type": "log", "vmin": 1.0, "vmax": 3000.0},
        "background_model_algorithm_version": model.algorithm_version,
        "background_model_primary_estimator_not_used_for_figure5": model.primary_estimator,
        "background_model_provisional": any(
            "provisional" in str(interval.get("source", "")).lower()
            or "provisional" in str(interval.get("label", "")).lower()
            for interval in model.approved_intervals
        ),
        "panels": diagnostics,
        "outputs": [str(png.resolve()), str(pdf.resolve())],
    }


def _parse_utc(value: str) -> float:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(text).timestamp()
