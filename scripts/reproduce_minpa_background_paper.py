"""Reproduce the MINPA background-paper method with the released DPF product.

The paper's conditional-nonzero estimator is retained as a sensitivity
product.  The primary local model includes zero DPF samples and uses the
median across intervals.  The script never modifies raw ``ori`` or existing
moment products.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.io import loadmat
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import (  # noqa: E402
    MODE1_ENERGY_EV,
    MODE1_MASS_AMU,
    MODE1_PITCH_CENTERS_DEG,
    MODE1_AZIMUTH_CENTERS_DEG,
    BackgroundConfig,
    BackgroundInterval,
    CandidateConfig,
    aggregate_energy_count_equivalent,
    concatenate_mode1_records,
    discover_background_candidates,
    estimate_background,
    mode1_ori_files_for_interval,
    read_mode1_ori_records,
    read_mode1_selected_dpf_channels,
    save_background_model,
    sha256_file,
)
from highE.constants import MARS_RADIUS_KM  # noqa: E402
from highE.minpa_figure5 import (  # noqa: E402
    QuietIntervalCoherenceConfig,
    plot_figure5,
    screen_hplus_coherent_signal,
)

DEFAULT_ORI_ROOT = Path(r"D:\Data\TW-1\result\MINPA\ori")
DEFAULT_DAY_SPE_ROOT = Path(r"D:\Data\TW-1\result\MINPA\day_spe")
DEFAULT_MOMAG_ROOT = Path(r"D:\Data\TW-1\result\MOMAG\C\01Hz_all")
DEFAULT_MAVEN_MAG_ROOT = Path(r"D:\Data\MAVEN\result\mag\ss1s")
DEFAULT_QUALITY_ROOT = ROOT / "outputs" / "tw1_minpa_quality_flags_all_species"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "minpa_background_paper_reproduction"
DEFAULT_CONFIG = ROOT / "config" / "minpa_background_reproduction.json"


def _finite_mean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    """Return a NaN-aware mean without warnings for entirely empty slices."""

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    total = np.where(finite, array, 0.0).sum(axis=axis)
    count = finite.sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=float),
        where=count > 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ori-root", type=Path, default=DEFAULT_ORI_ROOT)
    parser.add_argument("--day-spe-root", type=Path, default=DEFAULT_DAY_SPE_ROOT)
    parser.add_argument("--momag-root", type=Path, default=DEFAULT_MOMAG_ROOT)
    parser.add_argument("--maven-mag-root", type=Path, default=DEFAULT_MAVEN_MAG_ROOT)
    parser.add_argument("--quality-root", type=Path, default=DEFAULT_QUALITY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--use-provisional-candidates",
        action="store_true",
        help="Build a provisional model before the 40 candidate intervals receive manual approval.",
    )
    parser.add_argument("--skip-input-hash", action="store_true")
    parser.add_argument("--skip-example-spectrograms", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_json = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_root
    data_dir = output / "data"
    figure_dir = output / "figures"
    log_dir = output / "logs"
    for path in (data_dir, figure_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    paper_start = _parse_utc(cfg_json["paper"]["analysis_start_utc"])
    paper_stop = _parse_utc(cfg_json["paper"]["analysis_stop_utc"])
    day_paths = []
    for path in sorted(args.day_spe_root.glob("Ion_spe_*.mat")):
        token = path.stem.rsplit("_", 1)[-1]
        try:
            day_time = datetime.strptime(token, "%Y%m%d").replace(tzinfo=UTC).timestamp()
        except ValueError:
            continue
        if paper_start <= day_time < paper_stop:
            day_paths.append(path)
    candidates = discover_background_candidates(
        day_paths, CandidateConfig(**cfg_json["candidate_selection"])
    )
    _annotate_candidates_with_momag(candidates, args.momag_root)
    orbit_summary = _plot_figure2_orbits(
        args.momag_root,
        args.maven_mag_root,
        paper_start,
        paper_stop,
        figure_dir,
    )
    preliminary_intervals = [
        BackgroundInterval(
            start_utc=str(item["start_utc"]),
            stop_utc=str(item["stop_utc"]),
            approved=True,
            label=f"provisional-{index + 1:02d}",
            source="automatic_low_signal_preliminary_pool",
        )
        for index, item in enumerate(candidates)
    ]
    parts = []
    for index, interval in enumerate(preliminary_intervals, start=1):
        paths = mode1_ori_files_for_interval(args.ori_root, interval.start_utc, interval.stop_utc)
        part = read_mode1_ori_records(
            paths,
            interval.start_utc,
            interval.stop_utc,
            quality_root=args.quality_root,
            compute_sha256=False,
        )
        parts.append(part)
        print(
            f"read preliminary interval {index}/{len(preliminary_intervals)}: "
            f"{part.time_unix_s.size} records",
            flush=True,
        )

    model_config = BackgroundConfig(**cfg_json["background_model"])
    preliminary_records = concatenate_mode1_records(parts)
    preliminary_model = estimate_background(
        preliminary_records, preliminary_intervals, model_config
    )
    screen_config = QuietIntervalCoherenceConfig(
        count_threshold=float(cfg_json["coherent_signal_screen"]["count_threshold"]),
        maximum_energy_occupancy=float(
            cfg_json["coherent_signal_screen"]["maximum_energy_occupancy"]
        ),
        weak_count_threshold=float(
            cfg_json["coherent_signal_screen"]["weak_count_threshold"]
        ),
        maximum_weak_energy_occupancy=float(
            cfg_json["coherent_signal_screen"]["maximum_weak_energy_occupancy"]
        ),
    )
    retained_indices: list[int] = []
    rejected_indices: list[int] = []
    for index, (candidate, part) in enumerate(zip(candidates, parts)):
        diagnostic = screen_hplus_coherent_signal(
            part.dpf, preliminary_model.dpf_quantum, screen_config
        )
        candidate.update(diagnostic)
        candidate["approved"] = False
        if bool(diagnostic["coherent_signal_rejected"]):
            rejected_indices.append(index)
        else:
            retained_indices.append(index)

    _write_candidates(data_dir, candidates, cfg_json)
    _plot_candidate_montage(candidates, args.day_spe_root, figure_dir)
    coherence_summary = {
        "preliminary_candidate_count": len(candidates),
        "coherent_signal_rejected_count": len(rejected_indices),
        "coherent_signal_rejected_candidate_numbers": [
            index + 1 for index in rejected_indices
        ],
        "retained_candidate_count": len(retained_indices),
        "config": cfg_json["coherent_signal_screen"],
        "paper_example_2021_12_25_2310_2320": (
            "clean reference; exact times of the paper's other cases are not published"
        ),
    }
    if not args.use_provisional_candidates:
        summary = {
            "status": "candidate_review_required",
            "candidate_count": len(candidates),
            "coherent_signal_screen": coherence_summary,
            "candidate_json": str((data_dir / "background_interval_candidates.json").resolve()),
            "quality_policy": cfg_json["quality_policy"],
            "figure2_orbit": orbit_summary,
            "next_step": "Review retained intervals and set approved=true; rejected intervals contain a coherent H+ ridge.",
        }
        (log_dir / "reproduction_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False))
        return

    intervals = [
        BackgroundInterval(
            start_utc=preliminary_intervals[index].start_utc,
            stop_utc=preliminary_intervals[index].stop_utc,
            approved=True,
            label=f"provisional-clean-{index + 1:02d}",
            source="automatic_low_signal_plus_hplus_coherence_screen_provisional",
        )
        for index in retained_indices
    ]
    clean_parts = [parts[index] for index in retained_indices]
    records = concatenate_mode1_records(clean_parts)
    if not args.skip_input_hash:
        records.source_sha256 = {
            file_name: sha256_file(Path(file_name)) for file_name in records.source_files
        }
    model = estimate_background(records, intervals, model_config)
    model_path = data_dir / "minpa_mode1_background_model_provisional.npz"
    save_background_model(model_path, model)

    interval_nonzero, interval_zero = _interval_channel_means(clean_parts)
    _plot_figure4(interval_nonzero, interval_zero, intervals, figure_dir)
    _plot_figure9(interval_nonzero, figure_dir)
    poisson_summary = _plot_figure10_count_equivalent(
        args.ori_root, args.quality_root, model, figure_dir
    )
    figure5_summary: dict[str, object]
    if not args.skip_example_spectrograms:
        figure5_summary = plot_figure5(
            args.ori_root, args.quality_root, model, figure_dir
        )
        figure5_summary["background_model_path"] = str(model_path.resolve())
        figure5_summary["background_model_sha256"] = sha256_file(model_path)
        (log_dir / "figure05_reproduction.json").write_text(
            json.dumps(figure5_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _plot_figure3_context(
            args.ori_root, args.quality_root, args.momag_root, figure_dir
        )
    else:
        figure5_summary = {"performed": False, "reason": "example_plots_skipped"}

    positive_robust = model.robust_background_dpf > 0.0
    comparable = positive_robust & np.isfinite(model.paper_background_dpf)
    ratio = np.divide(
        model.paper_background_dpf,
        model.robust_background_dpf,
        out=np.full_like(model.paper_background_dpf, np.nan),
        where=comparable,
    )
    accepted_summaries = [item for item in model.interval_summaries if item["accepted"]]
    quality_record_counts = {
        str(bit): sum(
            int(item["project_quality_bit_record_counts"][str(bit)])
            for item in accepted_summaries
        )
        for bit in range(1, 6)
    }
    quality_interval_counts = {
        str(bit): sum(
            int(item["project_quality_bit_record_counts"][str(bit)]) > 0
            for item in accepted_summaries
        )
        for bit in range(1, 6)
    }
    summary = {
        "status": "provisional_model_complete" if model.valid else "provisional_model_invalid",
        "paper": cfg_json["paper"],
        "quality_policy": cfg_json["quality_policy"],
        "figure2_orbit": orbit_summary,
        "local_day_spe_files": len(day_paths),
        "local_day_spe_first": day_paths[0].name if day_paths else None,
        "local_day_spe_last": day_paths[-1].name if day_paths else None,
        "candidate_count": len(candidates),
        "coherent_signal_screen": coherence_summary,
        "accepted_interval_count": len(accepted_summaries),
        "accepted_record_count": sum(int(item["kept_records"]) for item in accepted_summaries),
        "accepted_project_quality_record_counts_by_bit": quality_record_counts,
        "accepted_project_quality_interval_counts_by_bit": quality_interval_counts,
        "model_valid": model.valid,
        "model_invalid_reasons": model.invalid_reasons,
        "model_path": str(model_path.resolve()),
        "primary_estimator": model.primary_estimator,
        "robust_background_zero_channel_fraction": float(np.mean(model.robust_background_dpf == 0.0)),
        "paper_to_robust_ratio_median_positive_channels": float(np.nanmedian(ratio)) if np.any(comparable) else None,
        "count_equivalent_model": {
            "quantum_valid_channel_fraction": float(np.mean(model.quantum_valid_mask)),
            "quantum_reconstruction_valid_fraction_median": float(
                np.nanmedian(model.quantum_reconstruction_valid_fraction)
            ),
            "lambda_median_count_equivalent_per_cell": float(
                np.nanmedian(model.background_lambda_count_equivalent)
            ),
            "interpretation": "Empirical DPF quanta recover integer-like multiples; these are count equivalents, not undocumented raw detector counts.",
        },
        "poisson_test": poisson_summary,
        "figure5": figure5_summary,
        "scope_exclusions": {
            "figures_6_8_swia": "downstream moment validation; does not affect MINPA background estimation",
            "figure_11_maxwellian": "downstream H+ fit validation; does not affect MINPA background estimation",
            "momag_role": "environmental context for interval review only; not a numerical background input",
        },
        "figures": [
            str((figure_dir / name).resolve())
            for name in (
                "candidate_interval_context_montage.png",
                "figure02_tw1_maven_orbits_mso.png",
                "figure03_orbit_context_dpf.png",
                "figure04_background_intervals_dpf.png",
                "figure05_hplus_count_equivalent_reproduction.png",
                "figure09_channel_dependence_dpf.png",
                "figure10_count_equivalent_poisson_fit.png",
            )
            if (figure_dir / name).exists()
        ],
    }
    (log_dir / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (log_dir / "interval_model_diagnostics.json").write_text(
        json.dumps(model.interval_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def _write_candidates(output: Path, candidates: list[dict[str, object]], config: dict[str, object]) -> None:
    payload = {
        "status": "provisional_not_approved",
        "quality_policy": config["quality_policy"],
        "candidate_selection": config["candidate_selection"],
        "coherent_signal_screen": config["coherent_signal_screen"],
        "intervals": candidates,
    }
    (output / "background_interval_candidates.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if candidates:
        with (output / "background_interval_candidates.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)


def _annotate_candidates_with_momag(
    candidates: list[dict[str, object]], momag_root: Path, mars_radius_km: float = 3397.0
) -> None:
    by_day: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        by_day.setdefault(str(candidate["day"]), []).append(candidate)
    for day, rows in by_day.items():
        path = momag_root / f"Bss{day}.mat"
        if not path.exists():
            continue
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        position = np.asarray(data["P_TW1"], dtype=float)
        magnetic = np.asarray(data["B_TW1"], dtype=float)
        for candidate in rows:
            midpoint = (float(candidate["start_unix_s"]) + float(candidate["stop_unix_s"])) / 2.0
            index = int(np.argmin(np.abs(position[:, 0] - midpoint)))
            dt = float(abs(position[index, 0] - midpoint))
            xyz_rm = position[index, 1:4] / mars_radius_km
            candidate["momag_match_dt_s"] = dt
            candidate["mso_x_rm"] = float(xyz_rm[0])
            candidate["mso_rho_rm"] = float(np.hypot(xyz_rm[1], xyz_rm[2]))
            candidate["mso_r_rm"] = float(np.linalg.norm(xyz_rm))
            b_index = int(np.argmin(np.abs(magnetic[:, 0] - midpoint)))
            candidate["bmag_nT"] = float(
                magnetic[b_index, 4]
                if magnetic.shape[1] >= 5
                else np.linalg.norm(magnetic[b_index, 1:4])
            )


def _plot_figure2_orbits(
    tw1_momag_root: Path,
    maven_mag_root: Path,
    start_s: float,
    stop_s: float,
    output: Path,
) -> dict[str, object]:
    """Reproduce the paper's cylindrical-MSO orbit coverage panel."""

    tw1, tw1_files = _load_position_track(
        tw1_momag_root, "P_TW1", start_s, stop_s, stride=120
    )
    maven, maven_files = _load_position_track(
        maven_mag_root, "Spss", start_s, stop_s, stride=120
    )
    if tw1.size == 0 or maven.size == 0:
        return {
            "performed": False,
            "reason": "missing Tianwen-1 or MAVEN position coverage",
            "tw1_file_count": len(tw1_files),
            "maven_file_count": len(maven_files),
        }

    tw1_rm = tw1[:, 1:4] / MARS_RADIUS_KM
    maven_rm = maven[:, 1:4] / MARS_RADIUS_KM
    fig, axis = plt.subplots(figsize=(9.5, 7.5), constrained_layout=True)
    axis.scatter(
        maven_rm[:, 0],
        np.hypot(maven_rm[:, 1], maven_rm[:, 2]),
        s=1.2,
        alpha=0.22,
        color="tab:blue",
        linewidths=0,
        label="MAVEN",
    )
    axis.scatter(
        tw1_rm[:, 0],
        np.hypot(tw1_rm[:, 1], tw1_rm[:, 2]),
        s=1.2,
        alpha=0.22,
        color="tab:red",
        linewidths=0,
        label="Tianwen-1",
    )

    # Trotignon et al. (2006), axisymmetric conic fits in cylindrical MSO.
    bow_shock = {"x0_rm": 0.6, "eccentricity": 1.026, "l_rm": 2.081}
    mpb_dayside = {"x0_rm": 0.64, "eccentricity": 0.77, "l_rm": 1.08}
    mpb_nightside = {"x0_rm": 1.60, "eccentricity": 1.009, "l_rm": 0.528}
    bs_x, bs_rho = _conic_curve(**bow_shock)
    mpb_day_x, mpb_day_rho = _conic_curve(**mpb_dayside)
    mpb_night_x, mpb_night_rho = _conic_curve(**mpb_nightside)
    bs_keep = (bs_x >= -5.0) & (bs_x <= 3.0) & (bs_rho <= 5.0)
    day_keep = (mpb_day_x >= 0.0) & (mpb_day_x <= 3.0) & (mpb_day_rho <= 5.0)
    night_keep = (mpb_night_x >= -5.0) & (mpb_night_x <= 0.0) & (mpb_night_rho <= 5.0)
    axis.plot(bs_x[bs_keep], bs_rho[bs_keep], "--", color="tab:orange", lw=1.8)
    axis.plot(
        mpb_day_x[day_keep], mpb_day_rho[day_keep], "-.", color="tab:orange", lw=1.8
    )
    axis.plot(
        mpb_night_x[night_keep],
        mpb_night_rho[night_keep],
        "-.",
        color="tab:orange",
        lw=1.8,
    )

    mars_x = np.linspace(-1.0, 1.0, 400)
    axis.fill_between(
        mars_x,
        0.0,
        np.sqrt(np.maximum(1.0 - mars_x**2, 0.0)),
        color="0.45",
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
    )
    axis.text(0.1, 0.22, "Mars", fontsize=11, zorder=6)
    axis.text(-3.55, 4.15, "Bow shock", color="tab:orange", fontsize=11)
    axis.text(-4.2, 2.05, "Induced magnetopause", color="tab:orange", fontsize=10)
    axis.text(-4.0, 3.55, "Magnetosheath", fontsize=10)
    axis.text(-4.0, 1.0, "Induced magnetosphere", fontsize=10)
    axis.text(1.05, 3.5, "Interplanetary space", fontsize=10)
    axis.set(
        xlim=(-5.0, 3.0),
        ylim=(0.0, 5.0),
        xlabel=r"$X_{MSO}$ ($R_M$)",
        ylabel=r"$\sqrt{Y_{MSO}^2+Z_{MSO}^2}$ ($R_M$)",
        title="Tianwen-1 and MAVEN orbit coverage, 2021-12-01 to 2022-01-31",
    )
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(loc="upper right", markerscale=5)
    fig.savefig(output / "figure02_tw1_maven_orbits_mso.png", dpi=200)
    fig.savefig(output / "figure02_tw1_maven_orbits_mso.pdf")
    plt.close(fig)

    return {
        "performed": True,
        "coordinate_system": "MSO cylindrical: X versus sqrt(Y^2+Z^2)",
        "mars_radius_km": MARS_RADIUS_KM,
        "decimation_stride_s": 120,
        "tw1_file_count": len(tw1_files),
        "maven_file_count": len(maven_files),
        "tw1_points": int(tw1.shape[0]),
        "maven_points": int(maven.shape[0]),
        "trotignon_2006_conics": {
            "bow_shock": bow_shock,
            "mpb_dayside": mpb_dayside,
            "mpb_nightside": mpb_nightside,
        },
    }


def _load_position_track(
    root: Path,
    variable: str,
    start_s: float,
    stop_s: float,
    *,
    stride: int,
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    source_files: list[str] = []
    day = datetime.fromtimestamp(start_s, UTC).date()
    stop_day = datetime.fromtimestamp(stop_s - 1.0, UTC).date()
    while day <= stop_day:
        path = root / f"Bss{day.strftime('%Y%m%d')}.mat"
        if path.exists():
            data = loadmat(path, squeeze_me=True, struct_as_record=False)
            position = np.asarray(data[variable], dtype=float)
            keep = (
                np.all(np.isfinite(position[:, :4]), axis=1)
                & (position[:, 0] >= start_s)
                & (position[:, 0] < stop_s)
            )
            selected = position[keep][::stride]
            if selected.size:
                rows.append(selected)
                source_files.append(str(path.resolve()))
        day += timedelta(days=1)
    return (
        np.concatenate(rows, axis=0) if rows else np.empty((0, 4), dtype=float),
        source_files,
    )


def _conic_curve(
    x0_rm: float,
    eccentricity: float,
    l_rm: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta_limit = (
        np.arccos(-1.0 / eccentricity) - 1.0e-4
        if eccentricity > 1.0
        else np.pi - 1.0e-4
    )
    theta = np.linspace(0.0, theta_limit, 5000)
    radius = l_rm / (1.0 + eccentricity * np.cos(theta))
    return x0_rm + radius * np.cos(theta), radius * np.sin(theta)


def _plot_candidate_montage(
    candidates: list[dict[str, object]], day_spe_root: Path, output: Path
) -> None:
    if not candidates:
        return
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    contexts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for candidate in candidates:
        day = str(candidate["day"])
        source_path = str(
            candidate.get("source_day_spe", day_spe_root / f"Ion_spe_{day}.mat")
        )
        segment = str(candidate.get("source_day_spe_segment", "num1"))
        cache_key = (source_path, segment)
        if cache_key not in cache:
            data = loadmat(source_path, squeeze_me=True, struct_as_record=False)
            item = getattr(data["H_spe_num"], segment)
            cache[cache_key] = (
                np.asarray(item.t, dtype=float),
                np.asarray(item.f, dtype=float),
                np.asarray(item.p, dtype=float),
            )
        time, energy, spectrum = cache[cache_key]
        mask = (
            (time >= float(candidate["start_unix_s"]) - 12.0 * 60.0)
            & (time < float(candidate["stop_unix_s"]) + 12.0 * 60.0)
        )
        contexts.append((time[mask], energy, spectrum[mask]))
    positive = np.concatenate(
        [spectrum[np.isfinite(spectrum) & (spectrum > 0.0)] for _, _, spectrum in contexts]
    )
    vmin, vmax = np.nanquantile(positive, [0.02, 0.995])
    norm = LogNorm(max(vmin, np.finfo(float).tiny), max(vmax, vmin * 10.0))
    fig, axes = plt.subplots(8, 5, figsize=(18, 22), constrained_layout=True, sharey=True)
    mesh = None
    for index, (axis, candidate, context) in enumerate(zip(axes.flat, candidates, contexts), start=1):
        time, energy, spectrum = context
        dt = [datetime.fromtimestamp(value, UTC) for value in time]
        mesh = axis.pcolormesh(dt, energy, spectrum.T, shading="auto", norm=norm, cmap="turbo")
        axis.axvline(
            datetime.fromtimestamp(float(candidate["start_unix_s"]), UTC),
            color="black",
            lw=1.3,
            zorder=6,
        )
        axis.axvline(
            datetime.fromtimestamp(float(candidate["stop_unix_s"]), UTC),
            color="black",
            lw=1.3,
            zorder=6,
        )
        axis.set_yscale("log")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.tick_params(labelsize=7)
        position = (
            f"x={float(candidate.get('mso_x_rm', np.nan)):.1f}, "
            f"rho={float(candidate.get('mso_rho_rm', np.nan)):.1f} Rm"
        )
        axis.set_title(f"{index:02d} {candidate['start_utc'][5:16].replace('T', ' ')}\n{position}", fontsize=8)
        if bool(candidate.get("coherent_signal_rejected", False)):
            axis.set_title(
                f"REJECT {index:02d} {candidate['start_utc'][5:16].replace('T', ' ')}\n"
                f"H+ ridge occ(>1)={float(candidate['maximum_weak_energy_occupancy']):.2f}",
                fontsize=8,
                color="red",
                fontweight="bold",
            )
            for spine in axis.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1.2)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, shrink=0.45, label="H+ DEF [1/(s cm² sr)]")
    fig.suptitle("Provisional background intervals: ±12 min H+ context; black lines delimit candidate")
    fig.savefig(output / "candidate_interval_context_montage.png", dpi=180)
    plt.close(fig)


def _interval_channel_means(parts: list) -> tuple[np.ndarray, np.ndarray]:
    nonzero, with_zero = [], []
    for part in parts:
        cube = np.asarray(part.dpf, dtype=float)
        finite = np.isfinite(cube)
        positive = finite & (cube > 0.0)
        nonzero.append(
            np.divide(
                np.where(positive, cube, 0.0).sum(axis=0),
                positive.sum(axis=0),
                out=np.full(cube.shape[1:], np.nan),
                where=positive.sum(axis=0) > 0,
            )
        )
        with_zero.append(_finite_mean(np.where(finite, cube, np.nan), axis=0))
    return np.stack(nonzero), np.stack(with_zero)


def _plot_figure4(
    interval_nonzero: np.ndarray,
    interval_zero: np.ndarray,
    intervals: list[BackgroundInterval],
    output: Path,
) -> None:
    nonzero_energy = _finite_mean(interval_nonzero, axis=(2, 3, 4))
    zero_energy = _finite_mean(interval_zero, axis=(2, 3, 4))
    midpoint = np.array([(item.start_s + item.stop_s) / 2.0 for item in intervals])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    colors = plt.cm.turbo(np.linspace(0, 1, len(intervals)))
    for index, color in enumerate(colors):
        axes[0].loglog(MODE1_ENERGY_EV, nonzero_energy[index], color=color, alpha=0.55, lw=0.8)
    axes[0].loglog(MODE1_ENERGY_EV, _finite_mean(nonzero_energy, axis=0), color="black", lw=2.0, label="paper-style mean")
    axes[0].loglog(MODE1_ENERGY_EV, np.nanmedian(zero_energy, axis=0), color="magenta", lw=2.0, label="zero-inclusive median")
    axes[0].set(xlabel="Energy (eV)", ylabel="Background DPF\n1/(s cm² sr eV)")
    axes[0].legend()
    axes[0].grid(alpha=0.2, which="both")
    axes[1].plot([datetime.fromtimestamp(value, UTC) for value in midpoint], _finite_mean(nonzero_energy, axis=1), "o", ms=4, label="conditional nonzero")
    axes[1].plot([datetime.fromtimestamp(value, UTC) for value in midpoint], _finite_mean(zero_energy, axis=1), "o", ms=4, label="including zero")
    axes[1].set(ylabel="Mean DPF over energy", xlabel="Candidate interval midpoint (UTC)")
    axes[1].set_yscale("log")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("MINPA Mode-1 background intervals: paper Figure 4 adaptation (DPF space)")
    fig.savefig(output / "figure04_background_intervals_dpf.png", dpi=180)
    fig.savefig(output / "figure04_background_intervals_dpf.pdf")
    plt.close(fig)


def _plot_figure9(interval_nonzero: np.ndarray, output: Path) -> None:
    projections = [
        (MODE1_ENERGY_EV, _finite_mean(interval_nonzero, axis=(2, 3, 4)), "Energy (eV)"),
        (MODE1_AZIMUTH_CENTERS_DEG, _finite_mean(interval_nonzero, axis=(1, 2, 4)), "Azimuth (deg)"),
        (MODE1_MASS_AMU, _finite_mean(interval_nonzero, axis=(1, 2, 3)), "Mass (amu)"),
        (MODE1_PITCH_CENTERS_DEG, _finite_mean(interval_nonzero, axis=(1, 3, 4)), "Pitch (deg)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (x, values, label) in zip(axes.flat, projections):
        for row in values:
            axis.plot(x, row, alpha=0.25, lw=0.7)
        axis.plot(x, _finite_mean(values, axis=0), color="black", lw=2.0)
        axis.set(xlabel=label, ylabel="Conditional-nonzero DPF")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[0, 0].set_xscale("log")
    fig.suptitle("Channel dependence of provisional MINPA background (paper Figure 9 adaptation)")
    fig.savefig(output / "figure09_channel_dependence_dpf.png", dpi=180)
    fig.savefig(output / "figure09_channel_dependence_dpf.pdf")
    plt.close(fig)


def _plot_figure3_context(
    ori_root: Path,
    quality_root: Path,
    momag_root: Path,
    output: Path,
) -> None:
    """Reproduce Figure 3 context with DPF-derived H+ DEF, MOMAG, and orbit."""

    start_utc = "2021-12-25T20:02:00Z"
    stop_utc = "2021-12-26T01:20:00Z"
    start_s, stop_s = _parse_utc(start_utc), _parse_utc(stop_utc)
    files = mode1_ori_files_for_interval(ori_root, start_utc, stop_utc)
    records = read_mode1_ori_records(
        files, start_utc, stop_utc, quality_root=quality_root, compute_sha256=False
    )
    raw_def = _h_def_spectrum(records.dpf)
    spectrum_time = [datetime.fromtimestamp(value, UTC) for value in records.time_unix_s]
    positive = raw_def[np.isfinite(raw_def) & (raw_def > 0.0)]
    vmin, vmax = np.nanquantile(positive, [0.02, 0.995])

    magnetic, position = _load_tw1_momag_interval(momag_root, start_s, stop_s)
    interface_utc = [
        "2021-12-25T20:23:00Z",
        "2021-12-25T21:45:00Z",
        "2021-12-25T22:33:00Z",
        "2021-12-25T23:10:00Z",
    ]
    interface_s = np.array([_parse_utc(value) for value in interface_utc])
    interface_dt = [datetime.fromtimestamp(value, UTC) for value in interface_s]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 8.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.8, 1.0]},
        sharex=True,
    )
    mesh = axes[0].pcolormesh(
        spectrum_time,
        MODE1_ENERGY_EV,
        raw_def.T,
        shading="auto",
        norm=LogNorm(max(vmin, np.finfo(float).tiny), max(vmax, vmin * 10.0)),
        cmap="turbo",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].set_title("(a) MINPA H+ DEF, summed with solid-angle weighting", loc="left")
    fig.colorbar(mesh, ax=axes[0], pad=0.01, label=r"H+ DEF [1/(s cm$^2$ sr)]")

    region_edges = [start_s, *interface_s.tolist(), stop_s]
    region_names = [
        "Interplanetary space",
        "Magnetosheath",
        "Induced magnetosphere",
        "Magnetosheath",
        "Interplanetary space",
    ]
    region_colors = ["#80c783", "#e8a66a", "#8ecae6", "#e8a66a", "#80c783"]
    for left, right, name, color in zip(
        region_edges[:-1], region_edges[1:], region_names, region_colors
    ):
        left_dt, right_dt = datetime.fromtimestamp(left, UTC), datetime.fromtimestamp(right, UTC)
        axes[0].axvspan(left_dt, right_dt, ymin=0.965, ymax=1.0, color=color, alpha=0.9)
        if right - left >= 35.0 * 60.0:
            midpoint = datetime.fromtimestamp((left + right) / 2.0, UTC)
            axes[0].text(
                midpoint,
                0.982,
                name,
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                fontweight="bold",
            )
    axes[0].annotate(
        "background example",
        xy=(datetime.fromtimestamp(_parse_utc("2021-12-25T23:10:00Z"), UTC), 500.0),
        xytext=(datetime.fromtimestamp(_parse_utc("2021-12-25T23:25:00Z"), UTC), 25.0),
        arrowprops={"arrowstyle": "->", "color": "red"},
        color="red",
        fontsize=9,
    )
    axes[0].annotate(
        "solar wind",
        xy=(datetime.fromtimestamp(_parse_utc("2021-12-26T00:00:00Z"), UTC), 700.0),
        xytext=(datetime.fromtimestamp(_parse_utc("2021-12-26T00:18:00Z"), UTC), 120.0),
        arrowprops={"arrowstyle": "->", "color": "red"},
        color="red",
        fontsize=9,
    )

    magnetic_time = [datetime.fromtimestamp(value, UTC) for value in magnetic[:, 0]]
    for column, label, color in zip(
        range(1, 5), (r"$B_x$", r"$B_y$", r"$B_z$", r"$|B|$"),
        ("tab:red", "tab:blue", "tab:green", "black")
    ):
        values = (
            magnetic[:, column]
            if column < magnetic.shape[1]
            else np.linalg.norm(magnetic[:, 1:4], axis=1)
        )
        axes[1].plot(magnetic_time, values, color=color, lw=0.7, label=label)
    axes[1].set(ylabel=r"$B_{MSO}$ (nT)", xlabel="UTC", title="(b) Tianwen-1 MOMAG")
    axes[1].legend(loc="upper left", ncol=4, fontsize=8)
    axes[1].grid(alpha=0.2)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    for axis in axes:
        for line_time in interface_dt:
            axis.axvline(line_time, color="red", linestyle="--", lw=1.1)

    orbit_axis = inset_axes(axes[1], width="34%", height="80%", loc="upper right", borderpad=1.0)
    position_rm = position[:, 1:4] / MARS_RADIUS_KM
    orbit_axis.plot(
        position_rm[:, 0], np.hypot(position_rm[:, 1], position_rm[:, 2]), color="tab:red", lw=1.4
    )
    for time_value, label in zip(interface_s, ("20:23", "21:45", "22:33", "23:10")):
        index = int(np.argmin(np.abs(position[:, 0] - time_value)))
        x_value = position_rm[index, 0]
        rho_value = float(np.hypot(position_rm[index, 1], position_rm[index, 2]))
        orbit_axis.plot(x_value, rho_value, "ko", ms=3.0)
        orbit_axis.annotate(label, (x_value, rho_value), xytext=(3, 3), textcoords="offset points", fontsize=6)
    _plot_trotignon_boundaries(orbit_axis)
    mars_x = np.linspace(-1.0, 1.0, 200)
    orbit_axis.fill_between(
        mars_x, 0.0, np.sqrt(np.maximum(1.0 - mars_x**2, 0.0)), color="0.6", edgecolor="black"
    )
    orbit_axis.set(xlim=(-5.0, 2.0), ylim=(0.0, 5.0), xlabel=r"$X/R_M$", ylabel=r"$\rho/R_M$")
    orbit_axis.tick_params(labelsize=6)
    orbit_axis.grid(alpha=0.25, linestyle="--")
    orbit_axis.set_title("(c) orbit", fontsize=8)

    fig.suptitle(
        "Paper Figure 3 adaptation: released DPF plus MOMAG and cylindrical-MSO context"
    )
    fig.savefig(output / "figure03_orbit_context_dpf.png", dpi=200)
    fig.savefig(output / "figure03_orbit_context_dpf.pdf")
    plt.close(fig)
    del records, raw_def
    gc.collect()


def _load_tw1_momag_interval(
    root: Path, start_s: float, stop_s: float
) -> tuple[np.ndarray, np.ndarray]:
    magnetic_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    day = datetime.fromtimestamp(start_s, UTC).date()
    stop_day = datetime.fromtimestamp(stop_s - 1.0, UTC).date()
    while day <= stop_day:
        path = root / f"Bss{day.strftime('%Y%m%d')}.mat"
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        magnetic = np.asarray(data["B_TW1"], dtype=float)
        position = np.asarray(data["P_TW1"], dtype=float)
        magnetic_parts.append(magnetic[(magnetic[:, 0] >= start_s) & (magnetic[:, 0] < stop_s)])
        position_parts.append(position[(position[:, 0] >= start_s) & (position[:, 0] < stop_s)])
        day += timedelta(days=1)
    return np.concatenate(magnetic_parts), np.concatenate(position_parts)


def _plot_trotignon_boundaries(axis: plt.Axes) -> None:
    bs_x, bs_rho = _conic_curve(0.6, 1.026, 2.081)
    day_x, day_rho = _conic_curve(0.64, 0.77, 1.08)
    night_x, night_rho = _conic_curve(1.60, 1.009, 0.528)
    bs = (bs_x >= -5.0) & (bs_x <= 3.0) & (bs_rho <= 5.0)
    day = (day_x >= 0.0) & (day_x <= 3.0) & (day_rho <= 5.0)
    night = (night_x >= -5.0) & (night_x <= 0.0) & (night_rho <= 5.0)
    axis.plot(bs_x[bs], bs_rho[bs], "--", color="tab:orange", lw=1.0)
    axis.plot(day_x[day], day_rho[day], "-.", color="tab:orange", lw=1.0)
    axis.plot(night_x[night], night_rho[night], "-.", color="tab:orange", lw=1.0)


def _plot_figure10_count_equivalent(
    ori_root: Path, quality_root: Path, model, output: Path
) -> dict[str, object]:
    start_utc = "2021-12-29T15:00:00Z"
    stop_utc = "2021-12-30T18:00:00Z"
    paper_energy_eV = np.array([3.55, 11.40, 9939.98, 25290.00])
    paper_lambda = np.array([4.10, 4.27, 4.55, 4.79])
    energy_indices = [
        int(np.argmin(np.abs(MODE1_ENERGY_EV - target))) for target in paper_energy_eV
    ]
    if not np.allclose(MODE1_ENERGY_EV[energy_indices], paper_energy_eV, rtol=0.0, atol=0.01):
        raise ValueError("The released Mode-1 energy table does not contain Figure 10 channels")
    channel_indices = [
        (energy, pitch, azimuth, mass)
        for energy in energy_indices
        for pitch in range(4)
        for azimuth in range(16)
        for mass in range(8)
    ]
    paths = mode1_ori_files_for_interval(ori_root, start_utc, stop_utc)
    _, dpf, flags, available = read_mode1_selected_dpf_channels(
        paths, start_utc, stop_utc, channel_indices, quality_root=quality_root
    )
    keep = available & ((flags & np.uint32(12)) == 0)
    selected = dpf[keep].reshape(-1, len(energy_indices), 4, 16, 8)
    quantum = np.asarray(model.dpf_quantum, dtype=float)[energy_indices]
    count_equivalent, reconstruction_fraction = aggregate_energy_count_equivalent(
        selected,
        quantum,
        residual_tolerance_counts=0.05,
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    diagnostics = []
    for column, (axis, energy_index, values) in enumerate(
        zip(axes.flat, energy_indices, count_equivalent.T)
    ):
        finite = np.isfinite(values) & (values >= 0.0)
        count_good = values[finite]
        lam = float(np.mean(count_good)) if count_good.size else np.nan
        maximum = int(max(12, np.nanquantile(count_good, 0.999))) if count_good.size else 12
        bins = np.arange(-0.5, maximum + 1.5)
        observed, _ = np.histogram(count_good, bins=bins)
        x = np.arange(maximum + 1)
        observed_frequency = observed / count_good.size if count_good.size else observed.astype(float)
        expected_frequency = poisson.pmf(x, lam) if np.isfinite(lam) else np.zeros_like(x)
        axis.bar(
            x,
            observed_frequency,
            width=0.9,
            color="salmon",
            alpha=0.75,
            label="count-equivalent distribution",
        )
        axis.plot(
            x,
            expected_frequency,
            color="tab:blue",
            marker="o",
            ms=2.5,
            lw=1.2,
            label=f"Poisson fit λ={lam:.2f}",
        )
        axis.set(
            xlabel="counts per Mode-1 record (inferred)",
            ylabel="frequency",
            title=(
                f"{MODE1_ENERGY_EV[energy_index]:.2f} eV; "
                f"paper λ={paper_lambda[column]:.2f}"
            ),
        )
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        diagnostics.append(
            {
                "energy_eV": float(MODE1_ENERGY_EV[energy_index]),
                "aggregation": "sum inferred counts over 4 pitch x 16 azimuth x 8 mass cells",
                "lambda_count_equivalent": lam,
                "paper_lambda": float(paper_lambda[column]),
                "relative_lambda_difference": float(
                    (lam - paper_lambda[column]) / paper_lambda[column]
                ),
                "records": int(count_good.size),
                "quantization_valid_fraction": float(
                    np.nanmean(reconstruction_fraction[:, column])
                ),
            }
        )
    fig.suptitle(
        "Paper Figure 10 reconstruction from released DPF: all-direction/all-mass count equivalents"
    )
    fig.savefig(output / "figure10_count_equivalent_poisson_fit.png", dpi=180)
    fig.savefig(output / "figure10_count_equivalent_poisson_fit.pdf")
    plt.close(fig)
    return {
        "performed": True,
        "space": (
            "per-record energy count equivalent inferred with channel-specific DPF quanta, "
            "then summed over all pitch/azimuth/mass cells"
        ),
        "raw_count_claim": False,
        "quality_reject_mask": 12,
        "records_before_quality_filter": int(dpf.shape[0]),
        "records_after_quality_filter": int(np.count_nonzero(keep)),
        "quality_rejected_records": int(np.count_nonzero(~keep)),
        "interval_utc": [start_utc, stop_utc],
        "channels": diagnostics,
    }


def _h_def_spectrum(dpf_cube: np.ndarray) -> np.ndarray:
    p_edges = np.linspace(0.0, 90.0, 5)
    dphi = np.deg2rad(360.0 / 16.0)
    omega_pitch = (
        np.cos(np.deg2rad(p_edges[:-1] + 90.0))
        - np.cos(np.deg2rad(p_edges[1:] + 90.0))
    ) * dphi
    omega = np.repeat(omega_pitch[:, None], 16, axis=1)
    h_dpf = np.asarray(dpf_cube, dtype=float)[..., 0]
    weighted = np.nansum(h_dpf * omega[None, None, :, :], axis=(2, 3)) / np.sum(omega)
    return weighted * MODE1_ENERGY_EV[None, :]


def _parse_utc(value: str) -> float:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(text).timestamp()


if __name__ == "__main__":
    main()
