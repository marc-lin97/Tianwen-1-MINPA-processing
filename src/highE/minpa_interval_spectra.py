"""Utilities for auditing spectra inside accepted MINPA background intervals.

The functions in this module deliberately operate on the published one-
dimensional ``day_spe`` DEF spectra.  They do not create a production
background model and they do not modify the accepted-interval decisions.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.io import loadmat

from .minpa_background import MODE1_ENERGY_EV


SPECIES = (("H", "H+", 0), ("O", "O+", 3), ("O2", "O2+", 5))
INTERVAL_SPECTRA_VERSION = "minpa-accepted-interval-spectra-v1"


def utc_seconds(text: str) -> float:
    """Convert an ISO UTC timestamp to Unix seconds."""

    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def zero_inclusive_time_mean(spectrum: np.ndarray) -> np.ndarray:
    """Return a time mean that includes finite zeros and ignores non-finite cells."""

    values = np.asarray(spectrum, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected [time, energy] spectrum, got {values.shape}")
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    return np.divide(
        np.where(finite, values, 0.0).sum(axis=0),
        count,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=count > 0,
    )


def unique_timestamp_time_mean(time: np.ndarray, spectrum: np.ndarray) -> tuple[np.ndarray, int]:
    """Average repeated timestamps first, then average unique observing times.

    Published ``day_spe`` contains one known group of four different spectra
    sharing a timestamp.  Collapsing that group avoids giving one observing
    time four times the weight and makes the record count agree with ``ori``.
    """

    time_values = np.asarray(time, dtype=float).reshape(-1)
    spectrum_values = np.asarray(spectrum, dtype=float)
    if spectrum_values.ndim != 2 or spectrum_values.shape[0] != time_values.size:
        raise ValueError(
            f"Time/spectrum shape mismatch: {time_values.shape}, {spectrum_values.shape}"
        )
    unique_time, inverse = np.unique(time_values, return_inverse=True)
    if unique_time.size == time_values.size:
        return zero_inclusive_time_mean(spectrum_values), 0
    grouped = np.stack([
        zero_inclusive_time_mean(spectrum_values[inverse == group_index])
        for group_index in range(unique_time.size)
    ])
    return zero_inclusive_time_mean(grouped), int(time_values.size - unique_time.size)


def interval_quantiles(
    mean_def: np.ndarray,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> np.ndarray:
    """Compute equal-interval DEF quantiles independently at every energy."""

    values = np.asarray(mean_def, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected [interval, species, energy], got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("At least one interval is required")
    return np.nanquantile(values, np.asarray(quantiles, dtype=float), axis=0)


def load_accepted_interval_rows(month_root: Path) -> list[dict[str, Any]]:
    """Join accepted candidates with their interval metrics across all months."""

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(month_root).glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            continue
        metrics = {
            str(item["candidate_id"]): item
            for item in summary.get("interval_metrics", [])
        }
        candidates = {
            str(item["candidate_id"]): item
            for item in summary.get("candidates", [])
            if bool(item.get("accepted"))
        }
        if set(metrics) != set(candidates):
            missing_candidates = sorted(set(metrics) - set(candidates))
            missing_metrics = sorted(set(candidates) - set(metrics))
            raise ValueError(
                f"Accepted candidate/metric mismatch in {path}: "
                f"missing candidates={missing_candidates}, missing metrics={missing_metrics}"
            )
        for candidate_id, metric in metrics.items():
            candidate = dict(candidates[candidate_id])
            start = str(candidate["final_start_utc"])
            stop = str(candidate["final_stop_utc"])
            midpoint = datetime.fromtimestamp(float(metric["midpoint_unix_s"]), UTC)
            candidate.update({
                "metric": metric,
                "start_unix_s": utc_seconds(start),
                "stop_unix_s": utc_seconds(stop),
                "year": midpoint.year,
                "quarter": f"{midpoint.year}Q{(midpoint.month - 1) // 3 + 1}",
            })
            rows.append(candidate)
    rows.sort(key=lambda item: (float(item["start_unix_s"]), str(item["candidate_id"])))
    return rows


def _read_segment(
    loaded: Mapping[str, Any], source: str, segment: str, species: str,
) -> dict[str, np.ndarray]:
    container_name = f"{species}_spe_num"
    if container_name not in loaded:
        raise ValueError(f"Missing {container_name} in {source}")
    container = loaded[container_name]
    if not hasattr(container, segment):
        raise ValueError(f"Missing {container_name}.{segment} in {source}")
    item = getattr(container, segment)
    mode = int(np.asarray(item.mod).reshape(-1)[0])
    time = np.asarray(item.t, dtype=float).reshape(-1)
    energy = np.asarray(item.f, dtype=float).reshape(-1)
    spectrum = np.asarray(item.p, dtype=float)
    if mode != 1:
        raise ValueError(f"Expected Mode 1, got Mode {mode}: {source} {segment} {species}")
    if energy.size != MODE1_ENERGY_EV.size or not np.allclose(
        energy, MODE1_ENERGY_EV, rtol=1.0e-5, atol=1.0e-5,
    ):
        raise ValueError(f"Non-standard Mode-1 energy table: {source} {segment} {species}")
    if spectrum.shape != (time.size, energy.size):
        raise ValueError(
            f"Spectrum shape mismatch in {source} {segment} {species}: "
            f"{spectrum.shape} versus {(time.size, energy.size)}"
        )
    # A few published day_spe segments contain repeated timestamps.  Preserve
    # those samples (matching the existing review figure), but reject an actual
    # time reversal because it would make interval selection ambiguous.
    if time.size > 1 and np.any(np.diff(time) < 0.0):
        raise ValueError(f"Time reversal in axis: {source} {segment} {species}")
    return {"time": time, "energy": energy, "spectrum": spectrum}


def extract_interval_mean_spectra(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Read all accepted intervals and return zero-inclusive time-mean DEF.

    The returned array has dimensions ``[interval, species, energy]``.  Every
    interval receives equal weight in later cross-interval summaries.
    """

    cache: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    means: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    reference_energy: np.ndarray | None = None
    for row in rows:
        source = str(row["source_day_spe"])
        segment = str(row["source_day_spe_segment"])
        key = (source, segment)
        if key not in cache:
            loaded = loadmat(source, squeeze_me=True, struct_as_record=False)
            cache[key] = {
                name: _read_segment(loaded, source, segment, name)
                for name, _, _ in SPECIES
            }
            times = [cache[key][name]["time"] for name, _, _ in SPECIES]
            if not all(np.array_equal(times[0], time) for time in times[1:]):
                raise ValueError(f"Species time-axis mismatch: {source} {segment}")
        start = float(row["start_unix_s"])
        stop = float(row["stop_unix_s"])
        species_means: list[np.ndarray] = []
        record_counts: list[int] = []
        finite_cell_counts: list[int] = []
        duplicate_time_counts: list[int] = []
        for name, _, _ in SPECIES:
            values = cache[key][name]
            mask = (values["time"] >= start) & (values["time"] < stop)
            if not np.any(mask):
                raise ValueError(
                    f"No day_spe records inside {row['candidate_id']}: {source} {segment}"
                )
            selected = values["spectrum"][mask]
            selected_time = values["time"][mask]
            mean_spectrum, duplicate_count = unique_timestamp_time_mean(
                selected_time, selected,
            )
            species_means.append(mean_spectrum)
            record_counts.append(int(mask.sum()))
            finite_cell_counts.append(int(np.isfinite(selected).sum()))
            duplicate_time_counts.append(duplicate_count)
            if reference_energy is None:
                reference_energy = values["energy"].copy()
        means.append(np.stack(species_means))
        audits.append({
            "candidate_id": str(row["candidate_id"]),
            "day_spe_record_count_h": record_counts[0],
            "day_spe_record_count_o": record_counts[1],
            "day_spe_record_count_o2": record_counts[2],
            "finite_cell_count_h": finite_cell_counts[0],
            "finite_cell_count_o": finite_cell_counts[1],
            "finite_cell_count_o2": finite_cell_counts[2],
            "duplicate_time_count_h": duplicate_time_counts[0],
            "duplicate_time_count_o": duplicate_time_counts[1],
            "duplicate_time_count_o2": duplicate_time_counts[2],
            "unique_day_spe_record_count_h": record_counts[0] - duplicate_time_counts[0],
            "unique_day_spe_record_count_o": record_counts[1] - duplicate_time_counts[1],
            "unique_day_spe_record_count_o2": record_counts[2] - duplicate_time_counts[2],
        })
    if reference_energy is None:
        raise ValueError("No accepted intervals were provided")
    return reference_energy, np.stack(means), audits
