#!/usr/bin/env python
"""Validate MINPA payload->spacecraft body->MSO against local products."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from highE.coordinates import tw1_orbiter_to_mso_matrix
from highE.minpa_io import read_ori_records
from highE.minpa_modes import minpa_to_body_direction
from highE.minpa_pipeline import AttitudeSeries
from highE.minpa_processing import compute_moments
from highE.tw1_minpa import load_momag_for_day


def _nearest_row(table: np.ndarray, time_s: float, tolerance_s: float = 1.0e-3) -> int | None:
    index = int(np.searchsorted(table[:, 0], time_s))
    candidates = [item for item in (index - 1, index) if 0 <= item < table.shape[0]]
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(table[item, 0] - time_s))
    return best if abs(table[best, 0] - time_s) <= tolerance_s else None


def _utc_seconds(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ori", type=Path, required=True, help="Read-only Mode-1 ori MAT file")
    parser.add_argument("--nv", type=Path, required=True, help="Legacy NV_YYYYMMDD.mat")
    parser.add_argument(
        "--nv-mso2",
        type=Path,
        required=True,
        help="Latest NV_MSO_2/NV_YYYYMMDD.mat with corrected energy widths",
    )
    parser.add_argument("--momag-root", type=Path, required=True, help="Directory containing BssYYYYMMDD.mat")
    parser.add_argument("--start", help="Optional inclusive UTC start, e.g. 2021-12-03T00:00:00Z")
    parser.add_argument("--stop", help="Optional exclusive UTC stop")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    records = read_ori_records(args.ori)
    if args.start:
        start_s = _utc_seconds(args.start)
        records = [record for record in records if record.time_unix_s >= start_s]
    if args.stop:
        stop_s = _utc_seconds(args.stop)
        records = [record for record in records if record.time_unix_s < stop_s]
    nv_data = loadmat(args.nv)
    mso_data = loadmat(args.nv_mso2)
    density_reference = np.asarray(mso_data["NH_TW1"], dtype=float)
    legacy_reference = np.asarray(nv_data["VH_TW1"], dtype=float)
    mso_reference = np.asarray(mso_data["VH_TW1_MSO"], dtype=float)
    day = args.nv.stem.split("_")[-1]
    source = load_momag_for_day(day, args.momag_root)
    if source is None:
        raise FileNotFoundError(f"No MOMAG attitude for {day} under {args.momag_root}")
    attitude = AttitudeSeries(source.time, source.roll, source.pitch, source.yaw, source.spacecraft_velocity_mso_km_s)

    density_error: list[float] = []
    density_ratio: list[float] = []
    legacy_vector_error: list[float] = []
    mso_vector_error: list[float] = []
    stored_legacy_to_mso_error: list[float] = []
    for record in records:
        legacy_row = _nearest_row(legacy_reference, record.time_unix_s)
        density_row = _nearest_row(density_reference, record.time_unix_s)
        if legacy_row is None or density_row is None:
            continue
        angles, _, _ = attitude.nearest(record.time_unix_s)
        result = compute_moments(record.ion_dpf, record.mode, "H+", attitude_roll_pitch_yaw_deg=angles)
        reference_density = density_reference[density_row, 1]
        density_error.append(abs(result.density_cm3 - reference_density))
        if np.isfinite(reference_density) and reference_density > 0.0:
            density_ratio.append(float(result.density_cm3 / reference_density))
        legacy_vector_error.append(float(np.linalg.norm(result.velocity_minpa_km_s - legacy_reference[legacy_row, 1:4])))
        mso_row = _nearest_row(mso_reference, record.time_unix_s)
        if mso_row is not None and np.all(np.isfinite(mso_reference[mso_row, 1:4])) and np.all(np.isfinite(result.velocity_mso_km_s)):
            mso_vector_error.append(float(np.linalg.norm(result.velocity_mso_km_s - mso_reference[mso_row, 1:4])))
            roll, pitch, yaw = angles
            body = minpa_to_body_direction(legacy_reference[legacy_row, 1:4][None, :])[0]
            transformed = tw1_orbiter_to_mso_matrix(roll, pitch, yaw) @ body
            stored_legacy_to_mso_error.append(float(np.linalg.norm(transformed - mso_reference[mso_row, 1:4])))

    def statistics(values: list[float]) -> dict[str, float | int | None]:
        raw = np.asarray(values, dtype=float)
        array = raw[np.isfinite(raw)]
        return {
            "input_count": int(raw.size),
            "count": int(array.size),
            "median": float(np.median(array)) if array.size else None,
            "maximum": float(np.max(array)) if array.size else None,
        }

    report = {
        "ori": str(args.ori),
        "nv": str(args.nv),
        "nv_mso2": str(args.nv_mso2),
        "interval_utc": {"start": args.start, "stop": args.stop},
        "coordinate_chain": "MINPA payload to spacecraft body [-V2,+V3,-V1]; Rz(-Yaw) Ry(-Pitch) Rx(-Roll) body to MSO",
        "energy_width_method": "per-channel logarithmic edges inferred from adjacent calibrated energy centers; not fixed dE/E=0.15",
        "density_absolute_error_cm3": statistics(density_error),
        "density_current_over_nv_mso2": statistics(density_ratio),
        "legacy_velocity_vector_error_km_s": statistics(legacy_vector_error),
        "mso_velocity_vector_error_km_s": statistics(mso_vector_error),
        "stored_legacy_to_mso_transform_error_km_s": statistics(stored_legacy_to_mso_error),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
