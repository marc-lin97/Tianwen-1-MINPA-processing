"""Build minimal public metadata for the frozen MINPA v2.2.0 release."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def calibration_time_rows(
    mode1_inventory: Mapping[str, Any],
    multimode_inventory: Mapping[str, Any],
) -> list[dict[str, str | int]]:
    """Return only the UTC times needed to identify approved calibration data."""

    for name, inventory in (
        ("Mode-1", mode1_inventory), ("Mode-4/12", multimode_inventory)
    ):
        if inventory.get("review_status") != "finalized":
            raise ValueError(f"{name} review inventory is not finalized")
    mode1 = list(mode1_inventory.get("approved_intervals", []))
    multimode = list(multimode_inventory.get("approved_intervals", []))
    if len(mode1) != 273:
        raise ValueError(f"Expected 273 Mode-1 intervals, got {len(mode1)}")
    if len(multimode) != 2901:
        raise ValueError(f"Expected 2901 Mode-4/12 species intervals, got {len(multimode)}")

    rows: list[dict[str, str | int]] = [
        {
            "mode": 1,
            "species": "H+&O+&O2+ (joint)",
            "start_utc": str(row["start_utc"]),
            "stop_utc": str(row["stop_utc"]),
        }
        for row in mode1
    ]
    rows.extend({
        "mode": int(row["mode"]),
        "species": str(row["species"]),
        "start_utc": str(row["start_utc"]),
        "stop_utc": str(row["stop_utc"]),
    } for row in multimode)
    keys = [
        (int(row["mode"]), str(row["species"]), str(row["start_utc"]), str(row["stop_utc"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Approved calibration time table contains exact duplicate rows")
    for row in rows:
        start = datetime.fromisoformat(str(row["start_utc"]).replace("Z", "+00:00"))
        stop = datetime.fromisoformat(str(row["stop_utc"]).replace("Z", "+00:00"))
        if start.tzinfo is None or stop.tzinfo is None:
            raise ValueError("Calibration interval timestamps must include UTC offsets")
        if start.astimezone(UTC) >= stop.astimezone(UTC):
            raise ValueError(f"Non-positive calibration interval: {row}")
    return sorted(rows, key=lambda row: (
        int(row["mode"]), str(row["species"]), str(row["start_utc"]), str(row["stop_utc"])
    ))


def write_calibration_time_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("mode", "species", "start_utc", "stop_utc"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def public_example_manifest(final_review: Mapping[str, Any]) -> dict[str, Any]:
    """Strip local review internals while retaining the final example set."""

    if final_review.get("status") != "finalized_manual_review":
        raise ValueError("Example review is not finalized")
    counts = dict(final_review["counts"])
    if counts != {
        "mode_species_groups": 9,
        "approved_noise_examples": 9,
        "approved_signal_examples": 9,
        "rejected_signal_candidates": 18,
    }:
        raise ValueError(f"Unexpected example review counts: {counts}")

    common = ("mode", "species", "start_utc", "stop_utc", "figure", "figure_sha256")
    signal_extra = (
        "strength_tier",
        "maximum_median_channel_signal_to_background",
        "signal_peak_retention_fraction",
        "ratio_peak_shift_channels",
    )
    signal = [
        {key: row[key] for key in (*common, *signal_extra)}
        for row in final_review["signal_examples"]
    ]
    return {
        "artifact_version": "minpa-typical-noise-signal-examples-v2.2.0-public",
        "status": "finalized_manual_review",
        "bundle_sha256": str(final_review["bundle_sha256"]),
        "units": str(final_review["units"]),
        "smoothing_or_interpolation": str(final_review["smoothing_or_interpolation"]),
        "counts": {
            "mode_species_groups": 9,
            "approved_before_after_examples": 9,
            "rejected_candidate_count": 18,
        },
        "signal_examples": signal,
        "omitted_review_intermediates": (
            "Noise-reference files, rejected-candidate rows, per-interval review results, "
            "scores, local paths, "
            "and source hashes are intentionally not distributed."
        ),
    }
