"""Finalize the file-presence review for MINPA v2.2.0 signal examples."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping


EXPECTED_GROUPS = {
    (mode, species)
    for mode in (1, 4, 12)
    for species in ("H+", "O+", "O2+")
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _basename(value: str) -> str:
    """Return a portable basename for either POSIX or Windows provenance paths."""

    return PureWindowsPath(value).name if "\\" in value else Path(value).name


def _portable_signal_row(row: Mapping[str, Any]) -> dict[str, Any]:
    validation = dict(row["native_validation"])
    return {
        "mode": int(row["mode"]),
        "species": str(row["species"]),
        "candidate_id": str(row["candidate_id"]),
        "start_utc": str(row["start_utc"]),
        "stop_utc": str(row["stop_utc"]),
        "duration_s": float(row["duration_s"]),
        "figure": f"signal/{row['plot_filename']}",
        "figure_sha256": str(row["plot_sha256"]),
        "strength_tier": str(validation["strength_tier"]),
        "maximum_median_channel_signal_to_background": float(
            validation["maximum_median_channel_signal_to_background"]
        ),
        "signal_peak_retention_fraction": float(
            validation["signal_peak_retention_fraction"]
        ),
        "ratio_peak_shift_channels": int(validation["ratio_peak_shift_channels"]),
        "source_ori": [
            {"filename": _basename(str(item["path"])), "sha256": str(item["sha256"])}
            for item in row.get("context_source_ori", [])
        ],
    }


def finalize_signal_example_review(
    review_index: Mapping[str, Any],
    pending_dir: Path,
) -> dict[str, Any]:
    """Infer final decisions from retained PNGs and fail closed on drift.

    Exactly one retained, byte-identical signal PNG is required for every
    Mode 1/4/12 and H+/O+/O2+ group. Missing expected files are rejected.
    Unexpected, renamed, modified, or nested PNGs are errors.
    """

    if review_index.get("status") != "pending_manual_signal_review":
        raise ValueError("Review index is not a pending manual signal review")
    groups = list(review_index.get("groups", []))
    group_keys = {(int(row["mode"]), str(row["species"])) for row in groups}
    if group_keys != EXPECTED_GROUPS or len(groups) != len(EXPECTED_GROUPS):
        raise ValueError("Review index must contain each Mode 1/4/12 and species group once")

    candidates: dict[str, Mapping[str, Any]] = {}
    group_for_filename: dict[str, tuple[int, str]] = {}
    noise_by_group: dict[tuple[int, str], Mapping[str, Any]] = {}
    for group in groups:
        key = (int(group["mode"]), str(group["species"]))
        noise_by_group[key] = group["noise_reference"]
        for row in group.get("signal_candidates", []):
            filename = str(row["plot_filename"])
            if filename in candidates:
                raise ValueError(f"Duplicate candidate filename: {filename}")
            if (int(row["mode"]), str(row["species"])) != key:
                raise ValueError(f"Candidate group mismatch: {filename}")
            candidates[filename] = row
            group_for_filename[filename] = key
    expected_count = int(review_index.get("pending_png_count", len(candidates)))
    if len(candidates) != expected_count:
        raise ValueError("Candidate count disagrees with pending_png_count")

    direct_pngs = {path.name: path for path in pending_dir.glob("*.png")}
    recursive_pngs = list(pending_dir.rglob("*.png"))
    if len(recursive_pngs) != len(direct_pngs):
        raise ValueError("Nested PNGs are not allowed in the flat review directory")
    extra = sorted(set(direct_pngs) - set(candidates))
    if extra:
        raise ValueError(f"Unexpected or renamed review PNGs: {extra}")

    retained: dict[tuple[int, str], list[Mapping[str, Any]]] = {
        key: [] for key in EXPECTED_GROUPS
    }
    rejected: list[Mapping[str, Any]] = []
    for filename, row in candidates.items():
        path = direct_pngs.get(filename)
        if path is None:
            rejected.append(row)
            continue
        actual_hash = sha256_file(path)
        if actual_hash != str(row["plot_sha256"]):
            raise ValueError(f"Retained PNG hash mismatch: {filename}")
        retained[group_for_filename[filename]].append(row)
    invalid_counts = {
        f"mode{mode:02d}_{species}": len(rows)
        for (mode, species), rows in sorted(retained.items())
        if len(rows) != 1
    }
    if invalid_counts:
        raise ValueError(f"Exactly one retained signal is required per group: {invalid_counts}")

    approved_rows = [
        _portable_signal_row(retained[key][0]) for key in sorted(EXPECTED_GROUPS)
    ]
    rejected_rows = [
        {
            "mode": int(row["mode"]),
            "species": str(row["species"]),
            "candidate_id": str(row["candidate_id"]),
            "start_utc": str(row["start_utc"]),
            "stop_utc": str(row["stop_utc"]),
            "expected_filename": str(row["plot_filename"]),
            "decision": "rejected_by_deleted_png",
        }
        for row in sorted(
            rejected,
            key=lambda value: (
                int(value["mode"]), str(value["species"]), str(value["candidate_id"])
            ),
        )
    ]
    noise_rows = []
    for key in sorted(EXPECTED_GROUPS):
        mode, species = key
        row = noise_by_group[key]
        noise_rows.append({
            "mode": mode,
            "species": species,
            "start_utc": str(row["start_utc"]),
            "stop_utc": str(row["stop_utc"]),
            "figure": f"noise/{_basename(str(row['figure']))}",
            "figure_sha256": str(row["figure_sha256"]),
            "decision": "manually_approved_noise",
        })

    return {
        "artifact_version": "minpa-typical-noise-signal-examples-v2.2.0",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "finalized_manual_review",
        "decision_rule": (
            "Retained byte-identical PNG=approved; missing expected PNG=rejected; "
            "exactly one approved signal per mode/species group"
        ),
        "bundle_sha256": str(review_index["bundle"]["sha256"]),
        "units": str(review_index["units"]),
        "smoothing_or_interpolation": str(review_index["smoothing_or_interpolation"]),
        "counts": {
            "mode_species_groups": len(EXPECTED_GROUPS),
            "approved_noise_examples": len(noise_rows),
            "approved_signal_examples": len(approved_rows),
            "rejected_signal_candidates": len(rejected_rows),
        },
        "noise_examples": noise_rows,
        "signal_examples": approved_rows,
        "rejected_signal_candidates": rejected_rows,
    }
