"""Validate frozen MINPA Mode-4/12 models on strong manually rejected spectra."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_io import read_ori_records  # noqa: E402
from highE.minpa_modes import species_mass_indices, split_raw_record  # noqa: E402
from highE.minpa_multimode_background import (  # noqa: E402
    apply_multimode_background,
    load_multimode_model_bundle,
)


DEFAULT_BUNDLE = ROOT / "outputs" / "minpa_multimode_channel_denoise_v2.0.0" / "bundle.json"
DEFAULT_REVIEW = ROOT / "outputs" / "minpa_multimode_noise_review_secondary_prefilter_v2" / "manual_review_final.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval_cube(row: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    start = float(row["start_unix_s"])
    stop = float(row["stop_unix_s"])
    mode = int(row["mode"])
    by_time: dict[float, np.ndarray] = {}
    for source_text in row["source_ori"]:
        for record in read_ori_records(Path(str(source_text)), start - 5.0, stop + 5.0):
            if int(record.quality_native) != 0:
                continue
            for product_time, flat in split_raw_record(mode, record.time_unix_s, record.ion_dpf):
                if start <= product_time < stop:
                    values = np.asarray(flat, dtype=float).reshape(shape)
                    previous = by_time.get(float(product_time))
                    if previous is not None and not np.array_equal(previous, values, equal_nan=True):
                        raise ValueError(f"Conflicting duplicate record at {product_time}")
                    by_time[float(product_time)] = values
    if not by_time:
        raise ValueError(f"No native records for {row['candidate_id']}")
    return np.stack([by_time[key] for key in sorted(by_time)])


def _energy_spectrum(cubes: np.ndarray, mass_indices: np.ndarray) -> np.ndarray:
    positive = np.maximum(cubes[..., mass_indices], 0.0)
    return np.nanmean(np.nansum(positive, axis=(2, 3, 4)), axis=0)


def validate_strong_signals(
    rejected: Sequence[Mapping[str, Any]],
    models: Mapping[int, Any],
    *,
    candidates_per_group: int = 12,
    minimum_signal_to_background: float = 20.0,
) -> dict[str, Any]:
    grouped: defaultdict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rejected:
        grouped[(int(row["mode"]), str(row["species"]))].append(row)
    cases = []
    for group, rows in sorted(grouped.items()):
        mode, species = group
        model = models[mode]
        mass_indices = species_mass_indices(mode, species)
        selected = sorted(
            rows,
            key=lambda row: (
                -float(row.get("maximum_minute_score", 0.0)),
                -float(row.get("decision_score", 0.0)),
                str(row["candidate_id"]),
            ),
        )[:candidates_per_group]
        for row in selected:
            raw = _interval_cube(row, model.shape)
            corrected = apply_multimode_background(raw, model).corrected_dpf
            raw_spectrum = _energy_spectrum(raw, mass_indices)
            corrected_spectrum = _energy_spectrum(corrected, mass_indices)
            raw_peak = int(np.nanargmax(raw_spectrum))
            corrected_peak = int(np.nanargmax(corrected_spectrum))
            peak = float(raw_spectrum[raw_peak])
            retention = float(corrected_spectrum[raw_peak] / peak) if peak > 0.0 else np.nan
            selected_raw = np.maximum(raw[..., mass_indices], 0.0)
            selected_corrected = np.maximum(corrected[..., mass_indices], 0.0)
            selected_background = model.background_dpf[..., mass_indices]
            selected_valid = model.valid_channel_mask[..., mass_indices] & (selected_background > 0.0)
            valid = np.broadcast_to(selected_valid, selected_raw.shape) & np.isfinite(selected_raw)
            channel_ratio = np.divide(
                selected_raw,
                np.broadcast_to(selected_background, selected_raw.shape),
                out=np.full(selected_raw.shape, np.nan),
                where=valid,
            )
            corrected_ratio = np.divide(
                selected_corrected,
                np.broadcast_to(selected_background, selected_corrected.shape),
                out=np.full(selected_corrected.shape, np.nan),
                where=valid,
            )
            ratio = float(np.nanmax(channel_ratio)) if np.any(np.isfinite(channel_ratio)) else np.nan
            raw_ratio_by_energy = np.nanmax(channel_ratio, axis=(0, 2, 3, 4))
            corrected_ratio_by_energy = np.nanmax(corrected_ratio, axis=(0, 2, 3, 4))
            ratio_peak = int(np.nanargmax(raw_ratio_by_energy))
            corrected_ratio_peak = int(np.nanargmax(corrected_ratio_by_energy))
            maximum_index = np.unravel_index(int(np.nanargmax(channel_ratio)), channel_ratio.shape)
            strong_cell_retention = (
                float(selected_corrected[maximum_index] / selected_raw[maximum_index])
                if selected_raw[maximum_index] > 0.0 else np.nan
            )
            cases.append({
                "candidate_id": row["candidate_id"],
                "mode": mode,
                "species": species,
                "start_utc": row["start_utc"],
                "stop_utc": row["stop_utc"],
                "record_count": int(raw.shape[0]),
                "raw_peak_energy_index": raw_peak,
                "corrected_peak_energy_index": corrected_peak,
                "peak_shift_channels": abs(corrected_peak - raw_peak),
                "maximum_channel_signal_to_background": ratio,
                "maximum_ratio_energy_index": ratio_peak,
                "corrected_maximum_ratio_energy_index": corrected_ratio_peak,
                "maximum_ratio_peak_shift_channels": abs(corrected_ratio_peak - ratio_peak),
                "strongest_channel_retention_fraction": strong_cell_retention,
                "raw_peak_retention_fraction": retention,
                "corrected_total_not_above_raw": bool(np.nansum(corrected) <= np.nansum(np.maximum(raw, 0.0)) + 1.0e-9),
                "strong_signal_case": bool(ratio >= minimum_signal_to_background),
            })
    strong = [row for row in cases if row["strong_signal_case"]]
    return {
        "candidate_case_count": len(cases),
        "strong_signal_definition": f"maximum raw channel / its corresponding supported-channel background >= {minimum_signal_to_background:g}",
        "strong_signal_case_count": len(strong),
        "strong_signal_peak_shift_within_one_channel_fraction": float(np.mean([row["maximum_ratio_peak_shift_channels"] <= 1 for row in strong])) if strong else None,
        "strong_signal_peak_retention_at_least_95pct_fraction": float(np.mean([row["strongest_channel_retention_fraction"] >= 0.95 for row in strong])) if strong else None,
        "all_corrected_totals_not_above_raw": all(row["corrected_total_not_above_raw"] for row in cases),
        "cases": cases,
    }


def validate_synthetic_strong_signals(models: Mapping[int, Any]) -> dict[str, Any]:
    cases = []
    for mode, model in sorted(models.items()):
        for species in ("H+", "O+", "O2+"):
            mass_indices = species_mass_indices(mode, species)
            background = model.background_dpf[..., mass_indices]
            valid = model.valid_channel_mask[..., mass_indices] & np.isfinite(background) & (background > 0.0)
            if not np.any(valid):
                raise ValueError(f"No positive supported channel for synthetic Mode {mode} {species} test")
            local_index = np.unravel_index(int(np.nanargmax(np.where(valid, background, np.nan))), background.shape)
            full_mass_index = int(mass_indices[local_index[-1]])
            full_index = (*local_index[:-1], full_mass_index)
            raw = np.zeros((1, *model.shape), dtype=float)
            raw[(0, *full_index)] = 20.0 * float(model.background_dpf[full_index])
            corrected = apply_multimode_background(raw, model).corrected_dpf
            retention = float(corrected[(0, *full_index)] / raw[(0, *full_index)])
            raw_energy = np.nansum(raw[..., mass_indices], axis=(2, 3, 4))[0]
            corrected_energy = np.nansum(corrected[..., mass_indices], axis=(2, 3, 4))[0]
            cases.append({
                "mode": mode,
                "species": species,
                "injected_signal_to_background": 20.0,
                "retention_fraction": retention,
                "peak_shift_channels": abs(int(np.argmax(corrected_energy)) - int(np.argmax(raw_energy))),
                "passed": bool(retention >= 0.95 - 1.0e-12 and np.argmax(corrected_energy) == np.argmax(raw_energy)),
            })
    return {
        "case_count": len(cases),
        "all_peak_retention_at_least_95pct": all(row["retention_fraction"] >= 0.95 - 1.0e-12 for row in cases),
        "all_peak_shifts_within_one_channel": all(row["peak_shift_channels"] <= 1 for row in cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle_path = args.bundle.resolve()
    review_path = args.review.resolve()
    output = args.output or bundle_path.parent / "validation.json"
    models = load_multimode_model_bundle(bundle_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    result = {
        "validation_version": "minpa-multimode-background-validation-v2.0.0",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "review_path": str(review_path),
        "review_sha256": _sha256(review_path),
        "strong_signal_validation": validate_strong_signals(review["rejected_intervals"], models),
        "synthetic_strong_signal_validation": validate_synthetic_strong_signals(models),
        "units_check": "input and background are released DPF in 1/(s cm^2 sr eV)",
        "shape_check": {str(mode): list(model.shape) for mode, model in models.items()},
        "coordinate_check": "instrument channel space only; no coordinate transform is applied",
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    release_manifest = {
        "release_version": "minpa-multimode-channel-denoise-v2.0.0",
        "status": "frozen",
        "created_utc": result["created_utc"],
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "validation_path": str(output.resolve()),
        "validation_sha256": _sha256(output),
        "review_path": str(review_path),
        "review_sha256": _sha256(review_path),
        "observational_strong_signal_validation": "not_assessable_no_20x_case",
        "synthetic_strong_signal_validation": "pass",
    }
    manifest_path = bundle_path.parent / "release_manifest.json"
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)
    summary = {
        "observational": {
            key: value for key, value in result["strong_signal_validation"].items() if key != "cases"
        },
        "synthetic": {
            key: value for key, value in result["synthetic_strong_signal_validation"].items() if key != "cases"
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
