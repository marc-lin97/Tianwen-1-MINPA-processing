"""Build and freeze the all-approved, time-invariant MINPA v2.2.0 bundle."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from highE.minpa_background import (  # noqa: E402
    BackgroundConfig,
    MODE1_SHAPE,
    SUPPORT_LOW,
    SUPPORT_SUPPORTED,
    SUPPORT_UNSUPPORTED,
    approved_intervals_from_review_inventory,
    concatenate_mode1_records,
    estimate_background,
    load_background_model,
    mode1_ori_files_for_interval,
    read_mode1_ori_records,
    save_background_model,
    sha256_file,
)
from highE.minpa_multimode_background import load_multimode_background_model  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/minpa_unified_static_channel_denoise_v2.2.0.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _review_decisions(workspace: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Infer retained/deleted decisions from a hash-locked flat review folder."""

    index_path = workspace / "indexes/candidate_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = list(payload["candidates"])
    by_name = {Path(row["plot_relative_path"]).name: row for row in rows}
    if len(by_name) != len(rows):
        raise RuntimeError(f"Duplicate review PNG filename in {index_path}")
    actual_paths = sorted((workspace / "pending").glob("*.png"))
    actual = {path.name for path in actual_paths}
    unknown = sorted(actual - set(by_name))
    nested = sorted((workspace / "pending").glob("**/*/*.png"))
    modified = [
        path.name for path in actual_paths
        if sha256_file(path) != str(by_name[path.name]["plot_sha256"])
    ]
    if unknown or nested or modified:
        raise RuntimeError(
            f"Invalid reviewed workspace: unknown={unknown[:5]} nested={len(nested)} "
            f"modified={modified[:5]}"
        )
    approved = [row for name, row in by_name.items() if name in actual]
    rejected = [row for name, row in by_name.items() if name not in actual]
    if not approved:
        raise RuntimeError(f"Reviewed workspace has no retained PNG decisions: {workspace}")
    return approved, rejected


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if sha256_file(source) != expected_sha256:
        raise RuntimeError(f"Source hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        raise RuntimeError(f"Copied hash mismatch: {destination}")
    temporary.replace(destination)


def _time_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["start_utc"]), str(row["stop_utc"])


def consolidate_approved_rows(
    approved: Sequence[tuple[str, Mapping[str, Any]]],
    rejected: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact-time deduplicate approvals while preserving aliases and reject collisions."""

    grouped: defaultdict[tuple[str, str], list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for source, row in approved:
        grouped[_time_key(row)].append((source, row))
    rejected_lookup: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for source, row in rejected:
        rejected_lookup[_time_key(row)].append(source)

    intervals: list[dict[str, Any]] = []
    reject_collisions: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(grouped), start=1):
        aliases = grouped[key]
        start, stop = key
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        stop_dt = datetime.fromisoformat(stop.replace("Z", "+00:00"))
        duration_s = (stop_dt - start_dt).total_seconds()
        if not 120.0 <= duration_s <= 600.0:
            raise ValueError(f"Approved interval duration outside 2-10 min: {key}")
        alias_rows = [
            {
                "review_source": source,
                "label": row.get("label"),
                "candidate_id": row.get("candidate_id"),
            }
            for source, row in aliases
        ]
        intervals.append({
            "label": f"all-approved-{index:04d}",
            "start_utc": start,
            "stop_utc": stop,
            "duration_s": duration_s,
            "source": "finalized_manual_reviews_all_approved_static_v2.2.0",
            "approval_aliases": alias_rows,
        })
        if key in rejected_lookup:
            reject_collisions.append({
                "start_utc": start,
                "stop_utc": stop,
                "approved_aliases": alias_rows,
                "rejected_review_sources": sorted(set(rejected_lookup[key])),
                "resolution": (
                    "retained because the authoritative v1.1 inventory explicitly lists the "
                    "physical interval in approved_intervals; rejected aliases remain provenance"
                ),
            })

    overlap_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(intervals):
        left_stop = datetime.fromisoformat(left["stop_utc"].replace("Z", "+00:00"))
        for right in intervals[left_index + 1:]:
            right_start = datetime.fromisoformat(right["start_utc"].replace("Z", "+00:00"))
            if right_start >= left_stop:
                break
            right_stop = datetime.fromisoformat(right["stop_utc"].replace("Z", "+00:00"))
            left_start = datetime.fromisoformat(left["start_utc"].replace("Z", "+00:00"))
            overlap_pairs.append({
                "left_label": left["label"],
                "right_label": right["label"],
                "overlap_s": (
                    min(left_stop, right_stop) - max(left_start, right_start)
                ).total_seconds(),
            })
    return intervals, {
        "imported_approved_count": len(approved),
        "exact_time_duplicate_count": len(approved) - len(intervals),
        "unique_approved_count": len(intervals),
        "exact_time_approved_rejected_alias_collision_count": len(reject_collisions),
        "exact_time_approved_rejected_alias_collisions": reject_collisions,
        "overlapping_nonidentical_approved_pair_count": len(overlap_pairs),
        "overlapping_nonidentical_approved_pairs": overlap_pairs,
    }


def _review_inventory(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    review_config_path = _resolve(config["mode1_review_sources_config"])
    review_config = json.loads(review_config_path.read_text(encoding="utf-8"))
    approved: list[tuple[str, Mapping[str, Any]]] = []
    rejected: list[tuple[str, Mapping[str, Any]]] = []
    provenance: list[dict[str, Any]] = []
    for value in review_config["review_sources"]:
        source = _resolve(value)
        if source.is_dir():
            source_approved, source_rejected = _review_decisions(source)
            index_path = source / "indexes/candidate_index.json"
            provenance.append({
                "path": str(source.resolve()),
                "candidate_index_sha256": sha256_file(index_path),
                "approved": len(source_approved),
                "rejected": len(source_rejected),
                "decision_source": "retained/deleted joint review PNGs",
            })
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload.get("review_status") != "finalized":
                raise ValueError(f"Review source is not finalized: {source}")
            source_approved = list(payload.get("approved_intervals", []))
            source_rejected = list(payload.get("rejected_intervals", []))
            provenance.append({
                "path": str(source.resolve()),
                "sha256": sha256_file(source),
                "approved": len(source_approved),
                "rejected": len(source_rejected),
                "decision_source": "finalized review inventory",
            })
        approved.extend((str(source.resolve()), row) for row in source_approved)
        rejected.extend((str(source.resolve()), row) for row in source_rejected)
    intervals, audit = consolidate_approved_rows(approved, rejected)
    return {
        "review_status": "finalized",
        "created_utc": _utc_now(),
        "release_version": config["release_version"],
        "decision_rule": "joint H+/O+/O2+ approval; exact UTC windows deduplicated",
        "counts": {"approved": len(intervals), "rejected": len(rejected)},
        "approved_intervals": intervals,
        "review_source_provenance": provenance,
        "audit": audit,
    }, provenance


def _support_counts(level: np.ndarray) -> dict[str, int]:
    return {
        "unsupported": int(np.count_nonzero(level == SUPPORT_UNSUPPORTED)),
        "low_support": int(np.count_nonzero(level == SUPPORT_LOW)),
        "supported": int(np.count_nonzero(level == SUPPORT_SUPPORTED)),
    }


def _compare_mode1(old_path: Path, new_model: Any) -> dict[str, Any]:
    old = load_background_model(old_path)
    finite = np.isfinite(old.background_dpf) & np.isfinite(new_model.background_dpf)
    ratio = np.divide(
        new_model.background_dpf,
        old.background_dpf,
        out=np.full(MODE1_SHAPE, np.nan),
        where=finite & (old.background_dpf > 0.0),
    )
    values = ratio[np.isfinite(ratio)]
    return {
        "previous_model_path": str(old_path.resolve()),
        "previous_model_sha256": sha256_file(old_path),
        "comparable_channel_count": int(values.size),
        "new_to_old_background_ratio_median": float(np.median(values)),
        "new_to_old_background_ratio_p05": float(np.quantile(values, 0.05)),
        "new_to_old_background_ratio_p95": float(np.quantile(values, 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen" or config["temporal_policy"] != "none":
        raise ValueError("Release must be frozen and time invariant")
    output = _resolve(config["output_root"])
    inventory, review_provenance = _review_inventory(config)
    inventory_path = output / "review_inventory/mode1_all_approved.json"
    _atomic_json(inventory_path, inventory)
    intervals = approved_intervals_from_review_inventory(inventory)

    ori_root = _resolve(config["ori_root"])
    quality_root = _resolve(config["quality_root"])
    parts = []
    loaded_record_count = 0
    for index, interval in enumerate(intervals, start=1):
        paths = mode1_ori_files_for_interval(ori_root, interval.start_utc, interval.stop_utc)
        if not paths:
            raise FileNotFoundError(f"No Mode-1 ori for {interval.label}")
        part = read_mode1_ori_records(
            paths, interval.start_utc, interval.stop_utc,
            quality_root=quality_root, compute_sha256=True,
        )
        parts.append(part)
        loaded_record_count += int(part.time_unix_s.size)
        if index % 25 == 0 or index == len(intervals):
            print(f"mode1_read={index}/{len(intervals)}", flush=True)
    records = concatenate_mode1_records(parts)
    model_config = replace(
        BackgroundConfig(**config["mode1_model"]),
        paper_pdf_path=str(_resolve(config["paper_pdf"]).resolve()),
    )
    model = estimate_background(records, intervals, model_config)
    if not model.valid:
        raise RuntimeError(f"Mode-1 model invalid: {model.invalid_reasons}")
    model_path = output / "models/minpa_mode01_background_v1.2.0.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(".tmp.npz")
    save_background_model(temporary_model, model)
    loaded = load_background_model(temporary_model)
    if loaded.background_dpf.shape != MODE1_SHAPE or not loaded.valid:
        raise RuntimeError("Saved Mode-1 model failed round-trip validation")
    temporary_model.replace(model_path)

    multimode_bundle_path = _resolve(config["multimode_bundle"])
    multimode_bundle = json.loads(multimode_bundle_path.read_text(encoding="utf-8"))
    model_items: dict[str, dict[str, Any]] = {
        "1": {
            "path": "models/minpa_mode01_background_v1.2.0.npz",
            "sha256": sha256_file(model_path),
            "model_kind": "mode1",
            "algorithm_version": model.algorithm_version,
            "policy_version": config["mode1_policy_version"],
        }
    }
    for mode in (4, 12):
        item = multimode_bundle["models"][str(mode)]
        source = Path(item["path"])
        if not source.is_absolute():
            source = multimode_bundle_path.parent / source
        destination = output / f"models/minpa_mode{mode:02d}_background_v2.0.0.npz"
        _copy_verified(source, destination, str(item["sha256"]))
        loaded_multimode = load_multimode_background_model(destination)
        if loaded_multimode.mode != mode or not loaded_multimode.valid:
            raise RuntimeError(f"Mode {mode} copied model failed validation")
        model_items[str(mode)] = {
            "path": f"models/{destination.name}",
            "sha256": str(item["sha256"]),
            "model_kind": "multimode",
            "algorithm_version": str(item["algorithm_version"]),
        }

    support = _support_counts(model.support_level)
    accepted_summaries = [row for row in model.interval_summaries if row["accepted"]]
    sanity = {
        "mode1_shape": list(model.background_dpf.shape),
        "mode1_units": model.units,
        "mode1_model_valid": model.valid,
        "mode1_nonnegative_finite_background": bool(
            np.all(model.background_dpf[np.isfinite(model.background_dpf)] >= 0.0)
        ),
        "mode1_support_counts": support,
        "mode1_support_total": int(sum(support.values())),
        "mode1_loaded_records_before_epoch_dedup": loaded_record_count,
        "mode1_unique_epoch_records": int(records.time_unix_s.size),
        "mode1_duplicate_epoch_records_removed": loaded_record_count - int(records.time_unix_s.size),
        "mode1_quality_accepted_intervals": len(accepted_summaries),
        "mode1_quality_accepted_records": int(sum(int(row["kept_records"]) for row in accepted_summaries)),
        "mode1_comparison": _compare_mode1(_resolve(config["previous_mode1_model"]), model),
    }
    if sanity["mode1_support_total"] != int(np.prod(MODE1_SHAPE)):
        raise RuntimeError("Mode-1 support accounting mismatch")
    if sanity["mode1_quality_accepted_intervals"] != len(intervals):
        raise RuntimeError("One or more manually approved Mode-1 intervals failed quality validation")

    bundle = {
        "bundle_version": config["release_version"],
        "status": "frozen",
        "created_utc": _utc_now(),
        "units": "1/(s cm^2 sr eV)",
        "dimension_order": ["energy", "pitch", "azimuth", "mass"],
        "primary_estimator": (
            "b_jc=mean(x_tc | finite x_tc>0); B_c=mean_j(b_jc), "
            "equal weight for every unique approved interval"
        ),
        "correction": config["correction"],
        "temporal_policy": "none; one static channel background per mode",
        "support_policy": {
            "unsupported": "preserve raw",
            "low_support": "subtract and flag",
            "supported": "subtract",
            "interpolation": "forbidden",
            "cross_mass_substitution": "forbidden",
        },
        "mode7_policy": "preserve raw",
        "default_policy": config["default_policy"],
        "production_policy": config["production_policy"],
        "models": model_items,
        "review_inventories": {
            "mode1": {
                "path": "review_inventory/mode1_all_approved.json",
                "sha256": sha256_file(inventory_path),
                "imported_approved_count": inventory["audit"]["imported_approved_count"],
                "unique_approved_count": inventory["audit"]["unique_approved_count"],
            },
            "mode04_mode12": {
                "path": multimode_bundle["review_inventory"],
                "sha256": multimode_bundle["review_inventory_sha256"],
                "approved_species_intervals": 2901,
            },
        },
    }
    bundle_path = output / "bundle.json"
    _atomic_json(bundle_path, bundle)
    release_manifest = {
        "release_version": config["release_version"],
        "status": "frozen",
        "created_utc": _utc_now(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "build_script_path": str(Path(__file__).resolve()),
        "build_script_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_files": [
            {
                "path": str((ROOT / relative).resolve()),
                "sha256": sha256_file(ROOT / relative),
            }
            for relative in (
                "src/highE/minpa_background.py",
                "src/highE/minpa_multimode_background.py",
                "src/highE/tw1_minpa.py",
                "scripts/run_tw1_highE_day.py",
            )
        ],
        "bundle_path": str(bundle_path.resolve()),
        "bundle_sha256": sha256_file(bundle_path),
        "review_source_provenance": review_provenance,
        "review_audit": inventory["audit"],
        "sanity_checks": sanity,
        "frozen_models": model_items,
        "scientific_scope": {
            "time_variation": "not modeled",
            "raw_data_mutation": "none",
            "mode1_species_approval": "joint H+/O+/O2+",
            "mode04_mode12_species_approval": "independent by species",
        },
    }
    manifest_path = output / "release_manifest.json"
    _atomic_json(manifest_path, release_manifest)
    readme = output / "README.md"
    readme.write_text(
        "# MINPA unified static channel denoise v2.2.0\n\n"
        "Frozen all-approved release. Mode 1 is rebuilt from 295 approved entries "
        "deduplicated to 273 exact UTC intervals; Mode 4/12 reuse the hash-locked "
        "2,901 species-specific approved-interval models. No temporal scaling is used.\n\n"
        "Correction is `max(raw-background, 0)` in DPF `1/(s cm^2 sr eV)`. "
        "NaNs are preserved; unsupported channels, Mode 7, and unreviewed mass bins "
        "remain unchanged. No smoothing, interpolation, or cross-mass substitution is allowed.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "release": config["release_version"],
        "bundle_sha256": sha256_file(bundle_path),
        "mode1_model_sha256": sha256_file(model_path),
        "mode1_intervals": len(intervals),
        "support_counts": support,
        "output": str(output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
