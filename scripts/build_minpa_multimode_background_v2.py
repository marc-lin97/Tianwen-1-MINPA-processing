"""Build and freeze finalized MINPA Mode-4/12 paper-channel background models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import SUPPORT_LOW, SUPPORT_SUPPORTED, SUPPORT_UNSUPPORTED, sha256_file  # noqa: E402
from highE.minpa_io import read_ori_records  # noqa: E402
from highE.minpa_modes import mode_layout, species_mass_indices, split_raw_record  # noqa: E402
from highE.minpa_multimode_background import (  # noqa: E402
    MULTIMODE_DENOISE_VERSION,
    MULTIMODE_MODEL_VERSION,
    SUPPORTED_MODES,
    SUPPORTED_SPECIES,
    apply_multimode_background,
    assemble_multimode_background_from_summaries,
    load_multimode_background_model,
    load_multimode_model_bundle,
    mode_channel_shape,
    save_multimode_background_model,
)


DEFAULT_CONFIG = ROOT / "config" / "minpa_multimode_background_v2.0.0.json"
GROUP_TOKENS = {
    (mode, species): f"m{mode:02d}_{species.replace('+', 'plus')}"
    for mode in SUPPORTED_MODES for species in SUPPORTED_SPECIES
}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _extract_arrays(source_path: str, references: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = Path(source_path)
    mode_values = {int(item["mode"]) for item in references}
    if len(mode_values) != 1:
        raise ValueError(f"Mixed modes for source {source}")
    mode = mode_values.pop()
    shape = mode_channel_shape(mode)
    by_group: defaultdict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in references:
        by_group[(mode, str(item["species"]))].append(item)
    for items in by_group.values():
        items.sort(key=lambda item: (float(item["start_unix_s"]), float(item["stop_unix_s"])))

    values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    interval_indices: defaultdict[str, list[int]] = defaultdict(list)
    times: defaultdict[str, list[float]] = defaultdict(list)
    raw_records = read_ori_records(source)
    invalid_shape_count = 0
    native_quality_rejected = 0
    for record in raw_records:
        if int(record.mode) != mode:
            raise ValueError(f"Filename/record mode mismatch in {source}")
        if int(record.quality_native) != 0:
            native_quality_rejected += 1
            continue
        try:
            subrecords = split_raw_record(mode, record.time_unix_s, record.ion_dpf)
        except ValueError:
            invalid_shape_count += 1
            continue
        for product_time, flat in subrecords:
            cube = np.asarray(flat, dtype=float).reshape(shape)
            for group, items in by_group.items():
                matches = [
                    item for item in items
                    if float(item["start_unix_s"]) <= product_time < float(item["stop_unix_s"])
                ]
                if len(matches) > 1:
                    raise ValueError(f"Overlapping approved intervals in {group} at {product_time}")
                if not matches:
                    continue
                item = matches[0]
                mass_indices = species_mass_indices(mode, group[1])
                token = GROUP_TOKENS[group]
                values[token].append(cube[..., mass_indices].reshape(-1))
                interval_indices[token].append(int(item["group_interval_index"]))
                times[token].append(float(product_time))
    payload: dict[str, Any] = {
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_size": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "mode": mode,
        "raw_record_count": len(raw_records),
        "invalid_shape_count": invalid_shape_count,
        "native_quality_rejected_count": native_quality_rejected,
        "rows_by_group": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for group, token in GROUP_TOKENS.items():
        if group[0] != mode or not values[token]:
            continue
        matrix = np.asarray(values[token], dtype=float)
        arrays[f"values__{token}"] = matrix
        arrays[f"interval__{token}"] = np.asarray(interval_indices[token], dtype=np.int32)
        arrays[f"time__{token}"] = np.asarray(times[token], dtype=float)
        payload["rows_by_group"][token] = int(matrix.shape[0])
    return {"metadata": payload, "arrays": arrays}


def _digest_extraction(task: tuple[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    result = _extract_arrays(*task)
    digests = {}
    for name, array in sorted(result["arrays"].items()):
        digest = hashlib.sha256()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.ascontiguousarray(array).view(np.uint8))
        digests[name] = digest.hexdigest()
    return {"source_path": result["metadata"]["source_path"], "digests": digests}


def _write_chunk(task: tuple[str, Sequence[Mapping[str, Any]], str]) -> dict[str, Any]:
    source, references, destination_text = task
    destination = Path(destination_text)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        if metadata["source_path"] != str(Path(source).resolve()):
            raise ValueError(f"Chunk/source mismatch: {destination}")
        if metadata["source_sha256"] != sha256_file(Path(source)):
            raise ValueError(f"Source changed since resumable chunk was written: {source}")
        return metadata
    result = _extract_arrays(source, references)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.array(json.dumps(result["metadata"], ensure_ascii=False)),
            **result["arrays"],
        )
    temporary.replace(destination)
    return result["metadata"]


def _build_references(approved: Sequence[Mapping[str, Any]]) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[tuple[int, str], list[dict[str, Any]]],
]:
    grouped: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for source in approved:
        mode = int(source["mode"])
        species = str(source["species"])
        if mode not in SUPPORTED_MODES or species not in SUPPORTED_SPECIES:
            raise ValueError(f"Unsupported approved group: {(mode, species)}")
        duration = float(source["duration_s"])
        if not 300.0 <= duration <= 600.0:
            raise ValueError(f"Approved duration is not 5-10 min: {source['candidate_id']}")
        grouped[(mode, species)].append(dict(source))
    for group, rows in grouped.items():
        rows.sort(key=lambda row: (float(row["start_unix_s"]), str(row["candidate_id"])))
        for index, row in enumerate(rows):
            row["group_interval_index"] = index
        if any(
            float(right["start_unix_s"]) < float(left["stop_unix_s"])
            for left, right in zip(rows, rows[1:])
        ):
            raise ValueError(f"Overlapping approved intervals in {group}")
    by_file: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in grouped.values():
        for row in rows:
            for source_path in row["source_ori"]:
                path = Path(str(source_path)).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                by_file[str(path)].append({
                    "mode": row["mode"],
                    "species": row["species"],
                    "start_unix_s": row["start_unix_s"],
                    "stop_unix_s": row["stop_unix_s"],
                    "group_interval_index": row["group_interval_index"],
                })
    return dict(by_file), dict(grouped)


def _benchmark(
    tasks: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    parallel_workers: int,
    speedup_required: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    single = [_digest_extraction(task) for task in tasks]
    single_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        parallel = list(executor.map(_digest_extraction, tasks))
    parallel_seconds = time.perf_counter() - start
    if single != parallel:
        raise RuntimeError("Single/parallel extraction benchmark is not deterministic")
    speedup = single_seconds / parallel_seconds if parallel_seconds > 0 else math.inf
    return {
        "file_count": len(tasks),
        "single_process_seconds": single_seconds,
        "parallel_process_seconds": parallel_seconds,
        "parallel_workers": parallel_workers,
        "speedup": speedup,
        "required_speedup": speedup_required,
        "values_shapes_and_hashes_identical": True,
        "selected_workers": parallel_workers if speedup >= speedup_required else 1,
    }


def _exact_pooled_median_mad(
    pool: np.memmap,
    rows: int,
    channels: int,
    *,
    chunk_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.full(channels, np.nan)
    mad = np.full(channels, np.nan)
    for start in range(0, channels, chunk_channels):
        stop = min(channels, start + chunk_channels)
        block = np.asarray(pool[:rows, start:stop], dtype=float)
        positive = np.where(np.isfinite(block) & (block > 0.0), block, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            center = np.nanmedian(positive, axis=0)
            spread = np.nanmedian(np.abs(positive - center[None, :]), axis=0)
        median[start:stop] = center
        mad[start:stop] = spread
    return median, mad


def _aggregate(
    chunk_paths: Sequence[Path],
    grouped_intervals: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    work_root: Path,
    *,
    median_channel_chunk: int,
) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    rows_expected = Counter()
    source_metadata = []
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        source_metadata.append(metadata)
        rows_expected.update(metadata["rows_by_group"])
        if int(metadata["invalid_shape_count"]) != 0:
            raise ValueError(f"Invalid ori science shape in {metadata['source_path']}")

    state: dict[tuple[int, str], dict[str, Any]] = {}
    for group, intervals in grouped_intervals.items():
        mode, species = group
        selected_shape = mode_channel_shape(mode)[:-1] + (species_mass_indices(mode, species).size,)
        channels = int(np.prod(selected_shape))
        token = GROUP_TOKENS[group]
        pool_path = work_root / "pools" / f"{token}.float64.dat"
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool = np.memmap(pool_path, mode="w+", dtype=np.float64, shape=(max(1, rows_expected[token]), channels))
        state[group] = {
            "selected_shape": selected_shape,
            "channels": channels,
            "pool": pool,
            "pool_path": pool_path,
            "cursor": 0,
            "seen_times": set(),
            "sum_positive": np.zeros((len(intervals), channels), dtype=np.float64),
            "positive_count": np.zeros((len(intervals), channels), dtype=np.int32),
            "finite_count": np.zeros((len(intervals), channels), dtype=np.int32),
            "record_count": np.zeros(len(intervals), dtype=np.int32),
        }

    for chunk_number, path in enumerate(chunk_paths, start=1):
        with np.load(path, allow_pickle=False) as data:
            for group, item in state.items():
                token = GROUP_TOKENS[group]
                value_key = f"values__{token}"
                if value_key not in data.files:
                    continue
                values = np.asarray(data[value_key], dtype=float)
                indices = np.asarray(data[f"interval__{token}"], dtype=np.int32)
                times = np.asarray(data[f"time__{token}"], dtype=float)
                keep_values = []
                for value in times:
                    scalar = float(value)
                    keep_values.append(scalar not in item["seen_times"])
                    item["seen_times"].add(scalar)
                keep = np.asarray(keep_values, dtype=bool)
                values = values[keep]
                indices = indices[keep]
                if not values.size:
                    continue
                cursor = int(item["cursor"])
                item["pool"][cursor:cursor + values.shape[0]] = values
                item["cursor"] = cursor + values.shape[0]
                for interval_index in np.unique(indices):
                    block = values[indices == interval_index]
                    finite = np.isfinite(block)
                    positive = finite & (block > 0.0)
                    item["sum_positive"][interval_index] += np.where(positive, block, 0.0).sum(axis=0)
                    item["positive_count"][interval_index] += positive.sum(axis=0)
                    item["finite_count"][interval_index] += finite.sum(axis=0)
                    item["record_count"][interval_index] += block.shape[0]
        if chunk_number % 100 == 0:
            print(f"aggregate chunks {chunk_number}/{len(chunk_paths)}", flush=True)

    summaries: dict[int, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    support_input: dict[str, Any] = {"source_files": source_metadata, "groups": {}}
    for group, item in state.items():
        mode, species = group
        intervals = grouped_intervals[group]
        cursor = int(item["cursor"])
        item["pool"].flush()
        expected_records = np.asarray([int(row["native_record_count"]) for row in intervals], dtype=np.int32)
        if not np.array_equal(item["record_count"], expected_records):
            mismatch = np.flatnonzero(item["record_count"] != expected_records)
            raise ValueError(
                f"Mode {mode} {species} native record-count mismatch in {mismatch.size} intervals; "
                f"first={mismatch[:5].tolist()}"
            )
        means = np.divide(
            item["sum_positive"],
            item["positive_count"],
            out=np.full_like(item["sum_positive"], np.nan),
            where=item["positive_count"] > 0,
        )
        median, mad = _exact_pooled_median_mad(
            item["pool"], cursor, item["channels"], chunk_channels=median_channel_chunk
        )
        selected_shape = item["selected_shape"]
        summaries[mode][species] = {
            "interval_means_dpf": means.reshape((len(intervals), *selected_shape)),
            "sample_count": item["finite_count"].sum(axis=0).reshape(selected_shape),
            "nonzero_sample_count": item["positive_count"].sum(axis=0).reshape(selected_shape),
            "nonzero_median_dpf": median.reshape(selected_shape),
            "nonzero_mad_dpf": mad.reshape(selected_shape),
        }
        support_input["groups"][f"mode{mode:02d}_{species}"] = {
            "approved_interval_count": len(intervals),
            "unique_product_record_count": cursor,
            "record_count_min": int(item["record_count"].min()),
            "record_count_max": int(item["record_count"].max()),
            "record_count_sum": int(item["record_count"].sum()),
        }
    return dict(summaries), support_input


def _support_report(model: Any) -> dict[str, Any]:
    report = {"mode": model.mode, "valid": model.valid, "groups": {}}
    for species in SUPPORTED_SPECIES:
        mass = species_mass_indices(model.mode, species)
        support = model.support_level[..., mass]
        background = model.background_dpf[..., mass]
        report["groups"][species] = {
            "channel_count": int(support.size),
            "supported": int(np.count_nonzero(support == SUPPORT_SUPPORTED)),
            "low_support": int(np.count_nonzero(support == SUPPORT_LOW)),
            "unsupported": int(np.count_nonzero(support == SUPPORT_UNSUPPORTED)),
            "background_finite_fraction": float(np.mean(np.isfinite(background))),
            "background_positive_median_dpf": float(np.nanmedian(background[background > 0.0])) if np.any(background > 0.0) else None,
            "background_positive_p95_dpf": float(np.nanquantile(background[background > 0.0], 0.95)) if np.any(background > 0.0) else None,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["release_version"] != MULTIMODE_DENOISE_VERSION:
        raise ValueError("Config/release version mismatch")
    review_path = _resolve(config["review_inventory"]).resolve()
    output_root = _resolve(config["output_root"]).resolve()
    work_root = _resolve(config["work_root"]).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {output_root}")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("review_status") != "finalized":
        raise ValueError("Manual review is not finalized")
    approved = review["approved_intervals"]
    by_file, grouped = _build_references(approved)
    work_root.mkdir(parents=True, exist_ok=True)
    chunks_root = work_root / "chunks"
    ordered_files = sorted(by_file)
    tasks = [(path, by_file[path]) for path in ordered_files]

    benchmark_path = work_root / "benchmark.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    else:
        count = min(int(config["benchmark_file_count"]), len(tasks))
        benchmark = _benchmark(
            tasks[:count],
            parallel_workers=int(config["parallel_workers"]),
            speedup_required=float(config["parallel_speedup_required"]),
        )
        benchmark["created_utc"] = _utc_now()
        _atomic_json(benchmark_path, benchmark)
    workers = int(benchmark["selected_workers"])
    print(json.dumps({"benchmark": benchmark, "source_file_count": len(tasks)}, ensure_ascii=False), flush=True)

    chunk_tasks = []
    chunk_paths = []
    for index, (source, references) in enumerate(tasks):
        name_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        destination = chunks_root / f"{index:05d}_{name_hash}.npz"
        chunk_paths.append(destination)
        chunk_tasks.append((source, references, str(destination)))
    if workers == 1:
        for index, task in enumerate(chunk_tasks, start=1):
            _write_chunk(task)
            if index % 50 == 0 or index == len(chunk_tasks):
                print(f"extract chunks {index}/{len(chunk_tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, _ in enumerate(executor.map(_write_chunk, chunk_tasks), start=1):
                if index % 50 == 0 or index == len(chunk_tasks):
                    print(f"extract chunks {index}/{len(chunk_tasks)}", flush=True)

    summaries, support_input = _aggregate(
        chunk_paths, grouped, work_root,
        median_channel_chunk=int(config["median_channel_chunk"]),
    )
    source_inventory = sorted(
        [
            {
                "path": item["source_path"],
                "sha256": item["source_sha256"],
                "size": item["source_size"],
                "mtime_ns": item["source_mtime_ns"],
                "mode": item["mode"],
            }
            for item in support_input["source_files"]
        ],
        key=lambda item: item["path"],
    )
    source_inventory_hash = hashlib.sha256(
        json.dumps(source_inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    review_hash = sha256_file(review_path)
    models = {}
    reports = {}
    for mode in SUPPORTED_MODES:
        approved_by_species = {
            species: [
                {key: value for key, value in row.items() if key != "plot_path"}
                for row in grouped[(mode, species)]
            ]
            for species in SUPPORTED_SPECIES
        }
        model = assemble_multimode_background_from_summaries(
            mode,
            summaries.get(mode, {}),
            approved_intervals_by_species=approved_by_species,
            bootstrap_replicates=int(config["bootstrap_replicates"]),
            bootstrap_seed=int(config["bootstrap_seed"]) + mode * 100,
            review_status="finalized",
            provenance={
                "review_version": review["review_version"],
                "review_inventory": str(review_path),
                "review_inventory_sha256": review_hash,
                "source_inventory_sha256": source_inventory_hash,
                "source_file_count": len(source_inventory),
                "minimum_duration_s": int(config["minimum_duration_s"]),
                "maximum_duration_s": int(config["maximum_duration_s"]),
                "bootstrap_replicates": int(config["bootstrap_replicates"]),
                "bootstrap_seed": int(config["bootstrap_seed"]) + mode * 100,
            },
        )
        if not model.valid or model.review_status != "finalized":
            raise ValueError(f"Mode {mode} model is not releasable: {model.invalid_reasons}")
        models[mode] = model
        reports[mode] = _support_report(model)

    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        model_entries = {}
        for mode, model in models.items():
            path = temporary_root / "models" / f"minpa_mode{mode:02d}_background_v2.0.0.npz"
            save_multimode_background_model(path, model)
            loaded = load_multimode_background_model(path)
            if loaded.mode != mode or not loaded.valid:
                raise RuntimeError(f"Mode {mode} model roundtrip failed")
            model_entries[str(mode)] = {
                "path": path.relative_to(temporary_root).as_posix(),
                "sha256": sha256_file(path),
                "algorithm_version": model.algorithm_version,
            }
        _atomic_json(temporary_root / "source_inventory.json", {"sha256": source_inventory_hash, "files": source_inventory})
        _atomic_json(temporary_root / "support_report.json", {"input": support_input["groups"], "models": reports})
        bundle = {
            "bundle_version": MULTIMODE_DENOISE_VERSION,
            "model_version": MULTIMODE_MODEL_VERSION,
            "status": "frozen",
            "created_utc": _utc_now(),
            "review_inventory": str(review_path),
            "review_inventory_sha256": review_hash,
            "source_inventory_sha256": source_inventory_hash,
            "units": config["units"],
            "primary_estimator": config["primary_estimator"],
            "dimension_order": ["energy", "pitch", "azimuth", "mass"],
            "models": model_entries,
            "mode7_policy": "unchanged",
            "unreviewed_mass_policy": "unchanged",
            "unsupported_channel_policy": "unchanged",
            "correction": "max(raw-background,0); NaN preserved",
            "benchmark": benchmark,
        }
        _atomic_json(temporary_root / "bundle.json", bundle)
        loaded_bundle = load_multimode_model_bundle(temporary_root / "bundle.json")
        if set(loaded_bundle) != set(SUPPORTED_MODES):
            raise RuntimeError("Frozen bundle roundtrip failed")
        for mode, model in loaded_bundle.items():
            raw = np.broadcast_to(np.nan_to_num(model.background_dpf, nan=1.0) + 1.0, (1, *model.shape)).copy()
            result = apply_multimode_background(raw, model)
            if np.any(result.corrected_dpf > raw) or np.any(result.corrected_dpf < 0.0):
                raise RuntimeError(f"Mode {mode} subtraction sanity check failed")
            unreviewed = ~np.broadcast_to(model.reviewed_mass_mask, model.shape)
            if not np.array_equal(result.corrected_dpf[0][unreviewed], raw[0][unreviewed]):
                raise RuntimeError(f"Mode {mode} unreviewed mass preservation failed")
        (temporary_root / "README.md").write_text(
            "# MINPA Mode 4/12 background bundle v2.0.0\n\n"
            "Frozen, hash-verified models built only from the finalized species-specific manual review. "
            "DPF units are 1/(s cm^2 sr eV); this does not reconstruct detector counts. "
            "Mode 12 is corrected after its two timed subrecords are split. Mode 7, unreviewed masses, "
            "and unsupported channels remain unchanged.\n",
            encoding="utf-8",
        )
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(json.dumps({"bundle": str((output_root / 'bundle.json').resolve()), "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
