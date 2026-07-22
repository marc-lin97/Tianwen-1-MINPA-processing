"""Run the resumable exploratory full-mission MINPA Mode-1 noise analysis."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background_temporal import (  # noqa: E402
    EXPLORATORY_STATUS,
    TEMPORAL_ALGORITHM_VERSION,
    _atomic_json,
    analyze_quarterly_noise,
    analyze_temporal_noise,
    benchmark_ori_files,
    build_mode1_inventory,
    discover_monthly_quiet_candidates,
    run_months_parallel,
    upgrade_monthly_lambda_pooling,
    write_quarterly_analysis_outputs,
    write_temporal_analysis_outputs,
)


DEFAULT_CONFIG = ROOT / "config" / "minpa_background_full_mission.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "minpa_background_temporal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "all", "inventory", "candidates", "benchmark", "months", "stats",
            "quarterly", "plot", "upgrade-lambda",
        ),
        default="all",
    )
    parser.add_argument(
        "--day-spe-root", type=Path,
        default=Path(os.environ["TW1_MINPA_DAY_SPE_ROOT"])
        if "TW1_MINPA_DAY_SPE_ROOT" in os.environ else None,
    )
    parser.add_argument(
        "--ori-root", type=Path,
        default=Path(os.environ["TW1_MINPA_ORI_ROOT"])
        if "TW1_MINPA_ORI_ROOT" in os.environ else None,
    )
    parser.add_argument(
        "--quality-root", type=Path,
        default=Path(os.environ["TW1_MINPA_QUALITY_ROOT"])
        if "TW1_MINPA_QUALITY_ROOT" in os.environ else None,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--months", nargs="*", help="Optional YYYYMM subset")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true",
        help="Reuse valid month checkpoints (default)",
    )
    resume_group.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Recompute requested months even when checkpoints are valid",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--benchmark-files", type=int, default=100)
    return parser.parse_args()


def _require_roots(args: argparse.Namespace) -> None:
    required: list[tuple[str, Path | None]] = []
    if args.stage in ("all", "inventory"):
        required.extend((
            ("--day-spe-root", args.day_spe_root),
            ("--ori-root", args.ori_root),
            ("--quality-root", args.quality_root),
        ))
    elif args.stage in ("benchmark", "months"):
        required.append(("--quality-root", args.quality_root))
    missing = [name for name, value in required if value is None]
    if missing:
        raise SystemExit(
            "Missing required local roots: " + ", ".join(missing)
            + ". Supply arguments or TW1_MINPA_* environment variables."
        )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _month_summaries(
    output_root: Path,
    monthly_candidates: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for month in sorted(monthly_candidates):
        path = output_root / "months" / month / "summary.json"
        if path.exists():
            summaries.append(_load_json(path))
        else:
            summaries.append({
                "month": month,
                "status": "missing_month_checkpoint",
                "candidate_count": len(monthly_candidates[month]),
                "accepted_interval_count": 0,
                "accepted_record_count": 0,
                "distinct_day_count": 0,
                "inference_eligible": False,
                "interval_metrics": [],
            })
    return summaries


def _write_manifest(
    output_root: Path,
    inventory: dict[str, object],
    monthly_candidates: dict[str, list[dict[str, object]]],
    month_results: list[dict[str, object]] | None = None,
    benchmark: dict[str, object] | None = None,
) -> None:
    results = month_results or _month_summaries(output_root, monthly_candidates)
    manifest = {
        "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
        "analysis_status": EXPLORATORY_STATUS,
        "updated_utc": datetime.now(UTC).isoformat(),
        "inventory_input_signature": inventory.get("input_signature"),
        "month_count": len(monthly_candidates),
        "candidate_count": sum(len(rows) for rows in monthly_candidates.values()),
        "month_status": {
            str(item["month"]): str(item.get("status", "unknown")) for item in results
        },
        "benchmark": {
            key: benchmark.get(key) for key in (
                "file_count", "workers", "single_seconds", "parallel_seconds",
                "speedup", "identical",
            )
        } if benchmark else None,
    }
    _atomic_json(output_root / "manifest.json", manifest)


def main() -> None:
    args = parse_args()
    _require_roots(args)
    config = _load_json(args.config)
    if config.get("algorithm_version") != TEMPORAL_ALGORITHM_VERSION:
        raise SystemExit("Config algorithm_version does not match the implementation")
    output = args.output_root.resolve()
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = data_dir / "mode1_inventory.json"
    candidates_path = data_dir / "monthly_candidates.json"
    benchmark_path = output / "logs" / "benchmark_100_ori_files.json"

    if args.stage == "upgrade-lambda":
        report = upgrade_monthly_lambda_pooling(output, config)
        print(json.dumps({
            "stage": "upgrade-lambda",
            "upgraded_months": len(report["upgraded_months"]),
            "raw_data_reread": report["raw_data_reread"],
        }, ensure_ascii=False), flush=True)
        return

    if args.stage in ("all", "inventory"):
        inventory = build_mode1_inventory(args.day_spe_root, args.ori_root, args.quality_root)
        _atomic_json(inventory_path, inventory)
        print(json.dumps({
            "stage": "inventory",
            "day_spe_files": inventory["day_spe_file_count"],
            "mode1_segments": inventory["strict_mode1_segment_count"],
            "mode1_records": inventory["strict_mode1_record_count"],
            "months": len(inventory["months"]),
            "errors": len(inventory["errors"]),
        }, ensure_ascii=False), flush=True)
        if args.stage == "inventory":
            return
    else:
        inventory = _load_json(inventory_path)

    if args.stage in ("all", "candidates"):
        monthly_candidates = discover_monthly_quiet_candidates(
            inventory, config["candidate_selection"]
        )
        _atomic_json(candidates_path, {
            "algorithm_version": TEMPORAL_ALGORITHM_VERSION,
            "analysis_status": EXPLORATORY_STATUS,
            "inventory_input_signature": inventory["input_signature"],
            "created_utc": datetime.now(UTC).isoformat(),
            "months": monthly_candidates,
        })
        print(json.dumps({
            "stage": "candidates",
            "months": len(monthly_candidates),
            "candidates": sum(len(rows) for rows in monthly_candidates.values()),
        }, ensure_ascii=False), flush=True)
        if args.stage == "candidates":
            _write_manifest(output, inventory, monthly_candidates)
            return
    else:
        monthly_candidates = _load_json(candidates_path)["months"]

    benchmark = None
    if args.stage == "benchmark" or (args.stage == "all" and not args.skip_benchmark):
        benchmark = benchmark_ori_files(
            inventory,
            file_count=args.benchmark_files,
            workers=max(1, args.workers),
            quality_root=args.quality_root,
        )
        _atomic_json(benchmark_path, benchmark)
        print(json.dumps({
            "stage": "benchmark",
            "files": benchmark["file_count"],
            "single_seconds": benchmark["single_seconds"],
            "parallel_seconds": benchmark["parallel_seconds"],
            "identical": benchmark["identical"],
        }, ensure_ascii=False), flush=True)
        if not bool(benchmark["identical"]):
            raise SystemExit("Single/multiprocess benchmark outputs differ")
        if args.stage == "benchmark":
            _write_manifest(output, inventory, monthly_candidates, benchmark=benchmark)
            return
    elif benchmark_path.exists():
        benchmark = _load_json(benchmark_path)

    if args.stage in ("all", "months"):
        manual_relative = Path(str(config["known_manual_review"]["expanded_candidates"]))
        manual_path = manual_relative if manual_relative.is_absolute() else ROOT / manual_relative
        month_results = run_months_parallel(
            monthly_candidates,
            inventory,
            args.quality_root,
            output,
            config,
            manual_path,
            workers=max(1, args.workers),
            resume=bool(args.resume),
            months=args.months,
        )
        _write_manifest(output, inventory, monthly_candidates, month_results, benchmark)
        print(json.dumps({
            "stage": "months",
            "requested": len(month_results),
            "complete": sum(item.get("status") == "complete" for item in month_results),
            "failed": sum(item.get("status") == "failed" for item in month_results),
            "resumed": sum(item.get("resume_action") == "skipped_valid_checkpoint" for item in month_results),
        }, ensure_ascii=False), flush=True)
        if args.stage == "months":
            return

    if args.stage in ("all", "stats"):
        summaries = _month_summaries(output, monthly_candidates)
        result = analyze_temporal_noise(summaries, config)
        outputs = write_temporal_analysis_outputs(output, result, config)
        print(json.dumps({
            "stage": "stats",
            "months": result["month_count"],
            "eligible_months": result["eligible_month_count"],
            "practically_significant_changes": sum(
                item.get("conclusion") == "practically_significant"
                for item in result["change_comparisons"]
            ),
            "outputs": outputs,
        }, ensure_ascii=False), flush=True)
        if args.stage == "stats":
            return

    if args.stage in ("all", "quarterly"):
        summaries = _month_summaries(output, monthly_candidates)
        quarterly = analyze_quarterly_noise(summaries, inventory, config)
        outputs = write_quarterly_analysis_outputs(output, quarterly)
        print(json.dumps({
            "stage": "quarterly",
            "quarters": quarterly["quarter_count"],
            "eligible_quarters": quarterly["eligible_quarter_count"],
            "practically_significant_changes": sum(
                item.get("conclusion") == "practically_significant"
                for item in quarterly["change_comparisons"]
            ),
            "practically_significant_pairwise_contrasts": sum(
                item.get("conclusion") == "practically_significant"
                for item in quarterly["eligible_quarter_pairwise_contrasts"]
            ),
            "outputs": outputs,
        }, ensure_ascii=False), flush=True)
        if args.stage == "quarterly":
            return

    if args.stage == "plot":
        result = _load_json(data_dir / "temporal_noise_analysis.json")
        outputs = write_temporal_analysis_outputs(output, result, config)
        print(json.dumps({"stage": "plot", "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
