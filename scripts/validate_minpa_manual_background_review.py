"""Resolve and validate manually approved MINPA Mode-1 background windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import (  # noqa: E402
    BackgroundConfig,
    approved_intervals_from_manual_review,
    concatenate_mode1_records,
    estimate_background,
    mode1_ori_files_for_interval,
    read_mode1_ori_records,
)


DEFAULT_ORI = Path(r"D:\Data\TW-1\result\MINPA\ori")
DEFAULT_QUALITY = ROOT / "outputs" / "tw1_minpa_quality_flags_all_species"
DEFAULT_CONFIG = ROOT / "config" / "minpa_background_reproduction.json"
DEFAULT_REVIEW = ROOT / "config" / "minpa_background_manual_review.json"
DEFAULT_ORIGINAL = (
    ROOT / "outputs" / "minpa_background_paper_reproduction" / "data"
    / "background_interval_candidates.json"
)
DEFAULT_EXPANDED = (
    ROOT / "outputs" / "minpa_background_paper_reproduction" / "data"
    / "expanded_quiet_window_candidates.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "minpa_background_paper_reproduction" / "data"
    / "approved_background_intervals_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ori-root", type=Path, default=DEFAULT_ORI)
    parser.add_argument("--quality-root", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--original-candidates", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--expanded-candidates", type=Path, default=DEFAULT_EXPANDED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config_payload = _read_json(args.config)
    review = _read_json(args.manual_review)
    original = _read_json(args.original_candidates)
    expanded = _read_json(args.expanded_candidates)
    intervals = approved_intervals_from_manual_review(review, original, expanded)

    parts = []
    for interval in intervals:
        paths = mode1_ori_files_for_interval(
            args.ori_root, interval.start_utc, interval.stop_utc
        )
        parts.append(read_mode1_ori_records(
            paths,
            interval.start_utc,
            interval.stop_utc,
            quality_root=args.quality_root,
            compute_sha256=False,
        ))
    records = concatenate_mode1_records(parts)
    model_config = BackgroundConfig(**config_payload["background_model"])
    model = estimate_background(records, intervals, model_config)

    accepted_summaries = [
        item for item in model.interval_summaries if bool(item["accepted"])
    ]
    accepted_records = sum(int(item["kept_records"]) for item in accepted_summaries)
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "requirements_satisfied"
            if model.valid
            else "reviewed_subset_not_yet_valid_for_production"
        ),
        "manual_review_source": str(args.manual_review.resolve()),
        "approved_interval_count": len(intervals),
        "quality_accepted_interval_count": len(accepted_summaries),
        "quality_accepted_record_count": accepted_records,
        "required_minimum_intervals": model_config.min_approved_intervals,
        "required_minimum_records": model_config.min_total_records,
        "minimum_intervals_satisfied": (
            len(accepted_summaries) >= model_config.min_approved_intervals
        ),
        "minimum_records_satisfied": accepted_records >= model_config.min_total_records,
        "model_valid": model.valid,
        "model_invalid_reasons": model.invalid_reasons,
        "approved_intervals": [asdict(interval) for interval in intervals],
        "interval_summaries": model.interval_summaries,
        "note": (
            "This file validates selection and quality only. No production model is "
            "saved unless all configured requirements are satisfied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        key: payload[key]
        for key in (
            "status",
            "approved_interval_count",
            "quality_accepted_interval_count",
            "quality_accepted_record_count",
            "minimum_intervals_satisfied",
            "minimum_records_satisfied",
            "model_valid",
            "model_invalid_reasons",
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
