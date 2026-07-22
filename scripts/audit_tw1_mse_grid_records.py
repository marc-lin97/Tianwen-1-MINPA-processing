"""Audit the complete Tianwen-1 MINPA MSE spatial-record product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.tw1_mse_grid_audit import (
    EXPECTED_QUALITY_ALGORITHM_VERSION,
    audit_tw1_mse_grid_records,
    cleanup_empty_staging_groups,
)
from highE.tw1_mse_grid_records import DEFAULT_GRID_OUTPUT_ROOT, DEFAULT_INPUT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_GRID_OUTPUT_ROOT)
    parser.add_argument("--expected-days", type=int, default=1029)
    parser.add_argument("--quality-algorithm-version", default=EXPECTED_QUALITY_ALGORITHM_VERSION)
    parser.add_argument("--cleanup-empty-staging", action="store_true")
    parser.add_argument("--allow-empty-staging", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cleanup_empty_staging:
        cleanup = cleanup_empty_staging_groups(args.output_root)
        print(json.dumps({"staging_cleanup": cleanup}, ensure_ascii=False))
        if cleanup["nonempty_staging_groups"]:
            return 2
    report = audit_tw1_mse_grid_records(
        args.input_root,
        args.output_root,
        expected_days=args.expected_days,
        expected_quality_algorithm_version=args.quality_algorithm_version,
        require_no_staging=not args.allow_empty_staging,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
