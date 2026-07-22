"""Audit the complete Tianwen-1 MINPA all-energy per-cell MSE archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.constants import DEFAULT_TW1_MOMAG_ROOT, DEFAULT_TW1_NV_MSO2_ROOT, DEFAULT_TW1_R_ROOT
from highE.tw1_all_energy_mse_grid_audit import (
    DEFAULT_OUTPUT_ROOT,
    audit_tw1_all_energy_mse_grid_records,
    cleanup_empty_staging_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_TW1_NV_MSO2_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--momag-root", default=str(DEFAULT_TW1_MOMAG_ROOT))
    parser.add_argument("--r-root", default=str(DEFAULT_TW1_R_ROOT))
    parser.add_argument("--dates", nargs="*", default=None, help="Optional NV product dates YYYYMMDD.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--expected-days", type=int, default=1029)
    parser.add_argument("--expected-excluded-days", type=int, default=22)
    parser.add_argument("--source-samples-per-day", type=int, default=3)
    parser.add_argument("--cleanup-empty-staging", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    cleanup = None
    if args.cleanup_empty_staging:
        cleanup = cleanup_empty_staging_groups(output_root)
        if cleanup["nonempty_staging_groups"]:
            raise RuntimeError(f"Refusing audit with nonempty staging groups: {cleanup}")
    report = audit_tw1_all_energy_mse_grid_records(
        input_root=Path(args.input_root),
        output_root=output_root,
        momag_root=Path(args.momag_root),
        r_root=Path(args.r_root),
        dates=args.dates,
        start_date=args.start_date,
        end_date=args.end_date,
        expected_days=args.expected_days,
        expected_excluded_days=args.expected_excluded_days,
        source_samples_per_day=args.source_samples_per_day,
    )
    print(json.dumps({"cleanup": cleanup, **report}, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
