"""Stream Tianwen-1 MINPA NV_MSO_2 all-energy moments into per-cell MSE HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.constants import DEFAULT_TW1_MOMAG_ROOT, DEFAULT_TW1_NV_MSO2_ROOT, DEFAULT_TW1_R_ROOT
from highE.tw1_all_energy_mse_grid_records import (
    DEFAULT_OUTPUT_ROOT,
    MseGridSpec,
    run_tw1_all_energy_mse_grid_records,
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
    parser.add_argument("--grid-min-rm", type=float, default=-5.0)
    parser.add_argument("--grid-max-rm", type=float, default=5.0)
    parser.add_argument("--grid-step-rm", type=float, default=0.1)
    parser.add_argument("--mars-radius-km", type=float, default=3397.0)
    parser.add_argument("--position-tolerance-s", type=float, default=5.0)
    parser.add_argument("--rotation-gap-s", type=float, default=8.0)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--max-days", type=int, default=None, help="Debug limit after geometry filtering.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing selected day groups.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid = MseGridSpec(args.grid_min_rm, args.grid_max_rm, args.grid_step_rm, args.mars_radius_km)
    summary = run_tw1_all_energy_mse_grid_records(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        momag_root=Path(args.momag_root),
        r_root=Path(args.r_root),
        dates=args.dates,
        start_date=args.start_date,
        end_date=args.end_date,
        grid=grid,
        position_tolerance_s=args.position_tolerance_s,
        rotation_gap_s=args.rotation_gap_s,
        min_free_gb=args.min_free_gb,
        overwrite=args.overwrite,
        max_days=args.max_days,
    )
    print(
        "TW1 all-energy MSE grid records: "
        f"processed={summary['processed_days']} "
        f"already_complete={summary['already_complete_days']} "
        f"excluded_geometry={summary['excluded_geometry_days']} "
        f"canonical_rows={summary['canonical_utc_rows']} "
        f"in_grid_rows={summary['in_grid_rows']} "
        f"max_daily_memory_MiB={summary['maximum_daily_working_memory_bytes'] / 1024**2:.2f} "
        f"free_GiB={summary['free_space_gb_after']:.2f} output={summary['output_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

