"""Classify MINPA by +Xb_MSE cones around +/-Z_MSE and plot X-Z means."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.constants import DEFAULT_TW1_NV_MSO2_ROOT, DEFAULT_TW1_R_ROOT
from highE.tw1_mse_grid_records import MseGridSpec
from highE.tw1_mse_xz_fov_stats import run_tw1_mse_xz_fov_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=r"D:\Data\highE\Tianwen-1")
    parser.add_argument(
        "--output-root",
        default=r"D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_mse_xz_fov_classes_y_m1_p1_5Rm",
    )
    parser.add_argument("--r-root", default=str(DEFAULT_TW1_R_ROOT))
    parser.add_argument("--nv-mso2-root", default=str(DEFAULT_TW1_NV_MSO2_ROOT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--y-min-rm", type=float, default=-1.0)
    parser.add_argument("--y-max-rm", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_tw1_mse_xz_fov_statistics(
        Path(args.input_root),
        Path(args.output_root),
        r_root=Path(args.r_root),
        nv_mso2_root=Path(args.nv_mso2_root),
        grid=MseGridSpec(),
        workers=args.workers,
        max_days=args.max_days,
        write_plots=not args.no_plots,
        progress_every=args.progress_every,
        y_min_rm=args.y_min_rm,
        y_max_rm=args.y_max_rm,
    )
    print(
        f"TW1 Xb_MSE FOV classes complete: days={summary['days_processed']} "
        f"valid_xb={summary['xb_valid_rows']} output={summary['output_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
