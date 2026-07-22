"""Plot full-energy H+/O+/O2+ X-Z maps for the two +Xb_MSE cone classes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.constants import DEFAULT_TW1_MOMAG_ROOT, DEFAULT_TW1_NV_MSO2_ROOT, DEFAULT_TW1_R_ROOT
from highE.tw1_all_energy_mse_xz_xb_stats import run_tw1_all_energy_mse_xz_xb_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_TW1_NV_MSO2_ROOT))
    parser.add_argument(
        "--output-root",
        default=r"D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_all_energy_mse_xz_xb_classes_y_m0p5_p0p5_5Rm",
    )
    parser.add_argument("--momag-root", default=str(DEFAULT_TW1_MOMAG_ROOT))
    parser.add_argument("--r-root", default=str(DEFAULT_TW1_R_ROOT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--y-min-rm", type=float, default=-0.5)
    parser.add_argument("--y-max-rm", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_tw1_all_energy_mse_xz_xb_statistics(
        Path(args.input_root),
        Path(args.output_root),
        momag_root=Path(args.momag_root),
        r_root=Path(args.r_root),
        workers=args.workers,
        max_days=args.max_days,
        write_plots=not args.no_plots,
        progress_every=args.progress_every,
        y_min_rm=args.y_min_rm,
        y_max_rm=args.y_max_rm,
    )
    print(
        f"TW1 all-energy Xb classes complete: days={summary['days_processed']} "
        f"in_grid={summary['rows_in_grid']} output={summary['output_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
