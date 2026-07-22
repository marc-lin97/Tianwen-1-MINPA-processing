"""Rebin classified Tianwen-1 MINPA MSE products from 0.1 to 0.2 Rm."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.tw1_mse_fov_rebin import run_rebin_classified_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=r"D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_mse_xz_fov_classes_y_m1_p1_5Rm",
    )
    parser.add_argument(
        "--output-root",
        default=r"D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_mse_xz_fov_classes_grid0p2_y_m0p5_p0p5_5Rm",
    )
    parser.add_argument("--y-min-rm", type=float, default=-0.5)
    parser.add_argument("--y-max-rm", type=float, default=0.5)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_rebin_classified_statistics(
        Path(args.source_root),
        Path(args.output_root),
        y_min_rm=args.y_min_rm,
        y_max_rm=args.y_max_rm,
        write_plots=not args.no_plots,
    )
    print(
        "TW1 classified MSE rebin complete: "
        f"ratio={summary['coarsening_ratio']} output={summary['output_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
