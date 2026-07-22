"""Compute Tianwen-1 MINPA MSE cell means and plot all-Y X-Z maps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.tw1_mse_xz_stats import run_tw1_mse_xz_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default=r"D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_grid_records_5Rm",
    )
    parser.add_argument(
        "--output-root",
        default=r"D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_mse_xz_means_5Rm",
    )
    parser.add_argument(
        "--quality-policy",
        choices=("valid", "strict-clean", "all-valid"),
        default="valid",
        help="Default valid policy retains bit-1 caution and rejects NV_MSO_2 bits 2-5.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None, help="Optional deterministic smoke-test limit.")
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_tw1_mse_xz_statistics(
        Path(args.input_root),
        Path(args.output_root),
        quality_policy=args.quality_policy,
        workers=args.workers,
        max_files=args.max_files,
        write_plots=not args.no_plots,
        progress_every=args.progress_every,
    )
    print(
        f"TW1 MSE X-Z statistics complete: files={summary['cell_files_processed']} "
        f"quality={summary['quality_policy']} output={summary['output_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
