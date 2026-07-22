"""Build MSE spatial grid statistics and center-slice figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.spatial_stats import MISSION_CONFIGS, SPECIES_IDS, run_spatial_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=r"E:\Data\highE", help="Root containing MAVEN and Tianwen-1 daily products.")
    parser.add_argument(
        "--output-root",
        default=r"E:\Data\highE\processed_mse_spatial_stats",
        help="Directory for grid statistics, figures, and logs.",
    )
    parser.add_argument("--missions", nargs="+", choices=sorted(MISSION_CONFIGS), default=sorted(MISSION_CONFIGS))
    parser.add_argument("--species", nargs="+", choices=SPECIES_IDS, default=list(SPECIES_IDS))
    parser.add_argument("--years", nargs="*", default=None, help="Optional years to scan inside each mission folder.")
    parser.add_argument("--dates", nargs="*", default=None, help="Optional YYYYMMDD dates; paths are built directly.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel file readers per mission/species.")
    parser.add_argument("--grid-step-rm", type=float, default=0.1)
    parser.add_argument("--density-min-cm3", type=float, default=0.001)
    parser.add_argument(
        "--output-mode",
        choices=("grid-records", "stats", "both"),
        default="grid-records",
        help="grid-records writes per-grid/date MAT intermediates; stats writes aggregated grids and figures.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write grid data and logs only.")
    parser.add_argument("--max-files-per-species", type=int, default=None, help="Debug limit for quick validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_spatial_statistics(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        missions=args.missions,
        species_ids=args.species,
        workers=args.workers,
        step_rm=args.grid_step_rm,
        density_min_cm3=args.density_min_cm3,
        write_plots=not args.no_plots,
        max_files_per_species=args.max_files_per_species,
        years=args.years,
        dates=args.dates,
        output_mode=args.output_mode,
    )
    for item in summary["products"]:
        if item.get("status") == "written":
            print(
                f"{item['mission']} {item['species']} {item['fov_group']}: "
                f"density_samples={item['density_samples']} velocity_samples={item['velocity_samples']} "
                f"nonempty_density_bins={item['nonempty_density_bins']} data={item['data_file']}"
            )
        elif item.get("status") == "grid_records_written":
            print(
                f"{item['mission']} {item['species']}: "
                f"grid_record_files={item['grid_record_file_count']} rows={item['grid_record_rows']} "
                f"root={item['grid_record_root']}"
            )
    print(f"Summary: {Path(args.output_root) / 'logs' / 'spatial_stats_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
