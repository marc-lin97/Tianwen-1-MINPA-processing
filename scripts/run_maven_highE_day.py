"""Run one UTC day of MAVEN STATIC-D1 high-energy O+/O2+ moments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.constants import DEFAULT_OUTPUT_ROOT
from highE.maven_static import MavEnPaths, SkipDay, process_maven_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="UTC date as YYYYMMDD.")
    parser.add_argument("--d1-root", default=r"D:\Data\MAVEN\rawdata\static\l2\d1-32e4d16a8m")
    parser.add_argument("--mag-root", default=r"D:\Data\MAVEN\result\mag\ss1s")
    parser.add_argument("--r-root", default=r"D:\Data\MAVEN\result\R_MSO2MSE")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--high-energy-min-eV", type=float, default=1000.0)
    parser.add_argument("--no-spacecraft-velocity-correction", action="store_true")
    parser.add_argument("--no-write", action="store_true", help="Run calculation and print summary without writing output files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = MavEnPaths(
        d1_root=Path(args.d1_root),
        mag_root=Path(args.mag_root),
        r_root=Path(args.r_root),
        output_root=Path(args.output_root),
    )
    try:
        _, summary = process_maven_day(
            args.date,
            paths,
            high_energy_min_eV=args.high_energy_min_eV,
            add_spacecraft_velocity=not args.no_spacecraft_velocity_correction,
            write_output=not args.no_write,
        )
    except SkipDay as exc:
        print(f"SKIP MAVEN {args.date}: {exc}")
        return 0
    print(
        f"MAVEN {args.date}: per_species_rows={summary['records']['per_species_rows']} "
        f"status={summary['records']['status_counts']} "
        f"density_median={summary['sanity_checks']['density_cm3_median']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
