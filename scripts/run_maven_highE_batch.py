"""Run MAVEN high-energy daily products with date-level parallelism."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.maven_static import MavEnPaths, process_maven_day


def _run_one(payload: tuple[str, str, str, str, str, float, bool]) -> tuple[str, dict]:
    date, d1_root, mag_root, r_root, output_root, high_e, add_sc_v = payload
    paths = MavEnPaths(Path(d1_root), Path(mag_root), Path(r_root), Path(output_root))
    _, summary = process_maven_day(date, paths, high_energy_min_eV=high_e, add_spacecraft_velocity=add_sc_v, write_output=True)
    return date, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dates", nargs="+", help="UTC dates as YYYYMMDD.")
    parser.add_argument("--d1-root", default=r"D:\Data\MAVEN\rawdata\static\l2\d1-32e4d16a8m")
    parser.add_argument("--mag-root", default=r"D:\Data\MAVEN\result\mag\ss1s")
    parser.add_argument("--r-root", default=r"D:\Data\MAVEN\result\R_MSO2MSE")
    parser.add_argument("--output-root", default=r"E:\Data\highE")
    parser.add_argument("--high-energy-min-eV", type=float, default=1000.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-spacecraft-velocity-correction", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payloads = [
        (
            date,
            args.d1_root,
            args.mag_root,
            args.r_root,
            args.output_root,
            args.high_energy_min_eV,
            not args.no_spacecraft_velocity_correction,
        )
        for date in args.dates
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, payload) for payload in payloads]
        for future in as_completed(futures):
            date, summary = future.result()
            print(f"{date}: per_species_rows={summary['records']['per_species_rows']} status={summary['records']['status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
