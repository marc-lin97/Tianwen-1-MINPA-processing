"""Run one UTC day of Tianwen-1 MINPA high-energy O+/O2+ moments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.constants import DEFAULT_OUTPUT_ROOT
from highE.minpa_background import load_background_model
from highE.tw1_minpa import SkipDay, Tw1Paths, process_tw1_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="UTC date as YYYYMMDD.")
    parser.add_argument("--ori-root", default=r"D:\Data\TW-1\result\MINPA\ori")
    parser.add_argument("--public-root", default=r"D:\Data\TW-1\rawdata\MINPA\public")
    parser.add_argument("--momag-root", default=r"D:\Data\TW-1\result\MOMAG\C\01Hz_all")
    parser.add_argument("--r-root", default=r"D:\Data\TW-1\result\R_MSO2MSE")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--project-quality-root",
        default=str(ROOT / "outputs" / "tw1_minpa_quality_flags_all_species"),
    )
    parser.add_argument("--high-energy-min-eV", type=float, default=1000.0)
    parser.add_argument(
        "--background-policy",
        choices=("none", "subtract-and-reject-uv"),
        default="none",
        help="Default 'none' is bitwise-compatible; correction must be selected explicitly.",
    )
    parser.add_argument("--background-model", type=Path)
    parser.add_argument(
        "--allow-provisional-background-model",
        action="store_true",
        help="Permit a model built from provisional, not manually approved, intervals for validation only.",
    )
    parser.add_argument("--no-spacecraft-velocity-correction", action="store_true")
    parser.add_argument("--no-write", action="store_true", help="Run calculation and print summary without writing output files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = Tw1Paths(
        ori_root=Path(args.ori_root),
        public_root=Path(args.public_root),
        momag_root=Path(args.momag_root),
        r_root=Path(args.r_root),
        output_root=Path(args.output_root),
        project_quality_root=Path(args.project_quality_root),
    )
    background_model = (
        load_background_model(args.background_model) if args.background_model is not None else None
    )
    try:
        _, summary = process_tw1_day(
            args.date,
            paths,
            high_energy_min_eV=args.high_energy_min_eV,
            add_spacecraft_velocity=not args.no_spacecraft_velocity_correction,
            write_output=not args.no_write,
            background_policy=args.background_policy,
            background_model=background_model,
            background_model_path=args.background_model,
            allow_provisional_background_model=args.allow_provisional_background_model,
        )
    except SkipDay as exc:
        print(f"SKIP TW1 {args.date}: {exc}")
        return 0
    print(
        f"TW1 {args.date}: source={summary['source_files']['minpa_source_type']} "
        f"per_species_rows={summary['records']['per_species_rows']} "
        f"status={summary['records']['status_counts']} "
        f"density_median={summary['sanity_checks']['density_cm3_median']} "
        f"background_policy={summary['assumptions']['background_policy']} "
        f"uv_rejected={summary['records']['uv_rejected_records']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
