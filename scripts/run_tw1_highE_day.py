"""Run one UTC day of Tianwen-1 MINPA high-energy O+/O2+ moments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.constants import DEFAULT_OUTPUT_ROOT
from highE.minpa_background import load_background_model
from highE.minpa_multimode_background import load_multimode_model_bundle
from highE.tw1_minpa import (
    SkipDay,
    Tw1Paths,
    load_unified_static_background_bundle,
    process_tw1_day,
)


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
        choices=(
            "none",
            "subtract-and-reject-uv",
            "paper-channel-subtract",
            "multimode-paper-channel-subtract",
            "all-approved-static-channel-subtract",
        ),
        default="none",
        help="Default 'none' is bitwise-compatible; correction must be selected explicitly.",
    )
    parser.add_argument("--background-model", type=Path)
    parser.add_argument(
        "--multimode-background-bundle",
        type=Path,
        help="Frozen v2.0.0 Mode-4/12 bundle; rc1 review manifests are refused.",
    )
    parser.add_argument(
        "--static-background-bundle",
        type=Path,
        help="Frozen v2.2.0 all-approved Mode-1/4/12 static bundle.",
    )
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
    multimode_models = (
        load_multimode_model_bundle(args.multimode_background_bundle)
        if args.multimode_background_bundle is not None
        else None
    )
    if args.static_background_bundle is not None:
        if args.background_model is not None or args.multimode_background_bundle is not None:
            raise ValueError("Static bundle cannot be combined with separate background model arguments")
        background_model, multimode_models = load_unified_static_background_bundle(
            args.static_background_bundle
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
            background_model_path=args.static_background_bundle or args.background_model,
            multimode_background_models=multimode_models,
            multimode_background_bundle_path=(
                args.static_background_bundle or args.multimode_background_bundle
            ),
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
