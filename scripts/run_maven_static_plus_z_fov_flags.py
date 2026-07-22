"""Generate per-epoch MAVEN STATIC +Z_MSE FOV flags from high-E MAT products."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.constants import STATIC_FOV_THETA_MAX_DEG, STATIC_FOV_THETA_MIN_DEG
from highE.coordinates import mse_plus_z_static_fov_flag, unix_s_to_utc_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=r"E:\Data\highE\MAVEN", help="Root containing MAVEN/YYYY high-E MAT products.")
    parser.add_argument("--output-root", default=None, help="Output root. Defaults to --input-root.")
    parser.add_argument("--dates", nargs="*", default=None, help="Optional YYYYMMDD dates. If omitted, scan input-root.")
    parser.add_argument("--species", choices=["Oplus", "O2plus"], default="Oplus", help="Species file used as the epoch/attitude source.")
    parser.add_argument("--theta-min-deg", type=float, default=STATIC_FOV_THETA_MIN_DEG, help="Fixed STATIC FOV lower theta edge in degrees.")
    parser.add_argument("--theta-max-deg", type=float, default=STATIC_FOV_THETA_MAX_DEG, help="Fixed STATIC FOV upper theta edge in degrees.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--min-source-age-seconds",
        type=float,
        default=60.0,
        help="Skip source MAT files modified more recently than this, to avoid half-written products.",
    )
    return parser.parse_args()


def discover_dates(input_root: Path, species: str) -> list[str]:
    dates: set[str] = set()
    pattern = re.compile(rf"maven_static_highE_{re.escape(species)}_(\d{{8}})\.mat$")
    for path in input_root.rglob(f"maven_static_highE_{species}_*.mat"):
        match = pattern.match(path.name)
        if match:
            dates.add(match.group(1))
    return sorted(dates)


def species_product_path(input_root: Path, species: str, date: str) -> Path:
    return input_root / date[:4] / f"maven_static_highE_{species}_{date}.mat"


def output_paths(output_root: Path, date: str) -> tuple[Path, Path]:
    out_dir = output_root / date[:4]
    return (
        out_dir / f"maven_static_plus_z_fov_flag_{date}.mat",
        out_dir / f"maven_static_plus_z_fov_flag_{date}_summary.json",
    )


def axes_mse_from_mat(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [
            np.column_stack([np.asarray(data[f"inst_{axis}_axis_mse_{comp}"], dtype=float).reshape(-1) for comp in "xyz"])
            for axis in "xyz"
        ],
        axis=1,
    )


def build_flag_payload(source_path: Path, theta_min_deg: float, theta_max_deg: float) -> tuple[dict[str, np.ndarray], dict]:
    variables = ["epoch_unix_s", "utc"] + [f"inst_{axis}_axis_mse_{comp}" for axis in "xyz" for comp in "xyz"]
    data = loadmat(source_path, variable_names=variables)
    axes_mse = axes_mse_from_mat(data)
    flag, theta_deg, target_static = mse_plus_z_static_fov_flag(axes_mse, theta_min_deg, theta_max_deg)
    epoch_unix_s = np.asarray(data["epoch_unix_s"], dtype=float).reshape(-1)
    utc = data.get("utc")
    if utc is None:
        utc = unix_s_to_utc_text(epoch_unix_s)
    else:
        utc = np.asarray(utc, dtype=object).reshape(-1)

    payload = {
        "epoch_unix_s": epoch_unix_s[:, None],
        "utc": utc[:, None],
        "mse_plus_z_in_static_fov_flag": flag[:, None],
        "mse_plus_z_static_theta_deg": theta_deg[:, None],
        "mse_plus_z_static_x": target_static[:, 0:1],
        "mse_plus_z_static_y": target_static[:, 1:2],
        "mse_plus_z_static_z": target_static[:, 2:3],
        "static_fov_theta_min_deg": np.full((epoch_unix_s.size, 1), float(theta_min_deg)),
        "static_fov_theta_max_deg": np.full((epoch_unix_s.size, 1), float(theta_max_deg)),
    }
    valid_flag = np.isfinite(flag)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_path),
        "rows": int(epoch_unix_s.size),
        "valid_flag_rows": int(np.count_nonzero(valid_flag)),
        "invalid_flag_rows": int(np.count_nonzero(~valid_flag)),
        "in_fov_rows": int(np.count_nonzero(flag == 1)),
        "out_fov_rows": int(np.count_nonzero(flag == 0)),
        "in_fov_fraction": float(np.nanmean(flag)) if np.any(valid_flag) else None,
        "theta_min_deg": float(theta_min_deg),
        "theta_max_deg": float(theta_max_deg),
        "method": "STATIC FOV is fixed in STATIC coordinates. MSE +Z is projected onto STATIC axes in MSE; flag=1 when theta_min_deg <= static theta <= theta_max_deg, 0 when outside, NaN when not determinable.",
    }
    return payload, summary


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root) if args.output_root else input_root
    dates = args.dates if args.dates is not None and len(args.dates) else discover_dates(input_root, args.species)
    if not dates:
        raise FileNotFoundError(f"No maven_static_highE_{args.species}_YYYYMMDD.mat files found under {input_root}")

    written = 0
    skipped_existing = 0
    skipped_missing = 0
    skipped_recent = 0
    failed_read = 0
    for date in dates:
        source_path = species_product_path(input_root, args.species, date)
        if not source_path.exists():
            skipped_missing += 1
            if not args.quiet:
                print(f"SKIP {date}: missing {source_path}")
            continue
        mat_path, json_path = output_paths(output_root, date)
        if not args.overwrite and (mat_path.exists() or json_path.exists()):
            skipped_existing += 1
            if not args.quiet:
                print(f"SKIP {date}: output exists; use --overwrite to replace")
            continue
        source_age_s = time.time() - source_path.stat().st_mtime
        if source_age_s < args.min_source_age_seconds:
            skipped_recent += 1
            if not args.quiet:
                print(f"SKIP {date}: source modified {source_age_s:.1f}s ago")
            continue
        try:
            payload, summary = build_flag_payload(source_path, args.theta_min_deg, args.theta_max_deg)
        except Exception as exc:
            failed_read += 1
            if not args.quiet:
                print(f"SKIP {date}: failed to read {source_path}: {type(exc).__name__}: {exc}")
            continue
        if not args.quiet:
            print(
                f"{date}: rows={summary['rows']} in_fov={summary['in_fov_rows']} "
                f"out_fov={summary['out_fov_rows']} invalid={summary['invalid_flag_rows']} "
                f"fraction={summary['in_fov_fraction']}"
            )
        if args.dry_run:
            continue
        mat_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(mat_path, payload)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
    print(
        f"finished: written={written} skipped_existing={skipped_existing} skipped_missing={skipped_missing} "
        f"skipped_recent={skipped_recent} failed_read={failed_read}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
