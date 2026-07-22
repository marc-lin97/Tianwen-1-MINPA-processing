"""Merge standalone MAVEN STATIC +Z_MSE FOV flags into O+ high-E MAT files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


FLAG_FIELDS = (
    "mse_plus_z_in_static_fov_flag",
    "mse_plus_z_static_theta_deg",
    "mse_plus_z_static_x",
    "mse_plus_z_static_y",
    "mse_plus_z_static_z",
    "static_fov_theta_min_deg",
    "static_fov_theta_max_deg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"E:\Data\highE\MAVEN", help="Root containing MAVEN/YYYY MAT products.")
    parser.add_argument("--dates", nargs="*", default=None, help="Optional YYYYMMDD dates. If omitted, scan root.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing integrated flag fields if they differ.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--min-source-age-seconds",
        type=float,
        default=120.0,
        help="Skip O+ or flag files modified more recently than this, to avoid half-written products.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON summary path. Defaults to root/logs/maven_plus_z_flag_integration_UTC.json.",
    )
    return parser.parse_args()


def discover_dates(root: Path) -> list[str]:
    dates: set[str] = set()
    pattern = re.compile(r"maven_static_highE_Oplus_(\d{8})\.mat$")
    for year_dir in root.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for path in year_dir.glob("maven_static_highE_Oplus_*.mat"):
            match = pattern.match(path.name)
            if match:
                dates.add(match.group(1))
    return sorted(dates)


def oplus_path(root: Path, date: str) -> Path:
    return root / date[:4] / f"maven_static_highE_Oplus_{date}.mat"


def flag_path(root: Path, date: str) -> Path:
    return root / date[:4] / f"maven_static_plus_z_fov_flag_{date}.mat"


def load_mat_payload(path: Path) -> dict[str, np.ndarray]:
    return {key: value for key, value in loadmat(path).items() if not key.startswith("__")}


def load_flag_payload(path: Path) -> dict[str, np.ndarray]:
    names = ["epoch_unix_s", *FLAG_FIELDS]
    return {key: value for key, value in loadmat(path, variable_names=names).items() if not key.startswith("__")}


def same_array(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return bool(np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True))
    return bool(np.array_equal(a, b))


def arrays_aligned_by_epoch(oplus: dict[str, np.ndarray], flag: dict[str, np.ndarray]) -> bool:
    left = np.asarray(oplus["epoch_unix_s"], dtype=float).reshape(-1)
    right = np.asarray(flag["epoch_unix_s"], dtype=float).reshape(-1)
    return left.shape == right.shape and bool(np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=False))


def existing_flags_match(oplus: dict[str, np.ndarray], flag: dict[str, np.ndarray]) -> bool:
    return all(field in oplus and same_array(oplus[field], flag[field]) for field in FLAG_FIELDS)


def source_is_recent(paths: tuple[Path, ...], min_age_seconds: float) -> bool:
    if min_age_seconds <= 0.0:
        return False
    now = time.time()
    return any(now - path.stat().st_mtime < min_age_seconds for path in paths)


def integrate_one(root: Path, date: str, *, dry_run: bool, overwrite: bool, min_source_age_seconds: float) -> dict[str, object]:
    op_path = oplus_path(root, date)
    fl_path = flag_path(root, date)
    row: dict[str, object] = {"date": date, "oplus_path": str(op_path), "flag_path": str(fl_path)}
    if not op_path.exists():
        return {**row, "status": "missing_oplus"}
    if not fl_path.exists():
        return {**row, "status": "missing_flag"}
    if source_is_recent((op_path, fl_path), min_source_age_seconds):
        return {**row, "status": "skipped_recent_source"}

    try:
        oplus = load_mat_payload(op_path)
        flag = load_flag_payload(fl_path)
    except Exception as exc:
        return {**row, "status": "read_failed", "error": f"{type(exc).__name__}: {exc}"}

    missing_fields = [field for field in FLAG_FIELDS if field not in flag]
    if missing_fields:
        return {**row, "status": "missing_flag_fields", "missing_fields": missing_fields}
    if not arrays_aligned_by_epoch(oplus, flag):
        return {
            **row,
            "status": "epoch_mismatch",
            "oplus_rows": int(np.asarray(oplus.get("epoch_unix_s", [])).reshape(-1).size),
            "flag_rows": int(np.asarray(flag.get("epoch_unix_s", [])).reshape(-1).size),
        }

    already_has_all = all(field in oplus for field in FLAG_FIELDS)
    if already_has_all and existing_flags_match(oplus, flag):
        return {**row, "status": "already_integrated", "rows": int(np.asarray(flag["epoch_unix_s"]).reshape(-1).size)}
    if already_has_all and not overwrite:
        return {**row, "status": "existing_fields_differ"}

    for field in FLAG_FIELDS:
        oplus[field] = flag[field]

    rows = int(np.asarray(flag["epoch_unix_s"]).reshape(-1).size)
    if dry_run:
        return {**row, "status": "would_write", "rows": rows}

    tmp_path = op_path.with_name(f"{op_path.stem}.tmp_{os.getpid()}.mat")
    try:
        savemat(tmp_path, oplus)
        os.replace(tmp_path, op_path)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return {**row, "status": "write_failed", "rows": rows, "error": f"{type(exc).__name__}: {exc}"}
    return {**row, "status": "written", "rows": rows}


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    dates = args.dates if args.dates is not None and len(args.dates) else discover_dates(root)
    if not dates:
        raise FileNotFoundError(f"No Oplus products found under {root}")

    started = time.perf_counter()
    rows = []
    counts: dict[str, int] = {}
    for date in dates:
        item = integrate_one(
            root,
            date,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            min_source_age_seconds=args.min_source_age_seconds,
        )
        rows.append(item)
        status = str(item["status"])
        counts[status] = counts.get(status, 0) + 1
        if not args.quiet:
            print(f"{date}: {status}")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "dates_requested": len(dates),
        "status_counts": counts,
        "elapsed_seconds": time.perf_counter() - started,
        "flag_fields": list(FLAG_FIELDS),
        "rows": rows,
    }
    summary_path = Path(args.summary_path) if args.summary_path else root / "logs" / f"maven_plus_z_flag_integration_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"finished: {counts} summary={summary_path if not args.dry_run else 'dry-run-not-written'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
