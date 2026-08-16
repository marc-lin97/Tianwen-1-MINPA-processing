"""Export calibration times and minimal example metadata for GitHub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_public_release import (  # noqa: E402
    calibration_time_rows,
    public_example_manifest,
    write_calibration_time_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode1-inventory",
        type=Path,
        default=ROOT / "outputs/minpa_unified_static_channel_denoise_v2.2.0/review_inventory/mode1_all_approved.json",
    )
    parser.add_argument(
        "--multimode-inventory",
        type=Path,
        default=ROOT / "outputs/minpa_multimode_noise_review_secondary_prefilter_v2/manual_review_final.json",
    )
    parser.add_argument(
        "--example-review",
        type=Path,
        default=ROOT / "outputs/minpa_typical_noise_signal_review_v2.2.0/manual_review_final.json",
    )
    parser.add_argument(
        "--time-output",
        type=Path,
        default=ROOT / "release/minpa_unified_static_channel_denoise_v2.2.0/calibration_intervals_approved.csv",
    )
    parser.add_argument(
        "--example-output",
        type=Path,
        default=ROOT / "examples/minpa_denoise_v2.2.0/manifest.json",
    )
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    rows = calibration_time_rows(_read(args.mode1_inventory), _read(args.multimode_inventory))
    write_calibration_time_csv(args.time_output, rows)
    public = public_example_manifest(_read(args.example_review))
    args.example_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.example_output.with_suffix(args.example_output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.example_output)
    counts: dict[str, int] = {}
    for row in rows:
        key = f"mode{int(row['mode']):02d}_{row['species']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"calibration_rows": len(rows), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
