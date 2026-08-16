"""Verify the committed MINPA v2.2.0 bundle and reviewed example assets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import apply_background  # noqa: E402
from highE.minpa_example_review import sha256_file  # noqa: E402
from highE.minpa_multimode_background import apply_multimode_background  # noqa: E402
from highE.tw1_minpa import load_unified_static_background_bundle  # noqa: E402


EXPECTED_BUNDLE_SHA256 = "0c21995c2b3baf7ba82e0893aa0aa5c81c17f268787b28e3d75181dcac468c71"
DEFAULT_RELEASE = ROOT / "release/minpa_unified_static_channel_denoise_v2.2.0"
DEFAULT_EXAMPLES = ROOT / "examples/minpa_denoise_v2.2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_path = args.release / "bundle.json"
    bundle_hash = sha256_file(bundle_path)
    if bundle_hash != EXPECTED_BUNDLE_SHA256:
        raise ValueError(f"Unexpected bundle SHA-256: {bundle_hash}")
    mode1, multimode = load_unified_static_background_bundle(bundle_path)

    raw1 = np.where(mode1.valid_channel_mask, 2.0 * mode1.background_dpf, 0.0)[None, ...]
    corrected1 = apply_background(raw1, mode1).corrected_dpf
    monotonic = bool(np.all(corrected1 <= raw1))
    mode_shapes = {"1": list(mode1.background_dpf.shape)}
    for mode in (4, 12):
        model = multimode[mode]
        raw = np.where(model.valid_channel_mask, 2.0 * model.background_dpf, 0.0)[None, ...]
        corrected = apply_multimode_background(raw, model).corrected_dpf
        monotonic = monotonic and bool(np.all(corrected <= raw))
        mode_shapes[str(mode)] = list(model.background_dpf.shape)
    if not monotonic:
        raise AssertionError("Synthetic corrected DPF exceeds raw DPF")

    manifest = json.loads((args.examples / "manifest.json").read_text(encoding="utf-8"))
    if manifest["bundle_sha256"] != bundle_hash:
        raise ValueError("Example manifest references a different bundle")
    verified_figures = 0
    for row in manifest["signal_examples"]:
        path = args.examples / str(row["figure"])
        actual = sha256_file(path)
        if actual != str(row["figure_sha256"]):
            raise ValueError(f"Example SHA-256 mismatch: {path}")
        verified_figures += 1
    if verified_figures != 9:
        raise ValueError(f"Expected 9 reviewed before/after figures, got {verified_figures}")

    with (args.release / "calibration_intervals_approved.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        calibration_rows = list(reader)
    expected_fields = ["mode", "species", "start_utc", "stop_utc"]
    if reader.fieldnames != expected_fields or len(calibration_rows) != 3174:
        raise ValueError(
            f"Unexpected public calibration time table: fields={reader.fieldnames}, "
            f"rows={len(calibration_rows)}"
        )

    report = {
        "status": "verified",
        "bundle_sha256": bundle_hash,
        "mode_shapes": mode_shapes,
        "review_counts": manifest["counts"],
        "verified_example_figures": verified_figures,
        "calibration_time_rows": len(calibration_rows),
        "synthetic_correction_not_above_raw": monotonic,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
