"""Reproduce Wang et al. (2024) Figure 5 from local MINPA Mode-1 DPF."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_background import load_background_model  # noqa: E402
from highE.minpa_figure5 import plot_figure5  # noqa: E402


DEFAULT_ORI_ROOT = Path(r"D:\Data\TW-1\result\MINPA\ori")
DEFAULT_QUALITY_ROOT = ROOT / "outputs" / "tw1_minpa_quality_flags_all_species"
DEFAULT_MODEL = (
    ROOT
    / "outputs"
    / "minpa_background_paper_reproduction"
    / "data"
    / "minpa_mode1_background_model_provisional.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "minpa_background_paper_reproduction" / "figures"
)
DEFAULT_LOG = (
    ROOT
    / "outputs"
    / "minpa_background_paper_reproduction"
    / "logs"
    / "figure05_reproduction.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ori-root", type=Path, default=DEFAULT_ORI_ROOT)
    parser.add_argument("--quality-root", type=Path, default=DEFAULT_QUALITY_ROOT)
    parser.add_argument("--background-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_background_model(args.background_model)
    summary = plot_figure5(args.ori_root, args.quality_root, model, args.output)
    summary.update(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "paper": {
                "title": "Analysis of the background signal in Tianwen-1 MINPA",
                "doi": "10.1016/j.asr.2024.07.080",
            },
            "input_policy": "local mission data read-only",
            "ori_root": str(args.ori_root.resolve()),
            "quality_root": str(args.quality_root.resolve()),
            "background_model": str(args.background_model.resolve()),
            "background_model_sha256": hashlib.sha256(
                args.background_model.read_bytes()
            ).hexdigest(),
        }
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
