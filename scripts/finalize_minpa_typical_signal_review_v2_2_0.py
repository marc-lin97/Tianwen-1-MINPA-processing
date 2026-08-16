"""Finalize the reviewed MINPA v2.2.0 signal examples from retained PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_example_review import finalize_signal_example_review  # noqa: E402


DEFAULT_WORKSPACE = ROOT / "outputs/minpa_typical_noise_signal_review_v2.2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-index", type=Path, default=DEFAULT_WORKSPACE / "review_index.json")
    parser.add_argument("--pending", type=Path, default=DEFAULT_WORKSPACE / "pending")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_WORKSPACE / "manual_review_final.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = json.loads(args.review_index.read_text(encoding="utf-8"))
    final = finalize_signal_example_review(index, args.pending)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(final["counts"], ensure_ascii=False))
    print(f"final={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
