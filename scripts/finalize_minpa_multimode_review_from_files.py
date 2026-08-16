"""Finalize independent MINPA Mode-4/12 decisions from retained PNG files."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.minpa_multimode_review import finalize_species_review  # noqa: E402


DEFAULT_ROOT = ROOT / "outputs" / "minpa_multimode_noise_review_v2.0.0-rc1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    index_path = args.workspace / "candidate_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    result = finalize_species_review(
        payload["candidates"],
        args.workspace / "pending",
        review_version=str(payload.get("review_version", "unknown")),
    )
    result["finalized_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    result["source_candidate_index"] = str(index_path.resolve())
    result["source_candidate_index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    output = args.output or args.workspace / "manual_review_final.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing final review: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
