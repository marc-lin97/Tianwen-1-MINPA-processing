"""Audit local inputs required by the MINPA background paper.

The command is read-only with respect to mission data.  It writes a compact
JSON manifest under the derived reproduction output directory and never
downloads mission data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "minpa_background_paper_reproduction"
    / "logs"
    / "data_availability_manifest.json"
)
DEFAULT_PAPER_PDF = ROOT / "output" / "pdf" / "wang2024_minpa_background.pdf"
PAPER_PDF_URL = (
    "https://space.ustc.edu.cn/users/"
    "1157234616JDEkdTA1LmZoMy4kUjdEZ2xBRXo4WDhYV3RaNjl5NUxZMA/"
    "publication/materials/20240712014437.048_1-s2.0-s0273117724007993-main.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ori-root", type=Path, default=Path(r"D:\Data\TW-1\result\MINPA\ori"))
    parser.add_argument(
        "--day-spe-root", type=Path, default=Path(r"D:\Data\TW-1\result\MINPA\day_spe")
    )
    parser.add_argument(
        "--tw1-momag-root",
        type=Path,
        default=Path(r"D:\Data\TW-1\result\MOMAG\C\01Hz_all"),
    )
    parser.add_argument(
        "--maven-mag-root",
        type=Path,
        default=Path(r"D:\Data\MAVEN\result\mag\ss1s"),
    )
    parser.add_argument("--paper-pdf", type=Path, default=DEFAULT_PAPER_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = date(2021, 12, 1)
    stop = date(2022, 2, 1)
    requested_days = [value.strftime("%Y%m%d") for value in _dates(start, stop)]

    ori_files = sorted(args.ori_root.glob("*MINPA-MOD1-*.mat"))
    ori_coverage = _ori_covered_days(ori_files, start, stop)
    ori_gaps = _mode1_filename_gaps(ori_files, start, stop)
    day_spe_files = sorted(args.day_spe_root.glob("Ion_spe_*.mat"))
    day_spe_coverage = sorted(
        path.stem.rsplit("_", 1)[-1]
        for path in day_spe_files
        if "20211201" <= path.stem.rsplit("_", 1)[-1] <= "20220131"
    )

    payload: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "acquisition_policy": {
            "mission_data_access": "local_read_only",
            "network_download_performed": False,
            "note": "MINPA and context-only MOMAG/MAVEN MAG inputs are read from local storage; SWIA is outside the background-estimation scope.",
        },
        "paper": {
            "title": "Analysis of the background signal in Tianwen-1 MINPA",
            "doi": "10.1016/j.asr.2024.07.080",
            "analysis_interval_utc": ["2021-12-01T00:00:00Z", "2022-02-01T00:00:00Z"],
            "data_statement": (
                "MAVEN SWIA from NASA PDS; Tianwen-1 MINPA and MOMAG may be "
                "requested from the CNSA Data Release System."
            ),
            "pdf_source_url": PAPER_PDF_URL,
            "local_pdf": _file_manifest(args.paper_pdf),
        },
        "local": {
            "requested_day_count": len(requested_days),
            "minpa_ori_mode1": _coverage_manifest(
                args.ori_root, ori_files, ori_coverage, requested_days
            ),
            "minpa_day_spe": _coverage_manifest(
                args.day_spe_root, day_spe_files, day_spe_coverage, requested_days
            ),
            "tw1_momag": _daily_mat_manifest(args.tw1_momag_root, requested_days),
            "maven_mag_position": _daily_mat_manifest(args.maven_mag_root, requested_days),
        },
        "official_documentation": {
            "data_structure_url": "https://www.swl.ac.cn/minpa/data-structure/structure/",
            "role": "metadata and dimensional interpretation only; not used to acquire mission data",
        },
    }
    payload["local"]["minpa_ori_mode1"].update(
        {
            "coverage_scope": "files physically present under the configured local root; not a statement about mission-wide availability",
            "continuous_gaps_from_filename_intervals": ori_gaps,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


def _dates(start: date, stop: date):
    current = start
    while current < stop:
        yield current
        current += timedelta(days=1)


def _ori_covered_days(files: list[Path], start: date, stop: date) -> list[str]:
    covered: set[str] = set()
    for path in files:
        match = re.search(r"_(\d{14})_(\d{14})_", path.name)
        if not match:
            continue
        file_start = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").date()
        file_stop = datetime.strptime(match.group(2), "%Y%m%d%H%M%S").date()
        current = max(start, file_start)
        final = min(stop - timedelta(days=1), file_stop)
        while current <= final:
            covered.add(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
    return sorted(covered)


def _coverage_manifest(
    root: Path,
    files: list[Path],
    covered_days: list[str],
    requested_days: list[str],
) -> dict[str, Any]:
    covered = set(covered_days)
    return {
        "root": str(root.resolve()),
        "file_count_in_root": len(files),
        "covered_days_in_paper_interval": sorted(covered),
        "covered_day_count": len(covered),
        "missing_days_in_paper_interval": [day for day in requested_days if day not in covered],
        "complete_for_paper_interval": covered.issuperset(requested_days),
    }


def _daily_mat_manifest(root: Path, requested_days: list[str]) -> dict[str, Any]:
    present = [day for day in requested_days if (root / f"Bss{day}.mat").exists()]
    return _coverage_manifest(
        root,
        [root / f"Bss{day}.mat" for day in present],
        present,
        requested_days,
    )


def _file_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.resolve()), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _mode1_filename_gaps(
    files: list[Path], start: date, stop: date
) -> list[dict[str, Any]]:
    start_s = datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp()
    stop_s = datetime.combine(stop, datetime.min.time(), tzinfo=UTC).timestamp()
    intervals: list[tuple[float, float, str]] = []
    for path in files:
        match = re.search(r"_(\d{14})_(\d{14})_", path.name)
        if not match:
            continue
        left = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()
        right = datetime.strptime(match.group(2), "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()
        if right > start_s and left < stop_s:
            intervals.append((max(left, start_s), min(right, stop_s), path.name))
    intervals.sort()
    gaps: list[dict[str, Any]] = []
    cursor = start_s
    preceding: str | None = None
    for left, right, name in intervals:
        if left > cursor:
            gaps.append(
                {
                    "start_utc": datetime.fromtimestamp(cursor, UTC).isoformat().replace("+00:00", "Z"),
                    "stop_utc": datetime.fromtimestamp(left, UTC).isoformat().replace("+00:00", "Z"),
                    "duration_s": left - cursor,
                    "preceding_file": preceding,
                    "following_file": name,
                }
            )
        if right > cursor:
            cursor = right
            preceding = name
    if cursor < stop_s:
        gaps.append(
            {
                "start_utc": datetime.fromtimestamp(cursor, UTC).isoformat().replace("+00:00", "Z"),
                "stop_utc": datetime.fromtimestamp(stop_s, UTC).isoformat().replace("+00:00", "Z"),
                "duration_s": stop_s - cursor,
                "preceding_file": preceding,
                "following_file": None,
            }
        )
    # The normal inter-orbit gaps are expected.  Keep only gaps at least one
    # day long so the manifest highlights missing local coverage, not cadence.
    return [item for item in gaps if float(item["duration_s"]) >= 86400.0]


if __name__ == "__main__":
    main()
