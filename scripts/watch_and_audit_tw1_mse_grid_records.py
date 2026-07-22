"""Wait for a Tianwen-1 grid writer, then verify and audit its full output."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highE.tw1_mse_grid_audit import audit_tw1_mse_grid_records, cleanup_empty_staging_groups
from highE.tw1_mse_grid_records import DEFAULT_GRID_OUTPUT_ROOT, DEFAULT_INPUT_ROOT


SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x00000102


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _windows_process_running(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
    if not handle:
        return False
    try:
        return int(kernel32.WaitForSingleObject(handle, 0)) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def _production_counts(output_root: Path) -> dict[str, Any]:
    progress_path = output_root / "logs" / "progress.json"
    final_path = output_root / "logs" / "latest_run_summary.json"
    progress = _read_json(progress_path) if progress_path.exists() else {}
    final = _read_json(final_path) if final_path.exists() else {}
    return {
        "progress_requested_days": progress.get("requested_days"),
        "progress_complete_days": progress.get("complete_or_already_complete_days"),
        "progress_last_date": progress.get("last_completed_date"),
        "final_requested_days": final.get("requested_days"),
        "final_processed_days": final.get("processed_days"),
        "final_already_complete_days": final.get("already_complete_days"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writer-pid", type=int, required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_GRID_OUTPUT_ROOT)
    parser.add_argument("--expected-days", type=int, default=1029)
    parser.add_argument("--writer-stderr", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    status_path = output_root / "logs" / "audit_watcher_status.json"
    stderr_path = args.writer_stderr or (output_root / "logs" / "full_run.stderr.log")
    base = {
        "watcher_pid": os.getpid(),
        "writer_pid": args.writer_pid,
        "input_root": str(args.input_root.resolve()),
        "output_root": str(output_root),
        "expected_days": args.expected_days,
        "writer_stderr": str(stderr_path.resolve()),
    }
    _atomic_json(status_path, {**base, "updated_utc": datetime.now(UTC).isoformat(), "status": "watching"})
    while _windows_process_running(args.writer_pid):
        counts = _production_counts(output_root)
        _atomic_json(
            status_path,
            {**base, **counts, "updated_utc": datetime.now(UTC).isoformat(), "status": "watching"},
        )
        time.sleep(max(1.0, args.poll_seconds))

    time.sleep(2.0)
    try:
        counts = _production_counts(output_root)
        stderr_bytes = stderr_path.stat().st_size if stderr_path.exists() else 0
        final_complete = int(counts.get("final_processed_days") or 0) + int(
            counts.get("final_already_complete_days") or 0
        )
        production_complete = (
            counts.get("progress_requested_days") == args.expected_days
            and counts.get("progress_complete_days") == args.expected_days
            and counts.get("final_requested_days") == args.expected_days
            and final_complete == args.expected_days
            and stderr_bytes == 0
        )
        if not production_complete:
            _atomic_json(
                status_path,
                {
                    **base,
                    **counts,
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "status": "production_incomplete",
                    "writer_stderr_bytes": stderr_bytes,
                },
            )
            return 2

        _atomic_json(
            status_path,
            {
                **base,
                **counts,
                "updated_utc": datetime.now(UTC).isoformat(),
                "status": "cleaning_empty_staging",
                "writer_stderr_bytes": stderr_bytes,
            },
        )
        cleanup = cleanup_empty_staging_groups(output_root)
        if cleanup["nonempty_staging_groups"]:
            _atomic_json(
                status_path,
                {
                    **base,
                    **counts,
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "status": "nonempty_staging_found",
                    "staging_cleanup": cleanup,
                },
            )
            return 3

        _atomic_json(
            status_path,
            {
                **base,
                **counts,
                "updated_utc": datetime.now(UTC).isoformat(),
                "status": "auditing",
                "staging_cleanup": cleanup,
            },
        )
        audit = audit_tw1_mse_grid_records(
            args.input_root,
            output_root,
            expected_days=args.expected_days,
            require_no_staging=True,
        )
        status = "complete" if audit["status"] == "pass" else "audit_failed"
        _atomic_json(
            status_path,
            {
                **base,
                **counts,
                "updated_utc": datetime.now(UTC).isoformat(),
                "status": status,
                "staging_cleanup": cleanup,
                "audit_summary": str(output_root / "logs" / "full_audit_summary.json"),
                "audit_error_count": audit["error_count"],
            },
        )
        return 0 if status == "complete" else 4
    except Exception as exc:  # pragma: no cover - production failure capture
        _atomic_json(
            status_path,
            {
                **base,
                "updated_utc": datetime.now(UTC).isoformat(),
                "status": "watcher_error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
