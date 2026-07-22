"""Run all available MAVEN high-E days with adaptive worker scaling."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class ActiveRun:
    date: str
    process: subprocess.Popen
    log_handle: TextIO
    started: float
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", default=r"D:\Data\MAVEN\rawdata\static\l2\d1-32e4d16a8m")
    parser.add_argument("--mag-root", default=r"D:\Data\MAVEN\result\mag\ss1s")
    parser.add_argument("--r-root", default=r"D:\Data\MAVEN\result\R_MSO2MSE")
    parser.add_argument("--output-root", default=r"E:\Data\highE")
    parser.add_argument("--start-date", default=None, help="Optional first date, YYYYMMDD.")
    parser.add_argument("--end-date", default=None, help="Optional last date, YYYYMMDD.")
    parser.add_argument("--initial-workers", type=int, default=4)
    parser.add_argument("--min-workers", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--worker-step", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--pressure-threshold-percent", type=float, default=80.0)
    parser.add_argument("--scale-window-seconds", type=float, default=600.0)
    parser.add_argument("--monitor-interval-seconds", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--date-file", default=None, help="Optional text file with one YYYYMMDD date per line.")
    return parser.parse_args()


def discover_dates(d1_root: Path, mag_root: Path, r_root: Path) -> list[str]:
    d1_dates = _dates_from_files(d1_root, "mvn_sta_l2_d1-32e4d16a8m_*_v*_r*.cdf", r"_(\d{8})_v")
    mag_dates = _dates_from_files(mag_root, "Bss*.mat", r"Bss(\d{8})\.mat")
    r_dates = _dates_from_files(r_root, "R_MSO2MSE_MAVEN_*.mat", r"R_MSO2MSE_MAVEN_(\d{8})\.mat")
    return sorted(d1_dates & mag_dates & r_dates)


def _dates_from_files(root: Path, glob_pattern: str, regex: str) -> set[str]:
    out: set[str] = set()
    compiled = re.compile(regex)
    for path in root.rglob(glob_pattern):
        match = compiled.search(path.name)
        if match:
            out.add(match.group(1))
    return out


def load_dates(args: argparse.Namespace) -> list[str]:
    if args.date_file:
        dates = [line.strip() for line in Path(args.date_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        dates = discover_dates(Path(args.d1_root), Path(args.mag_root), Path(args.r_root))
    if args.start_date:
        dates = [date for date in dates if date >= args.start_date]
    if args.end_date:
        dates = [date for date in dates if date <= args.end_date]
    return dates


def output_complete(output_root: Path, date: str) -> bool:
    out_dir = output_root / "MAVEN" / date[:4]
    return (
        (out_dir / f"maven_static_highE_Oplus_{date}.mat").exists()
        and (out_dir / f"maven_static_highE_O2plus_{date}.mat").exists()
        and (out_dir / f"maven_static_highE_{date}_summary.json").exists()
    )


def memory_pressure_percent() -> float | None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return float(status.dwMemoryLoad)
    return None


def disk_pressure_percent() -> float | None:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "(Get-Counter '\\PhysicalDisk(_Total)\\% Disk Time').CounterSamples.CookedValue",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        value = float(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None
    return max(0.0, min(100.0, value))


def launch_date(args: argparse.Namespace, date: str, logs_dir: Path) -> ActiveRun:
    log_path = logs_dir / f"maven_highE_{date}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_maven_highE_day.py"),
        date,
        "--d1-root",
        args.d1_root,
        "--mag-root",
        args.mag_root,
        "--r-root",
        args.r_root,
        "--output-root",
        args.output_root,
    ]
    process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    return ActiveRun(date=date, process=process, log_handle=log_handle, started=time.perf_counter(), log_path=log_path)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    logs_dir = output_root / "MAVEN" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = logs_dir / f"adaptive_run_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"

    all_dates = load_dates(args)
    dates = [date for date in all_dates if args.overwrite or not output_complete(output_root, date)]
    queue = list(dates)
    current_workers = min(args.initial_workers, args.max_workers)
    active: list[ActiveRun] = []
    completed: list[str] = []
    failed: dict[str, int] = {}
    retries: dict[str, int] = {}
    pressure_window: list[dict[str, float | None]] = []
    next_monitor = 0.0
    next_scale_check = time.perf_counter() + args.scale_window_seconds
    started = time.perf_counter()

    with run_log_path.open("w", encoding="utf-8") as run_log:
        _write_event(
            run_log,
            "start",
            {
                "discovered_dates": len(all_dates),
                "queued_dates": len(queue),
                "skipped_existing": len(all_dates) - len(queue),
                "initial_workers": current_workers,
                "max_workers": args.max_workers,
            },
        )
        while queue or active:
            now = time.perf_counter()
            while queue and len(active) < current_workers:
                run = launch_date(args, queue.pop(0), logs_dir)
                active.append(run)
                _write_event(run_log, "launch", {"date": run.date, "workers": current_workers, "log": str(run.log_path)})

            still_active: list[ActiveRun] = []
            for run in active:
                rc = run.process.poll()
                if rc is None:
                    still_active.append(run)
                    continue
                run.log_handle.close()
                elapsed = time.perf_counter() - run.started
                log_text = ""
                try:
                    log_text = run.log_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    log_text = ""
                if rc == 0:
                    if "SKIP MAVEN" in log_text:
                        _write_event(run_log, "skipped", {"date": run.date, "elapsed_seconds": elapsed, "reason": _last_nonempty_line(log_text)})
                    else:
                        completed.append(run.date)
                        _write_event(run_log, "complete", {"date": run.date, "elapsed_seconds": elapsed})
                else:
                    if "MemoryError" in log_text and retries.get(run.date, 0) < args.max_retries:
                        retries[run.date] = retries.get(run.date, 0) + 1
                        queue.insert(0, run.date)
                        old_workers = current_workers
                        current_workers = max(args.min_workers, current_workers - 1)
                        _write_event(
                            run_log,
                            "retry_memory_error",
                            {
                                "date": run.date,
                                "returncode": int(rc),
                                "elapsed_seconds": elapsed,
                                "retry": retries[run.date],
                                "old_workers": old_workers,
                                "new_workers": current_workers,
                            },
                        )
                    else:
                        failed[run.date] = int(rc)
                        _write_event(run_log, "failed", {"date": run.date, "returncode": int(rc), "elapsed_seconds": elapsed, "last_log_line": _last_nonempty_line(log_text)})
            active = still_active

            if now >= next_monitor:
                sample = {
                    "t_elapsed_seconds": now - started,
                    "memory_percent": memory_pressure_percent(),
                    "disk_percent": disk_pressure_percent(),
                    "workers": float(current_workers),
                    "active": float(len(active)),
                    "remaining": float(len(queue)),
                }
                pressure_window.append(sample)
                _write_event(run_log, "pressure", sample)
                next_monitor = now + args.monitor_interval_seconds

            if now >= next_scale_check:
                finite_mem = [s["memory_percent"] for s in pressure_window if s["memory_percent"] is not None]
                finite_disk = [s["disk_percent"] for s in pressure_window if s["disk_percent"] is not None]
                max_mem = max(finite_mem) if finite_mem else None
                max_disk = max(finite_disk) if finite_disk else None
                can_scale = (
                    current_workers < args.max_workers
                    and max_mem is not None
                    and max_disk is not None
                    and max_mem < args.pressure_threshold_percent
                    and max_disk < args.pressure_threshold_percent
                )
                if can_scale:
                    old_workers = current_workers
                    current_workers = min(args.max_workers, current_workers + args.worker_step)
                    _write_event(run_log, "scale_up", {"old_workers": old_workers, "new_workers": current_workers, "max_mem": max_mem, "max_disk": max_disk})
                elif max_mem is not None and max_mem >= 90.0 and current_workers > args.min_workers:
                    old_workers = current_workers
                    current_workers = max(args.min_workers, current_workers - 1)
                    _write_event(run_log, "scale_down_memory", {"old_workers": old_workers, "new_workers": current_workers, "max_mem": max_mem, "max_disk": max_disk})
                else:
                    _write_event(run_log, "hold_workers", {"workers": current_workers, "max_mem": max_mem, "max_disk": max_disk})
                pressure_window = []
                next_scale_check = now + args.scale_window_seconds

            time.sleep(1.0)

        _write_event(
            run_log,
            "finish",
            {
                "completed": len(completed),
                "failed": failed,
                "elapsed_seconds": time.perf_counter() - started,
                "run_log": str(run_log_path),
            },
        )
    print(f"Adaptive MAVEN batch finished: completed={len(completed)} failed={len(failed)} log={run_log_path}")
    return 0 if not failed else 1


def _write_event(handle: TextIO, event: str, payload: dict) -> None:
    row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
