from __future__ import annotations

import csv
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ORI_DIR = Path(r"D:\Data\TW-1\result\MINPA\ori")
OUT_CSV = Path("outputs/tw1_minpa_quality_flag_audit_altshape_20220531/bit5_hplus_candidates_hdf5.csv")
THRESHOLD = 1e10

ENERGY_MODE1 = np.array(
    [
        2.81, 3.548928, 4.482167, 5.660813, 7.149401, 9.029433, 11.40385,
        14.40264, 18.19001, 22.97332, 29.01447, 36.64422, 46.28032,
        58.45036, 73.82067, 93.23282, 117.7497, 148.7135, 187.8198,
        237.2095, 299.587, 378.3675, 477.8644, 603.5253, 762.2305,
        962.6694, 1215.816, 1535.532, 1939.321, 2449.292, 3093.366,
        3906.809, 4934.157, 6231.661, 7870.361, 9939.98, 12553.83,
        15855.03, 20024.33, 25290,
    ],
    dtype=float,
)


def decode_matlab_string(f: h5py.File, ref) -> str:
    arr = np.asarray(f[ref]).squeeze()
    return "".join(chr(int(x)) for x in arr)


def unix_from_utc(text: str) -> float:
    dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def scan_file(path: Path) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    with h5py.File(path, "r") as f:
        root = f["tw1_MINPA_data"]
        utc_group = f[root["value"][0, 0]]
        data_group = f[root["value"][41, 0]]
        t_refs = data_group["t"]
        utc_refs = utc_group["t"]
        energy_by_channel = np.repeat(ENERGY_MODE1, 64)

        for i in range(t_refs.shape[0]):
            row = np.asarray(f[t_refs[i, 0]]).squeeze()
            if row.size != 20480:
                continue
            h_flat = row[0::8]
            def_flat = h_flat * energy_by_channel
            max_idx = int(np.nanargmax(def_flat))
            max_def = float(def_flat[max_idx])
            if not math.isfinite(max_def) or max_def <= THRESHOLD:
                continue

            energy_idx = max_idx // 64
            angle_idx = max_idx % 64 + 1
            utc = decode_matlab_string(f, utc_refs[i, 0])
            hits.append(
                {
                    "source_file": str(path),
                    "record_index": i + 1,
                    "time_utc": utc,
                    "time_unix_s": unix_from_utc(utc),
                    "mode": 1,
                    "max_def_2d": max_def,
                    "max_def_energy_eV": float(ENERGY_MODE1[energy_idx]),
                    "max_def_angle_index": angle_idx,
                }
            )
    return hits


def main() -> None:
    files = sorted(ORI_DIR.glob("*MINPA-MOD1-DEF*.mat"))
    rows: list[dict[str, object]] = []
    for n, path in enumerate(files, start=1):
        try:
            hits = scan_file(path)
        except Exception as exc:  # Keep a broad scanner moving across mixed products.
            print(f"WARN {path.name}: {exc}")
            continue
        rows.extend(hits)
        if hits:
            print(f"hit file {n}/{len(files)} {path.name} hits={len(hits)} total={len(rows)}")
        elif n % 100 == 0:
            print(f"scanned {n}/{len(files)} total_hits={len(rows)}")
        if len(rows) >= 2:
            break

    rows.sort(key=lambda r: float(r["max_def_2d"]), reverse=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_file", "record_index", "time_utc", "time_unix_s", "mode",
            "max_def_2d", "max_def_energy_eV", "max_def_angle_index",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_CSV.resolve())
    print(f"hits={len(rows)}")


if __name__ == "__main__":
    main()
