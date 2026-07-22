"""Compare two TW-1 MINPA H+ quality-flag audit tables for bit-4 changes.

The script writes complete bit-4 added/removed tables and a smaller example
table suitable for plot_tw1_minpa_quality_flag_context_examples.m.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_file", "record_index", "time_utc"]
BIT4_COLUMN = "flag_bit4_evenodd_error"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--removed-max", type=int, default=20)
    parser.add_argument("--window-minutes", type=float, default=5.0)
    return parser.parse_args()


def read_records(folder: Path) -> pd.DataFrame:
    path = folder / "tw1_minpa_hplus_quality_flag_audit_records.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS + [BIT4_COLUMN] if col not in records.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return records


def row_keys(df: pd.DataFrame) -> pd.Series:
    return df[KEY_COLUMNS].astype(str).agg("\x1f".join, axis=1)


def select_evenly_in_time(df: pd.DataFrame, max_count: int) -> pd.DataFrame:
    if max_count <= 0 or len(df) <= max_count:
        return df.copy()
    ordered = df.sort_values("time_unix_s", kind="mergesort").reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, max_count).round().astype(int)
    indices = np.unique(indices)
    return ordered.iloc[indices].copy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_records = read_records(args.old_dir)
    new_records = read_records(args.new_dir)

    old_bit4 = old_records[old_records[BIT4_COLUMN] == 1].copy()
    new_bit4 = new_records[new_records[BIT4_COLUMN] == 1].copy()

    old_keys = set(row_keys(old_bit4))
    new_keys = set(row_keys(new_bit4))

    old_records = old_records.assign(_key=row_keys(old_records))
    new_records = new_records.assign(_key=row_keys(new_records))

    removed_all_old = old_records[old_records["_key"].isin(old_keys - new_keys)].copy()
    added_all_new = new_records[new_records["_key"].isin(new_keys - old_keys)].copy()

    new_for_merge = new_records.set_index("_key")
    old_for_merge = old_records.set_index("_key")

    removed_all = removed_all_old.merge(
        new_for_merge[
            [
                "quality_flag",
                "quality_flag_binary",
                BIT4_COLUMN,
                "alternating_extrema_fraction",
                "alternating_extrema_count",
                "alternating_phase_fraction",
                "alternating_phase_count",
            ]
        ],
        left_on="_key",
        right_index=True,
        how="left",
        suffixes=("_old", "_new"),
    )
    added_all = added_all_new.merge(
        old_for_merge[
            [
                "quality_flag",
                "quality_flag_binary",
                BIT4_COLUMN,
                "alternating_extrema_fraction",
                "alternating_extrema_count",
                "alternating_phase_fraction",
                "alternating_phase_count",
            ]
        ],
        left_on="_key",
        right_index=True,
        how="left",
        suffixes=("_new", "_old"),
    )

    removed_examples_old = select_evenly_in_time(removed_all_old, args.removed_max)
    added_examples_new = added_all_new.sort_values("time_unix_s", kind="mergesort").copy()

    examples = pd.concat(
        [
            new_records[new_records["_key"].isin(removed_examples_old["_key"])].assign(
                category="removed_old_bit4_now_nonbit4"
            ),
            added_examples_new.assign(category="added_old_nonbit4_now_bit4"),
        ],
        ignore_index=True,
    )
    examples = examples.sort_values(["category", "time_unix_s"], kind="mergesort")

    removed_all.drop(columns=["_key"], errors="ignore").to_csv(
        args.output_dir / "tw1_minpa_hplus_quality_flag_evenodd_removed_all.csv",
        index=False,
    )
    added_all.drop(columns=["_key"], errors="ignore").to_csv(
        args.output_dir / "tw1_minpa_hplus_quality_flag_evenodd_added_all.csv",
        index=False,
    )
    examples.drop(columns=["_key"], errors="ignore").to_csv(
        args.output_dir / "tw1_minpa_hplus_quality_flag_evenodd_diff_examples.csv",
        index=False,
    )

    window_s = args.window_minutes * 60.0
    mask = np.zeros(len(new_records), dtype=bool)
    for t0 in examples["time_unix_s"].astype(float):
        mask |= (new_records["time_unix_s"] >= t0 - window_s) & (
            new_records["time_unix_s"] <= t0 + window_s
        )
    context_records = new_records.loc[mask].drop(columns=["_key"], errors="ignore")
    context_records.to_csv(
        args.output_dir / "tw1_minpa_hplus_quality_flag_evenodd_diff_records.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "old_records": len(old_records),
                "new_records": len(new_records),
                "old_bit4": len(old_bit4),
                "new_bit4": len(new_bit4),
                "common_bit4": len(old_keys & new_keys),
                "removed_old_bit4_now_nonbit4": len(removed_all_old),
                "added_old_nonbit4_now_bit4": len(added_all_new),
                "removed_examples_selected": len(removed_examples_old),
                "added_examples_selected": len(added_examples_new),
                "window_minutes": args.window_minutes,
            }
        ]
    )
    summary.to_csv(
        args.output_dir / "tw1_minpa_hplus_quality_flag_evenodd_diff_summary.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    print(f"Wrote diff outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
