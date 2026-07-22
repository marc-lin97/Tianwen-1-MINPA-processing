# Tianwen-1 MINPA Per-Cell MSE Record Archive

## Purpose

This workflow streams the completed daily MINPA high-energy O+ and O2+ moment
products into spatially indexed HDF5 files. It preserves individual epochs and
does not sum densities across time records.

## Inputs and Coordinates

- Daily inputs:
  `E:\Data\highE\Tianwen-1\YYYY\tw1_minpa_highE_{Oplus,O2plus}_YYYYMMDD.mat`.
- Daily five-problem quality flags:
  `D:\Data\TW-1\result\MINPA\NV_MSO_2\NV_YYYYMMDD.mat`.
- High energy is inherited from the daily products as raw MINPA energy
  `> 1000 eV`.
- Each O+/O2+ daily pair must have identical epoch, MSO position, stored MSE
  position, and `mse_plus_z_in_minpa_fov_flag` arrays.
- The workflow reads the daily `R_MSO2MSE_Tianwen-1_YYYYMMDD.mat`, matches each
  particle epoch to the nearest matrix, and recomputes MSE position and both
  species' MSE velocity components from their MSO values.
- Recomputed MSE values must reproduce the stored daily-product MSE values
  within the configured absolute tolerance. Rotation orthonormality and
  determinant are also checked.
- Rotation epochs are stably sorted. At duplicate epochs, the matrix with the
  most finite elements is preferred; equally complete candidates retain source
  order, with conflict count and maximum difference recorded. This deterministic
  choice is accepted only when the final per-record MSE recomputation reproduces
  the stored daily products within tolerance.

## Grid

- Coordinate system: MSE.
- Mars radius: `3397 km`.
- Cell edges span `[-5, +5] Rm` on X, Y, and Z.
- Cell size: `0.1 x 0.1 x 0.1 Rm`.
- Shape: `100 x 100 x 100` cells; centers run from `-4.95` to `+4.95 Rm`.
- The lower and upper outer boundaries are accepted. A point exactly at
  `+5 Rm` is assigned to the final cell.
- Only non-empty cells are created.

## Record Policy and HDF5 Schema

All records with finite recomputed MSE position inside the grid are retained.
Density or velocity failures remain `NaN`; they are not removed and no
`0.001 cm^-3` density threshold is applied.

The audited production archive stores each non-empty cell in one
position-named HDF5 file under:

```text
D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_grid_records_5Rm\cells\ixNNN\
```

The filename records `ix/iy/iz` and the three cell-center coordinates. Each
file contains `days/YYYYMMDD/records`, one compressed HDF5 compound dataset
whose named fields include:

- Common fields: epoch, recomputed MSE position in km and Rm, MINPA FOV flag,
  rotation-matrix match time difference, finite-matrix flag, and the `8 s`
  time-match quality flag.
- `oplus_*` and `o2plus_*` fields: per-record density, recomputed MSE velocity
  X/Y/Z, original processing status code, derived density/velocity-valid flags,
  and the species-specific `NV_MSO_2` quality mask.
- Each species quality-mask block stores the uint32 mask, five explicit bit
  columns, `available`, epoch-match status, and epoch-match time difference.
  The five problems are: bit 1 sparse/caution, bit 2 sparse/invalid, bit 3 UV
  contamination, bit 4 odd/even acquisition error, and bit 5 distorted high
  channel. When `available=0`, a numeric mask of zero is only a placeholder and
  must not be interpreted as clean data.
- File and group attributes: cell edges and center, units, mission,
  instrument, input paths, rotation path, match method, and validation errors.

MATLAB can read datasets with `h5read`; Python can use `h5py`. These files are
HDF5 rather than traditional MATLAB v5 MAT files because HDF5 supports
incremental per-day updates without loading or rewriting historical records.
The compound layout preserves native float and integer types while avoiding
dozens of tiny HDF5 datasets per cell-day. Dataset attribute
`field_metadata_json` records units and descriptions for every named field.

## Streaming, Resume, and Provenance

The runner processes one UTC day at a time. Only the two daily species files,
the daily rotation file, and the current day's arrays are held in memory. It
opens only grid files touched by that day's spacecraft trajectory.

Per-day summaries are written atomically under `logs/days/YYYYMMDD.json` only
after all cell groups for that day are complete. A normal rerun skips completed
days. If a run stops before the day summary is written, the next run reuses
already completed cell-day groups and writes the remaining groups. Use
`--overwrite` only when intentionally replacing an existing date.

`logs/progress.json` is refreshed after each newly completed day, while
`logs/latest_run_summary.json` is finalized after the requested run completes.

The runner stops before the output disk falls below `--min-free-gb` (default
`20 GiB`). Do not run two writers against the same output root concurrently.

For large runs, per-cell HDF5 updates are dominated by small random I/O rather
than array memory. A verified read-only input copy and a separate output root
on NVMe may be used without changing the calculation. Never point two writers
at the same output root; record the copied-input hashes, storage hardware,
paths, and before/after throughput in the workflow log.

## Commands

One-day validation:

```powershell
python scripts\run_tw1_mse_grid_records.py --dates 20211203 --output-root D:\TEMP\tw1_mse_grid_records_smoke --min-free-gb 1
```

All available paired days:

```powershell
python scripts\run_tw1_mse_grid_records.py --input-root D:\Data\highE\Tianwen-1 --output-root D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_grid_records_5Rm
```

Full completion audit, run only after the production writer exits:

```powershell
python scripts\audit_tw1_mse_grid_records.py --input-root D:\Data\highE\Tianwen-1 --output-root D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_grid_records_5Rm --cleanup-empty-staging
```

The cleanup removes only empty transaction groups and stops if any non-empty
staging group is found. The audit then streams through every output file and
checks source/day coverage, schema v3 metadata, the exact 43-field record
layout, cell filename and edge consistency, every row's grid assignment,
km-to-Rm conversion, stored coordinate-recompute validation, rotation quality,
source-file provenance, density/velocity validity flags, and both species'
five NV bits against their bitmasks. It reconciles per-date and global HDF5 row
and cell-day totals with the atomic day summaries, and writes
`logs/full_audit_summary.json`.

For unattended completion, `scripts/watch_and_audit_tw1_mse_grid_records.py`
monitors a specific writer PID. After that process exits, it requires the
progress file, final run summary, and empty stderr log to prove 1029 completed
days before cleanup or audit begins. Its atomic state file is
`logs/audit_watcher_status.json`; an early writer exit is recorded as
`production_incomplete` and leaves the HDF5 files unchanged.

Optional controls include `--start-date`, `--end-date`, `--dates`,
`--max-days`, `--coordinate-atol`, `--rotation-gap-s`,
`--quality-flag-tolerance-s`, and path overrides including `--nv-mso2-root`.

## Validation Snapshot

The `20211203` smoke run used about `0.66 MiB` of day-level array memory,
retained `4382` in-grid records in `917` non-empty cells, and reproduced the
stored MSE position and both species' stored MSE velocities with zero maximum
absolute difference. Of `4467` input epochs, `4382` had finite matched rotation
matrices; the invalid rotation rows were excluded by the finite-position rule
and remained documented in the daily validation summary.

The final compound-record benchmark used about `0.82 MiB` of day-level arrays,
wrote the same `917` files in about `2.9 s`, and occupied `16.95 MiB`. A
field-by-field comparison against the earlier multi-dataset layout covered all
`917` cells, `4382` rows, and `43` fields per row with zero mismatches.

Across all `1029` input days, every date has an `NV_MSO_2` file and all quality
files use algorithm version
`2026-07-13-seven-band-all-species-v2-mode12-product-time`. Of `4,803,234`
daily-product epochs per species, `3,889,464` have an available matched flag;
unmatched/unavailable records are retained with `available=0`.

## Final Full-Product Audit

The NVMe production run completed all `1029` paired source dates in
`3410.41 s` (`56.84 min`) with zero writer stderr. The independent streaming
audit then passed with zero errors and verified:

- `1029` day summaries with no missing or extra dates;
- `156,033` position-named schema-v3 HDF5 files;
- `565,545` cell-day groups and `3,539,768` records, exactly matching all
  per-day summary totals;
- the required `100 x 100 x 100` grid, `[-5,+5] Rm` edges, `0.1 Rm` spacing,
  `Rm=3397 km`, and all `43` compound record fields;
- zero maximum position km/Rm inconsistency, zero maximum MSE position
  recomputation error, and zero maximum O+/O2+ MSE velocity recomputation
  error; rotation orthonormality and determinant errors were below
  `7e-16`;
- no staging groups, all provenance paths represented by retained cell-day
  groups present, and the single expected NV_MSO_2 algorithm version;
- `2,894,709` available quality-flag rows for each species, including
  `2,437,325` flagged O+ rows and `2,583,751` flagged O2+ rows.

Finite density counts were `1,905,465` for O+ and `1,347,569` for O2+.
Per-record densities remain unaggregated across epochs. The authoritative
machine-readable evidence is `logs/full_audit_summary.json`; the final test
suite also passed `26/26` tests.

Dates with zero in-grid cells have no HDF5 day group by design. The separate
`logs/day_summary_provenance_audit.json` closes that coverage boundary by
checking every one of the 1029 daily summaries, all O+/O2+/rotation/NV source
paths, NV metadata and algorithm version, and daily coordinate/rotation
validation values, including zero-cell dates.

That supplemental audit passed with zero errors: all `4,116` unique daily
source paths were present, `128` dates had zero in-grid cells, the summaries
covered `4,803,234` input epochs and `3,539,768` retained in-grid records, and
all validation maxima agreed with the full HDF5 audit.
