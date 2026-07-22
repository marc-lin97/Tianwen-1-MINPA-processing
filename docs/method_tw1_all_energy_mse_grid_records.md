# Tianwen-1 MINPA All-Energy Per-Cell MSE Record Archive

## Purpose and inputs

This workflow stores the full-energy MINPA moments already present in
`D:\Data\TW-1\result\MINPA\NV_MSO_2\NV_YYYYMMDD.mat`. It does not apply the
`>1 keV` selection used by the separate high-energy archive and does not
reintegrate the raw spectra.

For H+, O+, and O2+, the source variables are respectively:

```text
NH_TW1,  NO_TW1,  NO2_TW1       [Unix time, density] cm^-3
VH_TW1_MSO, VO_TW1_MSO,
VO2_TW1_MSO                     [Unix time, Vx, Vy, Vz] km/s
```

`NH_TW1(:,1)` is the authoritative row time. A velocity row may be entirely
NaN, including its time column; finite velocity time values must equal the
authoritative time in the same row. Density uses the mode-aware energy-width
correction documented in `method_tw1_minpa_nv_mso_2.md`; velocity is the legacy
MSO bulk velocity preserved by that product.

Spacecraft MSO position comes from the nearest `P_TW1` row in the actual UTC
day's MOMAG `BssYYYYMMDD.mat`, with a 5 s tolerance. Position, `Xb_MSO`, and
all three velocity vectors are rotated with the nearest actual-day
`R_MSO2MSE_Tianwen-1_YYYYMMDD.mat`. The rotation match time and the existing
8 s quality flag are retained.

## Canonical UTC rows and geometry coverage

Many `NV_MSO_2` products span adjacent UTC dates. Across all 1051 files there
are 6,549,739 physical rows but 5,548,505 unique epochs; 1,001,234 epochs occur
in two adjacent products. Duplicate density, mode, and quality fields agree,
but legacy velocities and `Xb_MSO` may differ because they were generated with
different filename-date attitude files.

The canonical rule is therefore:

```text
keep a row only when UTC(NH_TW1 time) equals the YYYYMMDD in NV_YYYYMMDD.mat
```

Every unique epoch has exactly one such canonical source row. This removes all
cross-file duplication while choosing the correctly dated attitude result.

Of the 1051 UTC dates, 1029 have both MOMAG position and R_MSO2MSE files. The
remaining 22 dates cannot be assigned a defensible MSE position and are
excluded with their row counts and missing paths in
`logs/geometry_coverage.json`.

## Grid and record schema

- Coordinate system: MSE.
- Mars radius: `3397 km`.
- X/Y/Z edges: `[-5,+5] Rm`, inclusive at both outer boundaries.
- Cell size: `0.1 x 0.1 x 0.1 Rm`.
- Shape: `100 x 100 x 100`; centers are `-4.95` through `+4.95 Rm`.
- Only non-empty cell files are created.

Each cell HDF5 file contains `days/YYYYMMDD/records`. The 59-field compound
dataset extends the existing 43-field high-energy schema: 11 common fields
plus 16 fields for each of `hplus`, `oplus`, and `o2plus`. Each species block
contains density, MSE Vx/Vy/Vz, status, density/velocity validity, the uint32
quality mask, five explicit quality bits, availability, and epoch-match fields.

The compatibility status is derived and explicitly labeled:

```text
1 = density is finite and positive, and all three MSE velocities are finite
5 = density is finite and positive, but velocity is incomplete
4 = density is non-finite or zero signal
```

The source density value, including zero or NaN, is preserved. Records are
stored per epoch; densities are never summed across time records and such a sum
must not be interpreted as local plasma density.

The MSE +Z FOV flag is reconstructed from
`Xb_MSE = R_MSO2MSE * Xb_MSO`, `mode_TW1`, and the established mode-specific
MINPA pitch edges. It remains NaN when `Xb_MSO` is unavailable.

## Commands and restart behavior

One-day validation:

```powershell
python scripts\run_tw1_all_energy_mse_grid_records.py --dates 20211203 --output-root D:\TEMP\tw1_all_energy_mse_grid_smoke_20211203 --min-free-gb 1
python scripts\audit_tw1_all_energy_mse_grid_records.py --dates 20211203 --output-root D:\TEMP\tw1_all_energy_mse_grid_smoke_20211203 --expected-days 1 --expected-excluded-days 0
```

Full production:

```powershell
python scripts\run_tw1_all_energy_mse_grid_records.py
python scripts\audit_tw1_all_energy_mse_grid_records.py --cleanup-empty-staging
```

Default output:

```text
D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_all_energy_grid_records_5Rm
```

The writer holds one source day in memory, opens only cells touched on that
day, writes through an HDF5 staging group, and creates the atomic day summary
only after all cell groups are complete. A normal rerun skips completed day
summaries; `--overwrite` intentionally replaces selected day groups. Never run
two writers against the same output root.

## Validation snapshot

The `20211203` benchmark retained 4,382 of 4,467 canonical rows in 917 cells,
used 1.05 MiB of day-array memory, and completed the writer in about 4.7 s on
the local D-drive NVMe. The independent audit passed all 59 fields and three
source-reconstructed rows with zero mismatches. Comparison with the audited
`>1 keV` archive covered all 917 cells and 4,382 common epochs: MSE position,
rotation match time, FOV flag, and O+/O2+ quality fields had zero differences.

## Final production audit

The D-drive production writer completed all 1029 processable dates in
4,313.1 s (71.9 min), with a maximum day-array footprint of 2.48 MiB. The
archive occupies about 11.23 GiB. Its counts are:

```text
input NV product dates                         1051
processable dates with MOMAG and R_MSO2MSE    1029
excluded geometry dates                         22
processable canonical UTC rows             5,452,228
excluded canonical UTC rows                   96,277
all unique canonical UTC rows              5,548,505
in-grid records                             3,979,338
non-empty HDF5 cell files                     155,145
cell-day groups                               558,551
```

The independent full audit ran in 6,117 s and passed with zero errors. It
verified every HDF5 row's 59-field schema, units, status mapping, quality-bit
reconstruction, grid assignment, km/Rm conversion, UTC date, provenance, and
rotation metadata. It found no staging groups. An independent source replay
reconstructed all 5,452,228 processable canonical rows and the same 3,979,338
in-grid rows, then compared 2,703 deterministic records across every UTC day
having in-grid coverage with zero field mismatches. Rotation orthonormal and
determinant errors were below `7e-16`.

The authoritative machine-readable results are
`logs/latest_run_summary.json`, `logs/geometry_coverage.json`, and
`logs/full_audit_summary.json` under the production output root.
