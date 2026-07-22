# Tianwen-1 MINPA All-Energy MSE X-Z Mean Maps

This workflow reads the audited full-energy `NV_MSO_2` per-cell HDF5 archive
and computes separate H+, O+, and O2+ MSE X-Z distributions. It uses the same
selection, record weighting, plot variables, and color-map rules as the
established `>1 keV` O+/O2+ X-Z workflow.

## Input, coordinates, and energy definition

- Input archive: `tw1_minpa_all_energy_grid_records_5Rm`.
- Energy selection: every energy bin represented by the `NV_MSO_2` moments;
  no `>1 keV` threshold is applied.
- Position and all velocity components are in MSE coordinates.
- Grid edges are `[-5,+5] Rm` on X, Y, and Z, spacing is `0.1 Rm`, and
  `Rm=3397 km`.
- The archive is read-only. Derived NPZ, PNG/PDF, and JSON products are written
  to a separate output directory.

## Default valid-data selection

For each species, the default `valid` policy requires a finite-quality rotation,
derived processing status 1, positive finite density, finite MSE velocity, and
the corresponding validity flags. Its five-bit `NV_MSO_2` quality flag must be
available and epoch matched. Problem classes 2 through 5 are rejected, while
class 1 is retained as a caution flag. No MINPA FOV selection is applied.

The optional `strict-clean` policy also rejects class 1. The `all-valid` policy
keeps valid status/rotation/moment records without applying the five-bit mask.

## Means, projection, and figures

Each non-empty 3-D grid cell is reduced to double-precision counts and sums.
For an X-Z pixel, sufficient statistics from every Y cell are added before the
mean is recomputed. This is a record-weighted all-Y projection, not an
equal-weight mean of Y-cell means.

Density is the arithmetic mean of per-record densities and is never a sum over
epochs interpreted as a local density. Speed is calculated per record as
`sqrt(Vx^2 + Vy^2 + Vz^2)` before averaging. For H+, O+, and O2+ separately,
the workflow writes:

- `log10(mean density)` in `cm^-3`;
- `log10(mean speed)` in `km s^-1`;
- mean MSE `Vx`, `Vy`, and `Vz` in `km s^-1`.

Density uses `viridis`, speed uses `magma`, and signed components use a
zero-centered `RdBu_r` scale. Color limits are the 2nd and 98th finite
percentiles and are recorded in the JSON summary. Every figure is saved as
both 220 dpi PNG and vector PDF.

## Command and products

```powershell
python scripts\plot_tw1_all_energy_mse_xz_means.py --workers 4
```

The default output root is:

```text
D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_all_energy_mse_xz_means_5Rm
```

It contains sparse 3-D and projected X-Z NPZ products under `data`, 15 PNG and
15 PDF maps under `figures`, and the complete selection, aggregation, grid,
color-scale, and record-count provenance under `logs`.

## Performance and validation

On an Intel Core Ultra 7 265KF system with 31.6 GiB RAM, a deterministic
2,000-file benchmark took 15.099 s with one worker and 2.965 s with four
workers. All 43 projected arrays were bitwise identical. The full four-worker
run processed 155,145 HDF5 cell files in 249.356 s.

The production validation reconstructed every X-Z sufficient-statistic array
from the sparse 3-D product with zero differences, recomputed 64 deterministic
source HDF5 cell samples with zero mismatches, verified
`mean(|V|) >= |mean(V)|`, and checked the grid, Mars radius, units, and all 30
figure files. The valid-record totals are 2,550,289 H+, 1,021,762 O+, and
747,676 O2+ records, covering 3,607, 3,554, and 3,513 finite X-Z pixels.
