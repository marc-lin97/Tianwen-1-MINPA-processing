# Tianwen-1 MINPA +Xb_MSE FOV-classified X-Z statistics

## Classification geometry

The Tianwen-1 orbiter-body `+Xb` direction is transformed to MSE as

```text
Xb_MSE = R_MSO2MSE Xb_MSO
Xb_MSO = R_MSO_from_body [1,0,0]^T
R_MSO_from_body = Rz(-Yaw) Ry(-Pitch) Rx(-Roll)
```

The epoch-native high-energy daily products store this direction in the
legacy-named fields `inst_x_axis_mse_x/y/z`. These fields are the orbiter-body
`+Xb` axis, not the MINPA `+V1` instrument axis. They are used as the
authoritative classification direction because they are evaluated at every
particle epoch. The independent `NV_MSO_2.Xb_MSO -> R_MSO2MSE -> Xb_MSE`
chain is evaluated wherever `Xb_MSO` is available and used as a validation.

For a finite unit vector `u = Xb_MSE / |Xb_MSE|`, the two mutually exclusive
classes are:

```text
xb_plus_z_le45:  acos(+u_z) <= 45 deg
xb_minus_z_le45: acos(-u_z) <= 45 deg
```

The 45-degree boundary is inclusive. Finite directions outside both cones are
counted for coverage but excluded from both classified means. NaN directions
are unclassified.

## Inputs, filters, and aggregation

The workflow streams the 1,029 paired O+/O2+ high-energy MINPA daily files.
For each day it loads the matching `NV_MSO_2` quality flags and
`R_MSO2MSE_Tianwen-1` matrices, validates the stored MSE positions and
velocities, constructs both Xb classes, and reduces records to the existing
`0.1 Rm` grid with edges `[-5,+5] Rm` and `Rm=3397 km`.

The valid-data policy is unchanged:

- density and velocity NaN/non-finite values are invalid;
- NV_MSO_2 problem classes 2-5 are invalid;
- class 1 remains valid with caution;
- flag-unavailable or epoch-unmatched rows are invalid;
- `processing_status_code == 1` and a finite rotation within 8 seconds are
  required.

O+ and O2+ are never combined. Each class stores per-cell counts, sums, mean
density, mean MSE `Vx/Vy/Vz`, and mean of the per-record MSE speed magnitude.
The X-Z projection Y range is configurable. The current requested product uses
the inclusive slab `-1 <= Y_MSE <= 1 Rm`, corresponding to the twenty grid
centers from `-0.95` through `+0.95 Rm`. Cells inside that slab are combined
with record-count weighting. Density and
speed are plotted as base-10 logarithms; signed components use zero-centered
diverging colors. The two FOV classes use identical color limits for each
species and variable so that their maps can be compared directly.

The density-speed flux is defined exactly as requested from the projected
means:

```text
flux = mean_density_cm3 * mean_speed_km_s * 1e5
```

The factor `1e5` converts km/s to cm/s, giving flux in `cm^-2 s^-1`. Flux is
stored in each NPZ and plotted as `log10(flux)` with a shared scale between the
two Xb classes.

## Command and outputs

```powershell
python scripts\plot_tw1_mse_xz_fov_classes.py --workers 4
```

Default output root:

```text
D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_mse_xz_fov_classes_y_m1_p1_5Rm
```

- `data`: one sparse 3-D NPZ and one Y-limited X-Z NPZ for each class;
- `figures`: 24 PNG and 24 PDF maps (2 classes x 2 species x 6 variables);
- `logs/tw1_minpa_mse_xz_fov_classified_y_m1_p1_summary.json`: inputs, definitions,
  daily coverage, validation maxima, shared color limits, and output paths.

## Full-run validation

The full run processed all 1,029 days and 4,803,234 particle epochs. It found
3,539,768 finite in-grid Xb directions:

- 563,895 in the `+Z_MSE` cone;
- 592,778 in the `-Z_MSE` cone;
- 2,383,095 outside both cones.

The three counts close exactly and the two cone masks never overlap. Across
every independently comparable record, the `NV_MSO_2.Xb_MSO` chain and the
epoch-native daily Xb axis had a maximum angular difference of 0.03725 degrees
and zero 45-degree class disagreements. The stored Xb unit-vector norm error
was below `5e-16`.

An independent output audit reconstructed every X-Z sufficient-statistic array
from the sparse 3-D NPZ files, verified count/mean masks, the mean-speed
inequality, the density-speed flux identity, and shared plot limits, and
reclassified deterministic raw daily samples. All checks passed. The repository
test suite passed 34/34.

For the requested `-1 <= Y_MSE <= 1 Rm` slab, the final valid moment counts
are:

| Xb class | O+ | O2+ |
|---|---:|---:|
| within 45 deg of `+Z_MSE` | 41,148 | 25,194 |
| within 45 deg of `-Z_MSE` | 50,577 | 27,656 |

The slab audit rebuilt every X-Z count and sum using only Y indices
`40` through `59`, confirmed that the underlying 3-D arrays are unchanged,
verified `flux = mean_density * mean_speed * 1e5` in every finite projected
cell, and passed all 34 repository tests.

## Derived 0.2 Rm X-Z product with the exact Y = +/-0.5 Rm slab

The coarser product is generated from the existing `0.1 Rm` sparse 3-D
sufficient statistics:

```powershell
python scripts\rebin_tw1_mse_xz_fov_classes.py
```

Source and output roots are:

```text
D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_mse_xz_fov_classes_y_m1_p1_5Rm
D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_mse_xz_fov_classes_grid0p2_y_m0p5_p0p5_5Rm
```

The full three-dimensional product is rebinned from `100 x 100 x 100` cells
at `0.1 Rm` to `50 x 50 x 50` cells at `0.2 Rm`. Counts, density sums, velocity
component sums, and per-record speed sums are added; all means and the
density-speed flux are then recomputed. Existing cell means are never averaged.

For the X-Z maps, the Y selection is applied before coarsening. Source-grid Y
indices 45-54, with centers `-0.45` through `+0.45 Rm` and physical edges
`-0.5` and `+0.5 Rm`, are projected first. X and Z are then coarsened by a
factor of two. This preserves the exact requested slab and avoids the
`-0.6` to `+0.6 Rm` effective coverage that would result from selecting the
six global `0.2 Rm` target cells whose centers lie between `-0.5` and `+0.5`.

The valid record counts therefore remain identical to the `0.1 Rm`, Y=+/-0.5
product:

| Xb class | O+ | O2+ |
|---|---:|---:|
| within 45 deg of `+Z_MSE` | 20,497 | 12,397 |
| within 45 deg of `-Z_MSE` | 26,062 | 14,845 |

The independent audit reconstructed every `0.2 Rm` X-Z sufficient-statistic
array directly from the `0.1 Rm` source, verified full three-dimensional
conservation and the flux identity, and checked all 24 PNG and 24 PDF files.
