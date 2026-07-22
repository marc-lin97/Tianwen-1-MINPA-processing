# Tianwen-1 MINPA All-Energy +Xb_MSE-classified X-Z Maps

This workflow separates full-energy `NV_MSO_2` H+, O+, and O2+ moments by the
orbiter-body `+Xb` direction in MSE coordinates and forms X-Z maps from the
inclusive `-0.5 <= Y_MSE <= 0.5 Rm` slab.

## Classification geometry

For each canonical UTC record,

```text
Xb_MSE = R_MSO2MSE Xb_MSO
```

is evaluated with the same nearest rotation matrix and 8 s validity tolerance
used by the audited all-energy archive. The two mutually exclusive classes are:

```text
xb_plus_z_le45:  angle(+Xb_MSE,+Z_MSE) <= 45 deg
xb_minus_z_le45: angle(+Xb_MSE,-Z_MSE) <= 45 deg
```

The 45-degree boundaries are inclusive. Finite `+Xb_MSE` directions outside
both cones remain in coverage counts but are excluded from both statistical
classes. Non-finite axes or axes without a valid rotation are unclassified.

## Input, selection, and projection

- Mission/instrument: Tianwen-1 MINPA.
- Moment source: all energy bins represented by `NV_MSO_2`; no `>1 keV`
  threshold.
- Position and velocity coordinates: MSE.
- Grid: `100 x 100 x 100`, edges `[-5,+5] Rm`, spacing `0.1 Rm`, and
  `Rm=3397 km`.
- Canonical row rule: retain only epochs whose true UTC date equals the
  `NV_YYYYMMDD.mat` filename date.
- Geometry coverage: 1,029 processable dates; 22 dates and 96,277 canonical
  rows without both MOMAG position and `R_MSO2MSE` are excluded.

The default valid-data rule requires processing status 1, finite positive
density, finite three-component MSE velocity, a finite rotation within 8 s,
and an available species-specific quality flag. Quality bit 1 remains a
caution; bits 2 through 5 are rejected.

The requested Y slab uses source-grid indices 45 through 54, with centers
`-0.45,-0.35,...,+0.45 Rm` and physical edges exactly at `-0.5` and `+0.5 Rm`.
Counts and physical sums are added across those ten layers before means are
computed, so the X-Z projection is record weighted.

Density is the mean of per-record densities, never a cross-epoch sum. Speed is
formed per record before averaging. The projected density-speed flux is:

```text
flux = mean_density_cm3 * mean_speed_km_s * 1e5
```

and has units `cm^-2 s^-1`.

## Figures and command

For each class and each species, six separate maps are written:

- `log10(mean density)`;
- `log10(mean speed)`;
- `log10(density-speed flux)`;
- mean MSE `Vx`, `Vy`, and `Vz`.

The two classes share color limits for the same species and variable. Density,
speed, and flux use sequential perceptually ordered color maps; signed velocity
components use zero-centered diverging colors. Each product is saved as PNG
and PDF.

```powershell
python scripts\plot_tw1_all_energy_mse_xz_xb_classes.py --workers 4
```

Default output root:

```text
D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_all_energy_mse_xz_xb_classes_y_m0p5_p0p5_5Rm
```

The output contains four NPZ files (sparse 3-D and projected X-Z data for each
class), 36 PNG and 36 PDF figures, and a JSON summary containing coverage,
classification, units, filters, shared color limits, daily provenance, and
validation diagnostics.

## Production performance and validation

On an Intel Core Ultra 7 265KF system with 31.6 GiB RAM, a deterministic
100-day benchmark took 58.594 s with one worker and 17.812 s with four workers.
All four NPZ files and 220 arrays were bitwise identical. The full 1,029-day
four-worker run took 176.363 s and reproduced the archived 3,979,338 in-grid
record count.

The in-grid `+Xb_MSE` coverage closes exactly:

| Classification | Records |
|---|---:|
| within 45 deg of `+Z_MSE` | 647,023 |
| within 45 deg of `-Z_MSE` | 673,460 |
| valid axis outside both cones | 2,658,855 |
| total | 3,979,338 |

Valid moment counts inside the requested Y slab are:

| Xb class | H+ | O+ | O2+ |
|---|---:|---:|---:|
| within 45 deg of `+Z_MSE` | 42,072 | 22,171 | 16,770 |
| within 45 deg of `-Z_MSE` | 44,767 | 28,023 | 19,377 |

The independent product audit reconstructed every X-Z count and sum from the
sparse 3-D arrays using only Y indices 45-54. Cone closure, cone overlap,
derived means, the speed inequality, flux identity, shared color limits,
coordinates, units, and all 72 figure files passed with zero errors.
