# Tianwen-1 MINPA MSE X-Z Mean Maps

This workflow reads the audited per-cell HDF5 archive and computes spatial
means for high-energy (`raw energy > 1000 eV`) O+ and O2+ separately. It never
reopens all grid files at once and never interprets a sum over records as a
local density.

## Inputs and coordinates

- Input archive: `tw1_minpa_grid_records_5Rm`.
- Coordinate system: MSE for position and all velocity components.
- Grid: `100 x 100 x 100`, edges `[-5,+5] Rm`, spacing `0.1 Rm`, and
  `Rm=3397 km`.
- Each HDF5 file is opened independently. Its UTC day groups are streamed one
  at a time and reduced to counts and double-precision sums.

## Default validity and quality selection

The default `valid` selection requires:

- `r_matrix_within_tolerance_flag == 1`;
- species `processing_status_code == 1`;
- positive finite density and `density_valid_flag == 1`;
- for velocity, all three MSE components finite and
  `velocity_valid_flag == 1`;
- the species NV_MSO_2 flag is available and epoch matched;
- none of NV_MSO_2 problem classes 2 through 5 is set. Class 1 is a caution
  flag and remains valid for this statistical product.

All NaN or otherwise non-finite density/velocity values are invalid. No MINPA
FOV selection is applied. The optional `strict-clean` policy also rejects
class 1, while `all-valid` retains valid status/rotation/moment records without
applying the NV_MSO_2 bitmask.

## Means and X-Z projection

The sparse 3-D product stores density and velocity record counts, sums, and
arithmetic means for every non-empty spatial cell. Density is the mean of the
per-record densities; it is not a sum across epochs.

For each `(X,Z)` pixel, sufficient statistics from every Y cell are combined.
The resulting X-Z mean is therefore weighted by the number of valid records,
not an equal-weight mean of Y-cell means. Speed is computed for each record as
`sqrt(Vx^2 + Vy^2 + Vz^2)` before averaging.

Separate figures are written for `log10(mean density)`, `log10(mean speed)`,
mean `Vx`, mean `Vy`, and mean `Vz`. Signed components use a zero-centered
diverging color map. Plot limits use the 2nd to 98th finite percentiles; the
exact limits are recorded in the JSON summary.

## Command and outputs

```powershell
python scripts\plot_tw1_mse_xz_means.py --workers 4
```

The default output root is:

```text
D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_mse_xz_means_5Rm
```

It contains compressed NPZ data under `data`, separate PNG/PDF figures under
`figures`, and a machine-readable provenance and validation summary under
`logs`.

## Full-production validation

The final `valid` run processed all `156,033` audited cell files. It retained
`952,844` O+ and `620,465` O2+ records for both density and vector velocity,
covering `3,553` and `3,510` finite X-Z pixels, respectively. An independent
audit reconstructed the X-Z arrays from all sparse 3-D sums/counts, checked
the mean-speed versus mean-vector inequality, and recomputed 64 deterministic
raw HDF5 cell samples with the NaN and bit-2-through-bit-5 filters. All checks
passed. The full Python test suite passed `29/29` tests.
