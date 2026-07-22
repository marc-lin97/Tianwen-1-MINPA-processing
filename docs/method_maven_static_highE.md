# MAVEN STATIC-D1 High-E O+/O2+ Method

This project writes derived products under `E:\Data\highE` and keeps raw mission
data read-only.

## Inputs

- STATIC-D1 CDF: `D:\Data\MAVEN\rawdata\static\l2\d1-32e4d16a8m`.
- MAVEN position: `D:\Data\MAVEN\result\mag\ss1s\BssYYYYMMDD.mat`.
  `Bss[:,0]` is the time axis and `Spss[:,1:4]` is MSO position in km.
- MSO-to-MSE matrix: `D:\Data\MAVEN\result\R_MSO2MSE\R_MSO2MSE_MAVEN_YYYYMMDD.mat`.

## Outputs

- One daily MAT file per species:
  `maven_static_highE_Oplus_YYYYMMDD.mat` and
  `maven_static_highE_O2plus_YYYYMMDD.mat`.
- Each species file contains one row per STATIC timestamp, with no duplicate
  time points.
- One daily JSON summary records both species files, source paths, assumptions,
  status counts, and sanity checks.
- A standalone legacy-product flag route can also write
  `maven_static_plus_z_fov_flag_YYYYMMDD.mat` and matching JSON summaries from
  existing species MAT files under `E:\Data\highE\MAVEN`.

## Moments

- Species: O+ uses `14.5-17.5 amu/e`; O2+ uses `30-34 amu/e`.
- High-energy selection uses raw STATIC `energy > 1000 eV`; `sc_pot` is not used
  for the threshold or speed.
- Cell density uses the local STATIC D1 expression
  `eflux / energy * denergy * domega / speed_cm_s`.
- Per-record bulk velocity is density-weighted over valid high-energy cells.
- Spacecraft velocity from finite differences of `Spss` is added in the STATIC
  frame before rotating the cell velocities to MSO. This can be disabled with
  `--no-spacecraft-velocity-correction`.

## Coordinates

- STATIC/App vectors are rotated to MSO with scalar-first `quat_mso`.
- Position and velocity are rotated to MSE with the nearest daily `R_MSO2MSE`
  matrix sample.
- Instrument FOV output includes the three STATIC/App coordinate axes in MSO and
  MSE.
- STATIC's deflected FOV is fixed in STATIC coordinates. The instrument manual
  gives an approximate `360 deg x 90 deg` FOV; the D1 theta channel centers and
  `dtheta` widths imply fixed theta bin-edge limits of
  `[-45.866667, +45.866667] deg`.
- The MSE +Z vector changes with spacecraft attitude and the MSO-to-MSE
  rotation. Each epoch projects MSE +Z into STATIC coordinates using the
  MSE-space STATIC axes, then tests whether
  `static_fov_theta_min_deg <= mse_plus_z_static_theta_deg <= static_fov_theta_max_deg`.
  The flag is `1` inside, `0` outside, and `NaN` when attitude or geometry is
  not determinable.
- The output keeps `mse_plus_z_static_x/y/z`,
  `mse_plus_z_static_theta_deg`, `static_fov_theta_min/max_deg`, and
  `mse_plus_z_in_static_fov_flag` for auditability.

## Acceleration

- Geometry masks are precomputed once for each `swp_ind` and species.
- The batch driver parallelizes across days with `ProcessPoolExecutor`.
- Per-day processing reads the CDF once and avoids per-species repeated support
  array I/O.
