# Tianwen-1 MINPA High-E O+ / O2+ Method

## Inputs

- MINPA source priority:
  - `D:\Data\TW-1\result\MINPA\ori`
  - `D:\Data\TW-1\rawdata\MINPA\public` only when no overlapping `ori` files exist
- MOMAG attitude and position:
  - `D:\Data\TW-1\result\MOMAG\C\01Hz_all\BssYYYYMMDD.mat`
  - variables: `P_TW1`, `Pitch_TW1`, `Roll_TW1`, `Yaw_TW1`
- MSO to MSE matrices:
  - `D:\Data\TW-1\result\R_MSO2MSE\R_MSO2MSE_Tianwen-1_YYYYMMDD.mat`
  - variables observed locally: `data1` for time, `data2` for matrices

## Moment Method

- High energy means MINPA table energy `> 1000 eV`; no spacecraft-potential
  correction is applied.
- `Ion_Count` is treated consistently with the prior MINPA workflow as
  differential particle flux.
- Energy widths use per-channel logarithmic edges inferred from adjacent
  calibrated energy centers; fixed `dE/E=0.15` is retained only as legacy
  provenance and is not used in current integration.
- Density contribution per energy-angle-mass cell is:

```text
DEF_i = Ion_Count_i * E_i
dn_i = Ion_Count_i * (E_high,i - E_low,i) * dOmega / v_i
```

- Bulk velocity is the `dn`-weighted cell velocity. Spacecraft velocity from
  MOMAG position finite differences is added in MSO by default, matching the
  MAVEN high-E product convention.

## Species

- O+ uses the MINPA O mass group, normally `16 amu`; mode 12 uses the legacy
  `15.56` and `16.45 amu` groups.
- O2+ uses the MINPA O2 mass group, normally `32 amu`; mode 12 uses the legacy
  `30.94` and `33.17 amu` groups.

## Coordinates and FOV

- MINPA look directions use the validated conversion from the prior workflow:
  - build the intermediate MINPA direction from pitch/azimuth bins
  - map MINPA payload vectors to spacecraft-body coordinates with the fixed
    matrix `X_b=-V2`, `Y_b=+V3`, `Z_b=-V1`
  - rotate body vectors to MSO with `Rz(-Yaw) * Ry(-Pitch) * Rx(-Roll)`
- MSO vectors are rotated to MSE with the nearest `R_MSO2MSE` sample.
- Output records spacecraft-body coordinate axes in both MSO and MSE:
  `inst_{x,y,z}_axis_{mso,mse}_{x,y,z}`.
- MINPA FOV is different from MAVEN STATIC. It is still handled with a
  theta-edge check, but with MINPA's own ion FOV:
  - ion FOV: `360 deg x 90 deg`
  - ion angular resolution from the instrument description: `22.5 deg x 5.4 deg`
  - after the validated payload-to-body mapping, the ion FOV occupies the
    `-X_body` hemisphere
  - the per-record flag `mse_plus_z_in_minpa_fov_flag` is `1` when the
    transformed MSE `+Z` direction is inside the mode-specific MINPA pitch/theta
    edges and `0` when it is outside
- The diagnostic solid-angle half-axis fractions in MSO/MSE are retained, but
  they are not the primary FOV gate.
- MINPA has a half-FOV blind region. Density and vector moments should be
  treated as coverage-limited unless FOV coverage is adequate for the science
  question.
- See `docs/method_tw1_minpa_instrument_to_orbiter.md` for the explicit
  payload-to-body mapping, MOMAG body-to-MSO attitude chain, and real-data
  comparison against stored `NV/NV_MSO` products.

## Output

```text
E:\Data\highE\Tianwen-1\YYYY\tw1_minpa_highE_Oplus_YYYYMMDD.mat
E:\Data\highE\Tianwen-1\YYYY\tw1_minpa_highE_O2plus_YYYYMMDD.mat
E:\Data\highE\Tianwen-1\YYYY\tw1_minpa_highE_YYYYMMDD_summary.json
```

Each species file has one row per unique MINPA timestamp. If overlapping raw
records produce duplicate timestamps, later duplicates are dropped and the
counts are recorded in the summary.

## 20211203 Validation Snapshot

- Source: `ori`
- Source files: 10
- Input MINPA records: 4471
- Output rows after duplicate-time drop: 4467 per species
- Duplicate extras dropped: 4 per species
- Pass rows:
  - O+: 4264
  - O2+: 4287
- Median high-E density:
  - O+: `0.0087956 cm^-3`
  - O2+: `0.0122645 cm^-3`
- Wall time for the first write test: about `14.28 s`
- Note: the initial 20211203 write test was produced before adding
  `mse_plus_z_in_minpa_fov_flag`. Tianwen-1 outputs should be regenerated with
  overwrite after this FOV correction.
