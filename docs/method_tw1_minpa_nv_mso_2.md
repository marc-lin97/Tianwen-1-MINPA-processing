# Tianwen-1 MINPA NV_MSO_2 Product

## Purpose

`NV_MSO_2` is a derivative of the legacy Tianwen-1 MINPA `NV_MSO` daily
product. It preserves the legacy MSO velocity components, updates the density
energy-width convention, and appends the spacecraft body-frame `+X_b` axis in
MSO coordinates.

Output directory:

```text
D:\Data\TW-1\result\MINPA\NV_MSO_2
```

Build script:

```text
scripts/build_tw1_minpa_nv_mso_2.m
```

## Inputs

- Legacy MSO product:
  `D:\Data\TW-1\result\MINPA\NV_MSO\NV_YYYYMMDD.mat`
- Legacy daily spectra with per-segment mode and time:
  `D:\Data\TW-1\result\MINPA\day_spe\Ion_spe_YYYYMMDD.mat`
- MOMAG attitude:
  `D:\Data\TW-1\result\MOMAG\C\01Hz_all\BssYYYYMMDD.mat`

Raw and legacy input files are read-only.

## Output Variables

The legacy variable names and shapes are retained:

```text
NH_TW1       [time_unix_s, density_cm^-3]
NO_TW1       [time_unix_s, density_cm^-3]
NO2_TW1      [time_unix_s, density_cm^-3]
VH_TW1_MSO   [time_unix_s, Vx, Vy, Vz] km/s
VO_TW1_MSO   [time_unix_s, Vx, Vy, Vz] km/s
VO2_TW1_MSO  [time_unix_s, Vx, Vy, Vz] km/s
```

Additional variables:

```text
Xb_MSO                  [time_unix_s, Xb_x, Xb_y, Xb_z]
mode_TW1                [time_unix_s, MINPA_mode]
density_scale_TW1       [time_unix_s, new_dE_over_E / 0.15]
energy_width_over_E_TW1 [time_unix_s, new_dE_over_E]
NV_MSO_2_info           metadata struct
```

## Density Update

The legacy `NV_MSO` density used a fixed `dE/E = 0.15`.

`NV_MSO_2` assigns each data point a MINPA mode by matching its timestamp to
the same day's `day_spe` segment times. It then rescales the legacy density by
the mode-specific numerical energy integration width:

```text
dE/E = sqrt(r) - 1/sqrt(r), where r = E(i+1)/E(i)
density_new = density_old * (dE/E)_mode / 0.15
```

This uses the adjacent logarithmic energy-center spacing. The per-mode values
are:

```text
mode(s)       dE/E       scale over 0.15
1,2           0.2339916  1.5599439
3,4,5,7,8     0.1446493  0.9643286
6             0.1940261  1.2935071
9,10,11       0.0998339  0.6655595
12            0.1509958  1.0066384
```

The 2021-12-03 validation had zero unmatched mode records.

## Velocity And Xb_MSO

The MSO velocity components are copied unchanged from the legacy `NV_MSO`
product.

`Xb_MSO` is computed with the same MOMAG attitude convention as the legacy
velocity rotation:

```text
R_MSO_from_body = Rz(-Yaw) * Ry(-Pitch) * Rx(-Roll)
Xb_MSO = R_MSO_from_body * [1, 0, 0]^T
```

The attitude match tolerance is 5 s. Rows without matching attitude keep
`NaN` in `Xb_MSO(:,2:4)`, matching the legacy behavior for unavailable MSO
velocity rotation.

## Build Summary

The full run processed 1051 daily products and 6,549,739 rows. No rows failed
mode matching. The saved summary is:

```text
D:\Data\TW-1\result\MINPA\NV_MSO_2\NV_MSO_2_build_summary.mat
```
