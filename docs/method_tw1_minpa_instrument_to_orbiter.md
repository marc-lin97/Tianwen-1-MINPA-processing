# Tianwen-1 MINPA payload to spacecraft body to MSO coordinates

## Authoritative local validation

The current convention is anchored to the latest local all-species comparison
of daily `NV` and `NV_MSO` products, not to an inferred mechanical mounting
orientation. The validation script is
`scripts/verify_tw1_minpa_velocity_mso_rotation.m`; its 2021-12-03 report is
`outputs/tw1_minpa_velocity_mso_rotation_validation/tw1_minpa_velocity_mso_rotation_20211203_summary.json`.

The source MINPA/`NV` velocity components are mapped directly into spacecraft
body coordinates
as

```text
[Xb, Yb, Zb]_body = [-V2, +V3, -V1]_MINPA

C_body_from_MINPA =
[  0  -1   0
   0   0   1
  -1   0   0 ].
```

The older textual one-step mapping `[-V3,-V1,+V2]` is not used: for H+ its
median vector error against stored `NV_MSO` is about `498 km/s`.

## Spacecraft body to MSO

MOMAG attitude supplies Roll, Pitch, and Yaw. With active right-handed column
vector rotations,

```text
C_MSO_from_body = Rz(-Yaw) Ry(-Pitch) Rx(-Roll)
V_MSO = C_MSO_from_body C_body_from_MINPA V_MINPA.
```

Each record uses its nearest attitude within 5 s. No interval-center attitude
is substituted for time-series moments or finite-cell VDF samples.

## Rotation audit results

For 2021-12-03 the exhaustive signed-permutation search reported:

| Species | Compared rows | Median vector error | Vector RMSE |
|---|---:|---:|---:|
| H+ | 4460 | `5.68e-14 km/s` | `3.94e-4 km/s` |
| O+ | 4436 | `3.97e-15 km/s` | `2.72e-5 km/s` |
| O2+ | 4438 | `2.51e-15 km/s` | `1.21e-5 km/s` |

The small H+ RMSE is caused by a few outliers; its 99th-percentile error is
`1.29e-13 km/s`. Speed preservation alone is not sufficient to select a
mapping because every signed permutation preserves speed.

## SWIA/NV_MSO_2 end-to-end check

The latest H+ comparison for 2021-12-03 00:00--01:00 UTC uses all MINPA energy
channels, `NV_MSO_2` density widths inferred from the energy table, and the
unchanged validated `VH_TW1_MSO` components. It reports:

- SWIA median speed: `423.815 km/s`;
- MINPA median speed: `420.796 km/s`;
- MINPA minus SWIA median speed: `-3.019 km/s`;
- corrected MINPA/SWIA median density ratio: `1.0313`.

The self-contained Python example is accepted only when it reproduces the
local `NV_MSO_2` density and `VH_TW1_MSO` components on matching timestamps.
