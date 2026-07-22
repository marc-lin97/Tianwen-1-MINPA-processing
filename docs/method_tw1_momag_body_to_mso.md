# Tianwen-1 MOMAG orbiter-body to MSO rotation

## Data and convention

The validation input is the read-only Tianwen-1 MOMAG 32 Hz MAT product named
`HX1-Or_GRAS_MOMAG-DB-32Hz_SCI_P_20250703164828_20250704003600_05060_C.mat`.
Its embedded PDS metadata gives 897,664 records from
`2025-07-03T16:48:28.772750Z` to `2025-07-04T00:36:00.581500Z`. The attitude
fields `Probe_Attitude_Roll_MSO`, `Probe_Attitude_Pitch_MSO`, and
`Probe_Attitude_Yaw_MSO` are in degrees. The label records the magnetic-field
unit as `none`, so this workflow reports residuals in the product's data unit
and does not silently assign nT.

Let

- `phi = Roll`, a rotation about X;
- `theta = Pitch`, a rotation about Y;
- `psi = Yaw`, a rotation about Z.

Use right-handed active elementary rotations acting on column vectors:

```text
Rx(a) = [1       0        0
         0  cos(a)  -sin(a)
         0  sin(a)   cos(a)]

Ry(a) = [ cos(a)  0  sin(a)
               0  1       0
          -sin(a)  0  cos(a)]

Rz(a) = [cos(a)  -sin(a)  0
         sin(a)   cos(a)  0
              0        0  1]
```

For this MOMAG product, the strict RPY convention that reproduces the stored
orbiter/MSO vector pair defines the MSO-to-orbiter frame rotation as

```text
C_Orbiter_from_MSO = Rx(phi) Ry(theta) Rz(psi).
```

The required vector transformation is the inverse rotation. Since a proper
rotation matrix is orthogonal,

```text
C_MSO_from_Orbiter
    = inverse(C_Orbiter_from_MSO)
    = transpose(Rx(phi) Ry(theta) Rz(psi))
    = Rz(-psi) Ry(-theta) Rx(-phi).
```

Writing `c_phi = cos(phi)`, `s_phi = sin(phi)`, and likewise for `theta` and
`psi`, the explicit matrix, arranged by rows, is

```text
[ c_psi c_theta,  c_psi s_theta s_phi + s_psi c_phi,  s_psi s_phi - c_psi s_theta c_phi ]
[-s_psi c_theta,  c_psi c_phi - s_psi s_theta s_phi,  s_psi s_theta c_phi + c_psi s_phi ]
[        s_theta,                         -c_theta s_phi,                         c_theta c_phi ]
```

Therefore, for every epoch,

```text
[Bx_MSO, By_MSO, Bz_MSO]^T
    = C_MSO_from_Orbiter [Bx_Orbiter, By_Orbiter, Bz_Orbiter]^T.
```

Angles must be converted from degrees to radians before evaluating sine and
cosine. The reverse transform is the transpose of this matrix.

## Validation

The validation uses the analytic matrix above and compares its transformed
orbiter-frame field with the independently stored `X_MSO`, `Y_MSO`, and
`Z_MSO`. It does not fit a matrix to the two field vectors. Sanity checks are:

- `C C^T = I`;
- `det(C) = +1`;
- vector magnitude is preserved;
- calculated MSO components agree with the stored MSO components within the
  product's finite ASCII precision (`0.001 deg` for attitude and three decimal
  places for field components).

Run the read-only check from MATLAB with:

```matlab
addpath('scripts');
verify_tw1_momag_attitude( ...
    'D:\Data\TW-1\result\MOMAG\ori_C\32Hz\HX1-Or_GRAS_MOMAG-DB-32Hz_SCI_P_20250703164828_20250704003600_05060_C.mat');
```

Derived outputs are written under `outputs/tw1_momag_attitude_validation` in
the repository. The source MAT file is only read, and the script checks that
its size and modification timestamp are unchanged.

For the example file, 896,816 of 897,664 records were finite and usable. The
full-file validation produced:

- component RMSE: `[3.9826e-4, 3.8787e-4, 4.0090e-4]` data units;
- vector RMSE: `6.8540e-4` data units;
- median / 99th-percentile vector error: `3.4804e-4 / 2.4298e-3` data units;
- maximum sampled `||C C^T - I||_F`: `4.1541e-16`;
- maximum sampled `|det(C) - 1|`: `4.4409e-16`;
- orbiter-to-calculated-MSO magnitude RMSE: `8.3259e-16` data units.

These results validate the analytic convention and are consistent with the
finite precision of the fields stored in the product.
