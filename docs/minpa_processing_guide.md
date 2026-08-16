# Tianwen-1 MINPA processing guide

## Scope and reproducibility contract

This workflow reads the released differential particle flux (DPF), preserves
the source file and diagnostics, and writes all derived products elsewhere.
It supports science Modes 1--12 at the interface level and rejects Mode 13.
The local archive has provided direct regression coverage for Modes 1, 4, 7,
and 12; other modes use the mission calibration tables but should receive an
event-level visual check before scientific publication.

Every output records the input path, UTC range, mode, species, energy limits,
quality policy, optional background-model version, and output paths in
`provenance.json`.  The NPZ contains the energy spectra; the CSV contains one
row per time with moment and flag columns.

## Data path and dimensions

`highE.minpa_io.read_records` reads either:

- local MATLAB v7.3 `ori/*.mat`; or
- the released fixed-width `.2B` product plus its `.2BL`/XML label.

The scientific value is DPF in `1/(s cm^2 sr eV)`, not a detector count.  The
four values `Ion_StartA_Count`, `Ion_StartB_Count`, `Ion_StopA_Count`, and
`Ion_StopB_Count` are preserved engineering monitor totals.  Neither the label
nor the website identifies them as independent background/dark measurements,
so the workflow never subtracts them from the science array.

The canonical split-record order is:

```text
[energy, pitch-major azimuth, mass]
```

| Modes | Energy bins | Pitch x azimuth | Mass bins | Resolved species |
|---|---:|---:|---:|---|
| 1--2 | 40 | 4 x 16 | 8 | H+, O+, O2+ |
| 3--5 | 64 | 4 x 16 | 8 | H+, O+, O2+ |
| 6 | 48 | 4 x 16 | 8 | H+, O+, O2+ |
| 7--8 | 64 | 16 x 16 | 16 | H+, O+, O2+ |
| 9--11 | 64 | 16 x 16 | 1 | H+ only |
| 12 | 48 | 1 x 1 | 32 | grouped H+, O+, O2+ |

Each raw Mode-12 record is split at `t+1.025 s` and `t+3.075 s` into two
`48 x 1 x 32 = 1536` subrecords.  Its one angular cell is azimuth-integrated.
It therefore supports spectra and density, but not independent vector velocity
or VDF reconstruction.  The API returns `status=direction_not_resolved` and
NaN vector components for Mode 12.

## Spectrum and moments

For each selected mass channel, the one-dimensional spectrum is the
solid-angle weighted mean.  Differential energy flux is

```text
DEF(E) = E * DPF(E)
```

The differential density contribution of a measured cell is

```text
dn = DPF * dE * dOmega / v
dE = E_high - E_low
E_edge(i+1/2) = sqrt(E_i E_(i+1))
v = sqrt(2 E e / (m mp))
```

where `v` is converted to cm/s for density in cm^-3.  The `m*mp` convention is
kept for numerical compatibility with the validated project processing.  Bulk
velocity is the `dn`-weighted mean of cell velocities.  Scalar temperature is
the trace-equivalent second central moment divided by three.  Values outside
`0 < DPF < 1e8`, non-finite values, and out-of-range energy bins do not
contribute.

MINPA observes a hemisphere, not `4*pi`.  The returned density is the integral
over the measured field of view; this pipeline does not invent an unmeasured
hemisphere correction.  The CLI also does not add spacecraft velocity unless
`--add-spacecraft-velocity` is explicitly supplied.

## Coordinate conversion

The latest local `NV/NV_MSO` signed-permutation audit fixes the MINPA-payload-to-body
mapping directly as

```text
[X_body, Y_body, Z_body] = [-V2_MINPA, +V3_MINPA, -V1_MINPA]
```

MOMAG attitude supplies the spacecraft-body-to-MSO matrix

```text
C_MSO_from_body = Rz(-yaw) @ Ry(-pitch) @ Rx(-roll)
```

and therefore

```text
q_payload = base_to_legacy_direction(theta, phi)
q_body    = [-q_payload_y, +q_payload_z, -q_payload_x]
V_MSO   = C_MSO_from_body @ q_body * speed
```

The helper name `base_to_legacy_direction` is retained for compatibility with
the historical geometry code; its returned components are the MINPA payload
components used by the explicit payload-to-body mapping above.

Attitude matching is nearest-neighbor with a 5 s tolerance.  Missing attitude
does not invalidate density: MINPA/body results remain available, MSO velocity
is NaN, and the status is `attitude_unavailable`.  Rotation tests verify
orthogonality, determinant +1, and speed preservation.

## Project flags and native Quality

The native `Quality` value is stored unmodified.  Its individual bit meanings
are not guessed.  Separately, `highE.minpa_quality` implements the approved
project definition (`2026-07-13-seven-band-all-species-v2-mode12-product-time`):

| Bit/value | Meaning | Default moment policy |
|---|---|---|
| 1 / 1 | 5--10 valid 2-D channels | retain with caution |
| 2 / 2 | fewer than 5 valid 2-D channels | reject |
| 3 / 4 | more than 80% of 1-D DEF bins exceed `1e5` | reject UV-like record |
| 4 / 8 | odd/even peak-valley acquisition pattern | reject |
| 5 / 16 | any 2-D DEF cell exceeds `1e10` | retain by default; configurable |

The odd/even test is performed independently in seven energy bands.  See
[`method_tw1_minpa_quality_flag.md`](method_tw1_minpa_quality_flag.md) for the
review thresholds and MATLAB audit workflow.  The Python and MATLAB cores use
the same thresholds.  Spectra are retained for rejected records so users can
see why a moment is missing.

For background-candidate estimation, the agreed project policy rejects bits
3--4, retains bits 1, 2, and 5, and additionally requires native `Quality==0`
at estimation time.

## Background noise

The frozen all-approved production release is v2.2.0 at
`release/minpa_unified_static_channel_denoise_v2.2.0/bundle.json`. Select
`all-approved-static-channel-subtract` with `--static-background-bundle` to
apply one time-invariant per-channel background to Mode 1, Mode 4, and Mode 12
in the same run. It does not use the exploratory quarterly temporal models.
Mode 1 uses 273 exact-time unique joint approvals; Mode 4/12 retain all 2,901
finalized species-specific approvals. Mode 7, unsupported channels, and
unreviewed mass bins remain unchanged.

Background is estimated and subtracted in four-dimensional Mode-1 DPF space,
never as a constant removed from DEF, density, or a final map:

```text
DPF_corrected = max(DPF_raw - DPF_background, 0)
```

The explicit CLI option `--background-model PATH` accepts only the frozen
Mode-1 model whose shape matches the input. With no model, values are
unchanged. The separate policy `multimode-paper-channel-subtract` accepts only
a finalized, hash-verified Mode-4/12 bundle. The frozen v2.0.0 bundle is
`outputs/minpa_multimode_channel_denoise_v2.0.0/bundle.json`; rc1 and prefilter
candidate workspaces remain review-only and are refused by the production
path. Mode 7, unreviewed Mode-4/12 mass bins, and unsupported channels remain
unchanged. Low-support reviewed channels are subtracted and explicitly marked.
UV-confirmed Mode-1 records are excluded rather than reconstructed
sector-by-sector.

The historical frozen Mode-1 v1.1.0 calibration contains 175 approved entries,
153 exact-time-deduplicated intervals, and 4,350 records. It is not rebuilt by
the multimode work. Mode 4/12 use independent H+, O+, and O2+ candidate lists;
automatic low-tail classification never constitutes approval.

See [`method_minpa_background_reproduction.md`](method_minpa_background_reproduction.md)
and [`method_minpa_background_temporal.md`](method_minpa_background_temporal.md)
for interval selection, zero-inclusive statistics, uncertainty, and temporal
variation analysis. See
[`method_minpa_multimode_noise_denoise.md`](method_minpa_multimode_noise_denoise.md)
for the review workflow, layouts, mass-bin scope, release gates, and rollback
rules. See
[`release_minpa_multimode_denoise_v2.0.0.md`](release_minpa_multimode_denoise_v2.0.0.md)
for finalized review counts, model hashes, channel support, and validation.
The superseding all-approved static release is documented in
[`release_minpa_unified_static_denoise_v2.2.0.md`](release_minpa_unified_static_denoise_v2.2.0.md).

## VDF figure

The VDF route follows the mature local MINPA implementation rather than using
one point at each channel center. For each accepted record it:

1. samples each pitch/azimuth cell on a `17 x 17` grid;
2. samples each energy cell at 9 points between logarithmic edges inferred
   from adjacent calibrated energy centers;
3. applies the validated `[-V2,+V3,-V1]` MINPA-payload-to-body mapping and that record's MOMAG
   attitude before combining records;
4. divides each parent cell's `dn` among its sub-samples, averages `dn` over
   records, and bins `sum(dn)/(dVa dVb)` in XY, XZ, and YZ;
5. plots finite geometric support in gray, positive signal in color, and the
   density-weighted bulk velocity from the same sub-samples.

White therefore means no sampled support, while gray means supported but no
positive signal. Since the released DPF is not converted here to a formally
calibrated phase-space-density variable, the color remains a reduced-density
proxy in `cm^-3 (km/s)^-2`; it must not be relabeled as absolute `f(v)`.
The example uses a one-minute VDF interval because the mature `17 x 17 x 9`
sub-sampling is intentionally denser than a channel-center quicklook.

## Extending the example

Change `INPUT`, UTC bounds, and `--species` in
[`examples/minpa_mode1_quickstart.py`](../examples/minpa_mode1_quickstart.py).
For a background-corrected Mode-1 product, add `--background-model` only after
the model's validity thresholds have been met.  For mixed-mode time spans,
read each file separately and call `process_records` per mode; energy grids
must not be concatenated into a false common spectrogram.
