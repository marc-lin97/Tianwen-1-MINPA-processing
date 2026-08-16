# MINPA unified static channel denoise v2.2.0

## Release status

`minpa-unified-static-channel-denoise-v2.2.0` is frozen. It uses every currently
approved interval, exact-time deduplicates equal windows, and estimates one
time-invariant background for every reviewed detector channel. Quarterly or
seasonal scaling is explicitly disabled.

Frozen bundle committed for portable use:

```text
release/minpa_unified_static_channel_denoise_v2.2.0/bundle.json
SHA-256: 0c21995c2b3baf7ba82e0893aa0aa5c81c17f268787b28e3d75181dcac468c71
```

Mode-1 model:

```text
models/minpa_mode01_background_v1.2.0.npz
SHA-256: 5436381e79c3ab52bc9e959534e638244634df74cb01d477edd4f5edb3673835
```

Mode 4 and Mode 12 retain the frozen v2.0.0 model hashes. The bundle stores the
three model hashes and release policy. Public interval metadata are deliberately
limited to `calibration_intervals_approved.csv`, containing only mode, species,
start UTC, and stop UTC; local review products and per-interval results are not
distributed.

## Estimator and correction

For approved interval `j` and detector channel `c`:

```text
b_jc = mean(x_tc | x_tc is finite and x_tc > 0)
B_c  = mean_j(b_jc) over finite interval estimates
corrected = max(raw - B_c, 0)
```

Intervals have equal statistical weight regardless of duration. Input and
background are DPF in `1/(s cm^2 sr eV)`. NaNs remain NaN. No smoothing,
interpolation, temporal scaling, or cross-mass substitution is allowed.

Support levels remain:

- `unsupported`: no positive approved sample; preserve the raw channel;
- `low_support`: fewer than 3 supporting intervals or 5 positive samples;
  subtract and retain the flag;
- `supported`: at least 3 supporting intervals and 5 positive samples.

## Approved interval inventory

Mode 1 combines seven completed joint H+/O+/O2+ review sources. It imports 295
approved entries and exact-time deduplicates them to 273 independent windows.
All 273 pass project-quality availability, project bits 3--4, native
`Quality==0`, Mode-1 layout, and the approved 2--10 minute duration range.

The authoritative v1.1 inventory contains 13 physical windows that appear in
both its approved list and its retained historical-reject provenance. They
remain included because the authoritative `approved_intervals` membership was
already used by v1.1; the collisions are explicitly recorded rather than
silently discarded. Three pairs of approved windows overlap without identical
boundaries. Both approvals retain equal interval weight, while 78 duplicate raw
epochs loaded through overlaps are removed before estimation.

Mode 4/12 reuse all 2,901 finalized species-specific approved intervals from
v2.0.0. No automatic or pending candidate enters this release.

## Channel scope and validation

Mode-1 support over all 20,480 channels is:

| Level | Channels |
|---|---:|
| supported | 9,148 |
| low_support | 6,818 |
| unsupported | 4,514 |

Compared with Mode-1 v1.1.0, 15,094 channels have finite positive backgrounds
in both models. The new/old background ratio has median 1.000 and 5th--95th
percentiles 0.9947--1.000. The larger inventory primarily improves channel
support rather than shifting the typical subtraction amplitude.

Mode 4 corrects only reviewed 1, 16, and 32 amu bins. Mode 12 corrects the two
reviewed mass bins for each H+, O+, and O2+ species after native record
splitting. Mode 7, unreviewed mass bins, and unsupported channels remain raw.

Round-trip loading validates Modes 1/4/12, and a synthetic three-mode check
confirms every finite corrected value is no greater than its raw value. The
full project test suite must pass before publishing the GitHub release assets.

## Nine reviewed before/after examples

The observational-signal review retained one of three five-minute candidates
for every Mode 1/4/12 and H+/O+/O2+ combination: 9 approved and 18 rejected in
total. The nine retained PNG files are committed under
`examples/minpa_denoise_v2.2.0/signal/`; `manifest.json` preserves the final
example times and figure hashes plus the aggregate rejected count. Every PNG
already contains raw and corrected spectrogram panels, an interval-mean spectrum,
and a native-channel signal/background diagnostic. Separate historical noise-
window images and rejected-candidate rows remain local. This example review does
not alter the frozen model.

## Usage

```powershell
python scripts\run_tw1_highE_day.py YYYYMMDD `
  --background-policy all-approved-static-channel-subtract `
  --static-background-bundle release\minpa_unified_static_channel_denoise_v2.2.0\bundle.json
```

The default remains `none`. The new policy is explicit opt-in and retains the
existing whole-record Mode-1 UV rejection.

## Verification, rebuild, and data availability

Verify the committed model and all nine reviewed figures without mission data:

```powershell
python -m pip install -e .[test]
python scripts\verify_minpa_v2_2_0_release.py
```

Rebuild locally with:

```powershell
python scripts\build_minpa_unified_static_v2_2_0.py
```

Exact model re-estimation additionally requires the local read-only MINPA
archive and the completed manual-review inventories; those raw mission inputs
are not redistributed. The frozen model files are committed so readers can
apply and verify the published correction without rebuilding the calibration.
Do not publish any changed bundle or model hash under v2.2.0.
