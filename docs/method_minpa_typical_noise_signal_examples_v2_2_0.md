# MINPA v2.2.0 typical noise/signal example review

## Purpose and final status

This workflow pairs the existing manually approved noise reference for every
Mode 1/4/12 and H+/O+/O2+ combination with three observational signal
candidates. The human review is complete: exactly one signal image was retained
for each combination, giving 9 approved and 18 rejected signal candidates. It
is an example set, not a new background model or production classifier. The
frozen v2.2.0 bundle remains read-only and time independent.

## Candidate discovery

The full local 2021--2025 `day_spe` inventory is split into non-overlapping,
UTC-aligned five-minute windows. A window requires project quality to be
available, project bits 3--4 to be clear, at least 80% nominal cadence
coverage, and no gap longer than 1.5 nominal cadences. Native `Quality==0` is
checked later from the corresponding `ori` data.

The scan found 474,661 quality-clean five-minute windows. `day_spe` and the
native per-channel model use different angular/mass aggregation, so their
absolute amplitudes are never divided. Dimensionless spectral concentration,
three-energy-band persistence, and total-flux rank are used only to order the
windows. The top 500 windows per mode/species enter native validation.

## Native-channel validation

For each ranked interval, the raw DPF and frozen channel background use the
same units, `1/(s cm^2 sr eV)`, and the same instrument channel ordering. A
candidate must have:

- median maximum native-channel signal/background of at least 3 in at least
  two adjacent energy channels;
- signal/background exceedance in at least 50% of valid records;
- corrected ratio-peak movement of no more than one energy channel;
- at least 65% retention at the selected signal energy;
- native `Quality==0` and the cadence/coverage rules above.

Candidates are labelled rather than silently treated as equally strong:

- `strong_20x`: at least 20 times background and at least 95% peak retention;
- `clear_5x`: at least 5 times background and at least 80% peak retention;
- `moderate_3x`: passes only the minimum 3-times/65% gate.

The 3-times and retention thresholds are linked by the subtraction law:
subtracting one background unit from a three-background-unit observation has
a theoretical retention of 2/3. The lower tier is included for human review,
not as proof that the feature is physical.

## Figures and approval

The flat `pending/` folder contains 27 PNGs, three per mode/species group. Each
figure shows raw and corrected energy-time spectra on a shared LogNorm scale,
the five-minute interval mean spectrum, and the native-channel
signal/background spectrum. Black lines delimit the candidate; available
context extends by up to 12 minutes on each side. Data gaps are inserted as
NaN and remain blank. No smoothing or interpolation is used.

The decision rule was file-presence based: a retained, byte-identical PNG was
approved and a deleted expected PNG was rejected. Unexpected, renamed,
modified, or nested files fail finalization. The nine noise references already
come from manually approved noise intervals; historical rejected intervals
were not relabelled as signals.

Final retained signal intervals:

| Mode | Species | Start UTC | Tier | Max median channel S/B | Peak retained |
|---:|---|---|---|---:|---:|
| 1 | H+ | 2021-11-29 07:35 | strong_20x | 519.5 | 99.3% |
| 1 | O+ | 2024-10-04 17:10 | strong_20x | 304.0 | 99.0% |
| 1 | O2+ | 2024-01-23 17:45 | clear_5x | 26.9 | 88.9% |
| 4 | H+ | 2025-03-10 07:00 | strong_20x | 141.9 | 98.0% |
| 4 | O+ | 2022-04-25 16:30 | clear_5x | 28.1 | 91.3% |
| 4 | O2+ | 2022-07-22 07:00 | clear_5x | 58.9 | 95.1% |
| 12 | H+ | 2022-07-31 13:50 | moderate_3x | 4.0 | 70.2% |
| 12 | O+ | 2022-04-01 15:55 | moderate_3x | 3.0 | 71.4% |
| 12 | O2+ | 2024-05-04 06:40 | strong_20x | 35.6 | 97.2% |

## Limitations

Mode 12 is azimuth integrated. Its H+ and O+ candidates are substantially
weaker than most Mode 1/4 cases and require especially careful visual review.
Altitude, solar zenith angle, local time, MSO/MSE position, bow-shock/MPB
location, and upstream propagation are not selection variables because this
artifact tests instrument-channel subtraction, not a Mars-region event class.
Those quantities would be required before interpreting a retained signal as a
specific plasma population or boundary event.

## Reproduction

```powershell
python scripts\prepare_minpa_typical_signal_review_v2_2_0.py
```

Finalize a local retained/deleted workspace with:

```powershell
python scripts\finalize_minpa_typical_signal_review_v2_2_0.py
```

The portable final index is `examples/minpa_denoise_v2.2.0/manifest.json`.
Only the nine retained signal PNGs are published because each already includes
raw and corrected panels. Historical noise-reference images, rejected candidate
rows, and other review intermediates remain local.
The frozen bundle SHA-256 must remain
`0c21995c2b3baf7ba82e0893aa0aa5c81c17f268787b28e3d75181dcac468c71`.
