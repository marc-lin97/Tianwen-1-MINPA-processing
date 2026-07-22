# MINPA Density Audit: Differential Number Flux Handling

## Question

For the 2021-12-03 00:00-01:00 UTC MAVEN SWIA and Tianwen-1 MINPA comparison,
the MINPA H+ density is much lower than SWIA while the MSO velocity components
and speed agree closely. This note checks whether the MINPA moment code
misinterprets the MINPA differential number flux.

## Moment Formula

If MINPA `Ion_Count` is differential number flux,

```text
J(E) [cm^-2 s^-1 sr^-1 eV^-1],
```

then the density contribution of one energy-angle cell is

```text
dn = J(E) * dE * dOmega / v .
```

The inspected Python and legacy MATLAB code compute

```text
DEF = J(E) * E
dn = DEF * (dE/E) * dOmega / v
```

which is algebraically the same expression, because `DEF * (dE/E) = J(E) * dE`.
The unit check is

```text
(cm^-2 s^-1 sr^-1 eV^-1) * eV * sr / (cm s^-1) = cm^-3 .
```

Therefore the current implementation is consistent with `Ion_Count` being
differential number flux, not differential energy flux.

## Code Evidence

- Current Python MINPA moment code (`src/highE/tw1_minpa.py` and
  `src/highE/minpa_processing.py`) computes logarithmic channel edges from the
  calibrated energy centers and integrates
  `density = flux * (E_high-E_low) * omega / speed_cm_s` per channel.
- Current method note:
  `docs/method_tw1_minpa_highE.md`
  documents `Ion_Count` as differential particle flux and uses
  `DEF = Ion_Count * E`, `dn = DEF * (dE/E) * dOmega / v`.
- Prior MINPA workflow note:
  `D:\codex\处理MAVEN数据\docs\minpa_moments_notes.md`
  documents the same convention.
- Legacy MATLAB:
  `D:\Code\TW-1\TW1_MINPA_deal_v3.m`
  computes `e = Energy .* ion.count`, uses `dE_E = 0.15`, and
  integrates `N = e .* dE_E .* omega / speed_cm_s`.

## 2021-12-03 00:00-01:00 Check

Using the existing 1-minute median comparison table
`D:\Data\TW-1\result_codex\comparisons\maven_swia_tw1_minpa_20211203_0000_0100\swia_minpa_hplus_1min_median_20211203_0000_0100.csv`:

- The 00:02 UTC MINPA point has density about `0.00235 cm^-3` and speed about
  `52.5 km/s`; it is an obvious bad/partial sample for this solar-wind
  comparison.
- After excluding rows with `MINPA density <= 0.1 cm^-3` or
  `MINPA speed <= 300 km/s`, 57 one-minute rows remain.
- Median MINPA/SWIA density ratio: `0.6611`.
- Mean MINPA/SWIA density ratio: `0.6787`.
- A constant density scale factor of about `1.51` would align the median MINPA
  density with SWIA for this interval.
- Median speed difference, MINPA minus SWIA: `-3.37 km/s`.

This pattern is consistent with a scale-like density issue while the velocity
moment remains robust. A common multiplicative flux, geometric factor, energy
width, or solid-angle error scales density directly but largely cancels in the
normalized bulk velocity.

## Energy Spacing Check

The historical code used a fixed `dE/E = 0.15`; the current code no longer
does. A fixed value can mean the effective
relative energy-bin integration width, but it is not identical for every MINPA
mode to the spacing between adjacent energy centers.

For a logarithmically spaced energy table with adjacent center ratio
`r = E(i+1)/E(i)`, common estimates are:

```text
lower-center spacing:      (E(i+1)-E(i))/E(i)                 = r - 1
upper-center spacing:      (E(i+1)-E(i))/E(i+1)               = 1 - 1/r
arithmetic-center spacing: (E(i+1)-E(i))/((E(i)+E(i+1))/2)    = 2(r-1)/(r+1)
log-bin full width:        (sqrt(r)*E - E/sqrt(r))/E          = sqrt(r)-1/sqrt(r)
```

The legacy MATLAB spectrum code uses the last expression: it forms the upper
and lower logarithmic half-bin edges and divides the resulting full bin width
by center energy.

Using the legacy `ENERGY_EV` tables:

```text
mode(s)       bins  adjacent ratio r   log-bin dE/E   arithmetic-center spacing
1,2           40    1.2629636          0.2339916      0.2324064
3,4,5,7,8     64    1.1554888          0.1446493      0.1442724
6             48    1.2137600          0.1940261      0.1931194
9,10,11       64    1.1049416          0.0998339      0.0997098
12            48    1.1628253          0.1509958      0.1505673
```

The 2021-12-03 00:00-01:00 comparison falls in a file named `MOD1-DEF`, so
the mode-1 energy-center spacing gives an effective logarithmic full-bin
`dE/E` of about `0.234`, close to but not exactly `0.227`. The value `0.227`
came from forcing the median MINPA/SWIA density ratio to unity:
`0.15 / 0.661 ~= 0.227`.

The current implementation uses each channel's actual logarithmic width
`E_high-E_low`. For Mode 1, the table is logarithmically uniform and the
resulting median `dE/E` is `0.2339916`. This reproduces the local `NV_MSO_2`
energy-width convention and removes the former fixed-15% density bias.

## Likely Non-Formula Causes

1. MINPA field-of-view and angular coverage. If part of the proton distribution
   is outside the active look directions, the density integral is biased low,
   while the measured beam direction and velocity can still look very close.
2. Absolute calibration or geometric factor. A multiplicative calibration factor
   affects density, but cancels to first order in the bulk velocity
   `sum(dn*v)/sum(dn)`.
3. Effective `dOmega`, angular response, or active-sector weighting. Energy
   widths are now taken from the calibrated center table; angular-response
   uncertainties can still affect density while cancelling in bulk velocity.
4. Spacecraft separation. MAVEN SWIA and Tianwen-1 MINPA are not co-located;
   solar-wind density structures can vary more strongly than bulk flow velocity.
5. Product-level flags, low counts, background subtraction, mass-channel
   separation, or partial accumulations. These mostly perturb absolute density
   and can leave velocity relatively stable.

## Recommended Next Checks

1. Recompute the 2021-12-03 00:00-01:00 H+ density directly from the raw/public
   MINPA differential number flux, with an explicit per-bin audit table
   containing `J`, `E`, `dE`, `dOmega`, `v`, and `dn`.
2. Compare that direct integration with the local `NV_MSO` density product
   before comparing to SWIA.
3. Add a per-time coverage metric: active angular bins, total effective
   `dOmega`, beam direction relative to MINPA FOV, and whether the core beam is
   near a FOV edge.
4. Repeat the SWIA comparison on several quiet upstream intervals. A nearly
   constant MINPA/SWIA ratio points to calibration or geometric factor; a ratio
   that depends on attitude/FOV points to angular coverage.

## Current implementation validation

The recommended raw-to-product check has now been completed for 2021-12-03
00:00--01:00 UTC. Using all Mode-1 energy channels and per-channel logarithmic
widths, 210 finite H+ records have a median current/`NV_MSO_2` density ratio of
`0.99999983`; the median absolute density difference is `8.75e-7 cm^-3`.
The checked report is `docs/assets/minpa_mode1_example/validation_nv_mso_2.json`.
