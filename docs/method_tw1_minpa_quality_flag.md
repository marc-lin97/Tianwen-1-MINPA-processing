# Tianwen-1 MINPA Quality Flag

## Purpose

This note defines a per-time MINPA data-quality flag for the derived
`NV_MSO_2` moments and daily spectra products. Raw `ori` files remain
read-only. The flag is computed from the original energy-angle-mass data and
then attached to derived products after visual audit.

## Bit Encoding

The final flag is a five-bit unsigned integer bitmask. The bit order follows
the five problem definitions in the user request:

```text
bit 1, value 1  : valid 2D channel count is 5-10, use with caution
bit 2, value 2  : valid 2D channel count is <5, invalid sample
bit 3, value 4  : more than 80% of 1D energy channels have DEF > 1e5
bit 4, value 8  : 1D peak/valley alternation with odd/even phase ordering
bit 5, value 16 : any 2D DEF channel exceeds 1e10
```

`quality_flag_binary` stores the same information as a five-character string
in problem order. For example:

```text
00000 -> no flagged problem
10000 -> bit 1 is set, integer flag 1
01000 -> bit 2 is set, integer flag 2
00001 -> bit 5 is set, integer flag 16
01001 -> bits 2 and 5 are set, integer flag 18
```

Explicit boolean columns are also written:

```text
flag_bit1_sparse_caution
flag_bit2_sparse_invalid
flag_bit3_uv_contamination
flag_bit4_evenodd_error
flag_bit5_high_channel
```

These columns prevent leading-zero loss when CSV files are read by tools that
infer `quality_flag_binary` as a number.

## Computation

Audit script:

```text
scripts/audit_tw1_minpa_quality_flags.m
```

The review/audit script computes H+. The production core applies the same
definition independently to H+, O+, and O2+ in a single read of each ori
record. For each species and record:

- The valid 2D channel count uses raw DPF/count-like energy-angle values:
  `0 < DPF < 1e8`.
- The high-channel check uses 2D differential energy flux:
  `DEF = DPF * E`.
- The 1D UV check uses a solid-angle weighted 1D DEF spectrum.
- The odd/even acquisition-error check uses the shape of the solid-angle
  weighted 1D DEF spectrum. For interior energy channel `i`, the channel is
  counted as an alternating extrema channel if `DEF_i` is greater than both
  neighbors and `DEF_i >= 1e5`, or if `DEF_i` is less than 50% of both
  neighboring channels. Endpoints of each tested band are not counted because
  they cannot be tested as local peaks or valleys. Bit 4 is not triggered by
  UV-contamination context and does not depend on neighboring bit-3 points. It
  is set from the current record alone.
- Bit 4 is now evaluated independently in seven bands: all energies,
  `0-30 eV`, `30-300 eV`, `300 eV` to the highest available energy in that
  mode, `0-50 eV`, `50-500 eV`, and `500-5000 eV`. If any one band satisfies
  both conditions, bit 4 is set. The conditions are:
  at least 52.5% of finite interior channels in that band are local peaks or
  valleys, and at least 55% of those local extrema fit one odd/even phase
  pattern. The two accepted phase patterns are odd-index peaks with even-index
  valleys, or even-index peaks with odd-index valleys.
- In the audit table, the legacy columns `alternating_extrema_fraction` and
  `alternating_phase_fraction` continue to describe the all-energy band.
  Additional columns preserve the band-specific decisions and metrics:
  `flag_bit4_evenodd_all_energy`, `flag_bit4_evenodd_0_30eV`,
  `flag_bit4_evenodd_30_300eV`, `flag_bit4_evenodd_300_maxeV`,
  `flag_bit4_evenodd_0_50eV`, `flag_bit4_evenodd_50_500eV`, and
  `flag_bit4_evenodd_500_5000eV`, plus matching `alternating_extrema_*` and
  `alternating_phase_*` columns for each band.
  These thresholds preserve reviewed boundary examples such as
  `2022-07-16T00:30:55.949Z`, while reducing false positives relative to the
  looser 0.5/0.5 threshold pair.
- Modes 1, 4, and 7 retain their mode-specific energy-angle grids. Mode 12 is
  split into two sub-records by the source format; each split row contains
  `32 mass bins x 48 energy bins x 1 angular sample = 1536` values. Therefore
  mode 12 quality metrics use one angle per split record. Treating it as 16
  angles makes every extracted species cube fail its dimension check and is
  invalid.

For quick visual audit, run with early stop enabled:

```matlab
summary = audit_tw1_minpa_quality_flags({}, 300, true);
```

For production all-record flag generation, run without early stop:

```matlab
summary = audit_tw1_minpa_quality_flags({}, inf, false);
```

The audit writes tables and example figures under:

```text
outputs/tw1_minpa_quality_flag_audit
```

For full production processing, use the day-by-day driver instead of one
global audit table:

```text
scripts/run_tw1_minpa_quality_flags_all.m
```

Dry-run one day:

```matlab
summary = run_tw1_minpa_quality_flags_all({'20211203'}, true, 1);
```

Full run after audit approval:

```matlab
summary = run_tw1_minpa_quality_flags_all({}, false, inf);
```

The driver writes per-day audit outputs under:

```text
outputs/tw1_minpa_quality_flags_full\YYYYMMDD
```

Processing days independently avoids keeping all MINPA records in memory.

## Monthly Sample Validation

For review across the full local ori collection, run:

```matlab
summary = run_tw1_minpa_monthly_quality_flag_audit();
```

This deterministic validation selects the middle available start date in each
`yyyyMM` group, computes the current strict H+ quality flag for all selected
days, and then selects up to five representative examples for each issue bit.
The context figures use +/-5 min windows with the project `day_spe`
spectrogram, the flag panel, NV_MSO_2 density, and NV_MSO_2 speed magnitude.

Outputs are written under:

```text
outputs/tw1_minpa_quality_flag_monthly_sample_strict
```

Key review files:

```text
tw1_minpa_monthly_sample_dates.csv
tw1_minpa_hplus_quality_flag_audit_records.csv
tw1_minpa_hplus_quality_flag_issue_examples.csv
tw1_minpa_hplus_quality_flag_issue_example_counts.csv
issue_classes_1_5_context_5min
```

For a broader mode-filtered review, use the three-days-per-month mode 1/4/12
driver:

```matlab
summary = run_tw1_minpa_monthly3_mode1412_quality_flag_audit();
```

This selects up to three available start dates per month after filtering to
MINPA modes 1, 4, and 12. The selected dates are deterministic: first,
middle, and last available start date in each month, or all dates when fewer
than three are available. Requested context examples are bit1=10, bit2=10,
bit3=20, bit4=20, and bit5=10.

Default output:

```text
outputs/tw1_minpa_quality_flag_monthly3_mode1412
```

Key review files:

```text
tw1_minpa_monthly3_mode1412_sample_dates.csv
tw1_minpa_hplus_quality_flag_audit_records.csv
tw1_minpa_hplus_quality_flag_issue_examples.csv
tw1_minpa_hplus_quality_flag_issue_example_counts.csv
issue_classes_1_5_context_5min
```

## Product Insertion

The approved three-species production driver is:

```text
scripts/run_tw1_minpa_quality_flags_all_species.m
```

One-product dry-run:

```matlab
summary = run_tw1_minpa_quality_flags_all_species({'20250916'}, true, 1, true);
```

Full restartable production run:

```matlab
summary = run_tw1_minpa_quality_flags_all_species({}, false, inf, false);
```

Each ori file is read once and saved as a compact three-species checkpoint
under `outputs/tw1_minpa_quality_flags_all_species/ori_flag_checkpoints`.
Derived MAT files are written to a temporary file, schema-checked, and then
replaced. Raw ori files remain read-only.

The matcher uses `(UTC, mode)`, not the date embedded in a derived filename.
This is required because some local `NV_MSO_2` files span adjacent UTC dates
and some `day_spe` files contain segments whose true UTC is far from the
filename date. Matching tolerance is 1 ms.

Variables added to `NV_MSO_2` for `S = H, O, O2`:

```text
quality_flag_S_TW1           [time_unix_s, uint32_bitmask]
quality_flag_bit_S_TW1       [time_unix_s, bit1, bit2, bit3, bit4, bit5]
quality_flag_binary_S_TW1    string vector
quality_flag_available_S_TW1 [time_unix_s, 0_or_1]
MINPA_quality_flag_info metadata struct
```

The corresponding H+, O+, or O2+ flag is added to every segment of
`H_spe_num`, `O_spe_num`, or `O2_spe_num`:

```text
quality_flag
quality_flag_bit
quality_flag_binary
quality_flag_available
quality_flag_species
```

`quality_flag_available=false` and `quality_flag_binary="99999"` mark an
unmatched or unavailable species. The numeric zero stored in that case is only
a placeholder and must not be interpreted as a good-quality sample.

Regression script:

```text
scripts/validate_tw1_minpa_quality_flags_all_species.m
```

The H+ vectorized core matches the reviewed scalar audit point-for-point for
modes 1, 4, 7, and corrected mode 12. The fixed regression set contains 2577
records with zero H+ flag mismatches.

## Current Audit Status

The seven-band bit-4 audit examples cover all five problem classes. A full
mode-1/4/12 monthly-three-day review is preserved under
`outputs/tw1_minpa_quality_flag_monthly3_mode1412_7band_bit4`. That directory
predates the mode-12 one-angle correction, so its mode-12 rows are superseded;
mode-1/4 rows remain useful. Corrected mode-12 regression outputs are under
`outputs/tw1_minpa_quality_flag_regression_mode4_12`. Production validation on
product `20250916` matched every NV and spectra timestamp for all three species
and then passed a write/read schema and bit-reconstruction check.

## Full Production Result

Algorithm version:

```text
2026-07-13-seven-band-all-species-v2-mode12-product-time
```

The completed run produced 9553 source checkpoints containing 5,795,135 ori
records. All checkpoints reopened successfully, and comparison with current
source file size and modification time found zero changed raw ori files.

Source-record flag totals are:

```text
species  flagged     bit1      bit2       bit3    bit4     bit5
H+       2,171,618   592,527   1,005,466  24,690  574,939  70
O+       4,957,345   1,314,992 3,558,354  13,370  101,656  70
O2+      5,140,597   1,206,233 3,873,684  16,779   59,761  70
```

Bits are not mutually exclusive, so bit-column totals do not sum to the
flagged-record count.

All 1051 `NV_MSO_2` and 1051 `day_spe` product pairs were written or already
held the validated version. Independent reopening found zero schema failures,
zero bit-reconstruction failures, and zero unmatched records for every
species. Per species, the verified products contain 6,549,739 NV rows and
7,284,379 spectra rows, all with `quality_flag_available=true`.

Machine-readable production summaries are under:

```text
outputs/tw1_minpa_quality_flags_all_species
outputs/tw1_minpa_quality_flags_all_species_validation
```
