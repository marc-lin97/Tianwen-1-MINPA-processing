# Codex Workflow Log

## 2026-07-15 MINPA spectrum and moments algorithm documentation

- Added one source-of-truth Markdown method summary and a reproducible DOCX
  build script covering the full route from raw MINPA DPF through species DEF
  spectra, density, bulk velocity, scalar temperature, MSO/MSE rotations,
  quality flags, and downstream products.
- The documentation explicitly separates the legacy/day_spe production chain,
  the mode-width-rescaled `NV_MSO_2` chain, and the independent raw-data Python
  route. It does not describe `NV_MSO_2` as a raw per-bin reintegration.
- Recorded the mode-12 product-layout hazard: the corrected quality route uses
  two 1536-value subrecords with `32 mass x 48 energy x 1 angle`; a generic
  16-azimuth reshape is not a production-safe assumption for those rows.
- Recorded that current production moments stop at density and bulk velocity;
  scalar temperature exists only in the independent direct processor, while
  the full pressure tensor and heat flux remain unimplemented production
  quantities.

## 2026-07-14 Tianwen-1 MINPA all-energy MSE grid archive

- Added a separate all-energy H+/O+/O2+ per-cell HDF5 route sourced directly
  from `NV_MSO_2`; it preserves per-record density and never treats a
  cross-epoch sum as local density.
- The 1051 NV products contain 6,549,739 rows but 5,548,505 unique epochs.
  Every unique epoch has exactly one row whose true UTC date matches the NV
  filename date; using that row removes 1,001,234 adjacent-file duplicates and
  selects the correctly dated legacy velocity and `Xb_MSO`.
- Geometry discovery found the same 1029 usable UTC dates as the high-energy
  route and 22 dates missing both MOMAG position and R_MSO2MSE. The latter are
  explicitly reported rather than assigned guessed coordinates.
- The `20211203` single-day benchmark wrote 4,382 in-grid records to 917 files
  in about 4.7 s with 1.05 MiB of day-array memory. Its independent audit
  passed 59 fields and source-recomputed samples. Comparison against the
  audited high-energy archive found zero MSE-position, rotation-time, FOV, or
  O+/O2+ quality-field differences across all 4,382 common records.
- The D-drive writer completed all 1029 processable dates in 4,313.1 s
  (71.9 min), with 2.48 MiB maximum day-array memory. It retained 5,452,228
  canonical rows before spatial selection and 3,979,338 in-grid rows. The 22
  excluded geometry dates contained 96,277 canonical rows, closing the global
  unique count exactly at 5,548,505.
- The 11.23 GiB final archive has 155,145 HDF5 files and 558,551 cell-day
  groups. The 6,117 s independent audit passed with zero errors, no staging
  groups, exact day/global count closure, and 2,703 source-recomputed record
  comparisons with zero field mismatches. Rotation orthonormality and
  determinant errors remained below `7e-16`; `logs/full_audit_summary.json` is
  the authoritative report.

## 2026-07-14 Tianwen-1 MSE grid storage acceleration

- The original production root on `E:` was identified as a Toshiba external
  USB disk, while `D:` is a Fanxiang S790 NVMe SSD. Recent days touched about
  846 cell files per day and only 11.4% of consecutive-day files overlapped,
  so a large open-file cache would provide little benefit and add HDF5 risk.
- Copied the read-only daily high-E input tree from `E:` to
  `D:\Data\highE\Tianwen-1`. All 4,120 files and 1,964,847,288 bytes matched,
  followed by a per-file SHA-256 comparison with zero mismatches.
- Started an isolated full production root at
  `D:\Data\highE\processed_mse_spatial_stats\tw1_minpa_grid_records_5Rm`.
  The E-drive writer was left intact as a temporary backup; the two processes
  never share an output file. The D-drive route uses the identical schema,
  filters, coordinate checks, units, flags, and daily-resume logic.
- The initial 45-second D-drive benchmark advanced from 18 to 30 completed
  days, or 15.99 days/minute, with both writer and audit-watcher stderr logs
  empty. The final run completed 1029/1029 dates in 3410.41 s (56.84 min).
- After D caught the E backup at 258 days, compared 25 spatial cell files from
  five representative dates (`20211203`, `20211225`, `20220517`, `20220815`,
  and `20220923`), including duplicate-rotation-axis cases. All 91 sampled rows,
  compound dtypes, shapes, and every named field were exactly equal with NaNs
  treated consistently. Coordinate/velocity recomputation and quality metadata
  attributes also matched. Expected differences were only input/output drive
  paths, completion timestamps, and one legacy E-day summary lacking the later
  duplicate-rotation conflict-count audit key; no scientific values differed.
- Final streaming audit passed with zero errors across 156,033 schema-v3 HDF5
  files, 565,545 cell-day groups, and 3,539,768 records. It reconciled all 1029
  source dates and day summaries, found no staging groups, verified all 43
  fields and both species' five NV bits, and found zero position and O+/O2+
  velocity recomputation error. Rotation orthonormality/determinant errors were
  below 7e-16. The final pytest run passed 26/26 tests. Authoritative evidence
  is the D-drive `logs/full_audit_summary.json` and
  `logs/audit_watcher_status.json`.
- A focused daily-provenance audit then covered the 128 dates with no in-grid
  HDF5 group. It passed all 1029 summaries and 4,116 unique O+/O2+/rotation/NV
  source paths, confirmed one NV algorithm version, reconciled 4,803,234 input
  epochs to 3,539,768 in-grid records, and reproduced the full audit's error
  maxima. Evidence is `logs/day_summary_provenance_audit.json`.
- After the D-drive archive passed both audits, the redundant E-drive writer
  was stopped at 351/1029 days without deleting its files. Its watcher recorded
  `production_incomplete`; the E output is non-authoritative and the completed
  D output is the sole final archive.

## 2026-07-13 Tianwen-1 MINPA three-species quality flags

- The approved five-bit quality definition is now computed independently for
  H+, O+, and O2+ with seven bit-4 energy intervals: all energy, 0-30,
  30-300, 300-max, 0-50, 50-500, and 500-5000 eV.
- Mode 12 source rows are split into two records of 1536 values each. Their
  actual layout is 32 mass bins x 48 energy bins x 1 angular sample; using the
  normal 16-azimuth assumption makes species extraction invalid.
- Do not use the `NV_MSO_2` or `day_spe` filename date as the matching key.
  Local files can span adjacent dates, and at least one `day_spe` file contains
  stale segments from a distant date. Product insertion must match actual UTC
  and mode with a 1 ms tolerance.
- Full processing uses one atomic checkpoint per ori file, computes all three
  species in one read, and writes derived MAT files through a checked temporary
  file. `quality_flag_available=false` plus binary `99999` marks missing flag
  coverage; numeric zero alone is not sufficient.
- H+ regression against the reviewed scalar audit passed for modes 1, 4, 7,
  and corrected mode 12. The 20250916 product write/read test had zero
  unmatched NV or spectra records for all three species.
- Full production completed 9553/9553 ori checkpoints and 1051/1051
  NV_MSO_2/day_spe product pairs. Independent reopening verified 6,549,739 NV
  rows and 7,284,379 spectra rows per species with zero unavailable flags,
  zero schema failures, and zero bit-reconstruction failures. Raw ori size and
  modification-time changes were zero.

## 2026-06-13 Tianwen-1 MINPA high-E route

- Added `src/highE/tw1_minpa.py`, `scripts/run_tw1_highE_day.py`, and
  `scripts/run_tw1_highE_adaptive.py` after MAVEN completion.
- Tianwen-1 source priority is `D:\Data\TW-1\result\MINPA\ori` first and
  `D:\Data\TW-1\rawdata\MINPA\public` only when no overlapping `ori` files
  exist for the UTC day.
- MINPA FOV is handled separately from MAVEN STATIC: output records MINPA probe
  coordinate axes in MSO/MSE and solid-angle half-axis fractions, not STATIC
  theta-edge FOV flags.
- `20211203` write test used 10 `ori` files, processed 4471 MINPA records, and
  wrote 4467 unique-time rows per species after dropping 4 duplicate extra
  records per species. Wall time was about 14.28 s.
- Current Tianwen-1 common-date discovery found 1029 runnable days with local
  MINPA, MOMAG, and R_MSO2MSE coverage; all discovered common days currently
  resolve to `ori` source files.
- Correction after user review: Tianwen-1 FOV should also use theta-edge logic,
  but with MINPA's own `360 deg x 90 deg` ion FOV hemisphere, not MAVEN
  STATIC's `+-45 deg` theta range. The first adaptive Tianwen-1 run was stopped
  before completion; outputs produced before this correction should be treated
  as superseded and regenerated with overwrite.
- Final Tianwen-1 overwrite pass completed 1029/1029 common days. Thirteen days
  initially failed because some MINPA `ori` records had blank or malformed UTC
  fields or mismatched UTC/Ion_Count record counts; the reader now discards
  those bad source records and keeps valid paired records. The 13 failed days
  were rerun, with the final 6-day retry finishing with `failed={}`.
- 2026-07-12 validation against local legacy MINPA `NV`/`NV_MSO` products for
  `20211203` confirmed the Probe-to-MSO attitude rotation
  `Rz(-Yaw)*Ry(-Pitch)*Rx(-Roll)`. The local `NV_20211203.mat` columns reproduce
  stored `NV_MSO` only when mapped to Probe as `[-V2,+V3,-V1]`; the older
  `TW1_MINPA_redeal_3.m` text mapping `[-V3,-V1,+V2]` preserves speed but not
  MSO components for this local file. Validation outputs are under
  `outputs/tw1_minpa_velocity_mso_rotation_validation`.
- 2026-07-12 follow-up: the `[-V2,+V3,-V1]` mapping is the fixed MINPA
  instrument-to-orbiter mounting matrix. The previous high-E code's two
  `legacy_to_probe_direction` calls are algebraically equivalent to this
  matrix, so the effective transform was correct; the implementation now names
  the fixed matrix explicitly in `highE.coordinates`.
- 2026-07-13: added the Tianwen-1 per-cell MSE record archive. It uses strict
  `[-5,+5] Rm` edges, `0.1 Rm` cells, and `Rm=3397 km`; reads one paired O+/O2+
  day at a time; recomputes MSE position and velocity from MSO using the nearest
  daily `R_MSO2MSE`; and writes only the touched position-named HDF5 cell files.
  All finite-position in-grid records are retained without the spatial-stats
  `0.001 cm^-3` cutoff. Each record preserves MINPA FOV, species processing,
  finite-rotation, and rotation-time-gap flags. Per-day JSON summaries make the
  long run resumable without loading historical cell datasets.
- 2026-07-13 quality-flag follow-up: each O+ and O2+ cell-day group also carries
  the daily `NV_MSO_2` five-problem flag mask, five explicit bits, availability,
  and epoch-match diagnostics. `available=0` is preserved because its numeric
  zero mask is only a missing-value placeholder, not a clean-data judgment.
- 2026-07-13 rotation-axis follow-up: many daily `R_MSO2MSE` files contain
  duplicate epochs and a few contain slight local reordering. The grid writer
  uses stable time sorting and collapses duplicates only after selecting the
  matrix with the most finite elements. Equally complete candidates retain
  stable source order; their conflict count and maximum matrix difference are
  recorded rather than hidden. Every day records duplicate/reorder diagnostics,
  and the recomputed MSE vectors must still reproduce the stored daily-product
  MSE vectors before any cell output is accepted.
- 2026-07-13 HDF5 acceleration follow-up: replaced roughly 40 small datasets
  per cell-day with one compound `records` dataset containing the same 43 named
  fields and native dtypes. The 20211203 benchmark improved from about 7.9 s
  and 31.22 MiB to 2.9 s and 16.95 MiB. A complete comparison of 917 cells and
  4382 rows found zero field-value differences. The streaming rule remains one
  input day in memory and only trajectory-touched cell files opened.
- 2026-07-13 completion-audit follow-up: added a streaming full-product audit
  that reconciles all source dates, day summaries, cell-day groups, and rows;
  validates the exact 43-field schema, every row's grid assignment, coordinate
  and rotation tolerances, provenance paths, validity flags, and both species'
  five NV bits. Empty transaction groups can be removed after the writer exits;
  non-empty staging content is never removed automatically.
- 2026-07-14 X-Z statistics rule: for the final valid-data spatial means,
  non-finite density or velocity values are invalid, NV_MSO_2 problem classes
  2-5 are invalid, and class 1 remains usable as a caution flag. Flag-unavailable
  or epoch-unmatched rows are not treated as clean. Density uses per-record
  arithmetic means, speed is formed per record before averaging, and the X-Z
  map combines all Y cells with record-count weighting.
- 2026-07-14 Xb/FOV classification: the daily high-energy fields named
  `inst_x_axis_mse_x/y/z` are the orbiter-body `+Xb` direction at the particle
  epoch. Use them to define the inclusive 45-degree cones around `+Z_MSE` and
  `-Z_MSE`; do not infer these classes from the older `mse_plus_z_in_minpa_fov`
  flag. Validate the axis independently with `NV_MSO_2.Xb_MSO` and
  `R_MSO2MSE`. Small independent attitude-match differences can occur (full
  run maximum 0.03725 degrees), so report angular and class disagreement; the
  full run had zero class disagreements.
- 2026-07-14 Y-slab projection: the requested classified X-Z maps use the
  inclusive interval `-0.2 <= Y_MSE <= 0.2 Rm`. With the `0.1 Rm` grid this is
  exactly Y indices 48-51 with centers `-0.15,-0.05,0.05,0.15 Rm`. Keep the
  sparse 3-D sufficient statistics unchanged and restrict only the Y summation
  used to form X-Z maps; record the selected indices and centers in every NPZ
  and JSON summary.
- 2026-07-14 Y-slab and flux update: the current requested classified X-Z
  product supersedes the earlier plotting slab and uses the inclusive interval
  `-0.5 <= Y_MSE <= 0.5 Rm`, exactly Y indices 45-54 with centers from `-0.45`
  through `+0.45 Rm`. The density-speed flux is computed after projection as
  `mean_density_cm3 * mean_speed_km_s * 1e5`, has units `cm^-2 s^-1`, and is
  plotted as `log10(flux)`. The sparse 3-D statistics and prior output products
  remain unchanged.
- 2026-07-14 Y-slab expansion: the current requested classified X-Z product
  supersedes the earlier plotting slab and uses the inclusive interval
  `-1 <= Y_MSE <= 1 Rm`, exactly Y indices 40-59 with centers from `-0.95`
  through `+0.95 Rm`. All filters, FOV classes, record-weighted means, flux
  definition, and sparse 3-D sufficient statistics remain unchanged; prior
  Y-slab output directories are retained.
- 2026-07-14 0.2 Rm rebin rule: derive coarser classified products by summing
  the existing 0.1 Rm counts and physical sums, then recompute means and flux;
  never average cell means. For the exact `-0.5 <= Y_MSE <= 0.5 Rm` X-Z slab,
  select source Y indices 45-54 before coarsening X and Z by two. Selecting
  global 0.2 Rm target-cell centers would instead include six Y cells spanning
  edges `-0.6` to `+0.6 Rm`, so it is not equivalent to the requested slab.
  Preserve full-statistic conservation and require the projected record counts
  to match the corresponding 0.1 Rm slab product.
- 2026-07-14 full-energy three-ion X-Z maps: the all-energy `NV_MSO_2` archive
  reuses the established all-Y, record-weighted MSE X-Z algorithm for H+, O+,
  and O2+. Preserve species-specific status and five-bit quality selection;
  retain bit 1 as caution and reject bits 2-5 under the default policy. Density
  remains the mean of per-record density, while speed is formed per record
  before averaging. A 2,000-file one-worker/four-worker benchmark took
  15.099/2.965 s with 43 bitwise-identical arrays; the full 155,145-file run
  took 249.356 s. Full X-Z reconstruction, 64 source-cell samples, units,
  coordinates, speed inequality, and 15 PNG plus 15 PDF products all passed.
- 2026-07-14 full-energy +Xb classification: classify canonical `NV_MSO_2`
  H+/O+/O2+ records with `Xb_MSE = R_MSO2MSE * Xb_MSO` into the inclusive,
  mutually exclusive 45-degree cones around `+Z_MSE` and `-Z_MSE`. The exact
  requested `-0.5 <= Y_MSE <= 0.5 Rm` slab is source Y indices 45-54 with
  centers `-0.45` through `+0.45 Rm`. Keep axes outside both cones in coverage
  counts but out of both means. A deterministic 100-day one/four-worker
  benchmark took 58.594/17.812 s with 220 bitwise-identical arrays; the full
  1,029-day run took 176.363 s. Cone closure, all X-Z sufficient statistics,
  flux, speed inequality, shared color limits, units, and 36 PNG plus 36 PDF
  figures passed independent validation.
- 2026-07-15 MINPA background-quality policy: for Mode 1 background interval
  construction, retain project quality bits 1, 2, and 5; reject only bit 3
  (UV candidate) and bit 4 (alternating-energy-channel error). The resulting
  reject mask is decimal 12 (`0x0C`). Store per-interval counts for all five
  bits in model metadata so later policy changes remain auditable. The current
  40 provisional intervals contain 752 bit-1 records, 580 bit-2 records, and
  no bit-5 records, so relaxing bit 5 changes eligibility but not this
  provisional model numerically. Native nonzero `Quality` records remain
  omitted without assigning undocumented meanings to native bits.
- 2026-07-15 MINPA background integration and paper reproduction: added a
  channel-resolved Mode-1 DPF background model, explicit pre-moment
  `max(raw-background, 0)` correction, and whole-record directional UV
  rejection. Historical results remain unchanged unless
  `subtract-and-reject-uv` is selected; provisional models require an explicit
  allow switch and produce version-suffixed files. The background interval
  veto remains project bits 3--4 only (mask `0x0C`), retaining bits 1, 2, and
  5. A 2021-12-31 integration run wrote separate O+/O2+ products with no
  corrected-density monotonicity violations. The full Python suite passed 55
  tests. Exact paper Figures 6--8 remain pending because local numerical MINPA
  coverage lacks 2022-01-02 through 2022-01-31; official quicklooks are useful
  for manual context only and cannot replace the numerical science array.
- 2026-07-15 paper-fidelity audit: downloaded and hashed the author-hosted
  final PDF, extracted exact figure channels and quantitative targets, and
  separated paper facts from DPF-space inference. Added the full 62-day
  Tianwen-1/MAVEN cylindrical-MSO Figure 2 and the Figure 3 MINPA/MOMAG/orbit
  context. Corrected Figure 10 from a single detector cell to a per-record sum
  over all 4x16x8 pitch/azimuth/mass cells at the paper's exact energies. The
  inferred Poisson lambda values are 3.71, 3.79, 4.28, and 4.33 versus the
  paper's 4.10, 4.27, 4.55, and 4.79; channel quantization reconstruction is
  above 99.98%. Added a reproducible data-availability manifest: TW1/MOMAG and
  MAVEN positions cover all 62 days, while numerical MINPA remains missing for
  2022-01-02 through 2022-01-31. The public MINPA API currently advertises
  quicklook PNGs, not a numerical science-array download endpoint.
- 2026-07-15 MINPA background scope correction: SWIA/Figures 6--8 and the H+
  Maxwellian/Figure 11 fit are downstream validation only and do not enter
  background interval selection, DPF/count-equivalent estimation, UV detection,
  or subtraction. They are excluded from the background-method acceptance gate.
  MOMAG remains context-only. Mission inputs are read from local storage; the
  configured `ori` copy has a Mode-1 gap after 2022-01-01 05:48:33 UTC within
  the paper interval, and no mission-data download is performed.
- 2026-07-15 MINPA Figure 5 and quiet-interval correction: the earlier combined
  raw/corrected DEF plot did not use the paper's nonzero-channel estimator; its
  primary sparse-cell model was zero in most channels, so the two panels were
  nearly identical. Replaced it with a standalone count-equivalent Figure 5
  using `max(DPF-paper_background_dpf, 0)`, the paper time ranges, and a shared
  `1--3000` log scale. Corrected the paper quiet reference from a boundary-
  crossing 23:06--23:15 interval to 2021-12-25 23:10--23:20. Added two-level
  coherent H+ ridge screening: reject an interval if any energy exceeds five
  count equivalents in at least 10% of records, or one count equivalent in at
  least 40%. Of 40 low-total-count candidates, 19 are rejected and 21 (608
  records) remain provisional. The quiet-reference summed residual is 1.35%;
  the strong-solar-wind peak remains at 762.23 eV and retains 98.94% of its
  maximum. Sixty Python tests and the moment/background verifier pass.
- 2026-07-16 MINPA manual background-window review: using the original
  one-based 40-candidate montage numbering, reject candidates 5, 6, 9, 16,
  17, 18, 24, 25, 32, 35, and 37 because a weak real spectral ridge remains.
  Manually approve candidates 10, 13, 23, 26, 27, 29, 30, 33, 34, and 38 as
  having no visible real ridge between the black boundary lines. The approved
  subset contains 10 intervals and 292 kept records, below the required 20
  intervals and 300 records, so it must not produce or promote a valid model.
  The decisions are recorded in `config/minpa_background_manual_review.json`.
- 2026-07-16 MINPA pure-background classifier: the manually contaminated
  windows have both a prominent time-mean H+ DEF peak and concentration of H+
  count equivalents in three adjacent energy channels. A transparent AND rule
  (`DEF peak/median > 2.75` and adjacent-three-energy fraction `> 0.133`)
  matches all 21 manual labels; strict nested leave-one-out threshold fitting
  reaches 85.7%, with candidates 18, 37, and 38 defining the narrow boundary.
  Scores within 10% of the decision boundary remain manual-review cases.
  Added feature CSV/JSON, a three-panel diagnostic, and synthetic tests.
- 2026-07-16 MINPA `day_spe` mode correction: `num1` through `num9` are time
  segments, not fixed mode identifiers. Some files alternate `mod=4` 64-energy
  segments and `mod=1` 40-energy segments. Candidate discovery now traverses
  every H/O/O2 segment and requires all three `mod` values to equal 1 plus an
  exact Mode-1 energy-grid match. The prior `num1` assumption both admitted
  non-Mode-1 data and missed later Mode-1 segments. With the corrected parser,
  the expanded 50th-percentile pool contains 94 Mode-1 windows; the classifier
  selects 12 high-confidence and 8 borderline pure-background candidates (579
  records), including the 10 manual references and 10 new intervals awaiting
  review. Predictions remain non-production until manually approved.
- 2026-07-16 MINPA expanded manual review and borderline-boundary refinement:
  manually approve expanded candidates 3, 6, 15, 19, 27, and 40; reject 25;
  use manually adjusted intervals 1=08:32--08:38, 7=07:46--07:52, and
  16=15:05--15:15 UTC on their respective dates. A deterministic secondary
  search now applies only to initially borderline candidates. It moves either
  boundary on a 60-s grid, never shortens below 50% of the original duration,
  requires at least 18 usable DPF records, and selects the longest subwindow
  that reaches high confidence and passes the gross-coherence and quality-bit
  3--4 vetoes; equal durations are ranked by lower decision score. All eight
  borderline candidates reached high confidence after trimming. Manual bounds
  override automatic results for expanded 1, 7, and 16; automatic trims are
  used for original candidates 10, 23, 27, 34, and 38. Exact `ori`/quality
  validation accepts all 19 reviewed intervals and 522 records. The record
  threshold passes, but the production model remains invalid because one more
  approved interval is required.
- 2026-07-16 MINPA full-mission Mode 1 temporal-noise workflow: added strict
  traversal of every `day_spe` segment, monthly-local quiet candidate discovery,
  resumable atomic month checkpoints, count-equivalent/quantization/density and
  engineering diagnostics, day-block bootstrap and rank-permutation inference,
  BIC stable-period analysis split at gaps longer than 45 days, FDR control,
  and PNG/PDF/CSV/JSON/NPZ outputs. The local inventory independently recovers
  1051 day files, 7055 strict Mode 1 segments, 4,680,015 records, and 43 months
  with zero parse errors. A 100-file selected-channel and quality-mask benchmark
  is bitwise identical between one and four processes (40.49 s versus 10.25 s,
  3.95x). Complete early/middle/late representative-month outputs are also
  identical across execution modes after excluding runtime timestamps and root
  paths. Results remain explicitly exploratory because only the 2021-12 windows
  have human review; no production background model or moment file is written.
- 2026-07-16 MINPA quarterly temporal-noise extension and lambda v2 correction:
  the interval count-equivalent rate now takes an arithmetic mean across the
  already energy-pooled directional channels. The previous angular median
  collapsed sparse ion rates to exact zero whenever more than half the angular
  cells were empty. All 43 month checkpoints were migrated losslessly from the
  stored per-interval `lambda_pam` arrays without rereading `ori`; candidates,
  bounds, background DPF, density contributions, and source hashes are
  unchanged. Calendar-quarter analysis pools intervals and then uses independent
  day medians. Three of 16 quarters pass the strict thresholds: 2021Q4, 2022Q2,
  and 2024Q1. They belong to separate coverage blocks, so no continuous-period
  change point is fitted. Cross-gap pairwise contrasts find global lambda lower
  by 65.3%, 99.5%, and 98.6% respectively (`q=2.80e-4` for each); these are
  period-distribution differences, not evidence for a continuous ageing trend.
  The v2 repeat of the 100-file benchmark remained identical for values, quality
  masks, and metadata: 30.63 s single-process versus 10.73 s with four workers
  (2.86x on the current machine).
- 2026-07-16 MINPA quarterly interval visual audit: selected five finite
  global-lambda strata (10/30/50/70/90 percent ranks, distinct days preferred)
  from each of 2021Q4, 2022Q2, and 2024Q1. Three-species spectrograms use common
  per-species color limits, eight-minute context on each side, and black retained
  bounds; a companion figure shows zero-inclusive time-mean DEF versus energy.
  The montage exposes residual structured spectra in multiple automatically
  accepted intervals, especially O+/O2+ structures that the H+-only classifier
  cannot reject, plus visible H+ structure in some high-lambda 2021Q4 windows.
  Therefore a quarter passing the numerical sample threshold is not equivalent
  to a human-validated pure-background quarter. Multi-species visual review or a
  multi-species coherence veto is required before promoting quarterly temporal
  differences to an instrument-background evolution claim.
- 2026-07-16 MINPA review-only DEF noise reference: reconstructed the
  solid-angle-weighted 1D background DEF from the 19 manually referenced
  2021-12 intervals using `E * dpf_quantum * lambda(pitch,azimuth,mass)`, then
  took the energy median within each interval and the median across intervals.
  H+, O+, and O2+ are 4.75e3, 3.78e3, and 4.16e3 `1/(s cm^2 sr)`. Use 5e3 as
  a rounded typical 5--10 minute time/angle-averaged reference and 1e4 as the
  robust upper reference across interval medians. Pooling all records first
  gives 1.30e4, 6.47e3, and 7.95e3, respectively, and is retained only as a
  sensitivity bound because high-lambda structured windows receive more weight.
  This is not a raw-cell subtraction constant:
  one nonzero four-dimensional count quantum is about 1.5e6 DEF, and its
  solid-angle-projected one-event contribution has a median near 2.1e4 DEF.
  The reference remains review-only because 19 intervals do not meet the
  configured 20-interval production threshold and visual sampling found
  residual structured spectra.
- 2026-07-16 MINPA all-accepted-interval DEF spectra: read all 244 accepted
  intervals (6,719 unique observing times) from the 43 completed monthly
  checkpoints and revalidated Mode 1, 40 energies, the standard energy table,
  spectrum shapes, and cross-species time alignment. Time means include finite
  zeros. One published day_spe interval, `202112-0016`, has 40 rows but only 37
  unique timestamps; its four spectra sharing one timestamp are averaged first,
  making the unique-time count agree with the 37 accepted ori records. The
  full-mission and 13-quarter figures overlay every interval and a black
  per-energy median, with 10--90% and 25--75% envelopes. Across the 40-bin
  median curves the representative levels are 8.19e3 (H+), 3.14e3 (O+), and
  2.46e3 (O2+) `1/(s cm^2 sr)`. These are exploratory accepted-candidate
  summaries, not production subtraction constants, because 225 intervals have
  no cross-year manual validation and some retain real multi-species structure.

## 2026-06-12

- Created the initial MAVEN STATIC-D1 high-energy O+/O2+ route.
- Used `Mars_Oplus_escape_experiment` as the mature reference for STATIC D1
  geometry, scalar-first `quat_mso`, density expression, and `Spss` position
  handling.
- MAVEN position source is `D:\Data\MAVEN\result\mag\ss1s\BssYYYYMMDD.mat`;
  `Bss[:,0]` is the time axis and `Spss[:,1:4]` is MSO position in km.
- Added acceleration by precomputing geometry masks for every `swp_ind` and
  species, and by using date-level parallelism in the batch script.
- Output was changed to one daily MAT file per species so each file has one row
  per STATIC timestamp and no duplicated time points.
- Added a MAVEN STATIC +Z_MSE FOV flag. The STATIC FOV is treated as a fixed
  instrument-frame theta-edge range; each epoch projects MSE +Z into STATIC
  coordinates and writes `1`, `0`, or `NaN` for inside, outside, or invalid.
- Tianwen-1 MINPA is intentionally not implemented yet; it will be added after
  MAVEN product review.
## 2026-07-16 — self-contained MINPA raw-to-product route

- Internalized calibrated Mode 1--12 energy/mass/angular layouts; removed the
  runtime dependency on `D:\codex\处理MAVEN数据\scripts\tw1_minpa_moments.py`.
- Added read-only `ori`/`.2B` parsing, H+/O+/O2+ spectra, density, scalar
  temperature, MINPA/body/MSO velocity, quality evaluation, and a
  density-conserving diagnostic VDF projection.
- Preserved native `Quality`, START/STOP totals, five HV monitors, and solar
  angles without assigning unpublished bit/physical meanings.
- Mode 12 is split into two 1536-value product-time rows.  Its azimuth-integrated
  single angular cell supports density but not independent vector velocity or
  a 2-D VDF; the code returns an explicit unavailable status.
- Background-window policy remains: reject project bits 3--4, retain 1, 2, and
  5, and require native `Quality==0` for estimation.  Background correction is
  explicit, Mode-1-only, nonnegative, and disabled by default.
- Added a real 2021-12-31 example with spectrum, moments, VDF projection,
  coordinate figure, NPZ/CSV output, and JSON provenance, plus focused unit
  tests for layouts, splitting, flags, density, rotation, and VDF closure.
- A nonzero 2021-12-31 Mode-1 H+ record was compared with the former external
  reference implementation using identity attitude: density, all three
  velocity components, and scalar temperature agreed to floating-point roundoff
  (maximum observed relative difference `3.2e-16`).

## 2026-07-17 — MINPA coordinate-chain and VDF correction

- Replaced the ambiguous collapsed coordinate description with the mature
  project chain: construct the `TW1_MINPA_deal_v5.m` intermediate direction,
  apply `legacy_to_probe_direction([x,y,z])=[-z,-x,+y]` twice, then apply the
  nearest MOMAG `Rz(-Yaw) Ry(-Pitch) Rx(-Roll)` attitude matrix.
- Real-data regression on the 2021-12-31 Mode-1 file matched all 1,225 local
  `NV` rows with maximum density and velocity differences `4.33e-8 cm^-3` and
  `1.79e-6 km/s`. Transforming the stored intermediate velocity itself isolates
  the coordinate chain and matches `NV_MSO` at about `1e-13 km/s`.
- Replaced the former channel-center-only VDF quicklook with the mature finite
  cell workflow: default `17 x 17` angular and 9 energy sub-samples, separate
  geometric-coverage mask, record-wise MSO rotation, per-record mean density
  weighting, common-log-scale XY/XZ/YZ projections, contours, and a bulk marker
  derived from exactly the plotted cells.
- MINPA VDF color remains a reduced-density proxy, not an absolute calibrated
  PSD. White denotes unsampled velocity space; gray denotes sampled support
  without positive signal. Mode 12 remains ineligible for a 2-D VDF.
- The public quickstart time-series example was moved to 2021-12-03
  00:00--01:00 UTC using local Mode-1 source file 00639. Its finite-cell VDF
  uses 00:30--00:31 UTC within the same hour; figure provenance records both
  intervals and the unchanged quality/energy selection.

## 2026-07-17 — MINPA payload/body terminology and energy-width correction

- User-confirmed coordinate meaning: `[-V2,+V3,-V1]` maps MINPA payload
  components into the Tianwen-1 spacecraft body frame. MOMAG attitude is a
  separate body-to-MSO rotation `Rz(-Yaw) Ry(-Pitch) Rx(-Roll)`.
- Main processing, VDF rotation, FOV diagnostics, figure labels, provenance,
  and method notes now use this explicit two-stage terminology. Historical
  `probe` helpers remain compatibility aliases only and are not used by the
  main processing chain.
- Replaced fixed `dE/E=0.15` in the self-contained moment, VDF, high-energy,
  and Mode-1 background-density paths with per-channel widths from logarithmic
  edges inferred from adjacent calibrated energy centers.
- Read-only regression for 2021-12-03 00:00--01:00 against local `NV_MSO_2`
  found a median current/reference H+ density ratio of `0.99999983` over 210
  finite records. Over 202 comparable MSO velocity rows, the median vector
  difference was `1.03e-5 km/s`; rotating the stored payload velocity itself
  reproduced `NV_MSO_2` at a median `6.36e-14 km/s`.
