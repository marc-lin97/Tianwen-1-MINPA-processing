# Reviewed MINPA v2.2.0 examples

This directory contains the nine examples retained in the final human review:
one observational-signal interval for every combination of Mode 1, 4, and 12
with H+, O+, and O2+.

- `signal/` contains the nine retained PNG files. Every file includes the raw
  and v2.2.0-corrected spectrogram panels, an interval-mean spectrum, and a
  native-channel signal/background diagnostic.
- `manifest.json` records the final example UTC intervals, figure hashes,
  signal tiers, signal/background ratios, and peak retention. Rejected-candidate
  rows and local review intermediates are intentionally omitted.

Each figure uses native DPF in `1/(s cm^2 sr eV)`. Raw and corrected
spectrograms share a color scale; no smoothing or interpolation is used.
Mode 12 is an azimuth-integrated product. The `moderate_3x` Mode-12 H+ and O+
examples are intentionally weaker than the other retained examples and should
not be interpreted as a plasma-region classification.

Verify all model and nine figure hashes without mission data:

```powershell
python -m pip install -e .[test]
python scripts/verify_minpa_v2_2_0_release.py
```

To regenerate figures from mission data, place the read-only `ori/*.mat`,
`day_spe/*.mat`, and project quality products locally and run:

```powershell
python scripts/plot_minpa_denoise_examples_3modes_3species.py `
  --ori-root D:\Data\TW-1\result\MINPA\ori `
  --quality-root outputs\tw1_minpa_quality_flags_all_species `
  --bundle release\minpa_unified_static_channel_denoise_v2.2.0\bundle.json

python scripts/prepare_minpa_typical_signal_review_v2_2_0.py `
  --day-spe-root D:\Data\TW-1\result\MINPA\day_spe `
  --ori-root D:\Data\TW-1\result\MINPA\ori
```

Exact deterministic noise-example reselection also requires the completed local
Mode-1 and Mode-4/12 review inventories accepted by the plotting script. The
second command intentionally rebuilds the 27-candidate local review workspace;
the committed manifest contains only the final 9 retained signal examples and
the aggregate rejected count. Raw Tianwen-1 data and local review intermediates
are not redistributed by this repository.
