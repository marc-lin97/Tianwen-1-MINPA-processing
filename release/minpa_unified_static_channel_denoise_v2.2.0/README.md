# MINPA unified static channel denoise v2.2.0

This is the frozen, time-invariant production bundle for Tianwen-1/MINPA
Modes 1, 4, and 12. Its `bundle.json` SHA-256 is:

```text
0c21995c2b3baf7ba82e0893aa0aa5c81c17f268787b28e3d75181dcac468c71
```

The correction is performed in native instrument-channel space:

```text
b_jc = mean(x_tc | x_tc is finite and x_tc > 0)
B_c  = mean_j(b_jc)
corrected = max(raw - B_c, 0)
```

Approved intervals receive equal weight. NaNs remain NaN; unsupported and
unreviewed channels remain raw. There is no smoothing, interpolation,
cross-mass substitution, quarterly scaling, or correction for Mode 7.

Load and verify the bundle from the repository root:

```powershell
python -m pip install -e .[test]
python scripts/verify_minpa_v2_2_0_release.py
```

The model files are intentionally committed so applying the frozen correction
does not require rebuilding from local review workspaces. The only published
calibration-interval metadata are the Mode/species/start/stop UTC columns in
`calibration_intervals_approved.csv`; per-interval review products and
intermediate statistics are intentionally omitted. See
[`../../docs/release_minpa_unified_static_denoise_v2.2.0.md`](../../docs/release_minpa_unified_static_denoise_v2.2.0.md)
for channel scope, approval counts, validation, limitations, and provenance.
