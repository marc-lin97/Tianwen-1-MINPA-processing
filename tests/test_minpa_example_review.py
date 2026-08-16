from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from highE.minpa_example_review import finalize_signal_example_review


SPECIES = ("H+", "O+", "O2+")


def _index(tmp_path: Path) -> dict:
    groups = []
    for mode in (1, 4, 12):
        for species in SPECIES:
            candidates = []
            for number in range(1, 4):
                filename = f"mode{mode:02d}_{species}_{number}.png"
                content = f"{mode}-{species}-{number}".encode()
                candidates.append({
                    "mode": mode,
                    "species": species,
                    "candidate_id": f"signal-{mode}-{species}-{number}",
                    "start_utc": "2024-01-01T00:00:00Z",
                    "stop_utc": "2024-01-01T00:05:00Z",
                    "duration_s": 300.0,
                    "plot_filename": filename,
                    "plot_sha256": hashlib.sha256(content).hexdigest(),
                    "context_source_ori": [{"path": r"D:\data\source.mat", "sha256": "a" * 64}],
                    "native_validation": {
                        "strength_tier": "strong_20x",
                        "maximum_median_channel_signal_to_background": 25.0,
                        "signal_peak_retention_fraction": 0.96,
                        "ratio_peak_shift_channels": 0,
                    },
                    "_content": content,
                })
            groups.append({
                "mode": mode,
                "species": species,
                "noise_reference": {
                    "start_utc": "2023-01-01T00:00:00Z",
                    "stop_utc": "2023-01-01T00:05:00Z",
                    "figure": str(tmp_path / f"noise-{mode}-{species}.png"),
                    "figure_sha256": "b" * 64,
                },
                "signal_candidates": candidates,
            })
    return {
        "status": "pending_manual_signal_review",
        "pending_png_count": 27,
        "bundle": {"sha256": "c" * 64},
        "units": "DPF",
        "smoothing_or_interpolation": "none",
        "groups": groups,
    }


def _retain_first_per_group(index: dict, pending: Path) -> None:
    pending.mkdir()
    for group in index["groups"]:
        row = group["signal_candidates"][0]
        (pending / row["plot_filename"]).write_bytes(row["_content"])


def test_finalize_signal_review_accepts_one_per_group(tmp_path: Path) -> None:
    index = _index(tmp_path)
    pending = tmp_path / "pending"
    _retain_first_per_group(index, pending)
    final = finalize_signal_example_review(index, pending)
    assert final["counts"] == {
        "mode_species_groups": 9,
        "approved_noise_examples": 9,
        "approved_signal_examples": 9,
        "rejected_signal_candidates": 18,
    }
    assert final["signal_examples"][0]["source_ori"][0]["filename"] == "source.mat"


def test_finalize_signal_review_rejects_modified_png(tmp_path: Path) -> None:
    index = _index(tmp_path)
    pending = tmp_path / "pending"
    _retain_first_per_group(index, pending)
    first = next(pending.glob("*.png"))
    first.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        finalize_signal_example_review(index, pending)


def test_finalize_signal_review_requires_one_per_group(tmp_path: Path) -> None:
    index = _index(tmp_path)
    pending = tmp_path / "pending"
    _retain_first_per_group(index, pending)
    first = index["groups"][0]["signal_candidates"][1]
    (pending / first["plot_filename"]).write_bytes(first["_content"])
    with pytest.raises(ValueError, match="Exactly one retained signal"):
        finalize_signal_example_review(index, pending)


def test_finalize_signal_review_rejects_extra_png(tmp_path: Path) -> None:
    index = _index(tmp_path)
    pending = tmp_path / "pending"
    _retain_first_per_group(index, pending)
    (pending / "renamed.png").write_bytes(b"extra")
    with pytest.raises(ValueError, match="Unexpected or renamed"):
        finalize_signal_example_review(index, pending)
