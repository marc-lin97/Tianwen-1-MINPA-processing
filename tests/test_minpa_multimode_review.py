import hashlib
from pathlib import Path

import numpy as np
import pytest

from highE.minpa_io import _quality_scalar
from highE.minpa_multimode_review import (
    MinuteWindow,
    finalize_species_review,
    select_and_merge_low_tail,
)


def test_native_quality_hex_text_is_decoded() -> None:
    values = [np.asarray([[ord(char)] for char in "0x00"], dtype=np.uint16)]
    assert _quality_scalar(values, 0) == 0.0
    values = [np.asarray([[ord(char)] for char in "0x08"], dtype=np.uint16)]
    assert _quality_scalar(values, 0) == 8.0


def _minute(mode: int, species: str, start: float, score: float) -> MinuteWindow:
    return MinuteWindow(
        mode=mode,
        species=species,
        month="202201",
        start_unix_s=start,
        stop_unix_s=start + 60.0,
        score=score,
        record_count=30 if mode == 12 else 5,
        source_day_spe=("day.mat",),
        source_segments=("num1",),
    )


def test_low_tail_is_species_independent_includes_ties_and_merges_without_cap() -> None:
    rows = []
    for index in range(20):
        rows.append(_minute(4, "H+", index * 60.0, 0.0 if index < 3 else float(index)))
        rows.append(_minute(4, "O+", index * 60.0, float(20 - index)))
    candidates, thresholds = select_and_merge_low_tail(rows, quantile=0.10)
    h = [item for item in candidates if item["species"] == "H+"]
    o = [item for item in candidates if item["species"] == "O+"]
    assert len(h) == 1
    assert h[0]["duration_s"] == 180.0
    assert len(o) == 1
    assert o[0]["start_unix_s"] != h[0]["start_unix_s"]
    assert all(item["selection_quantile"] == 0.10 for item in candidates)
    assert len(thresholds) == 2


def test_file_presence_finalization_is_independent_by_species(tmp_path: Path) -> None:
    candidates = [
        {"candidate_id": "h", "species": "H+", "plot_filename": "h.png"},
        {"candidate_id": "o", "species": "O+", "plot_filename": "o.png"},
    ]
    (tmp_path / "h.png").write_bytes(b"png")
    result = finalize_species_review(candidates, tmp_path)
    assert [item["candidate_id"] for item in result["approved_intervals"]] == ["h"]
    assert [item["candidate_id"] for item in result["rejected_intervals"]] == ["o"]
    (tmp_path / "unexpected.png").write_bytes(b"png")
    with pytest.raises(ValueError, match="Unexpected"):
        finalize_species_review(candidates, tmp_path)


def test_file_presence_finalization_checks_hash_and_preserves_version(tmp_path: Path) -> None:
    content = b"original png"
    candidate = {
        "candidate_id": "h",
        "species": "H+",
        "plot_filename": "h.png",
        "plot_sha256": hashlib.sha256(content).hexdigest(),
    }
    (tmp_path / "h.png").write_bytes(content)
    result = finalize_species_review([candidate], tmp_path, review_version="secondary-v2")
    assert result["review_version"] == "secondary-v2"
    (tmp_path / "h.png").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Modified"):
        finalize_species_review([candidate], tmp_path, review_version="secondary-v2")
