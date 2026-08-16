from scripts.build_minpa_unified_static_v2_2_0 import consolidate_approved_rows


def test_consolidate_approved_rows_deduplicates_and_audits_reject_alias() -> None:
    approved = [
        ("a.json", {"start_utc": "2024-01-01T00:00:00Z", "stop_utc": "2024-01-01T00:02:00Z", "label": "a"}),
        ("b.json", {"start_utc": "2024-01-01T00:00:00Z", "stop_utc": "2024-01-01T00:02:00Z", "candidate_id": "b"}),
        ("b.json", {"start_utc": "2024-01-01T00:03:00Z", "stop_utc": "2024-01-01T00:08:00Z", "candidate_id": "c"}),
    ]
    rejected = [
        ("legacy.json", {"start_utc": "2024-01-01T00:00:00Z", "stop_utc": "2024-01-01T00:02:00Z"})
    ]
    rows, audit = consolidate_approved_rows(approved, rejected)
    assert len(rows) == 2
    assert rows[0]["label"] == "all-approved-0001"
    assert len(rows[0]["approval_aliases"]) == 2
    assert audit["imported_approved_count"] == 3
    assert audit["exact_time_duplicate_count"] == 1
    assert audit["exact_time_approved_rejected_alias_collision_count"] == 1


def test_consolidate_approved_rows_reports_nonidentical_overlap() -> None:
    approved = [
        ("a", {"start_utc": "2024-01-01T00:00:00Z", "stop_utc": "2024-01-01T00:05:00Z"}),
        ("b", {"start_utc": "2024-01-01T00:04:00Z", "stop_utc": "2024-01-01T00:08:00Z"}),
    ]
    _, audit = consolidate_approved_rows(approved, [])
    assert audit["overlapping_nonidentical_approved_pair_count"] == 1
    assert audit["overlapping_nonidentical_approved_pairs"][0]["overlap_s"] == 60.0
