import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.eligibility_agent import check_eligibility  # noqa: E402

MEMBERS = pd.DataFrame(
    [
        {
            "member_id": "M00001",
            "plan_id": "PLAN001",
            "eligibility_status": "active",
            "coverage_start": "2025-01-01",
            "coverage_end": "2026-12-31",
        },
        {
            "member_id": "M00002",
            "plan_id": "PLAN002",
            "eligibility_status": "lapsed",
            "coverage_start": "2024-01-01",
            "coverage_end": "2024-12-31",
        },
    ]
)

BASE_CLAIM = {
    "patient_id": "M00001",
    "member_plan_id": "PLAN001",
    "submission_date": "2026-01-15",
}


def test_active_member_in_window_is_eligible():
    result = check_eligibility(BASE_CLAIM, MEMBERS)
    assert result.eligible
    assert result.reasons == []


def test_member_not_found():
    claim = {**BASE_CLAIM, "patient_id": "M99999"}
    result = check_eligibility(claim, MEMBERS)
    assert not result.eligible
    assert "member not found on file" in result.reasons


def test_lapsed_member_ineligible():
    claim = {"patient_id": "M00002", "member_plan_id": "PLAN002", "submission_date": "2024-06-01"}
    result = check_eligibility(claim, MEMBERS)
    assert not result.eligible
    assert any("lapsed" in r for r in result.reasons)


def test_plan_mismatch_flagged():
    claim = {**BASE_CLAIM, "member_plan_id": "PLAN003"}
    result = check_eligibility(claim, MEMBERS)
    assert not result.eligible
    assert any("plan mismatch" in r for r in result.reasons)


def test_submission_outside_coverage_window_flagged():
    claim = {**BASE_CLAIM, "submission_date": "2027-01-01"}
    result = check_eligibility(claim, MEMBERS)
    assert not result.eligible
    assert any("outside" in r for r in result.reasons)
