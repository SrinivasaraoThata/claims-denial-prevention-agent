import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.validation_agent import validate_claim  # noqa: E402

GOOD_CLAIM = {
    "claim_id": "C000001",
    "patient_id": "M00001",
    "procedure_code": "99213",
    "diagnosis_code": "E11.9",
    "provider_id": "P0001",
    "submission_date": "2026-01-15",
    "prior_auth_flag": False,
    "member_plan_id": "PLAN001",
    "billed_amount": 110.0,
}


def test_valid_claim_passes():
    result = validate_claim(GOOD_CLAIM)
    assert result.passed
    assert result.errors == []


def test_hcpcs_procedure_code_passes():
    claim = {**GOOD_CLAIM, "procedure_code": "J1745"}
    result = validate_claim(claim)
    assert result.passed


def test_missing_field_fails():
    claim = dict(GOOD_CLAIM)
    del claim["billed_amount"]
    result = validate_claim(claim)
    assert not result.passed
    assert any("billed_amount" in e for e in result.errors)


def test_invalid_procedure_code_fails():
    claim = {**GOOD_CLAIM, "procedure_code": "ABC"}
    result = validate_claim(claim)
    assert not result.passed
    assert any("procedure_code" in e for e in result.errors)


def test_invalid_diagnosis_code_fails():
    claim = {**GOOD_CLAIM, "diagnosis_code": "diabetes"}
    result = validate_claim(claim)
    assert not result.passed
    assert any("diagnosis_code" in e for e in result.errors)


def test_negative_billed_amount_fails():
    claim = {**GOOD_CLAIM, "billed_amount": -5}
    result = validate_claim(claim)
    assert not result.passed
    assert any("billed_amount" in e for e in result.errors)


def test_future_submission_date_fails():
    claim = {**GOOD_CLAIM, "submission_date": "2099-01-01"}
    result = validate_claim(claim)
    assert not result.passed
    assert any("future" in e for e in result.errors)


def test_malformed_date_fails():
    claim = {**GOOD_CLAIM, "submission_date": "not-a-date"}
    result = validate_claim(claim)
    assert not result.passed
    assert any("submission_date" in e for e in result.errors)
