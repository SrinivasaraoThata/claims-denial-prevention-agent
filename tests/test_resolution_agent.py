import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.denial_risk_agent import DenialRiskResult  # noqa: E402
from agents.eligibility_agent import EligibilityResult  # noqa: E402
from agents.resolution_agent import HUMAN_REVIEW, NEEDS_INFO, READY_TO_SUBMIT, resolve  # noqa: E402
from agents.validation_agent import ValidationResult  # noqa: E402

VALID = ValidationResult(passed=True)
ELIGIBLE = EligibilityResult(eligible=True)
LOW_RISK = DenialRiskResult(risk_score=0.1, risk_band="low", missing_prior_auth=False)
MEDIUM_RISK = DenialRiskResult(risk_score=0.45, risk_band="medium", missing_prior_auth=False)
HIGH_RISK = DenialRiskResult(risk_score=0.8, risk_band="high", missing_prior_auth=False)


def test_clean_low_risk_claim_is_ready_to_submit():
    result = resolve(VALID, ELIGIBLE, LOW_RISK)
    assert result.decision == READY_TO_SUBMIT
    assert result.reasons == []


def test_validation_failure_takes_priority():
    invalid = ValidationResult(passed=False, errors=["missing required field: billed_amount"])
    result = resolve(invalid, ELIGIBLE, HIGH_RISK)
    assert result.decision == NEEDS_INFO
    assert result.reasons == invalid.errors


def test_ineligible_member_needs_info():
    ineligible = EligibilityResult(eligible=False, reasons=["member coverage status is lapsed"])
    result = resolve(VALID, ineligible, LOW_RISK)
    assert result.decision == NEEDS_INFO
    assert result.reasons == ineligible.reasons


def test_missing_prior_auth_needs_info_even_at_low_risk():
    risk = DenialRiskResult(risk_score=0.1, risk_band="low", missing_prior_auth=True)
    result = resolve(VALID, ELIGIBLE, risk)
    assert result.decision == NEEDS_INFO
    assert "prior authorization" in result.reasons[0]


def test_medium_risk_routes_to_human_review():
    result = resolve(VALID, ELIGIBLE, MEDIUM_RISK)
    assert result.decision == HUMAN_REVIEW


def test_high_risk_routes_to_human_review():
    result = resolve(VALID, ELIGIBLE, HIGH_RISK)
    assert result.decision == HUMAN_REVIEW
