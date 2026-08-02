"""Resolution agent: combines the upstream agent outputs into one of three
outcomes: ready to submit, needs info, or human review.

- needs_info: something fixable is wrong (incomplete/malformed claim, an
  eligibility mismatch, or a missing prior authorization). These are
  concrete, actionable gaps rather than a judgment call.
- human_review: the claim is well-formed and the member is eligible, but
  the denial-risk model isn't confident it's clean. Routing risk instead of
  guessing is the human-in-the-loop safety net.
- ready_to_submit: passed every check and scored low risk.
"""

from dataclasses import dataclass, field
from typing import Optional

from agents.denial_risk_agent import DenialRiskResult
from agents.eligibility_agent import EligibilityResult
from agents.validation_agent import ValidationResult

READY_TO_SUBMIT = "ready_to_submit"
NEEDS_INFO = "needs_info"
HUMAN_REVIEW = "human_review"

REVIEW_RISK_BANDS = {"medium", "high"}


@dataclass
class ResolutionResult:
    decision: str  # "ready_to_submit" | "needs_info" | "human_review"
    reasons: list[str] = field(default_factory=list)


def resolve(
    validation: ValidationResult,
    eligibility: Optional[EligibilityResult],
    denial_risk: Optional[DenialRiskResult],
) -> ResolutionResult:
    """eligibility and denial_risk may be None when validation failed and the
    pipeline skipped straight to resolution without running those agents."""
    if not validation.passed:
        return ResolutionResult(decision=NEEDS_INFO, reasons=list(validation.errors))

    if not eligibility.eligible:
        return ResolutionResult(decision=NEEDS_INFO, reasons=list(eligibility.reasons))

    if denial_risk.missing_prior_auth:
        return ResolutionResult(
            decision=NEEDS_INFO,
            reasons=["prior authorization is required for this procedure but is not on file"],
        )

    if denial_risk.risk_band in REVIEW_RISK_BANDS:
        return ResolutionResult(
            decision=HUMAN_REVIEW,
            reasons=[f"denial-risk score {denial_risk.risk_score:.2f} ({denial_risk.risk_band} risk)"],
        )

    return ResolutionResult(decision=READY_TO_SUBMIT, reasons=[])
