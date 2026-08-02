"""Pydantic request/response models for the claims API."""

from typing import Optional

from pydantic import BaseModel


class ClaimIn(BaseModel):
    claim_id: str
    patient_id: str
    procedure_code: str
    diagnosis_code: str
    provider_id: str
    submission_date: str
    prior_auth_flag: bool
    member_plan_id: str
    billed_amount: float


class ValidationOut(BaseModel):
    passed: bool
    errors: list[str]


class EligibilityOut(BaseModel):
    eligible: bool
    reasons: list[str]


class PolicyOut(BaseModel):
    query: str
    answer: str
    llm_used: bool
    sources: list[str]


class DenialRiskOut(BaseModel):
    risk_score: float
    risk_band: str
    missing_prior_auth: bool


class ResolutionOut(BaseModel):
    decision: str
    reasons: list[str]


class ClaimDecisionResponse(BaseModel):
    claim_id: str
    decision: str
    reasons: list[str]
    validation: ValidationOut
    eligibility: Optional[EligibilityOut] = None
    policy: Optional[PolicyOut] = None
    denial_risk: Optional[DenialRiskOut] = None
