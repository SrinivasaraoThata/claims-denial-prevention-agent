"""Validation agent: checks claim completeness and code formats.

Rule-based, no LLM call. Runs first in the pipeline since there's no point
checking eligibility or policy for a claim that's malformed.
"""

import re
from dataclasses import dataclass, field
from datetime import date

REQUIRED_FIELDS = [
    "claim_id",
    "patient_id",
    "procedure_code",
    "diagnosis_code",
    "provider_id",
    "submission_date",
    "prior_auth_flag",
    "member_plan_id",
    "billed_amount",
]

# CPT codes are 5 digits; HCPCS Level II codes are 1 letter + 4 digits (e.g. J1745).
PROCEDURE_CODE_RE = re.compile(r"^(\d{5}|[A-Z]\d{4})$")

# ICD-10-CM: a letter, two digits, then an optional decimal with up to four
# alphanumeric characters (e.g. I10, E11.9, M17.11, C50.911).
DIAGNOSIS_CODE_RE = re.compile(r"^[A-Z]\d{2}(\.\d{1,4})?$")


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


def validate_claim(claim: dict) -> ValidationResult:
    errors = []

    for field_name in REQUIRED_FIELDS:
        value = claim.get(field_name)
        if value is None or value == "":
            errors.append(f"missing required field: {field_name}")

    if errors:
        return ValidationResult(passed=False, errors=errors)

    procedure_code = str(claim["procedure_code"])
    if not PROCEDURE_CODE_RE.match(procedure_code):
        errors.append(f"invalid procedure_code format: {procedure_code}")

    diagnosis_code = str(claim["diagnosis_code"])
    if not DIAGNOSIS_CODE_RE.match(diagnosis_code):
        errors.append(f"invalid diagnosis_code format: {diagnosis_code}")

    try:
        submission_date = date.fromisoformat(str(claim["submission_date"]))
        if submission_date > date.today():
            errors.append("submission_date is in the future")
    except ValueError:
        errors.append(f"invalid submission_date format: {claim['submission_date']}")

    try:
        billed_amount = float(claim["billed_amount"])
        if billed_amount <= 0:
            errors.append("billed_amount must be greater than zero")
    except (TypeError, ValueError):
        errors.append(f"invalid billed_amount: {claim['billed_amount']}")

    return ValidationResult(passed=len(errors) == 0, errors=errors)
