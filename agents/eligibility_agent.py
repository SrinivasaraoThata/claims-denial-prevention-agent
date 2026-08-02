"""Eligibility agent: checks member plan status and coverage dates.

Rule-based lookup against the member roster, no LLM call. Assumes the claim
has already passed validation (submission_date and member_plan_id are
well-formed).
"""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class EligibilityResult:
    eligible: bool
    reasons: list[str] = field(default_factory=list)


def check_eligibility(claim: dict, members_df: pd.DataFrame) -> EligibilityResult:
    reasons = []

    member_rows = members_df[members_df["member_id"] == claim["patient_id"]]
    if member_rows.empty:
        return EligibilityResult(eligible=False, reasons=["member not found on file"])

    member = member_rows.iloc[0]

    if member["eligibility_status"] != "active":
        reasons.append(f"member coverage status is {member['eligibility_status']}")

    if str(member["plan_id"]) != str(claim["member_plan_id"]):
        reasons.append(
            f"plan mismatch: claim lists {claim['member_plan_id']}, "
            f"member is enrolled in {member['plan_id']}"
        )

    submission_date = date.fromisoformat(str(claim["submission_date"]))
    coverage_start = date.fromisoformat(str(member["coverage_start"]))
    coverage_end = date.fromisoformat(str(member["coverage_end"]))
    if not (coverage_start <= submission_date <= coverage_end):
        reasons.append(
            f"submission_date {submission_date.isoformat()} falls outside "
            f"coverage window {coverage_start.isoformat()} to {coverage_end.isoformat()}"
        )

    return EligibilityResult(eligible=len(reasons) == 0, reasons=reasons)
