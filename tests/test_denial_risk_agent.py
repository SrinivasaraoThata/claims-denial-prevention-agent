import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.denial_risk_agent import load_model, score_claim  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def members_df():
    return pd.read_csv(DATA_DIR / "members.csv")


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_score_claim_returns_probability_and_band(members_df, model):
    member = members_df.iloc[0]
    claim = {
        "claim_id": "C999999",
        "patient_id": member["member_id"],
        "procedure_code": "99213",
        "diagnosis_code": "E11.9",
        "provider_id": "P0001",
        "submission_date": member["coverage_start"],
        "prior_auth_flag": True,
        "member_plan_id": member["plan_id"],
        "billed_amount": 110.0,
    }
    result = score_claim(claim, members_df, model)

    assert 0.0 <= result.risk_score <= 1.0
    assert result.risk_band in {"low", "medium", "high"}


def test_missing_prior_auth_scores_higher_risk(members_df, model):
    member = members_df[members_df["eligibility_status"] == "active"].iloc[0]
    base_claim = {
        "claim_id": "C999998",
        "patient_id": member["member_id"],
        "procedure_code": "27447",  # total knee replacement, requires prior auth
        "diagnosis_code": "M17.11",
        "provider_id": "P0002",
        "submission_date": member["coverage_start"],
        "member_plan_id": member["plan_id"],
        "billed_amount": 33500.0,
    }
    with_auth = score_claim({**base_claim, "prior_auth_flag": True}, members_df, model)
    without_auth = score_claim({**base_claim, "prior_auth_flag": False}, members_df, model)

    assert without_auth.risk_score > with_auth.risk_score
    assert with_auth.missing_prior_auth is False
    assert without_auth.missing_prior_auth is True


def test_score_is_consistent_between_single_and_batch_scoring(members_df, model):
    member = members_df.iloc[3]
    claim = {
        "claim_id": "C999997",
        "patient_id": member["member_id"],
        "procedure_code": "70551",
        "diagnosis_code": "I10",
        "provider_id": "P0010",
        "submission_date": member["coverage_start"],
        "prior_auth_flag": False,
        "member_plan_id": member["plan_id"],
        "billed_amount": 1750.0,
    }
    first = score_claim(claim, members_df, model)
    second = score_claim(claim, members_df, model)

    assert first.risk_score == second.risk_score
