import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from generate_synthetic_data import (  # noqa: E402
    DENIAL_REASONS,
    DIAGNOSES,
    PLANS,
    PROCEDURES,
    generate_claims_and_denials,
    generate_members,
)
import random  # noqa: E402
import numpy as np  # noqa: E402


def _generate(seed=123):
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    members_df = generate_members(py_rng, np_rng)
    claims_df, denials_df = generate_claims_and_denials(py_rng, np_rng, members_df)
    return members_df, claims_df, denials_df


def test_members_schema_and_values():
    members_df, _, _ = _generate()
    expected_columns = {
        "member_id",
        "plan_id",
        "eligibility_status",
        "coverage_start",
        "coverage_end",
    }
    assert expected_columns == set(members_df.columns)
    assert members_df["member_id"].is_unique
    assert set(members_df["plan_id"]) <= set(PLANS.keys())
    assert set(members_df["eligibility_status"]) <= {"active", "lapsed", "terminated"}
    starts = pd.to_datetime(members_df["coverage_start"])
    ends = pd.to_datetime(members_df["coverage_end"])
    assert (ends > starts).all()


def test_claims_schema_and_values():
    _, claims_df, _ = _generate()
    expected_columns = {
        "claim_id",
        "patient_id",
        "procedure_code",
        "diagnosis_code",
        "provider_id",
        "submission_date",
        "prior_auth_flag",
        "member_plan_id",
        "billed_amount",
    }
    assert expected_columns == set(claims_df.columns)
    assert claims_df["claim_id"].is_unique
    assert set(claims_df["procedure_code"]) <= set(PROCEDURES.keys())
    assert set(claims_df["diagnosis_code"]) <= set(DIAGNOSES.keys())
    assert (claims_df["billed_amount"] > 0).all()


def test_historical_denials_schema_and_rate():
    _, _, denials_df = _generate()
    expected_columns = {
        "claim_id",
        "patient_id",
        "procedure_code",
        "diagnosis_code",
        "provider_id",
        "submission_date",
        "prior_auth_flag",
        "member_plan_id",
        "billed_amount",
        "outcome",
        "denial_reason",
    }
    assert expected_columns == set(denials_df.columns)
    assert set(denials_df["outcome"]) == {"approved", "denied"}
    denied = denials_df[denials_df["outcome"] == "denied"]
    assert set(denied["denial_reason"]) <= set(DENIAL_REASONS)
    approved = denials_df[denials_df["outcome"] == "approved"]
    assert (approved["denial_reason"] == "").all()

    denial_rate = (denials_df["outcome"] == "denied").mean()
    # Sanity bound: realistic claim denial rates generally fall well under 50%.
    assert 0.05 < denial_rate < 0.35


def test_generation_is_reproducible_given_seed():
    _, claims_a, denials_a = _generate(seed=7)
    _, claims_b, denials_b = _generate(seed=7)
    pd.testing.assert_frame_equal(claims_a, claims_b)
    pd.testing.assert_frame_equal(denials_a, denials_b)


def test_missing_prior_auth_correlates_with_higher_denial_rate():
    _, _, denials_df = _generate()
    rates = denials_df.groupby("prior_auth_flag")["outcome"].apply(
        lambda s: (s == "denied").mean()
    )
    assert rates[False] > rates[True]
