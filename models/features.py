"""Shared feature engineering for the denial-risk model.

Used by both the training script and (in a later step) the denial-risk agent,
so the features a claim is scored on at inference time match what the model
was trained on.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from generate_synthetic_data import PROCEDURES  # noqa: E402

CATEGORICAL_COLUMNS = ["procedure_code", "diagnosis_code", "member_plan_id", "provider_id"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + [
    "prior_auth_flag",
    "billed_amount",
    "requires_prior_auth",
    "missing_prior_auth",
    "member_ineligible",
]


def _requires_prior_auth(procedure_code):
    return PROCEDURES.get(procedure_code, (None, False, None, None))[1]


def build_features(claims_df, members_df):
    """Build the model-ready feature frame for a set of claims.

    claims_df must have: procedure_code, diagnosis_code, provider_id,
    submission_date, prior_auth_flag, member_plan_id, billed_amount,
    patient_id.
    members_df must have: member_id, coverage_start, coverage_end.
    """
    df = claims_df.merge(
        members_df[["member_id", "coverage_start", "coverage_end"]],
        left_on="patient_id",
        right_on="member_id",
        how="left",
    )

    submission_date = pd.to_datetime(df["submission_date"])
    coverage_start = pd.to_datetime(df["coverage_start"])
    coverage_end = pd.to_datetime(df["coverage_end"])
    df["member_ineligible"] = ~submission_date.between(coverage_start, coverage_end)

    df["requires_prior_auth"] = df["procedure_code"].map(_requires_prior_auth)
    df["missing_prior_auth"] = df["requires_prior_auth"] & ~df["prior_auth_flag"].astype(bool)

    features = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        features[col] = features[col].astype("category")
    for col in ["prior_auth_flag", "requires_prior_auth", "missing_prior_auth", "member_ineligible"]:
        features[col] = features[col].astype(bool)

    return features


def build_labels(denials_df):
    return (denials_df["outcome"] == "denied").astype(int)
