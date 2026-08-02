import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_features, build_labels  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def denials_and_members():
    denials_df = pd.read_csv(DATA_DIR / "historical_denials.csv", dtype={"denial_reason": str})
    members_df = pd.read_csv(DATA_DIR / "members.csv")
    return denials_df, members_df


def test_build_features_shape_and_dtypes(denials_and_members):
    denials_df, members_df = denials_and_members
    X = build_features(denials_df, members_df)

    assert len(X) == len(denials_df)
    assert set(X.columns) == set(FEATURE_COLUMNS)
    for col in CATEGORICAL_COLUMNS:
        assert X[col].dtype.name == "category"
    for col in ["prior_auth_flag", "requires_prior_auth", "missing_prior_auth", "member_ineligible"]:
        assert X[col].dtype == bool


def test_build_labels_binary(denials_and_members):
    denials_df, _ = denials_and_members
    y = build_labels(denials_df)
    assert set(y.unique()) <= {0, 1}
    assert len(y) == len(denials_df)


def test_missing_prior_auth_flags_expected_rows(denials_and_members):
    denials_df, members_df = denials_and_members
    X = build_features(denials_df, members_df)
    # Rows where prior auth was required but not obtained must be flagged.
    flagged = X[X["missing_prior_auth"]]
    assert (flagged["requires_prior_auth"]).all()
    assert (~flagged["prior_auth_flag"]).all()


def test_model_trains_and_meets_minimum_quality_bar():
    from train_denial_risk_model import train_and_evaluate

    metrics = train_and_evaluate()

    assert metrics["roc_auc"] > 0.7
    assert metrics["precision"] > 0.3
    assert metrics["recall"] > 0.3


def test_categorical_codes_are_stable_for_a_single_row_batch(denials_and_members):
    # Category codes are how XGBoost encodes categorical splits. If a batch
    # of one row got its own category codes instead of the fixed universe's,
    # a claim scored on its own (as the denial-risk agent does at inference
    # time) would silently get a different, wrong feature encoding than the
    # same claim scored as part of a larger batch.
    denials_df, members_df = denials_and_members
    X_batch = build_features(denials_df, members_df)

    row = denials_df.iloc[[0]]
    X_single = build_features(row, members_df)

    for col in CATEGORICAL_COLUMNS:
        assert list(X_single[col].cat.categories) == list(X_batch[col].cat.categories)
        assert X_single[col].cat.codes.iloc[0] == X_batch[col].cat.codes.iloc[0]
