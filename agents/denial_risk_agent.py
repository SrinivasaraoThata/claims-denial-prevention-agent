"""Denial-risk agent: scores a claim's denial probability with the trained
XGBoost model. Direct function call, no LLM involved.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from features import build_features  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "denial_risk_model.json"

LOW_RISK_MAX = 0.3
MEDIUM_RISK_MAX = 0.6


@dataclass
class DenialRiskResult:
    risk_score: float
    risk_band: str  # "low" | "medium" | "high"
    missing_prior_auth: bool


def _band_for_score(score: float) -> str:
    if score < LOW_RISK_MAX:
        return "low"
    if score < MEDIUM_RISK_MAX:
        return "medium"
    return "high"


def load_model(model_path: Path = MODEL_PATH) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def score_claim(claim: dict, members_df: pd.DataFrame, model: XGBClassifier) -> DenialRiskResult:
    claim_df = pd.DataFrame([claim])
    features = build_features(claim_df, members_df)
    risk_score = float(model.predict_proba(features)[:, 1][0])
    return DenialRiskResult(
        risk_score=risk_score,
        risk_band=_band_for_score(risk_score),
        missing_prior_auth=bool(features["missing_prior_auth"].iloc[0]),
    )
