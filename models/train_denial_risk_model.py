"""Train and evaluate the denial-risk XGBoost model on synthetic historical claims.

Usage:
    python models/train_denial_risk_model.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, build_labels  # noqa: E402

MODELS_DIR = Path(__file__).parent
DATA_DIR = MODELS_DIR.parent / "data"
SEED = 42
DECISION_THRESHOLD = 0.5


def load_data():
    denials_df = pd.read_csv(DATA_DIR / "historical_denials.csv", dtype={"denial_reason": str})
    members_df = pd.read_csv(DATA_DIR / "members.csv")
    return denials_df, members_df


def train_and_evaluate():
    denials_df, members_df = load_data()

    X = build_features(denials_df, members_df)
    y = build_labels(denials_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    preds_test = (proba_test >= DECISION_THRESHOLD).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds_test, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)

    metrics = {
        "test_set_size": int(len(y_test)),
        "test_denial_rate": float(y_test.mean()),
        "decision_threshold": DECISION_THRESHOLD,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }

    model.save_model(MODELS_DIR / "denial_risk_model.json")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = train_and_evaluate()
    for key, value in metrics.items():
        print(f"{key}: {value}")
