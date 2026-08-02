import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import app  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def active_member():
    members_df = pd.read_csv(DATA_DIR / "members.csv")
    return members_df[members_df["eligibility_status"] == "active"].iloc[0]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_submit_clean_claim_ready_to_submit(client, active_member):
    claim = {
        "claim_id": "APITEST001",
        "patient_id": active_member["member_id"],
        "procedure_code": "99213",
        "diagnosis_code": "E11.9",
        "provider_id": "P0001",
        "submission_date": active_member["coverage_start"],
        "prior_auth_flag": False,
        "member_plan_id": active_member["plan_id"],
        "billed_amount": 110.0,
    }
    response = client.post("/claims/submit", json=claim)
    assert response.status_code == 200
    body = response.json()
    assert body["claim_id"] == "APITEST001"
    assert body["decision"] in {"ready_to_submit", "needs_info", "human_review"}
    assert body["validation"]["passed"] is True
    assert body["eligibility"]["eligible"] is True
    assert body["policy"] is not None
    assert body["denial_risk"] is not None


def test_submit_malformed_claim_needs_info(client, active_member):
    claim = {
        "claim_id": "APITEST002",
        "patient_id": active_member["member_id"],
        "procedure_code": "NOTACODE",
        "diagnosis_code": "E11.9",
        "provider_id": "P0001",
        "submission_date": active_member["coverage_start"],
        "prior_auth_flag": False,
        "member_plan_id": active_member["plan_id"],
        "billed_amount": 110.0,
    }
    response = client.post("/claims/submit", json=claim)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "needs_info"
    assert body["validation"]["passed"] is False
    assert body["eligibility"] is None


def test_status_returns_prior_decision(client, active_member):
    claim = {
        "claim_id": "APITEST003",
        "patient_id": active_member["member_id"],
        "procedure_code": "99213",
        "diagnosis_code": "E11.9",
        "provider_id": "P0001",
        "submission_date": active_member["coverage_start"],
        "prior_auth_flag": False,
        "member_plan_id": active_member["plan_id"],
        "billed_amount": 110.0,
    }
    client.post("/claims/submit", json=claim)

    response = client.get("/claims/APITEST003/status")
    assert response.status_code == 200
    assert response.json()["claim_id"] == "APITEST003"


def test_status_unknown_claim_returns_404(client):
    response = client.get("/claims/DOES-NOT-EXIST/status")
    assert response.status_code == 404
