import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.denial_risk_agent import load_model  # noqa: E402
from agents.graph import build_graph  # noqa: E402
from agents.policy_rag_agent import build_index  # noqa: E402
from agents.resolution_agent import NEEDS_INFO  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TEST_PERSIST_DIR = Path(__file__).resolve().parent / "_tmp_chroma_db_graph"


@pytest.fixture(scope="module")
def members_df():
    return pd.read_csv(DATA_DIR / "members.csv")


@pytest.fixture(scope="module")
def graph(members_df):
    shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)
    model = load_model()
    collection = build_index(persist_dir=TEST_PERSIST_DIR)
    compiled = build_graph(members_df, model, collection)
    yield compiled
    shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)


def test_malformed_claim_skips_to_needs_info(graph):
    claim = {
        "claim_id": "C1",
        "patient_id": "M00001",
        "procedure_code": "BAD",
        "diagnosis_code": "E11.9",
        "provider_id": "P0001",
        "submission_date": "2026-01-15",
        "prior_auth_flag": False,
        "member_plan_id": "PLAN001",
        "billed_amount": 100.0,
    }
    final_state = graph.invoke({"claim": claim})

    assert final_state["resolution"].decision == NEEDS_INFO
    assert final_state.get("eligibility") is None
    assert final_state.get("denial_risk") is None


def test_valid_claim_runs_full_pipeline(graph, members_df):
    member = members_df.iloc[0]
    claim = {
        "claim_id": "C2",
        "patient_id": member["member_id"],
        "procedure_code": "27447",  # total knee replacement, high billed amount
        "diagnosis_code": "M17.11",
        "provider_id": "P0001",
        "submission_date": member["coverage_start"],
        "prior_auth_flag": False,
        "member_plan_id": member["plan_id"],
        "billed_amount": 33500.0,
    }
    final_state = graph.invoke({"claim": claim})

    assert final_state["validation"].passed
    assert final_state["eligibility"] is not None
    assert final_state["policy"] is not None
    assert final_state["denial_risk"] is not None
    # No prior auth on a procedure that requires it -> needs_info, regardless of risk band.
    assert final_state["resolution"].decision == NEEDS_INFO


def test_ineligible_member_routes_to_needs_info(graph, members_df):
    lapsed = members_df[members_df["eligibility_status"] != "active"].iloc[0]
    claim = {
        "claim_id": "C3",
        "patient_id": lapsed["member_id"],
        "procedure_code": "99213",
        "diagnosis_code": "E11.9",
        "provider_id": "P0001",
        "submission_date": "2026-01-15",
        "prior_auth_flag": False,
        "member_plan_id": lapsed["plan_id"],
        "billed_amount": 110.0,
    }
    final_state = graph.invoke({"claim": claim})

    assert final_state["resolution"].decision == NEEDS_INFO
