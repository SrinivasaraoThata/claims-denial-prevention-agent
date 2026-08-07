"""FastAPI app exposing the claims denial-prevention pipeline.

Loads the member roster, the trained denial-risk model, and the policy
vector index once at startup, then runs each submitted claim through the
LangGraph pipeline (validation -> eligibility -> policy -> denial risk ->
resolution).
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from agents.denial_risk_agent import load_model  # noqa: E402
from agents.graph import build_graph  # noqa: E402
from agents.policy_rag_agent import load_collection  # noqa: E402
from api.schemas import (  # noqa: E402
    ClaimDecisionResponse,
    ClaimIn,
    DenialRiskOut,
    EligibilityOut,
    PolicyOut,
    ValidationOut,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# In-memory store of past decisions, keyed by claim_id. Fine for a portfolio
# demo; a real deployment would back this with a database.
_decision_store: dict[str, ClaimDecisionResponse] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    members_df = pd.read_csv(DATA_DIR / "members.csv")
    risk_model = load_model()
    policy_collection = load_collection()
    app.state.graph = build_graph(members_df, risk_model, policy_collection)
    yield


app = FastAPI(title="Claims Denial-Prevention API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/claims/submit", response_model=ClaimDecisionResponse)
def submit_claim(claim: ClaimIn):
    final_state = app.state.graph.invoke({"claim": claim.model_dump()})

    validation = final_state["validation"]
    eligibility = final_state.get("eligibility")
    policy = final_state.get("policy")
    denial_risk = final_state.get("denial_risk")
    resolution = final_state["resolution"]

    response = ClaimDecisionResponse(
        claim_id=claim.claim_id,
        decision=resolution.decision,
        reasons=resolution.reasons,
        validation=ValidationOut(passed=validation.passed, errors=validation.errors),
        eligibility=(
            EligibilityOut(eligible=eligibility.eligible, reasons=eligibility.reasons)
            if eligibility is not None
            else None
        ),
        policy=(
            PolicyOut(
                query=policy.query, answer=policy.answer, llm_used=policy.llm_used, sources=policy.sources
            )
            if policy is not None
            else None
        ),
        denial_risk=(
            DenialRiskOut(
                risk_score=denial_risk.risk_score,
                risk_band=denial_risk.risk_band,
                missing_prior_auth=denial_risk.missing_prior_auth,
            )
            if denial_risk is not None
            else None
        ),
    )

    _decision_store[claim.claim_id] = response
    return response


@app.get("/claims/{claim_id}/status", response_model=ClaimDecisionResponse)
def get_claim_status(claim_id: str):
    response = _decision_store.get(claim_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"no decision on file for claim_id {claim_id}")
    return response
