"""LangGraph pipeline wiring the five claims-processing agents together.

validation -> eligibility -> policy -> denial_risk -> resolution

A claim that fails validation skips straight to resolution: eligibility and
denial-risk scoring both assume a well-formed submission_date and known
codes, and a malformed claim can't be trusted enough to run those checks
against.
"""

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.denial_risk_agent import DenialRiskResult, score_claim
from agents.eligibility_agent import EligibilityResult, check_eligibility
from agents.policy_rag_agent import PolicyRagResult, check_policy
from agents.resolution_agent import ResolutionResult, resolve
from agents.validation_agent import ValidationResult, validate_claim


class ClaimState(TypedDict, total=False):
    claim: dict
    validation: ValidationResult
    eligibility: Optional[EligibilityResult]
    policy: Optional[PolicyRagResult]
    denial_risk: Optional[DenialRiskResult]
    resolution: ResolutionResult


def build_graph(members_df, risk_model, policy_collection):
    """Compile the claims pipeline, bound to the given data/model/index."""

    def validation_node(state: ClaimState) -> dict:
        return {"validation": validate_claim(state["claim"])}

    def eligibility_node(state: ClaimState) -> dict:
        return {"eligibility": check_eligibility(state["claim"], members_df)}

    def policy_node(state: ClaimState) -> dict:
        return {"policy": check_policy(state["claim"], policy_collection)}

    def denial_risk_node(state: ClaimState) -> dict:
        return {"denial_risk": score_claim(state["claim"], members_df, risk_model)}

    def resolution_node(state: ClaimState) -> dict:
        result = resolve(state["validation"], state.get("eligibility"), state.get("denial_risk"))
        return {"resolution": result}

    def route_after_validation(state: ClaimState) -> str:
        return "eligibility" if state["validation"].passed else "resolution"

    graph = StateGraph(ClaimState)
    graph.add_node("validate_claim", validation_node)
    graph.add_node("check_eligibility", eligibility_node)
    graph.add_node("check_policy", policy_node)
    graph.add_node("score_denial_risk", denial_risk_node)
    graph.add_node("resolve_claim", resolution_node)

    graph.add_edge(START, "validate_claim")
    graph.add_conditional_edges(
        "validate_claim",
        route_after_validation,
        {"eligibility": "check_eligibility", "resolution": "resolve_claim"},
    )
    graph.add_edge("check_eligibility", "check_policy")
    graph.add_edge("check_policy", "score_denial_risk")
    graph.add_edge("score_denial_risk", "resolve_claim")
    graph.add_edge("resolve_claim", END)

    return graph.compile()
