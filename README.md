# Multi-Agent Claims Denial-Prevention System

## Problem statement
Health insurance claims get denied after the fact for reasons that are often
predictable before submission: missing prior authorization, a member whose
coverage lapsed, a diagnosis and procedure that don't line up, incomplete
documentation. This project runs a claim through a pipeline of specialized
agents before it's ever submitted, catching the preventable denials early and
routing the genuinely uncertain cases to a human instead of guessing.

All data used here is synthetic, generated in this repo. No real or
proprietary claims, member, or policy data is involved.

## Architecture
5-agent pipeline built with LangGraph (see [CLAUDE.md](CLAUDE.md) for the
full design):
1. Validation agent - checks claim completeness and code formats
2. Eligibility agent - checks member plan status and coverage dates
3. Policy RAG agent - retrieves relevant policy text (prior auth rules, etc.)
4. Denial-risk agent - scores denial probability with a trained XGBoost model
5. Resolution agent - decides ready to submit / needs info / human review

A claim that fails validation skips straight to resolution rather than
running eligibility and risk scoring against data that can't be trusted.
Everything else runs the full pipeline and returns a decision plus the
reasoning trace from each agent.

Retrieval for the policy RAG agent uses Chroma with a TF-IDF embedding
function fit on the (small, fixed) policy corpus, so indexing and tests run
offline with no model download. Answer synthesis calls Gemini when a
`GOOGLE_API_KEY` is set, grounded strictly in the retrieved policy text; with
no key configured it falls back to returning the retrieved excerpts directly
so the rest of the pipeline still works without a keyed dependency.

Architecture diagram: TBD (see `docs/`).

## Results / metrics
Denial-risk model (XGBoost, trained on 4,000 synthetic historical claims,
evaluated on a held-out 1,000-claim test set, 15% denial rate):

| Metric | Value |
|---|---|
| ROC-AUC | 0.81 |
| PR-AUC | 0.48 |
| Precision @ 0.5 threshold | 0.50 |
| Recall @ 0.5 threshold | 0.33 |

Precision/recall at the default 0.5 threshold are modest, which is expected
given the model is meant to feed a risk score into the resolution agent
rather than act as a hard yes/no classifier. The resolution agent routes by
score band (low / medium / high) rather than a single threshold, so recall
on the highest-risk band matters more than precision at the midpoint. Full
numbers are in `models/metrics.json`, regenerated each time the training
script runs.

## Quick start
```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py
python models/train_denial_risk_model.py
uvicorn api.main:app --reload
pytest tests/ -v
```

The first command regenerates `data/claims.csv`, `data/members.csv`, and
`data/historical_denials.csv` with a fixed random seed. The second trains the
denial-risk model on `data/historical_denials.csv` and writes
`models/denial_risk_model.json` and `models/metrics.json`. The third starts
the API at `http://127.0.0.1:8000`, with interactive docs at `/docs`. The
test suite covers the generated data's schema, feature engineering, each
agent, the LangGraph pipeline, and the API endpoints.

Try it once the API is running:

```bash
curl -X POST http://127.0.0.1:8000/claims/submit \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "C000001",
    "patient_id": "M00001",
    "procedure_code": "70551",
    "diagnosis_code": "I10",
    "provider_id": "P0001",
    "submission_date": "2025-05-01",
    "prior_auth_flag": false,
    "member_plan_id": "PLAN001",
    "billed_amount": 1750.0
  }'
```

## Tech stack
- Agent orchestration: LangGraph
- LLM: Google Gemini (free tier, via Google AI Studio)
- Vector store: Chroma
- Risk model: XGBoost
- API: FastAPI
- Hosting: Google Cloud Run (Always Free tier)
- CI: GitHub Actions

## Live demo
TBD once deployed.

## Why I built this
I've worked on claims processing systems professionally and wanted a public,
from-scratch version of the same idea: catch preventable denials before
submission instead of appealing them after the fact. Everything here is
built with synthetic data so it's shareable without touching anything
proprietary.

## Limitations / next steps
- Synthetic data only; real claims data has messier edge cases than what's
  modeled here.
- Precision/recall on the denial-risk model are modest at a single threshold;
  the resolution agent mitigates this by routing on risk band rather than a
  hard cutoff.
- Policy retrieval uses TF-IDF over a 9-document corpus, which works well
  here because the corpus is small and each document maps cleanly to a
  procedure category. It wouldn't scale to a large, overlapping policy
  library without moving to a real embedding model.
- Decisions are stored in memory, not persisted; they don't survive an API
  restart.
- Not yet deployed; no live demo link, no Docker image, no CI workflow.
- No production-grade auth, rate limiting, or monitoring on the API.

## License
MIT
