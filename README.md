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
5-agent pipeline built with LangGraph (in progress, see [CLAUDE.md](CLAUDE.md)
for the full design):
1. Validation agent - checks claim completeness and code formats
2. Eligibility agent - checks member plan status and coverage dates
3. Policy RAG agent - retrieves relevant policy text (prior auth rules, etc.)
4. Denial-risk agent - scores denial probability with a trained XGBoost model
5. Resolution agent - decides ready to submit / needs info / human review

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
rather than act as a hard yes/no classifier. The resolution agent (Day 2)
will route by score band rather than a single threshold, so recall on the
highest-risk band matters more than precision at the midpoint. Full numbers
are in `models/metrics.json`, regenerated each time the training script runs.

## Quick start
Data generation and the denial-risk model are runnable so far:

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py
python models/train_denial_risk_model.py
pytest tests/ -v
```

The first command regenerates `data/claims.csv`, `data/members.csv`, and
`data/historical_denials.csv` with a fixed random seed. The second trains the
denial-risk model on `data/historical_denials.csv` and writes
`models/denial_risk_model.json` and `models/metrics.json`. The test suite
checks the generated data's schema, the model's feature engineering, and a
minimum quality bar on held-out ROC-AUC/precision/recall.

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
  the resolution agent should route by risk band instead of one cutoff.
- RAG retrieval quality hasn't been evaluated yet (still to build in Day 2).
- No production-grade auth, rate limiting, or monitoring on the API.

## License
MIT
