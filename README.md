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
TBD once the risk model and pipeline are built.

## Quick start
Data generation is the only runnable piece so far:

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py
pytest tests/ -v
```

This regenerates `data/claims.csv`, `data/members.csv`, and
`data/historical_denials.csv` with a fixed random seed, and runs the test
suite that checks the generated data's schema and shape.

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
- Denial-risk model and RAG retrieval quality haven't been evaluated yet
  (Day 1 covers data generation only).
- No production-grade auth, rate limiting, or monitoring on the API.

## License
MIT
