"""Generate synthetic claims, member, and historical denial data for this project.

Everything here is fabricated. Procedure/diagnosis codes are real public code
formats (CPT-style, ICD-10-style) but the claims, members, and outcomes tied to
them are randomly generated and carry no relation to any real patient, provider,
or payer. Run with a fixed seed so the output is reproducible.
"""

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# procedure_code -> (description, requires_prior_auth, base_billed_amount, category)
PROCEDURES = {
    "99213": ("Office visit, established patient, low complexity", False, 110, "office_visit"),
    "99214": ("Office visit, established patient, moderate complexity", False, 165, "office_visit"),
    "93000": ("Electrocardiogram, routine", False, 95, "diagnostic"),
    "71046": ("Chest X-ray, 2 views", False, 120, "diagnostic"),
    "36415": ("Venipuncture, routine blood draw", False, 25, "lab"),
    "90837": ("Psychotherapy, 60 minutes", False, 175, "behavioral_health"),
    "45378": ("Diagnostic colonoscopy", True, 1450, "endoscopy"),
    "43239": ("Upper GI endoscopy with biopsy", True, 1600, "endoscopy"),
    "70551": ("MRI brain without contrast", True, 1750, "imaging"),
    "70553": ("MRI brain with and without contrast", True, 2400, "imaging"),
    "29881": ("Knee arthroscopy with meniscectomy", True, 5200, "surgery"),
    "27447": ("Total knee replacement", True, 33500, "surgery"),
    "19120": ("Breast biopsy, excisional", True, 2100, "surgery"),
    "97110": ("Therapeutic exercise, physical therapy", True, 90, "physical_therapy"),
    "J1745": ("Infliximab infusion, per 10mg", True, 4800, "specialty_drug"),
}

# diagnosis_code -> description
DIAGNOSES = {
    "E11.9": "Type 2 diabetes mellitus without complications",
    "I10": "Essential (primary) hypertension",
    "M54.5": "Low back pain",
    "J06.9": "Acute upper respiratory infection, unspecified",
    "M17.11": "Unilateral primary osteoarthritis, right knee",
    "K21.9": "Gastro-esophageal reflux disease without esophagitis",
    "F41.1": "Generalized anxiety disorder",
    "N18.3": "Chronic kidney disease, stage 3",
    "C50.911": "Malignant neoplasm of unspecified site, right female breast",
    "Z00.00": "Encounter for general adult medical exam without abnormal findings",
}

PLANS = {
    "PLAN001": "PPO Gold",
    "PLAN002": "HMO Standard",
    "PLAN003": "EPO Bronze",
    "PLAN004": "POS Silver",
}

DENIAL_REASONS = [
    "missing_prior_authorization",
    "member_ineligible_on_service_date",
    "diagnosis_procedure_mismatch",
    "duplicate_claim",
    "non_covered_service_for_plan",
    "documentation_incomplete",
]

N_MEMBERS = 500
N_PROVIDERS = 100
N_HISTORICAL_CLAIMS = 5000
N_OPEN_CLAIMS = 200

DATA_DIR = Path(__file__).parent


def make_rng(seed):
    return random.Random(seed), np.random.default_rng(seed)


def random_date(rng, start, end):
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta + 1)))


def generate_members(py_rng, np_rng):
    rows = []
    coverage_window_start = date(2023, 1, 1)
    coverage_window_end = date(2026, 6, 1)
    for i in range(1, N_MEMBERS + 1):
        member_id = f"M{i:05d}"
        plan_id = py_rng.choice(list(PLANS.keys()))
        coverage_start = random_date(np_rng, coverage_window_start, date(2025, 12, 1))
        # Most members are active; a minority have lapsed or future-dated coverage.
        status_roll = py_rng.random()
        if status_roll < 0.82:
            eligibility_status = "active"
            coverage_end = coverage_start + timedelta(days=int(np_rng.integers(365, 1095)))
        elif status_roll < 0.93:
            eligibility_status = "lapsed"
            coverage_end = coverage_start + timedelta(days=int(np_rng.integers(30, 400)))
        else:
            eligibility_status = "terminated"
            coverage_end = coverage_start + timedelta(days=int(np_rng.integers(30, 200)))
        rows.append(
            {
                "member_id": member_id,
                "plan_id": plan_id,
                "eligibility_status": eligibility_status,
                "coverage_start": coverage_start.isoformat(),
                "coverage_end": coverage_end.isoformat(),
            }
        )
    return pd.DataFrame(rows)


def pick_provider(py_rng):
    return f"P{py_rng.randint(1, N_PROVIDERS):04d}"


def build_high_denial_providers(py_rng):
    # A small subset of providers has a structurally higher denial rate,
    # meant to simulate billing-quality variance across a provider network.
    all_providers = [f"P{i:04d}" for i in range(1, N_PROVIDERS + 1)]
    return set(py_rng.sample(all_providers, k=12))


def simulate_claim(py_rng, np_rng, members_df, high_denial_providers, claim_index, historical):
    member_row = members_df.sample(1, random_state=int(np_rng.integers(0, 1_000_000))).iloc[0]
    member_id = member_row["member_id"]
    plan_id = member_row["plan_id"]
    coverage_start = date.fromisoformat(member_row["coverage_start"])
    coverage_end = date.fromisoformat(member_row["coverage_end"])

    procedure_code = py_rng.choice(list(PROCEDURES.keys()))
    proc_desc, requires_pa, base_amount, category = PROCEDURES[procedure_code]
    diagnosis_code = py_rng.choice(list(DIAGNOSES.keys()))

    provider_id = pick_provider(py_rng)

    # Submission dates: historical claims spread over the last two years,
    # open claims cluster around the most recent month. Most claims are
    # submitted while the member's coverage is active; a minority land
    # outside the coverage window, which is what actually produces
    # eligibility-driven denials below.
    historical_window_start = date(2024, 1, 1)
    historical_window_end = date(2026, 6, 30)
    if historical:
        window_start = max(coverage_start, historical_window_start)
        window_end = min(coverage_end, historical_window_end)
        if window_start < window_end and py_rng.random() < 0.85:
            submission_date = random_date(np_rng, window_start, window_end)
        else:
            submission_date = random_date(np_rng, historical_window_start, historical_window_end)
    else:
        submission_date = random_date(np_rng, date(2026, 7, 1), date(2026, 7, 31))

    # Prior auth is obtained most of the time when required, but not always.
    if requires_pa:
        prior_auth_flag = py_rng.random() < 0.72
    else:
        prior_auth_flag = py_rng.random() < 0.05

    billed_amount = round(base_amount * float(np_rng.normal(1.0, 0.12)), 2)
    billed_amount = max(billed_amount, 10.0)

    claim = {
        "claim_id": f"C{claim_index:06d}",
        "patient_id": member_id,
        "procedure_code": procedure_code,
        "diagnosis_code": diagnosis_code,
        "provider_id": provider_id,
        "submission_date": submission_date.isoformat(),
        "prior_auth_flag": prior_auth_flag,
        "member_plan_id": plan_id,
        "billed_amount": billed_amount,
    }

    if not historical:
        return claim, None

    # --- outcome simulation for historical claims only ---
    member_ineligible = not (coverage_start <= submission_date <= coverage_end)
    missing_pa = requires_pa and not prior_auth_flag
    high_value_no_pa = requires_pa and not prior_auth_flag and billed_amount > 2000
    risky_provider = provider_id in high_denial_providers
    diagnosis_mismatch = py_rng.random() < 0.04  # rare data-entry mismatches

    denial_score = 0.02  # base denial rate for a clean claim
    if member_ineligible:
        denial_score += 0.45
    if missing_pa:
        denial_score += 0.30
    if high_value_no_pa:
        denial_score += 0.10
    if risky_provider:
        denial_score += 0.12
    if diagnosis_mismatch:
        denial_score += 0.25
    denial_score = min(denial_score, 0.95)

    is_denied = py_rng.random() < denial_score

    if is_denied:
        if member_ineligible:
            reason = "member_ineligible_on_service_date"
        elif missing_pa:
            reason = "missing_prior_authorization"
        elif diagnosis_mismatch:
            reason = "diagnosis_procedure_mismatch"
        else:
            reason = py_rng.choice(DENIAL_REASONS)
        outcome = "denied"
    else:
        reason = ""
        outcome = "approved"

    denial_record = {
        "claim_id": claim["claim_id"],
        "patient_id": member_id,
        "procedure_code": procedure_code,
        "diagnosis_code": diagnosis_code,
        "provider_id": provider_id,
        "submission_date": claim["submission_date"],
        "prior_auth_flag": prior_auth_flag,
        "member_plan_id": plan_id,
        "billed_amount": billed_amount,
        "outcome": outcome,
        "denial_reason": reason,
    }
    return claim, denial_record


def generate_claims_and_denials(py_rng, np_rng, members_df):
    high_denial_providers = build_high_denial_providers(py_rng)

    historical_rows = []
    claim_idx = 1
    for _ in range(N_HISTORICAL_CLAIMS):
        _, denial_record = simulate_claim(
            py_rng, np_rng, members_df, high_denial_providers, claim_idx, historical=True
        )
        historical_rows.append(denial_record)
        claim_idx += 1

    open_rows = []
    for _ in range(N_OPEN_CLAIMS):
        claim, _ = simulate_claim(
            py_rng, np_rng, members_df, high_denial_providers, claim_idx, historical=False
        )
        open_rows.append(claim)
        claim_idx += 1

    return pd.DataFrame(open_rows), pd.DataFrame(historical_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    members_df = generate_members(py_rng, np_rng)
    claims_df, denials_df = generate_claims_and_denials(py_rng, np_rng, members_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    members_df.to_csv(args.out_dir / "members.csv", index=False)
    claims_df.to_csv(args.out_dir / "claims.csv", index=False)
    denials_df.to_csv(args.out_dir / "historical_denials.csv", index=False)

    denial_rate = (denials_df["outcome"] == "denied").mean()
    print(f"members: {len(members_df)}")
    print(f"open claims: {len(claims_df)}")
    print(f"historical claims: {len(denials_df)} (denial rate: {denial_rate:.1%})")


if __name__ == "__main__":
    main()
