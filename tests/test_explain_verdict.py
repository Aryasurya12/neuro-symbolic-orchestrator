"""Tests for Canonical Execution Verdict Explainer (src/reporting/explain_verdict.py).

Verifies:
1. All 9 canonical labels are accurately produced from offline fixtures:
   - CORRECT
   - CORRECT_BUT_COSTLIER
   - CORRECT_REFUSAL
   - WRONG_PLAN
   - WRONG_OUTCOME
   - RIGHT_ANSWER_WRONG_READING
   - UNPARSEABLE
   - PROVIDER_FAILURE
   - NOT_GRADED
2. Real figures (costs, vCPUs, RAM, % gap, latency, SLA) are dynamically generated into the explanation.
3. Terminal box, terminal summary, and dashboard consume identical strings from explain_run().
4. No overall winner, score, or ranking is generated.
5. Zero live network calls during test execution.
"""

from __future__ import annotations

import json
import pytest
from typing import Any, Dict

from src.reporting.explain_verdict import (
    ALL_LABELS,
    DISCLAIMER_RULE,
    LABEL_COLORS,
    LABEL_CORRECT,
    LABEL_CORRECT_BUT_COSTLIER,
    LABEL_CORRECT_REFUSAL,
    LABEL_LEGEND,
    LABEL_NOT_GRADED,
    LABEL_PROVIDER_FAILURE,
    LABEL_RIGHT_ANSWER_WRONG_READING,
    LABEL_TOOLTIPS,
    LABEL_UNPARSEABLE,
    LABEL_WRONG_OUTCOME,
    LABEL_WRONG_PLAN,
    explain_run,
    find_manifest_entry,
    format_plain_english_box,
    format_plain_english_summary,
)
from src.verifiers.canonical_record import (
    CanonicalExecutionRecord,
    CheckStatus,
    ExplanationSource,
    FeasibilityStatus,
    NormalizationStatus,
    OptimalityStatus,
)


# =============================================================================
# Offline Fixtures for All 9 Canonical Labels
# =============================================================================

@pytest.fixture
def manifest_key_q01() -> Dict[str, Any]:
    """Ground truth key for Q01 (Feasible VM query: 8 vCPUs, 16 GB, $300 max, optimal $121.47)."""
    return {
        "query_id": "Q01_VM_BURSTABLE",
        "query_text": "Deploy 8 vCPUs and 16GB RAM for under $300 on AWS",
        "problem_type": "Z3_MILP_VM_Placement",
        "expected_outcome": "FEASIBLE",
        "expected_optimal_cost_usd": 121.47,
        "optimum_source": "INDEPENDENT_SEARCH",
        "required_vcpus": 8,
        "required_ram_gb": 16.0,
        "budget_max_usd": 300.0,
        "key_notes": "4x t3.medium at $121.47/month is the global optimum",
    }


@pytest.fixture
def manifest_key_q05_clarification() -> Dict[str, Any]:
    """Ground truth key for Q05 (Missing budget -> Clarification Required)."""
    return {
        "query_id": "Q05_VM_NO_BUDGET",
        "query_text": "Need 16 vCPUs and 32GB RAM on AWS",
        "problem_type": "Z3_MILP_VM_Placement",
        "expected_outcome": "CLARIFICATION_REQUIRED",
        "expected_optimal_cost_usd": None,
        "optimum_source": None,
        "required_vcpus": 16,
        "required_ram_gb": 32.0,
        "budget_max_usd": None,
        "key_notes": "Query omitted budget limit; policy requires asking clarification",
    }


@pytest.fixture
def manifest_key_q07_infeasible() -> Dict[str, Any]:
    """Ground truth key for Q07 (Infeasible budget: $10/month for 32 vCPUs)."""
    return {
        "query_id": "Q07_VM_INFEASIBLE_BUDGET",
        "query_text": "Need 32 vCPUs and 64GB RAM for under $10 on AWS",
        "problem_type": "Z3_MILP_VM_Placement",
        "expected_outcome": "INFEASIBLE",
        "expected_optimal_cost_usd": None,
        "optimum_source": None,
        "required_vcpus": 32,
        "required_ram_gb": 64.0,
        "budget_max_usd": 10.0,
        "key_notes": "Cheapest 32 vCPU allocation costs >$200; $10 is impossible",
    }


@pytest.fixture
def manifest_key_dr_q14() -> Dict[str, Any]:
    """Ground truth key for DR query with strict latency & SLA."""
    return {
        "query_id": "Q14_DR_MULTI_REGION",
        "query_text": "Multi-region DR with 50ms latency, 99.99% SLA under $600",
        "problem_type": "Graph_Disaster_Recovery",
        "expected_outcome": "FEASIBLE",
        "expected_optimal_cost_usd": 243.00,
        "optimum_source": "FORMULA",
        "budget_max_usd": 600.0,
        "latency_max_ms": 50.0,
        "sla_availability_pct": 99.99,
        "key_notes": "Valid DR topology at $243/month",
    }


# =============================================================================
# 1. Label Tests: CORRECT
# =============================================================================

def test_explain_verdict_correct_optimal(manifest_key_q01):
    """Verifies CORRECT label when plan meets requirements and cost is within tolerance of optimum."""
    record = {
        "mode": 4,
        "mode_name": "Mode 4 (Neuro-Symbolic)",
        "feasibility": "PASS",
        "recomputed_cost_usd": 121.47,
        "claimed_cost_usd": 121.47,
        "normalization_status": "SUCCESS",
        "requirements": {
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": 300.0,
        },
        "violations": [],
        "summary_status": "FEASIBLE_OPTIMAL",
    }

    verdict = explain_run(record, key=manifest_key_q01)

    assert verdict["label"] == LABEL_CORRECT
    assert "Mode 4 (Neuro-Symbolic)" in verdict["headline"]
    assert "verified optimal plan" in verdict["headline"]
    assert "$121.47" in verdict["what_happened"]
    assert "$300" in verdict["what_happened"] or "300" in verdict["what_happened"]
    assert "within $0.00" in verdict["what_happened"] or "$121.47" in verdict["what_happened"]
    assert verdict["check_this"] is None
    assert verdict["color"] == LABEL_COLORS[LABEL_CORRECT]
    assert verdict["tooltip"] == LABEL_TOOLTIPS[LABEL_CORRECT]


# =============================================================================
# 2. Label Tests: CORRECT_BUT_COSTLIER
# =============================================================================

def test_explain_verdict_correct_but_costlier(manifest_key_q01):
    """Verifies CORRECT_BUT_COSTLIER when valid LLM plan costs more than optimum."""
    record = {
        "mode": 2,
        "mode_name": "Mode 2 (Schema LLM)",
        "feasibility": "PASS",
        "recomputed_cost_usd": 242.94,
        "claimed_cost_usd": 242.94,
        "normalization_status": "SUCCESS",
        "requirements": {
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": 300.0,
        },
        "violations": [],
        "summary_status": "FEASIBLE_SUBOPTIMAL",
    }

    verdict = explain_run(record, key=manifest_key_q01)

    assert verdict["label"] == LABEL_CORRECT_BUT_COSTLIER
    assert "costs 100.0% more than the optimum" in verdict["headline"] or "% more" in verdict["headline"]
    assert "$242.94" in verdict["what_happened"]
    assert "$121.47" in verdict["what_happened"]
    assert "100.0%" in verdict["what_happened"] or "wastes money" in verdict["what_happened"]
    assert verdict["check_this"] is not None
    assert "$121.47" in verdict["check_this"] or "cheaper SKU" in verdict["check_this"]


# =============================================================================
# 3. Label Tests: CORRECT_REFUSAL
# =============================================================================

def test_explain_verdict_correct_refusal_clarification(manifest_key_q05_clarification):
    """Verifies CORRECT_REFUSAL when mode appropriately requests clarification for missing budget."""
    record = {
        "mode": 4,
        "mode_name": "Mode 4 (Neuro-Symbolic)",
        "feasibility": "NOT_EVALUATED",
        "recomputed_cost_usd": None,
        "claimed_cost_usd": None,
        "normalization_status": "CLARIFICATION_REQUIRED",
        "requirements": {},
        "violations": [],
        "summary_status": "Clarification required: Missing budget",
    }

    verdict = explain_run(record, key=manifest_key_q05_clarification)

    assert verdict["label"] == LABEL_CORRECT_REFUSAL
    assert "correctly refused or requested clarification" in verdict["headline"]
    assert "CLARIFICATION_REQUIRED" in verdict["what_happened"]
    assert "requested clarification" in verdict["what_happened"]
    assert "Refusing impossible requests" in verdict["why_it_matters"]


def test_explain_verdict_correct_refusal_infeasible(manifest_key_q07_infeasible):
    """Verifies CORRECT_REFUSAL when solver proves request is mathematically infeasible."""
    record = {
        "mode": 3,
        "mode_name": "Mode 3 (Symbolic Local)",
        "feasibility": "INFEASIBLE",
        "recomputed_cost_usd": None,
        "claimed_cost_usd": None,
        "normalization_status": "SOLVER_INFEASIBLE",
        "requirements": {"required_vcpus": 32, "required_ram_gb": 64.0, "budget_max_usd": 10.0},
        "violations": ["Budget $10.00 is below minimum achievable cost $242.94"],
        "summary_status": "PROVEN_INFEASIBLE",
    }

    verdict = explain_run(record, key=manifest_key_q07_infeasible)

    assert verdict["label"] == LABEL_CORRECT_REFUSAL
    assert "INFEASIBLE" in verdict["what_happened"]
    assert "proved the constraints are mathematically infeasible" in verdict["what_happened"]


# =============================================================================
# 4. Label Tests: RIGHT_ANSWER_WRONG_READING (Modes 3/4)
# =============================================================================

def test_explain_verdict_right_answer_wrong_reading(manifest_key_dr_q14):
    """Verifies RIGHT_ANSWER_WRONG_READING when Mode 3 outputs valid answer using default parameters."""
    # Key expected budget $600, latency 50ms, SLA 99.99%
    # Mode 3 used defaults: budget $500, latency 100ms, SLA 99.9%
    record = {
        "mode": 3,
        "mode_name": "Mode 3 (Symbolic Local)",
        "feasibility": "PASS",
        "recomputed_cost_usd": 243.00,
        "claimed_cost_usd": 243.00,
        "normalization_status": "SUCCESS",
        "requirements": {
            "budget_max_usd": 500.0,
            "latency_max_ms": 100.0,
            "sla_availability_pct": 99.9,
        },
        "violations": [],
        "summary_status": "FEASIBLE_OPTIMAL",
    }

    verdict = explain_run(record, key=manifest_key_dr_q14)

    assert verdict["label"] == LABEL_RIGHT_ANSWER_WRONG_READING
    assert "matched outcome by luck" in verdict["headline"] or "misread" in verdict["headline"]
    assert "$243.00" in verdict["what_happened"]
    assert "differed from your key" in verdict["what_happened"]
    assert "budget" in verdict["what_happened"] or "latency" in verdict["what_happened"]
    assert "luck" in verdict["what_happened"]
    assert verdict["check_this"] is not None


# =============================================================================
# 5. Label Tests: WRONG_PLAN
# =============================================================================

def test_explain_verdict_wrong_plan(manifest_key_q01):
    """Verifies WRONG_PLAN when allocation violates a constraint (e.g. Budget overflow)."""
    record = {
        "mode": 1,
        "mode_name": "Mode 1 (Raw LLM)",
        "feasibility": "FAIL",
        "recomputed_cost_usd": 450.00,
        "claimed_cost_usd": 250.00,
        "normalization_status": "SUCCESS",
        "requirements": {"required_vcpus": 8, "required_ram_gb": 16.0, "budget_max_usd": 300.0},
        "violations": ["Budget exceeded: actual cost $450.00 exceeds limit $300.00"],
        "summary_status": "CONSTRAINT_VIOLATION",
    }

    verdict = explain_run(record, key=manifest_key_q01)

    assert verdict["label"] == LABEL_WRONG_PLAN
    assert "returned an invalid plan violating requirement" in verdict["headline"]
    assert "$450.00" in verdict["what_happened"]
    assert "Budget exceeded" in verdict["what_happened"]
    assert "violates SLA thresholds, causes outages, or breaches budgets" in verdict["why_it_matters"]
    assert "Budget exceeded" in verdict["check_this"]


# =============================================================================
# 6. Label Tests: WRONG_OUTCOME
# =============================================================================

def test_explain_verdict_wrong_outcome_guessed_instead_of_asking(manifest_key_q05_clarification):
    """Verifies WRONG_OUTCOME when mode guesses an allocation instead of asking clarification."""
    record = {
        "mode": 1,
        "mode_name": "Mode 1 (Raw LLM)",
        "feasibility": "PASS",
        "recomputed_cost_usd": 242.94,
        "claimed_cost_usd": 150.00,
        "normalization_status": "SUCCESS",
        "requirements": {"required_vcpus": 16, "required_ram_gb": 32.0},
        "violations": [],
        "summary_status": "FABRICATED_PLAN",
    }

    verdict = explain_run(record, key=manifest_key_q05_clarification)

    assert verdict["label"] == LABEL_WRONG_OUTCOME
    assert "returned a plan when the key expected CLARIFICATION_REQUIRED" in verdict["headline"]
    assert "fabricated an allocation costing $242.94" in verdict["what_happened"]
    assert "Fabricating numbers or guessing missing constraints" in verdict["why_it_matters"]


def test_explain_verdict_wrong_outcome_claimed_infeasible_when_feasible(manifest_key_q01):
    """Verifies WRONG_OUTCOME when mode erroneously halts on a feasible query."""
    record = {
        "mode": 2,
        "mode_name": "Mode 2 (Schema LLM)",
        "feasibility": "FAIL",
        "recomputed_cost_usd": None,
        "claimed_cost_usd": None,
        "normalization_status": "SOLVER_INFEASIBLE",
        "requirements": {},
        "violations": [],
        "summary_status": "Erred out",
    }

    verdict = explain_run(record, key=manifest_key_q01)

    assert verdict["label"] == LABEL_WRONG_OUTCOME
    assert "failed to find a plan for a feasible request" in verdict["headline"]
    assert "$300.00" in verdict["what_happened"]


# =============================================================================
# 7. Label Tests: UNPARSEABLE
# =============================================================================

def test_explain_verdict_unparseable_prose(manifest_key_q01):
    """Verifies UNPARSEABLE when LLM prose cannot be structured into SKU instances."""
    record = {
        "mode": 1,
        "mode_name": "Mode 1 (Raw LLM)",
        "feasibility": "NOT_EVALUABLE",
        "recomputed_cost_usd": None,
        "claimed_cost_usd": None,
        "normalization_status": "UNPARSEABLE",
        "normalization_errors": ["No concrete instance type detected in markdown table"],
        "requirements": {},
        "violations": [],
        "summary_status": "PARSE_ERROR",
    }

    verdict = explain_run(record, key=manifest_key_q01)

    assert verdict["label"] == LABEL_UNPARSEABLE
    assert "generated unparseable prose" in verdict["headline"]
    assert "could not be mapped to instance SKUs" in verdict["what_happened"]
    assert "No concrete instance type detected" in verdict["what_happened"]
    assert "NOT counted as a wrong answer" in verdict["tooltip"] or "cannot be verified" in verdict["why_it_matters"]


# =============================================================================
# 8. Label Tests: PROVIDER_FAILURE
# =============================================================================

def test_explain_verdict_provider_failure_429():
    """Verifies PROVIDER_FAILURE when API returns HTTP 429 Rate Limit / Quota Exhaustion."""
    key = {"expected_outcome": "FEASIBLE", "expected_optimal_cost_usd": 121.47}
    record = {
        "mode": 1,
        "mode_name": "Mode 1 (Raw LLM)",
        "feasibility": "NOT_EVALUABLE",
        "normalization_status": "API_FAILURE",
        "normalization_errors": ["Groq HTTP 429: rate limit exceeded, daily quota exhausted"],
        "violations": [],
        "summary_status": "HTTP 429 Rate Limit",
    }

    verdict = explain_run(record, key=key)

    assert verdict["label"] == LABEL_PROVIDER_FAILURE
    assert "external provider infrastructure failure" in verdict["headline"]
    assert "HTTP 429 Rate Limit" in verdict["what_happened"]
    assert "not an algorithmic or reasoning error" in verdict["why_it_matters"]


def test_explain_verdict_provider_failure_timeout():
    """Verifies PROVIDER_FAILURE on provider timeout."""
    key = {"expected_outcome": "FEASIBLE"}
    record = {
        "mode": 2,
        "mode_name": "Mode 2 (Schema LLM)",
        "feasibility": "NOT_EVALUABLE",
        "normalization_status": "API_FAILURE",
        "normalization_errors": ["API request timeout after 30.0s deadline exceeded"],
        "violations": [],
        "summary_status": "Timeout",
    }

    verdict = explain_run(record, key=key)

    assert verdict["label"] == LABEL_PROVIDER_FAILURE
    assert "Timeout" in verdict["what_happened"]


def test_explain_verdict_provider_failure_truncation():
    """Verifies PROVIDER_FAILURE on generation truncation (max tokens)."""
    key = {"expected_outcome": "FEASIBLE"}
    record = {
        "mode": 1,
        "mode_name": "Mode 1 (Raw LLM)",
        "feasibility": "NOT_EVALUABLE",
        "normalization_status": "TRUNCATION_FAILURE",
        "normalization_errors": ["Response cut off midway through JSON output"],
        "violations": [],
        "summary_status": "TRUNCATION",
    }

    verdict = explain_run(record, key=key)

    assert verdict["label"] == LABEL_PROVIDER_FAILURE
    assert "Truncation" in verdict["what_happened"]


# =============================================================================
# 9. Label Tests: NOT_GRADED
# =============================================================================

def test_explain_verdict_not_graded_no_key():
    """Verifies NOT_GRADED when key is None."""
    record = {
        "mode": 4,
        "mode_name": "Mode 4 (Neuro-Symbolic)",
        "feasibility": "PASS",
        "recomputed_cost_usd": 121.47,
        "claimed_cost_usd": 121.47,
        "normalization_status": "SUCCESS",
        "summary_status": "FEASIBLE_OPTIMAL",
    }

    verdict = explain_run(record, key=None)

    assert verdict["label"] == LABEL_NOT_GRADED
    assert "completed without an evaluation ground-truth key" in verdict["headline"]
    assert "$121.47" in verdict["what_happened"]
    assert "No ground-truth manifest record was matched" in verdict["what_happened"]


def test_explain_verdict_not_graded_empty_outcome_key():
    """Verifies NOT_GRADED when key exists but expected_outcome is None or NOT_GRADED."""
    key = {"expected_outcome": "NOT_GRADED"}
    record = {
        "mode": 3,
        "mode_name": "Mode 3 (Symbolic Local)",
        "feasibility": "PASS",
        "recomputed_cost_usd": 85.00,
        "normalization_status": "SUCCESS",
        "summary_status": "FEASIBLE",
    }

    verdict = explain_run(record, key=key)

    assert verdict["label"] == LABEL_NOT_GRADED
    assert "unkeyed query" in verdict["headline"]
    assert "$85.00" in verdict["what_happened"]


# =============================================================================
# 10. Consistency & Realism Tests (Terminal vs Dashboard)
# =============================================================================

def test_terminal_and_dashboard_string_identity(manifest_key_q01):
    """Asserts that terminal formatters and dashboard view consume identical fields from explain_run."""
    record = {
        "mode": 4,
        "mode_name": "Mode 4 (Neuro-Symbolic)",
        "feasibility": "PASS",
        "recomputed_cost_usd": 121.47,
        "claimed_cost_usd": 121.47,
        "normalization_status": "SUCCESS",
        "requirements": {
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": 300.0,
        },
        "violations": [],
        "summary_status": "FEASIBLE_OPTIMAL",
    }

    # Canonical dictionary
    verdict = explain_run(record, key=manifest_key_q01)

    # Terminal box formatting
    box_output = format_plain_english_box(verdict)
    clean_box = " ".join([line.strip("| \t") for line in box_output.splitlines() if not line.startswith("+")])
    assert verdict["label"] in box_output
    assert verdict["headline"] in clean_box or " ".join(verdict["headline"].split()) in " ".join(clean_box.split())
    assert verdict["why_it_matters"] in clean_box or " ".join(verdict["why_it_matters"].split()) in " ".join(clean_box.split())

    # Terminal summary formatting
    summary_output = format_plain_english_summary([record], key=manifest_key_q01)
    assert verdict["label"] in summary_output
    assert verdict["headline"] in summary_output
    assert DISCLAIMER_RULE in summary_output

    # Verify no winner / ranking / score is printed in summary
    assert "winner" not in summary_output.lower()
    assert "ranking" not in summary_output.lower()
    assert "score:" not in summary_output.lower()


def test_manifest_entry_lookup():
    """Verifies that find_manifest_entry accurately matches query prefixes and IDs."""
    # Matches Q01 by number or prefix
    entry_1 = find_manifest_entry("Q01")
    assert entry_1 is not None
    assert "Q01" in entry_1.get("query_id", "")

    entry_num = find_manifest_entry("1")
    assert entry_num is not None
    assert "Q01" in entry_num.get("query_id", "")

    entry_text = find_manifest_entry("Deploy 8 vCPUs and 16GB RAM for under $300 on AWS")
    assert entry_text is not None


def test_all_labels_and_colors_complete():
    """Verifies that all 9 canonical labels have associated colors, tooltips, and legends."""
    assert len(ALL_LABELS) == 9
    for lbl in ALL_LABELS:
        assert lbl in LABEL_COLORS
        assert lbl in LABEL_TOOLTIPS
        assert lbl in LABEL_LEGEND
        assert LABEL_COLORS[lbl].startswith("#")
        assert len(LABEL_TOOLTIPS[lbl]) > 10
        assert len(LABEL_LEGEND[lbl]) > 10


# =============================================================================
# 11. Mock Mode Banner Tests (--mock-llm / is_mock)
# =============================================================================

def test_mock_llm_banners_and_boxes(manifest_key_q01):
    """Verifies that --mock-llm / is_mock=True displays MOCK DATA banner in header and all boxes."""
    record_mock = {
        "mode": 1,
        "mode_name": "Mode 1: Raw LLM",
        "feasibility": "FAIL",
        "normalization_status": "UNPARSEABLE",
        "is_mock": True,
        "provider": "Mock",
        "summary_status": "mock_fixture",
    }
    record_live = {
        "mode": 1,
        "mode_name": "Mode 1: Raw LLM",
        "feasibility": "FAIL",
        "normalization_status": "UNPARSEABLE",
        "is_mock": False,
        "provider": "Groq",
        "summary_status": "live_groq",
    }

    # 1. explain_run output
    v_mock = explain_run(record_mock, key=manifest_key_q01)
    v_live = explain_run(record_live, key=manifest_key_q01)
    assert v_mock["is_mock"] is True
    assert v_live["is_mock"] is False

    # 2. Plain English Box output
    box_mock = format_plain_english_box(v_mock)
    box_live = format_plain_english_box(v_live)
    assert "[MOCK DATA]" in box_mock
    assert "[MOCK DATA - OFFLINE BENCHMARK FIXTURE - ZERO LIVE CALLS]" in box_mock
    assert "[MOCK DATA]" not in box_live
    assert "OFFLINE BENCHMARK FIXTURE" not in box_live

    # 3. Plain English Summary output
    sum_mock = format_plain_english_summary([record_mock], key=manifest_key_q01)
    sum_live = format_plain_english_summary([record_live], key=manifest_key_q01)
    assert "[MOCK DATA]" in sum_mock
    assert "[MOCK DATA]" not in sum_live


# =============================================================================
# 12. Stated Parameter Extraction & Mismatch Tests (Modes 1 & 2)
# =============================================================================

def test_stated_parameters_not_stated_handling(manifest_key_dr_q14):
    """Verifies that unstated parameters in Mode 1 & 2 produce 'not stated' without guessing or inheriting Mode 3."""
    from run_all_modes_comparative import (
        extract_stated_parameters_from_prose,
        extract_stated_parameters_from_json,
    )

    # Mode 1 prose with only regions and SLA stated (budget & latency omitted)
    prose_text = "Deploy DR between us-east-1 and us-central1 with 99.99% SLA."
    stated_m1 = extract_stated_parameters_from_prose(prose_text)
    assert stated_m1.get("sla_availability_pct") == 99.99
    assert stated_m1.get("budget_max_usd") is None
    assert stated_m1.get("latency_max_ms") is None

    rec_m1 = {
        "mode": 1,
        "mode_name": "Mode 1: Raw LLM",
        "feasibility": "FAIL",
        "normalization_status": "AMBIGUOUS",
        "requirements": stated_m1,
        "summary_status": "Missing budget",
    }
    v_m1 = explain_run(rec_m1, key=manifest_key_dr_q14)
    # Explanation note must say 'it answered for budget not stated; you asked $600.00'
    assert "budget not stated" in v_m1["what_happened"] or "not stated" in v_m1["what_happened"]
    assert "$600.00" in v_m1["what_happened"]

    # Mode 2 JSON with budget=$100 (differs from key $600)
    json_obj = {
        "primary_region": "us-east-1",
        "secondary_region": "us-central1",
        "budget_max_usd": 100.0,
        "achieved_sla_pct": 99.99,
    }
    stated_m2 = extract_stated_parameters_from_json(json_obj)
    assert stated_m2.get("budget_max_usd") == 100.0
    rec_m2 = {
        "mode": 2,
        "mode_name": "Mode 2: Schema LLM",
        "feasibility": "FAIL",
        "normalization_status": "SUCCESS",
        "requirements": stated_m2,
        "summary_status": "Rejected",
    }
    v_m2 = explain_run(rec_m2, key=manifest_key_dr_q14)
    assert "it answered for budget $100.00; you asked $600.00" in v_m2["what_happened"]


# =============================================================================
# 13. Per-Mode Success Rates (Never Pooled Across Modes)
# =============================================================================

def test_per_mode_metrics_never_pooled(manifest_key_q01):
    """Verifies that format_plain_english_summary computes per-mode metrics with sample size n."""
    r1 = {
        "mode": 1,
        "mode_name": "Mode 1: Raw LLM",
        "normalization_status": "UNPARSEABLE",
        "feasibility": "FAIL",
        "task_outcome": "UNGRADED-UNPARSEABLE",
    }
    r2 = {
        "mode": 2,
        "mode_name": "Mode 2: Schema LLM",
        "normalization_status": "SUCCESS",
        "feasibility": "PASS",
        "task_outcome": "SUCCESS",
        "requirements": {"required_vcpus": 8, "required_ram_gb": 16.0, "budget_max_usd": 300.0},
    }
    r3 = {
        "mode": 3,
        "mode_name": "Mode 3: Pure Symbolic",
        "normalization_status": "SUCCESS",
        "feasibility": "PASS",
        "task_outcome": "SUCCESS",
        "requirements": {"required_vcpus": 8, "required_ram_gb": 16.0, "budget_max_usd": 300.0},
    }
    r4 = {
        "mode": 4,
        "mode_name": "Mode 4: Neuro-Symbolic",
        "normalization_status": "SUCCESS",
        "feasibility": "PASS",
        "task_outcome": "SUCCESS",
        "requirements": {"required_vcpus": 8, "required_ram_gb": 16.0, "budget_max_usd": 300.0},
    }

    summary = format_plain_english_summary([r1, r2, r3, r4], key=manifest_key_q01)

    # Must contain per-mode breakdown headers
    assert "BENCHMARK METRICS BY EXECUTION MODE (NEVER POOLED ACROSS MODES)" in summary
    assert "Mode 1: Raw LLM (n=1)" in summary
    assert "Mode 2: Schema LLM (n=1)" in summary
    assert "Mode 3: Pure Symbolic (n=1)" in summary
    assert "Mode 4: Neuro-Symbolic (n=1)" in summary

    # Mode 1: 0% strict, unparseable excluded in lenient
    assert "Task Success (Strict)   : 0.0% (0/1)" in summary
    assert "1 unparseable excluded" in summary

    # Modes 2, 3, 4: 100% strict and lenient
    assert "Task Success (Strict)   : 100.0% (1/1)" in summary
    assert "Interpretation Accuracy : 100.0% (1/1)" in summary


# =============================================================================
# 14. Unicode Hyphen Normalization & Labelled Total Tests (Mode 1 DR)
# =============================================================================

def test_unicode_hyphen_normalization_mode1_dr(manifest_key_dr_q14):
    """Verifies that Unicode hyphens (U+2011 non-breaking, U+2013 en-dash) are normalised and labelled total is extracted."""
    from src.semantic.normalizer import OutputNormalizer

    # Real LLM style prose containing Unicode non-breaking hyphens \u2011, en-dash \u2013, narrow spaces \u202f, and base component prices
    prose = (
        "**Provider & Regions**\n"
        "- **AWS \u2013 us\u2011east\u20111** (base DR cost\u202f$120\u202f/\u202fmo)\n"
        "- **GCP \u2013 us\u2011central1** (base DR cost\u202f$115\u202f/\u202fmo)\n\n"
        "**Latency & SLA**\n"
        "- Peer latency\u202f=\u202f32\u202fms \u2264\u202f50\u202fms\n"
        "- Combined SLA\u202f=\u202f99.999975%\n\n"
        "**Cost Calculation**\n"
        "| Item | Cost (USD) |\n"
        "|------|------------|\n"
        "| AWS us\u2011east\u20111 base DR | $120.00 |\n"
        "| GCP us\u2011central1 base DR | $115.00 |\n"
        "| Latency surcharge (32 ms \u00d7 $0.25) | $8.00 |\n"
        "| **Total Monthly Cost** | **$243.00** |\n\n"
        "**Recommendation**\n"
        "Deploy the disaster\u2011recovery pair across **AWS us\u2011east\u20111** and **GCP us\u2011central1** "
        "with a total cost of **$243 per month**."
    )

    norm_status, ext_cost, norm_decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=prose,
        problem_type="Z3_Graph_Disaster_Recovery",
        contract_data=manifest_key_dr_q14,
    )

    assert norm_status == NormalizationStatus.SUCCESS
    assert ext_cost == 243.0  # Must be labelled total $243.00, NEVER component price $120.00
    assert norm_decision is not None
    assert norm_decision.get("primary_region") == "us-east-1"
    assert norm_decision.get("secondary_region") == "us-central1"


def test_mode1_cost_from_chosen_plan_not_alternative():
    """Verifies that cost is taken from the chosen/recommended plan, not alternative options."""
    from src.semantic.normalizer import OutputNormalizer

    prose = (
        "Option 1 (Alternative rejected): 4x t3.xlarge would cost $485.88 monthly, which exceeds budget.\n"
        "Recommendation (Chosen Plan):\n"
        "We recommend deploying 2x t3.xlarge instances on AWS for 8 vCPUs and 16GB RAM.\n"
        "Estimated total monthly cost: $242.94/month."
    )

    norm_status, ext_cost, norm_decision, errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=prose,
        problem_type="ILP_VM_Allocation",
    )

    assert norm_status == NormalizationStatus.SUCCESS
    assert ext_cost == 242.94  # Must be $242.94 from chosen plan, not $485.88 from alternative


def test_metrics_omitted_when_no_graded_queries():
    """Verifies that format_plain_english_summary prints nothing for benchmark metrics when queries are ungraded."""
    r_ungraded = {
        "mode": 1,
        "mode_name": "Mode 1: Raw LLM",
        "normalization_status": "SUCCESS",
        "feasibility": "PASS",
        "task_outcome": "NOT_GRADED",
        "expected_outcome": "NOT_GRADED",
    }

    summary = format_plain_english_summary([r_ungraded], key={"expected_outcome": "NOT_GRADED"})
    assert "BENCHMARK METRICS BY EXECUTION MODE" not in summary


def test_stage_4_5_report_and_source_agreement(manifest_key_q01):
    """Verifies that Stage 4.5 report source header and ExplanationSource agree."""
    from src.semantic.explainer import FinOpsExplainer
    from src.semantic.schemas import CloudOptimizationContract

    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        budget_max_usd=300.0,
        required_vcpus=8,
        required_ram_gb=16.0,
    )
    solver_res = {
        "status": "OPTIMAL",
        "solver": "SciPy_MILP_HiGHS",
        "total_monthly_cost_usd": 121.47,
        "allocated_vms": [{"instance_type": "t3.medium", "provider": "AWS", "count": 4, "vcpus_per_vm": 2, "ram_gb_per_vm": 4.0, "monthly_cost": 121.47}],
    }
    check = {"feasible_against_contract": True, "violations": []}

    # Offline / template mode
    report_text, exp_source = FinOpsExplainer.generate_report_with_source(
        contract=contract,
        solver_result=solver_res,
        check_result=check,
        enable_llm_explainer=False,
        offline=True,
    )
    assert exp_source == ExplanationSource.LOCAL_TEMPLATE
    assert "Source: Local Rule-Based Template" in report_text


def test_stage_4_5_live_provider_agreement(monkeypatch):
    """Verifies that a successful live Groq call logs live provider and includes response ID without fallback."""
    from src.semantic.explainer import FinOpsExplainer
    from src.semantic.schemas import CloudOptimizationContract

    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        budget_max_usd=300.0,
        required_vcpus=8,
        required_ram_gb=16.0,
    )
    solver_res = {
        "status": "OPTIMAL",
        "solver": "SciPy_MILP_HiGHS",
        "total_monthly_cost_usd": 121.48,
        "allocated_vms": [{"instance_type": "t3.medium", "provider": "AWS", "count": 4, "vcpus_per_vm": 2, "ram_gb_per_vm": 4.0, "monthly_cost": 121.48}],
    }
    check = {"feasible_against_contract": True, "violations": []}

    mock_recs = [
        "Commit to AWS 1-Year Savings Plans to achieve 30% discounts.",
        "Set CloudWatch alerts at 80% of budget.",
        "Track CPU credit balances on t3.medium instances.",
    ]
    monkeypatch.setattr(
        FinOpsExplainer,
        "generate_llm_recommendations",
        classmethod(lambda cls, c, s, **kw: (mock_recs, "chatcmpl-test-id-12345", 1.25, ExplanationSource.LIVE_PROVIDER)),
    )

    report_text, exp_source = FinOpsExplainer.generate_report_with_source(
        contract=contract,
        solver_result=solver_res,
        check_result=check,
        enable_llm_explainer=True,
        offline=False,
    )
    assert exp_source == ExplanationSource.LIVE_PROVIDER
    assert "response_id: chatcmpl-test-id-12345" in report_text
    assert "Fallback" not in report_text
    assert "Live provider call unavailable or failed" not in report_text


def test_success_rates_graded_only():
    """Verifies that ungraded runs do not output 0.0% success rate tables."""
    from src.benchmarks.visualizer import BenchmarkVisualizer

    # Mock records for an ungraded query
    r_ungraded = CanonicalExecutionRecord(
        mode=1,
        mode_name="Mode 1: Raw LLM",
        original_query="Ungraded custom query",
        requirement_source="neural_contract",
        problem_type="ILP_VM_Allocation",
        requirements={"query_id": "Q_UNGRADED_CUSTOM"},
        execution_path="Path",
        normalization_status=NormalizationStatus.SUCCESS,
        feasibility=FeasibilityStatus.PASS,
        task_outcome="NOT_GRADED",
    )
    manifest = [{"query_id": "Q_UNGRADED_CUSTOM", "category": "COLLOQUIAL_HINGLISH", "expected_outcome": "NOT_GRADED"}]
    
    analytics = BenchmarkVisualizer.compute_analytics([r_ungraded], manifest)
    assert analytics["category_breakdown"]["COLLOQUIAL_HINGLISH"]["total_queries"] == 0
    text_summary = BenchmarkVisualizer.render_cli_summary(analytics)
    assert "0.0% (0/0)" not in text_summary
    assert "0.0% (0/4)" not in text_summary
    assert "TASK SUCCESS RATE BY QUERY CATEGORY" not in text_summary


def test_alignment_line_both_infeasible(capsys):
    """Verifies that when both Mode 3 and Mode 4 are infeasible, 'Both infeasible' is printed and no $0.00 plan is described."""
    import run_all_modes_comparative as rmc

    m3_rec = CanonicalExecutionRecord(
        mode=3,
        mode_name="Mode 3: Pure Symbolic",
        original_query="Infeasible query",
        requirement_source="parsed_contract",
        problem_type="ILP_VM_Allocation",
        requirements={},
        execution_path="Local SCOPE",
        normalization_status=NormalizationStatus.SOLVER_INFEASIBLE,
        feasibility=FeasibilityStatus.FAIL,
        task_outcome="SUCCESS",
        recomputed_cost_usd=None,
    )
    m4_rec = CanonicalExecutionRecord(
        mode=4,
        mode_name="Mode 4: Neuro-Symbolic",
        original_query="Infeasible query",
        requirement_source="neural_contract",
        problem_type="ILP_VM_Allocation",
        requirements={},
        execution_path="Groq API",
        normalization_status=NormalizationStatus.SOLVER_INFEASIBLE,
        feasibility=FeasibilityStatus.FAIL,
        task_outcome="SUCCESS",
        recomputed_cost_usd=None,
    )

    rmc.print_comparison_table([m3_rec, m4_rec])
    captured = capsys.readouterr().out
    assert "Both infeasible" in captured
    assert "$0.00" not in captured
    assert "Fully Aligned" not in captured


def test_alignment_line_coincidental_cost_match(capsys):
    """Verifies that when costs match ($243.00) but SLA differs, Coincidental Cost Match is printed."""
    import run_all_modes_comparative as rmc
    from src.semantic.schemas import CloudOptimizationContract

    m3_c = CloudOptimizationContract(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "GCP"],
        budget_max_usd=600.0,
        sla_availability_pct=99.9,
    )
    m4_c = CloudOptimizationContract(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "GCP"],
        budget_max_usd=600.0,
        sla_availability_pct=99.99,
    )

    m3_rec = CanonicalExecutionRecord(
        mode=3,
        mode_name="Mode 3: Pure Symbolic",
        original_query="DR query",
        requirement_source="parsed_contract",
        problem_type="Z3_Graph_Disaster_Recovery",
        requirements=m3_c.model_dump(),
        execution_path="Local SCOPE",
        normalization_status=NormalizationStatus.SUCCESS,
        feasibility=FeasibilityStatus.PASS,
        task_outcome="SUCCESS",
        recomputed_cost_usd=243.0,
    )
    m4_rec = CanonicalExecutionRecord(
        mode=4,
        mode_name="Mode 4: Neuro-Symbolic",
        original_query="DR query",
        requirement_source="neural_contract",
        problem_type="Z3_Graph_Disaster_Recovery",
        requirements=m4_c.model_dump(),
        execution_path="Groq API",
        normalization_status=NormalizationStatus.SUCCESS,
        feasibility=FeasibilityStatus.PASS,
        task_outcome="SUCCESS",
        recomputed_cost_usd=243.0,
    )

    rmc.print_comparison_table([m3_rec, m4_rec], m3_contract=m3_c, m4_contract=m4_c)
    captured = capsys.readouterr().out
    assert "Coincidental Cost Match with Interpretation Mismatch" in captured
    assert "Fully Aligned" not in captured


def test_explainer_wording_and_utf8_encoding():
    """Verifies that 'FEASIBLE & CERTIFIED' is replaced, and UTF-8 hyphens/percents are preserved without '?'."""
    from src.semantic.explainer import FinOpsExplainer
    from src.semantic.schemas import CloudOptimizationContract

    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS"],
        budget_max_usd=300.0,
        required_vcpus=8,
        required_ram_gb=16.0,
    )
    solver_res = {
        "status": "OPTIMAL",
        "solver": "SciPy_MILP_HiGHS",
        "total_monthly_cost_usd": 121.48,
        "allocated_vms": [{"instance_type": "t3.medium", "provider": "AWS", "count": 4, "vcpus_per_vm": 2, "ram_gb_per_vm": 4.0, "monthly_cost": 121.48}],
    }
    check = {"feasible_against_contract": True, "violations": []}

    report = FinOpsExplainer.generate_report(contract, solver_res, check_result=check, enable_llm_explainer=False)
    assert "FEASIBLE & CERTIFIED" not in report
    assert "FEASIBLE (Passed Independent Verification)" in report

    # Test Unicode en-dash / quotes / percent preservation
    raw_advice = (
        "1. Purchase a 1–year Savings Plan to save ~30–40% on compute.\n"
        "2. Set alerts at 80% threshold to prevent budget overruns.\n"
        "3. Right-size instance capacity for sustained workloads."
    )
    cleaned = FinOpsExplainer._clean_llm_recommendations(raw_advice)
    for rec in cleaned:
        assert "?" not in rec
        assert "%" in rec or "workloads" in rec
    assert "1-year" in cleaned[0] or "1–year" in cleaned[0]
    assert "30-40%" in cleaned[0] or "30–40%" in cleaned[0]



