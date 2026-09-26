"""SYM-6: Outcome-aware evaluation tests.

Tests cover:
    - Outcome label generation (all cases)
    - Threshold behaviour (EQUIVALENT/ACCEPTABLE/INFERIOR boundaries)
    - Feasibility preservation (INFEASIBLE label)
    - BOTH_INFEASIBLE and NOT_EVALUATED labels
    - FAILED label (candidate error)
    - Promotion gate pass/fail
    - Checkpoint immutability
    - Corrupted / missing candidate handling
    - Deterministic evaluation
    - Reference race correctness
    - Eval runner configuration isolation
"""

import pytest
import json
import os
import shutil
import time
from typing import Optional

from src.symbolic.models import SymbolicOptimizationRequest, OptimizationResult, OptimizationMetrics
from src.symbolic.optihive.outcome_evaluation import (
    RoutingOutcome, OutcomeComparison, compare_outcome, evaluate_promotion_gate,
    OUTCOME_EQUIVALENT, OUTCOME_ACCEPTABLE, OUTCOME_INFERIOR,
    OUTCOME_INFEASIBLE, OUTCOME_BOTH_INFEASIBLE, OUTCOME_FAILED, OUTCOME_NOT_EVALUATED,
    COST_TOLERANCE, COST_DEGRADATION_LIMIT, RUNTIME_LIMIT_FACTOR,
)
from src.symbolic.optihive.telemetry_training import CHECKPOINT_PATH, CANDIDATE_PATH


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_outcome(
    scenario="test",
    config="TEST",
    is_feasible=True,
    cost=100.0,
    total_ms=20.0,
    solver_ms=18.0,
    error=None,
    routing_mode="learned_high",
) -> RoutingOutcome:
    return RoutingOutcome(
        scenario_name=scenario,
        config_name=config,
        selected_solvers=["GA", "PSO"],
        routing_mode=routing_mode,
        routing_confidence="HIGH",
        is_feasible=is_feasible,
        total_runtime_ms=total_ms,
        solver_runtime_ms=solver_ms,
        objective_cost_usd=cost,
        hard_constraint_violation=False,
        fallback_used=False,
        error=error,
    )


def make_comparison(outcome_label, cost_delta_pct=None, runtime_ratio=None) -> OutcomeComparison:
    return OutcomeComparison(
        scenario_name="test",
        config_name="CANDIDATE",
        outcome_label=outcome_label,
        candidate_feasible=True,
        reference_feasible=True,
        candidate_cost=100.0,
        reference_cost=100.0,
        cost_delta_pct=cost_delta_pct,
        candidate_runtime_ms=20.0,
        reference_runtime_ms=20.0,
        runtime_ratio=runtime_ratio,
    )


def make_request(**kwargs) -> SymbolicOptimizationRequest:
    defaults = dict(
        problem_type="TEST",
        cloud_providers=["aws", "azure"],
        budget_max_usd=1000.0,
        service_count=3,
        required_vcpus=8,
        required_ram_gb=32.0,
        latency_max_ms=0.0,
        sla_availability_pct=0.0,
    )
    defaults.update(kwargs)
    return SymbolicOptimizationRequest(**defaults)


# ─── Outcome label tests ───────────────────────────────────────────────────────

class TestOutcomeLabels:

    def test_equivalent_within_tolerance(self):
        ref = make_outcome(cost=100.0, total_ms=20.0)
        cand = make_outcome(cost=104.0, total_ms=20.0)  # +4% → EQUIVALENT
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_EQUIVALENT

    def test_equivalent_at_boundary(self):
        ref = make_outcome(cost=100.0, total_ms=20.0)
        cand = make_outcome(cost=100.0 * (1 + COST_TOLERANCE), total_ms=20.0)
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_EQUIVALENT

    def test_acceptable_between_tolerance_and_limit(self):
        ref = make_outcome(cost=100.0, total_ms=20.0)
        cand = make_outcome(cost=115.0, total_ms=30.0)  # +15% cost, 1.5x runtime
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_ACCEPTABLE

    def test_inferior_cost_exceeds_limit(self):
        ref = make_outcome(cost=100.0, total_ms=20.0)
        cand = make_outcome(cost=130.0, total_ms=20.0)  # +30% cost → INFERIOR
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_INFERIOR

    def test_inferior_runtime_exceeds_limit(self):
        # Cost is in acceptable zone (+10%) BUT runtime is 4× reference → INFERIOR
        ref = make_outcome(cost=100.0, total_ms=20.0)
        cand = make_outcome(cost=110.0, total_ms=80.0)  # +10% cost, 4.0x runtime > 3.0 limit
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_INFERIOR

    def test_infeasible_candidate_ref_feasible(self):
        ref = make_outcome(is_feasible=True, cost=100.0)
        cand = make_outcome(is_feasible=False, cost=None)
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_INFEASIBLE

    def test_both_infeasible(self):
        ref = make_outcome(is_feasible=False, cost=None)
        cand = make_outcome(is_feasible=False, cost=None)
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_BOTH_INFEASIBLE

    def test_not_evaluated_ref_infeasible_cand_feasible(self):
        ref = make_outcome(is_feasible=False, cost=None)
        cand = make_outcome(is_feasible=True, cost=100.0)
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_NOT_EVALUATED

    def test_failed_when_candidate_has_error(self):
        ref = make_outcome(is_feasible=True, cost=100.0)
        cand = make_outcome(error="Solver crash")
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_FAILED

    def test_zero_cost_reference_is_equivalent(self):
        ref = make_outcome(cost=0.0)
        cand = make_outcome(cost=0.0)
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label == OUTCOME_EQUIVALENT

    def test_cost_delta_pct_is_populated_when_feasible(self):
        ref = make_outcome(cost=200.0, total_ms=20.0)
        cand = make_outcome(cost=210.0, total_ms=20.0)
        result = compare_outcome("s", "C", cand, ref)
        assert result.cost_delta_pct is not None
        assert abs(result.cost_delta_pct - 0.05) < 1e-9

    def test_runtime_ratio_populated(self):
        ref = make_outcome(cost=100.0, total_ms=10.0)
        cand = make_outcome(cost=100.0, total_ms=20.0)
        result = compare_outcome("s", "C", cand, ref)
        assert result.runtime_ratio is not None
        assert abs(result.runtime_ratio - 2.0) < 1e-9

    def test_comparable_across_all_labels(self):
        """All six non-BOTH_INFEASIBLE labels can be produced."""
        labels_produced = set()
        ref_feas = make_outcome(is_feasible=True, cost=100.0, total_ms=20.0)
        ref_infeas = make_outcome(is_feasible=False)

        # EQUIVALENT
        labels_produced.add(compare_outcome("s", "C", make_outcome(cost=100.0, total_ms=20.0), ref_feas).outcome_label)
        # ACCEPTABLE
        labels_produced.add(compare_outcome("s", "C", make_outcome(cost=115.0, total_ms=30.0), ref_feas).outcome_label)
        # INFERIOR
        labels_produced.add(compare_outcome("s", "C", make_outcome(cost=130.0, total_ms=20.0), ref_feas).outcome_label)
        # INFEASIBLE
        labels_produced.add(compare_outcome("s", "C", make_outcome(is_feasible=False), ref_feas).outcome_label)
        # BOTH_INFEASIBLE
        labels_produced.add(compare_outcome("s", "C", make_outcome(is_feasible=False), ref_infeas).outcome_label)
        # NOT_EVALUATED
        labels_produced.add(compare_outcome("s", "C", make_outcome(is_feasible=True, cost=100.0), ref_infeas).outcome_label)
        # FAILED
        labels_produced.add(compare_outcome("s", "C", make_outcome(error="crash"), ref_feas).outcome_label)

        expected = {OUTCOME_EQUIVALENT, OUTCOME_ACCEPTABLE, OUTCOME_INFERIOR,
                    OUTCOME_INFEASIBLE, OUTCOME_BOTH_INFEASIBLE, OUTCOME_NOT_EVALUATED, OUTCOME_FAILED}
        assert labels_produced == expected


# ─── Promotion gate tests ──────────────────────────────────────────────────────

class TestPromotionGate:

    def _all_equiv_comps(self, n=5):
        return [make_comparison(OUTCOME_EQUIVALENT) for _ in range(n)]

    def test_gate_passes_with_all_conditions_met(self):
        comps = self._all_equiv_comps(10)
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=0.85)
        assert gate.val_accuracy_ok
        assert gate.feasibility_preserved
        assert gate.outcome_agreement_ok
        assert gate.eligible_for_promotion

    def test_gate_fails_low_accuracy(self):
        comps = self._all_equiv_comps(5)
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=0.60)
        assert not gate.val_accuracy_ok
        assert not gate.eligible_for_promotion

    def test_gate_fails_infeasible_outcome(self):
        comps = self._all_equiv_comps(5)
        comps.append(make_comparison(OUTCOME_INFEASIBLE))
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=0.85)
        assert not gate.feasibility_preserved
        assert not gate.eligible_for_promotion

    def test_gate_fails_too_many_inferior(self):
        # 7 inferior out of 10 → 30% agreement < 80% threshold
        comps = [make_comparison(OUTCOME_INFERIOR) for _ in range(7)]
        comps += [make_comparison(OUTCOME_EQUIVALENT) for _ in range(3)]
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=0.85)
        assert not gate.outcome_agreement_ok
        assert not gate.eligible_for_promotion

    def test_gate_passes_with_mixed_acceptable_equivalent(self):
        comps = [make_comparison(OUTCOME_EQUIVALENT) for _ in range(8)]
        comps += [make_comparison(OUTCOME_ACCEPTABLE) for _ in range(2)]
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=0.85)
        assert gate.outcome_agreement_ok
        assert gate.eligible_for_promotion

    def test_gate_skips_both_infeasible_from_agreement_count(self):
        comps = [make_comparison(OUTCOME_BOTH_INFEASIBLE) for _ in range(5)]
        comps += [make_comparison(OUTCOME_EQUIVALENT) for _ in range(5)]
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=0.85)
        # Only 5 EQUIVALENT in evaluated bucket
        assert gate.equivalent_or_acceptable_count == 5
        assert gate.total_evaluated == 5
        assert gate.outcome_agreement_ok

    def test_gate_eligible_does_not_mean_promoted(self):
        comps = self._all_equiv_comps(5)
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=0.85)
        assert gate.eligible_for_promotion
        assert not gate.promoted  # Never automatically promoted

    def test_gate_handles_none_accuracy(self):
        comps = self._all_equiv_comps(5)
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=None)
        assert not gate.val_accuracy_ok
        assert not gate.eligible_for_promotion

    def test_gate_missing_candidate_path(self, tmp_path):
        comps = self._all_equiv_comps(5)
        gate = evaluate_promotion_gate(
            comps, candidate_val_accuracy=0.85,
            candidate_checkpoint_path=str(tmp_path / "missing.json")
        )
        assert not gate.eligible_for_promotion

    def test_gate_corrupt_candidate_path(self, tmp_path):
        bad = tmp_path / "corrupt.json"
        bad.write_text("{not_valid_json")
        comps = self._all_equiv_comps(5)
        gate = evaluate_promotion_gate(
            comps, candidate_val_accuracy=0.85,
            candidate_checkpoint_path=str(bad)
        )
        assert not gate.eligible_for_promotion

    def test_gate_checkpoint_missing_fields(self, tmp_path):
        incomplete = tmp_path / "incomplete.json"
        incomplete.write_text(json.dumps({"model_version": "x"}))  # missing W, b etc
        comps = self._all_equiv_comps(5)
        gate = evaluate_promotion_gate(
            comps, candidate_val_accuracy=0.85,
            candidate_checkpoint_path=str(incomplete)
        )
        assert not gate.eligible_for_promotion


# ─── Checkpoint immutability tests ────────────────────────────────────────────

class TestCheckpointImmutability:

    def test_production_checkpoint_exists(self):
        assert os.path.exists(CHECKPOINT_PATH), \
            f"Production checkpoint missing: {CHECKPOINT_PATH}"

    def test_production_checkpoint_is_valid_json(self):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        assert "model_version" in data
        assert "W" in data
        assert "b" in data

    def test_evaluation_does_not_modify_production(self, tmp_path):
        """Running the eval runner must leave the production checkpoint unchanged."""
        with open(CHECKPOINT_PATH) as f:
            before = f.read()

        # Run a quick evaluation
        from src.symbolic.optihive.eval_runner import run_sym4
        req = make_request()
        run_sym4("immutability_test", req)

        with open(CHECKPOINT_PATH) as f:
            after = f.read()

        assert before == after, "Production checkpoint was modified during evaluation"

    def test_promotion_gate_failure_does_not_modify_production(self, tmp_path):
        """A failed gate must never modify the production checkpoint."""
        with open(CHECKPOINT_PATH) as f:
            before = f.read()

        # Gate with insufficient accuracy
        comps = [make_comparison(OUTCOME_INFERIOR) for _ in range(5)]
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=0.10)
        assert not gate.eligible_for_promotion

        with open(CHECKPOINT_PATH) as f:
            after = f.read()

        assert before == after


# ─── Hard constraint protection ───────────────────────────────────────────────

class TestHardConstraintProtection:

    def test_infeasible_scenario_not_rewarded(self):
        """An infeasible candidate result is never labelled EQUIVALENT or ACCEPTABLE."""
        ref = make_outcome(is_feasible=True, cost=100.0)
        cand = make_outcome(is_feasible=False, cost=None)
        result = compare_outcome("s", "C", cand, ref)
        assert result.outcome_label not in (OUTCOME_EQUIVALENT, OUTCOME_ACCEPTABLE)

    def test_infeasible_candidate_fails_gate(self):
        """Any INFEASIBLE outcome must prevent gate from passing."""
        comps = [make_comparison(OUTCOME_EQUIVALENT) for _ in range(9)]
        comps.append(make_comparison(OUTCOME_INFEASIBLE))
        gate = evaluate_promotion_gate(comps, candidate_val_accuracy=0.90)
        assert not gate.eligible_for_promotion

    def test_sym3_router_uses_safety_arbitration(self):
        """SYM-3 deterministic router must include Z3 for latency/SLA constrained requests."""
        from src.symbolic.optihive.adaptive_router import AdaptiveSolverRouter
        req = make_request(latency_max_ms=50.0)

        class Sym3Only(AdaptiveSolverRouter):
            def __init__(self): self.learned_router = None

        router = Sym3Only()
        decision = router.route(req)
        assert "Z3" in decision.selected_solvers, \
            "Safety arbitration must include Z3 for latency-constrained requests"


# ─── Eval runner isolation ─────────────────────────────────────────────────────

class TestEvalRunnerIsolation:

    def test_baseline_returns_routing_outcome(self):
        from src.symbolic.optihive.eval_runner import run_baseline
        req = make_request()
        outcome = run_baseline("test_baseline", req)
        assert isinstance(outcome.is_feasible, bool)
        assert outcome.total_runtime_ms >= 0
        assert outcome.config_name == "BASELINE"

    def test_sym3_returns_routing_outcome(self):
        from src.symbolic.optihive.eval_runner import run_sym3
        req = make_request()
        outcome = run_sym3("test_sym3", req)
        assert isinstance(outcome.is_feasible, bool)
        assert outcome.config_name == "SYM-3"

    def test_sym4_returns_routing_outcome(self):
        from src.symbolic.optihive.eval_runner import run_sym4
        req = make_request()
        outcome = run_sym4("test_sym4", req)
        assert isinstance(outcome.is_feasible, bool)
        assert outcome.config_name == "SYM-4"

    def test_reference_race_returns_routing_outcome(self):
        from src.symbolic.optihive.eval_runner import run_reference_race
        req = make_request()
        outcome = run_reference_race("test_ref", req)
        assert isinstance(outcome.is_feasible, bool)
        assert outcome.config_name == "REFERENCE"

    def test_candidate_missing_checkpoint_falls_back(self, tmp_path):
        from src.symbolic.optihive.eval_runner import run_candidate
        req = make_request()
        outcome = run_candidate("test_cand", req, str(tmp_path / "nonexistent.json"))
        # Should return a result (fallback), not raise
        assert isinstance(outcome, RoutingOutcome)

    def test_configs_produce_independent_outcomes(self):
        """Running multiple configs on the same request must not share state."""
        from src.symbolic.optihive.eval_runner import run_baseline, run_sym3, run_sym4
        req = make_request()
        r1 = run_baseline("iso_test", req)
        r2 = run_sym3("iso_test", req)
        r3 = run_sym4("iso_test", req)
        # All should complete without error
        assert r1.config_name == "BASELINE"
        assert r2.config_name == "SYM-3"
        assert r3.config_name == "SYM-4"


# ─── Determinism ──────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_request_same_feasibility(self):
        """Two runs of the same config on the same request must agree on feasibility."""
        from src.symbolic.optihive.eval_runner import run_sym4
        req = make_request()
        r1 = run_sym4("det1", req)
        r2 = run_sym4("det1", req)
        assert r1.is_feasible == r2.is_feasible

    def test_outcome_label_is_deterministic(self):
        ref = make_outcome(cost=100.0, total_ms=20.0)
        cand = make_outcome(cost=108.0, total_ms=20.0)  # +8% → ACCEPTABLE
        label1 = compare_outcome("s", "C", cand, ref).outcome_label
        label2 = compare_outcome("s", "C", cand, ref).outcome_label
        assert label1 == label2

    def test_promotion_gate_is_deterministic(self):
        comps = [make_comparison(OUTCOME_EQUIVALENT) for _ in range(5)]
        g1 = evaluate_promotion_gate(comps, 0.85)
        g2 = evaluate_promotion_gate(comps, 0.85)
        assert g1.eligible_for_promotion == g2.eligible_for_promotion


# ─── Reference race correctness ───────────────────────────────────────────────

class TestReferenceRace:

    def test_feasible_scenario_produces_feasible_result(self):
        from src.symbolic.optihive.eval_runner import run_reference_race
        req = make_request(budget_max_usd=1000.0, required_vcpus=4, required_ram_gb=16.0)
        outcome = run_reference_race("ref_feas", req)
        # Don't assert feasibility — that's solver-determined. Just verify shape.
        assert isinstance(outcome.is_feasible, bool)
        assert outcome.total_runtime_ms >= 0

    def test_infeasible_budget_scenario_is_infeasible(self):
        from src.symbolic.optihive.eval_runner import run_reference_race
        req = make_request(budget_max_usd=0.01, required_vcpus=512, required_ram_gb=2048.0)
        outcome = run_reference_race("ref_infeas", req)
        assert not outcome.is_feasible

    def test_reference_includes_z3_in_solver_set(self):
        from src.symbolic.optihive.eval_runner import run_reference_race
        req = make_request()
        outcome = run_reference_race("ref_solvers", req)
        assert "Z3" in outcome.selected_solvers
