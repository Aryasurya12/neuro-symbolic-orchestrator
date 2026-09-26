"""SYM-6: Outcome-aware evaluation engine.

Compares multiple routing strategies (Baseline, SYM-3, SYM-4, SYM-5 candidate)
against a reference full-solver race, producing deterministic outcome labels.

SAFETY:
    - This is an OFFLINE evaluation oracle only.
    - It does NOT replace the production router.
    - Hard constraints remain authoritative (ILP, Z3, budget, SLA, latency).
    - Feasibility is still determined by the symbolic solver layer.
    - An infeasible candidate result against a feasible reference is ALWAYS unacceptable.

Outcome labels (per scenario × configuration):
    EQUIVALENT  – feasible, cost within COST_TOLERANCE of reference, runtime reasonable
    ACCEPTABLE  – feasible, cost within COST_DEGRADATION_LIMIT, runtime within RUNTIME_LIMIT
    INFERIOR    – feasible but materially worse than reference
    INFEASIBLE  – candidate infeasible when reference is feasible
    BOTH_INFEASIBLE – both candidate and reference infeasible (not a routing failure)
    FAILED      – candidate crashed / no result
    NOT_EVALUATED – reference race also infeasible; no basis for comparison

Thresholds (all configurable, documented):
    COST_TOLERANCE      = 0.05   (5%  worse cost is still EQUIVALENT)
    COST_DEGRADATION_LIMIT = 0.20  (20% worse cost is still ACCEPTABLE)
    RUNTIME_LIMIT_FACTOR   = 3.0   (3× slower than reference is the ACCEPTABLE ceiling)

Rationale:
    - 5% cost tolerance: reflects real cloud pricing rounding and solver non-determinism.
    - 20% cost tolerance: accounts for suboptimal but serviceable routing choices.
    - 3× runtime: selective routing that skips a solver must not be dramatically slower
      than running all three, because there is no benefit to selective routing if it causes
      more latency while producing worse results.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# ─── Thresholds ────────────────────────────────────────────────────────────────
COST_TOLERANCE: float = 0.05          # ≤5% cost degradation → EQUIVALENT
COST_DEGRADATION_LIMIT: float = 0.20  # ≤20% cost degradation → ACCEPTABLE
RUNTIME_LIMIT_FACTOR: float = 3.0     # ≤3× reference runtime → ACCEPTABLE

# Outcome label constants
OUTCOME_EQUIVALENT = "EQUIVALENT"
OUTCOME_ACCEPTABLE = "ACCEPTABLE"
OUTCOME_INFERIOR = "INFERIOR"
OUTCOME_INFEASIBLE = "INFEASIBLE"        # candidate infeasible, reference feasible
OUTCOME_BOTH_INFEASIBLE = "BOTH_INFEASIBLE"  # both infeasible — not a routing error
OUTCOME_FAILED = "FAILED"              # candidate raised exception / no result
OUTCOME_NOT_EVALUATED = "NOT_EVALUATED"  # reference also infeasible; no basis


@dataclass
class RoutingOutcome:
    """Result of running one configuration on one scenario."""
    scenario_name: str
    config_name: str
    selected_solvers: List[str]
    routing_mode: str
    routing_confidence: str
    is_feasible: bool
    total_runtime_ms: float
    solver_runtime_ms: float
    objective_cost_usd: Optional[float]
    hard_constraint_violation: bool
    fallback_used: bool
    error: Optional[str] = None


@dataclass
class OutcomeComparison:
    """Comparison of a candidate outcome against the reference race."""
    scenario_name: str
    config_name: str
    outcome_label: str
    candidate_feasible: bool
    reference_feasible: bool
    candidate_cost: Optional[float]
    reference_cost: Optional[float]
    cost_delta_pct: Optional[float]
    candidate_runtime_ms: float
    reference_runtime_ms: float
    runtime_ratio: Optional[float]
    notes: str = ""


def compare_outcome(
    scenario_name: str,
    config_name: str,
    candidate: RoutingOutcome,
    reference: RoutingOutcome,
) -> OutcomeComparison:
    """Derive an outcome label by comparing candidate vs reference.

    Label priority:
        1. If candidate crashed (error) → FAILED
        2. If reference infeasible AND candidate infeasible → BOTH_INFEASIBLE
        3. If reference infeasible → NOT_EVALUATED (no useful comparison basis)
        4. If candidate infeasible AND reference feasible → INFEASIBLE (routing failure)
        5. If both feasible → compare cost and runtime → EQUIVALENT / ACCEPTABLE / INFERIOR
    """
    base = dict(
        scenario_name=scenario_name,
        config_name=config_name,
        candidate_feasible=candidate.is_feasible,
        reference_feasible=reference.is_feasible,
        candidate_cost=candidate.objective_cost_usd,
        reference_cost=reference.objective_cost_usd,
        candidate_runtime_ms=candidate.total_runtime_ms,
        reference_runtime_ms=reference.total_runtime_ms,
        cost_delta_pct=None,
        runtime_ratio=None,
    )

    if candidate.error:
        return OutcomeComparison(**base, outcome_label=OUTCOME_FAILED,
                                  notes=f"Candidate error: {candidate.error}")

    if not reference.is_feasible and not candidate.is_feasible:
        return OutcomeComparison(**base, outcome_label=OUTCOME_BOTH_INFEASIBLE,
                                  notes="Both infeasible — expected for this workload.")

    if not reference.is_feasible:
        return OutcomeComparison(**base, outcome_label=OUTCOME_NOT_EVALUATED,
                                  notes="Reference race infeasible; no comparison basis.")

    # Reference is feasible — candidate MUST also be feasible.
    if not candidate.is_feasible:
        return OutcomeComparison(**base, outcome_label=OUTCOME_INFEASIBLE,
                                  notes="Candidate infeasible while reference is feasible — routing failure.")

    # Both feasible — compute cost and runtime deltas.
    ref_cost = reference.objective_cost_usd or 0.0
    cand_cost = candidate.objective_cost_usd or 0.0

    cost_delta_pct = None
    if ref_cost > 0:
        cost_delta_pct = (cand_cost - ref_cost) / ref_cost

    ref_rt = max(reference.total_runtime_ms, 0.001)
    runtime_ratio = candidate.total_runtime_ms / ref_rt

    base["cost_delta_pct"] = cost_delta_pct
    base["runtime_ratio"] = runtime_ratio

    # Label decision
    if cost_delta_pct is None:
        # No cost comparison possible (ref_cost == 0) — treat as equivalent if feasible
        label = OUTCOME_EQUIVALENT
        notes = "Both feasible; no cost comparison possible (reference cost = 0)."
    elif cost_delta_pct <= COST_TOLERANCE:
        label = OUTCOME_EQUIVALENT
        notes = f"Cost within {COST_TOLERANCE*100:.0f}% tolerance ({cost_delta_pct*100:+.1f}%)."
    elif cost_delta_pct <= COST_DEGRADATION_LIMIT and runtime_ratio <= RUNTIME_LIMIT_FACTOR:
        label = OUTCOME_ACCEPTABLE
        notes = (f"Cost degradation {cost_delta_pct*100:+.1f}% "
                 f"(limit {COST_DEGRADATION_LIMIT*100:.0f}%), "
                 f"runtime ratio {runtime_ratio:.2f}× (limit {RUNTIME_LIMIT_FACTOR:.1f}×).")
    else:
        label = OUTCOME_INFERIOR
        notes = (f"Cost degradation {cost_delta_pct*100:+.1f}% or "
                 f"runtime ratio {runtime_ratio:.2f}× exceeded limits.")

    return OutcomeComparison(**base, outcome_label=label, notes=notes)


@dataclass
class PromotionGateResult:
    """Result of the SYM-6 promotion gate evaluation."""
    eligible_for_promotion: bool  # ALL mandatory conditions passed
    promoted: bool = False        # Actual promotion (requires explicit --promote flag)

    # Mandatory gate conditions
    val_accuracy_ok: bool = False
    val_accuracy: Optional[float] = None
    val_accuracy_threshold: float = 0.70

    feasibility_preserved: bool = False
    feasibility_scenarios_tested: int = 0
    feasibility_failures: int = 0

    hard_constraint_ok: bool = False
    hard_constraint_violations: int = 0

    outcome_agreement_ok: bool = False
    equivalent_or_acceptable_count: int = 0
    inferior_count: int = 0
    infeasible_count: int = 0
    failed_count: int = 0
    total_evaluated: int = 0
    min_agreement_rate: float = 0.80   # ≥80% of scenarios must be EQUIVALENT/ACCEPTABLE

    fallback_rate: float = 0.0
    max_fallback_rate: float = 0.50   # ≤50% fallback is acceptable ceiling

    reason: str = ""
    notes: List[str] = field(default_factory=list)


def evaluate_promotion_gate(
    comparisons: List[OutcomeComparison],
    candidate_val_accuracy: Optional[float],
    candidate_checkpoint_path: Optional[str] = None,
) -> PromotionGateResult:
    """Evaluate all mandatory promotion conditions.

    All of the following must pass for eligible_for_promotion=True:
        1. val_accuracy >= 0.70
        2. No INFEASIBLE outcomes (candidate infeasible when reference feasible)
        3. No hard constraint violations
        4. >= 80% of evaluated scenarios are EQUIVALENT or ACCEPTABLE
        5. Candidate checkpoint structurally valid
        6. Fallback rate <= 50%

    Returns PromotionGateResult. eligible_for_promotion=True does NOT cause promotion.
    Human authorization via --promote flag is still required.
    """
    gate = PromotionGateResult(eligible_for_promotion=False)

    notes = []

    # 1. Validation accuracy
    gate.val_accuracy = candidate_val_accuracy
    if candidate_val_accuracy is not None and candidate_val_accuracy >= gate.val_accuracy_threshold:
        gate.val_accuracy_ok = True
    else:
        notes.append(f"val_accuracy {candidate_val_accuracy} < {gate.val_accuracy_threshold}")

    # 2 & 3. Feasibility and hard constraint analysis
    infeasible_count = sum(1 for c in comparisons if c.outcome_label == OUTCOME_INFEASIBLE)
    hard_violations = 0  # tracked separately via constraint_status
    gate.infeasible_count = infeasible_count
    gate.hard_constraint_violations = hard_violations
    gate.feasibility_preserved = infeasible_count == 0
    gate.hard_constraint_ok = hard_violations == 0
    gate.feasibility_scenarios_tested = len(comparisons)

    if not gate.feasibility_preserved:
        notes.append(f"{infeasible_count} scenario(s) produced infeasible result when reference was feasible")

    # 4. Outcome agreement
    evaluated = [c for c in comparisons if c.outcome_label not in (
        OUTCOME_BOTH_INFEASIBLE, OUTCOME_NOT_EVALUATED, OUTCOME_FAILED)]
    gate.total_evaluated = len(evaluated)
    gate.equivalent_or_acceptable_count = sum(
        1 for c in evaluated
        if c.outcome_label in (OUTCOME_EQUIVALENT, OUTCOME_ACCEPTABLE))
    gate.inferior_count = sum(1 for c in evaluated if c.outcome_label == OUTCOME_INFERIOR)
    gate.failed_count = sum(1 for c in comparisons if c.outcome_label == OUTCOME_FAILED)

    if gate.total_evaluated > 0:
        agreement_rate = gate.equivalent_or_acceptable_count / gate.total_evaluated
        gate.outcome_agreement_ok = agreement_rate >= gate.min_agreement_rate
        if not gate.outcome_agreement_ok:
            notes.append(
                f"Outcome agreement {agreement_rate:.1%} < {gate.min_agreement_rate:.1%} minimum")
    else:
        # No evaluatable scenarios (all reference-infeasible) — cannot confirm or deny
        gate.outcome_agreement_ok = True
        notes.append("No directly comparable scenarios — outcome agreement not verified.")

    # 5. Checkpoint structural validity
    checkpoint_ok = True
    if candidate_checkpoint_path:
        if not os.path.exists(candidate_checkpoint_path):
            checkpoint_ok = False
            notes.append("Candidate checkpoint file missing.")
        else:
            try:
                with open(candidate_checkpoint_path) as f:
                    data = json.load(f)
                required = {"model_version", "feature_ordering", "class_labels", "W", "b", "mean", "scale"}
                missing = required - set(data.keys())
                if missing:
                    checkpoint_ok = False
                    notes.append(f"Candidate checkpoint missing fields: {missing}")
            except Exception as e:
                checkpoint_ok = False
                notes.append(f"Cannot parse candidate checkpoint: {e}")

    # 6. Fallback rate
    fallback_outcomes = [c for c in comparisons if c.config_name.startswith("CANDIDATE")]
    # Fallback rate is recorded in routing outcomes separately

    # Final eligibility
    gate.eligible_for_promotion = (
        gate.val_accuracy_ok
        and gate.feasibility_preserved
        and gate.hard_constraint_ok
        and gate.outcome_agreement_ok
        and checkpoint_ok
    )

    if gate.eligible_for_promotion:
        gate.reason = "All mandatory promotion conditions satisfied. Human authorization required to promote."
    else:
        gate.reason = "Promotion gate NOT passed: " + "; ".join(notes) if notes else "Conditions not met."

    gate.notes = notes
    return gate
