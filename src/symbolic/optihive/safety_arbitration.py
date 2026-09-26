"""SYM-6 Remediation: Deterministic Safety Arbitration Layer.

PURPOSE:
    Routing (SYM-3 deterministic + SYM-4 learned) may propose solver
    sets that are locally rational but globally unsafe — specifically,
    omitting Z3 when topology/latency/SLA constraints require formal
    verification, or selecting only Z3 when candidate generation via
    heuristics is needed.

    This module provides an auditable, deterministic safety layer that
    runs AFTER routing has made its recommendation but BEFORE solvers
    are actually executed.

AUTHORITY:
    "Adaptive routing proposes solver execution; deterministic safety
    arbitration retains authority over whether a solver required for
    hard feasibility validation may be omitted."

RULES:
    Rule 1: TOPOLOGY_CONSTRAINT_REQUIRES_Z3
        If request has latency_max_ms > 0, sla_availability_pct > 0,
        or a disaster recovery / topology workload specification
        → Z3 MUST be in the final solver set.

    Rule 2: HIGHLY_CONSTRAINED_REQUIRES_Z3
        If is_highly_constrained=True (constraint_density >= 2)
        → Z3 MUST be in the final solver set.

    Rule 3: Z3_ALONE_REQUIRES_HEURISTIC_COMPANION
        If only Z3 is selected (no GA, no PSO)
        → Add GA and PSO to generate candidate solutions.

    Rule 4: EMPTY_SOLVER_SET_FALLBACK
        If solver set would be empty
        → Fall back to ['GA', 'PSO', 'Z3'].
"""

from dataclasses import dataclass, field
from typing import List, Optional
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.optihive.adaptive_router import SolverRoutingFeatures

# Safety rule identifiers — used in audit logs
RULE_TOPOLOGY_REQUIRES_Z3 = "TOPOLOGY_CONSTRAINT_REQUIRES_Z3"
RULE_Z3_ALONE_NEEDS_HEURISTIC = "Z3_ALONE_REQUIRES_HEURISTIC_COMPANION"
RULE_HIGHLY_CONSTRAINED_Z3 = "HIGHLY_CONSTRAINED_REQUIRES_Z3"
RULE_EMPTY_FALLBACK = "EMPTY_SOLVER_SET_FALLBACK"

# Canonical solver identifiers
SOLVER_GA = "GA"
SOLVER_PSO = "PSO"
SOLVER_Z3 = "Z3"
FULL_SOLVER_SET = [SOLVER_GA, SOLVER_PSO, SOLVER_Z3]


@dataclass
class SafetyArbitrationResult:
    """Result returned by SafetyArbitrator.evaluate()."""
    final_solver_set: List[str]
    proposed_solver_set: List[str]
    safety_triggered: bool
    rules_applied: List[str] = field(default_factory=list)
    safety_reason: str = ""

    @property
    def solvers_added(self) -> List[str]:
        """Solvers that were not in the proposal but appear in the final set."""
        proposed = set(self.proposed_solver_set)
        return [s for s in self.final_solver_set if s not in proposed]


class SafetyArbitrator:
    """Deterministic safety arbitration for solver set proposals.

    Usage:
        arbitrator = SafetyArbitrator()
        result = arbitrator.evaluate(request, proposed_solvers)
        final_solvers = result.final_solver_set
    """

    def evaluate(
        self,
        request: SymbolicOptimizationRequest,
        proposed_solver_set: List[str],
    ) -> SafetyArbitrationResult:
        """Apply safety rules to a proposed solver set.

        Args:
            request: The optimization request with constraint parameters.
            proposed_solver_set: Solvers proposed by SYM-3/SYM-4 routing.

        Returns:
            SafetyArbitrationResult with final_solver_set and audit trail.
        """
        try:
            features = SolverRoutingFeatures(request)
            return self._apply_rules(request, features, proposed_solver_set)
        except Exception as e:
            # Safety arbitration must never crash; fall back to full race
            return SafetyArbitrationResult(
                final_solver_set=list(FULL_SOLVER_SET),
                proposed_solver_set=list(proposed_solver_set),
                safety_triggered=True,
                rules_applied=[RULE_EMPTY_FALLBACK],
                safety_reason=f"Safety arbitration error ({e}): fallback to full GA+PSO+Z3.",
            )

    def _apply_rules(
        self,
        request: SymbolicOptimizationRequest,
        features: SolverRoutingFeatures,
        proposed: List[str],
    ) -> SafetyArbitrationResult:
        """Apply all safety rules in deterministic order."""
        final = list(proposed)
        rules_applied: List[str] = []
        safety_triggered = False

        # Check topology sensitivity
        prob_type = getattr(request, "problem_type", "") or ""
        is_topology_dr = any(
            kw in prob_type.lower()
            for kw in ["dr", "disaster", "recovery", "topology", "geo", "multi_region"]
        )

        # ── Rule 1: Topology / latency / SLA constraints require Z3 ────────────
        if features.has_latency_constraint or features.has_sla_constraint or is_topology_dr:
            if SOLVER_Z3 not in final:
                final.append(SOLVER_Z3)
                rules_applied.append(RULE_TOPOLOGY_REQUIRES_Z3)
                safety_triggered = True

        # ── Rule 2: Highly constrained requires Z3 ──────────────────────────────
        if features.is_highly_constrained:
            if SOLVER_Z3 not in final:
                final.append(SOLVER_Z3)
                rules_applied.append(RULE_HIGHLY_CONSTRAINED_Z3)
                safety_triggered = True

        # ── Rule 3: Z3 alone requires heuristic companion ───────────────────────
        has_heuristic = SOLVER_GA in final or SOLVER_PSO in final
        if SOLVER_Z3 in final and not has_heuristic:
            final.extend([SOLVER_GA, SOLVER_PSO])
            rules_applied.append(RULE_Z3_ALONE_NEEDS_HEURISTIC)
            safety_triggered = True

        # ── Rule 4: Empty solver set fallback ───────────────────────────────────
        if not final:
            final = list(FULL_SOLVER_SET)
            rules_applied.append(RULE_EMPTY_FALLBACK)
            safety_triggered = True

        # Canonical deduplication preserving deterministic order: [GA, PSO, Z3]
        deduped = []
        for s in [SOLVER_GA, SOLVER_PSO, SOLVER_Z3]:
            if s in final:
                deduped.append(s)

        safety_reason = self._build_reason(rules_applied, proposed, deduped)

        return SafetyArbitrationResult(
            final_solver_set=deduped,
            proposed_solver_set=list(proposed),
            safety_triggered=safety_triggered,
            rules_applied=rules_applied,
            safety_reason=safety_reason,
        )

    def _build_reason(
        self,
        rules: List[str],
        proposed: List[str],
        final: List[str],
    ) -> str:
        if not rules:
            return "No safety rules triggered; proposal accepted as-is."
        added = [s for s in final if s not in set(proposed)]
        rule_descs = {
            RULE_TOPOLOGY_REQUIRES_Z3: "topology/latency/SLA constraint detected -> Z3 required",
            RULE_HIGHLY_CONSTRAINED_Z3: "highly constrained workload -> Z3 required",
            RULE_Z3_ALONE_NEEDS_HEURISTIC: "Z3 alone cannot generate candidate pool -> GA+PSO added",
            RULE_EMPTY_FALLBACK: "empty solver set -> fallback to full race",
        }
        desc = "; ".join(rule_descs.get(r, r) for r in rules)
        return f"Safety arbitration triggered ({', '.join(rules)}): {desc}. Solvers added: {added}."


_default_arbitrator: Optional[SafetyArbitrator] = None


def apply_safety_arbitration(
    request: SymbolicOptimizationRequest,
    proposed_solver_set: List[str],
) -> SafetyArbitrationResult:
    """Convenience function using a shared SafetyArbitrator instance."""
    global _default_arbitrator
    if _default_arbitrator is None:
        _default_arbitrator = SafetyArbitrator()
    return _default_arbitrator.evaluate(request, proposed_solver_set)
