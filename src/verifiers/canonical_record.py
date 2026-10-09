"""Canonical Execution Record and Structured Audit Event Definitions for Neurasym.

Provides a unified, auditable schema for single-run results across all 4 modes:
- Mode 1: Raw LLM (Groq API)
- Mode 2: Schema-Constrained LLM (Groq API)
- Mode 3: Pure Symbolic Pipeline (Local OR Solvers + IndependentChecker)
- Mode 4: Neuro-Symbolic Pipeline (Local OR Solvers + IndependentChecker + Groq Stage 6 Explainer)

Ensures that the CLI, Comparative Runner, Terminal Stage Trace, and Streamlit Dashboard
render the EXACT same record and verdicts without recalculating or re-solving.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class FeasibilityStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INFEASIBLE = "INFEASIBLE"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    NOT_EVALUATED = "NOT_EVALUATED"


class OptimalityStatus(str, Enum):
    PROVABLY_OPTIMAL = "Provably Optimal (MILP branch-and-bound exact)"
    EXHAUSTIVE_DISCRETE_MINIMUM = "Exact Discrete Minimum (Exhaustive Candidate Search)"
    MATCHES_INDEPENDENT_OPTIMUM = "Matches independent optimum (not proven by this mode)"
    HEURISTIC_FEASIBLE = "Optimality not established (Heuristic approximation)"
    INFEASIBLE = "Infeasible"
    UNVERIFIED = "Unverified (no optimality proof)"
    CLARIFICATION_REQUIRED = "Clarification Required"
    UNSUPPORTED = "Unsupported Workload"
    CONFLICTING = "Conflicting Requirements"
    INFRASTRUCTURE_FAILURE = "Infrastructure / API Failure"
    TRUNCATION_FAILURE = "Provider Truncation (Max Tokens)"


class NormalizationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NORMALIZATION_FAILURE = "NORMALIZATION_FAILURE"
    UNPARSEABLE = "UNPARSEABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    TASK_INCOMPATIBLE = "TASK_INCOMPATIBLE"
    SOLVER_INFEASIBLE = "SOLVER_INFEASIBLE"
    PROVEN_INFEASIBLE = "PROVEN_INFEASIBLE"
    API_FAILURE = "API_FAILURE"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    AMBIGUOUS = "AMBIGUOUS"
    TRUNCATION_FAILURE = "TRUNCATION_FAILURE"


class ExplanationSource(str, Enum):
    LIVE_PROVIDER = "LIVE_PROVIDER"
    CACHED_PROVIDER = "CACHED_PROVIDER"
    LOCAL_TEMPLATE = "LOCAL_TEMPLATE"
    UNAVAILABLE = "UNAVAILABLE"


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass
class AuditEvent:
    """Structured audit event for a single invariant check."""
    check_name: str
    observed_value: Any
    required_value: Any
    operator: str
    source: str
    formula: str
    recomputed_result: Any
    check_id: str = "CHK_GENERAL"
    run_id: Optional[str] = None
    mode: Optional[int] = None
    verifier_module: str = "src.verifiers.independent_checker"
    input_values: Optional[Dict[str, Any]] = None
    provenance: str = "catalog_and_contract"
    tolerance: Optional[str] = None
    signed_margin: Optional[str] = None
    status: CheckStatus = CheckStatus.FAIL
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "check_name": self.check_name,
            "run_id": self.run_id,
            "mode": self.mode,
            "verifier_module": self.verifier_module,
            "input_values": self.input_values or {},
            "provenance": self.provenance,
            "observed_value": self.observed_value,
            "required_value": self.required_value,
            "operator": self.operator,
            "source": self.source,
            "formula": self.formula,
            "recomputed_result": self.recomputed_result,
            "tolerance": self.tolerance,
            "signed_margin": self.signed_margin,
            "status": self.status.value if isinstance(self.status, CheckStatus) else str(self.status),
            "reason": self.reason,
        }


@dataclass
class ExtractedEvidence:
    """Provenance tracking for parsed numbers and entities."""
    field_name: str
    extracted_value: Any
    unit: str
    text_span: str
    confidence: str = "HIGH"  # HIGH, NEEDS_REVIEW, LOW

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalExecutionRecord:
    """Single canonical, auditable execution record for one run of any mode."""
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    mode: int = 4
    mode_name: str = "Mode 4: Neuro-Symbolic"
    original_query: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Requirement provenance & structured contract
    requirement_source: str = "parsed_contract"  # "human_reviewed" or "parsed_contract"
    problem_type: str = "ILP_VM_Allocation"
    requirements: Dict[str, Any] = field(default_factory=dict)

    # Execution provenance
    execution_path: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    solver_name: Optional[str] = None
    solver_seed: Optional[int] = None

    # Raw output
    original_response: Any = None

    # Normalization
    normalized_allocation: Any = None
    claimed_cost_usd: Optional[float] = None
    normalization_status: NormalizationStatus = NormalizationStatus.SUCCESS
    normalization_errors: List[str] = field(default_factory=list)
    extracted_evidence: List[ExtractedEvidence] = field(default_factory=list)

    # Independent Checker Result (Mode-Blind)
    feasibility: FeasibilityStatus = FeasibilityStatus.FAIL
    optimality_status: OptimalityStatus = OptimalityStatus.UNVERIFIED
    recomputed_cost_usd: Optional[float] = None
    cost_delta_usd: Optional[float] = None
    cost_error_pct: Optional[float] = None
    budget_headroom_usd: Optional[float] = None
    violations: List[str] = field(default_factory=list)
    audit_events: List[AuditEvent] = field(default_factory=list)
    summary_status: str = "Unverified"
    checked_against_human_ground_truth: bool = False

    # Manifest & Outcome Evaluation
    task_outcome: Optional[str] = None
    expected_outcome: Optional[str] = None
    query_id: Optional[str] = None

    # Timing Breakdown (ms)
    parsing_ms: float = 0.0
    solving_ms: float = 0.0
    verification_ms: float = 0.0
    explanation_ms: float = 0.0
    total_duration_ms: float = 0.0

    # Execution environment
    is_mock: bool = False

    # Explanation Provenance & Text
    explanation_source: ExplanationSource = ExplanationSource.UNAVAILABLE
    explanation_status: str = "SKIPPED"
    explanation_text: str = ""
    provider_response_id: Optional[str] = None
    explanation_elapsed_seconds: Optional[float] = None
    failed_stage: Optional[int] = None

    def __getitem__(self, item: str) -> Any:
        """Dict-like access for backwards compatibility with tests and callers."""
        if item == "passed":
            return self.feasibility == FeasibilityStatus.PASS
        if item == "cost":
            return self.recomputed_cost_usd if self.recomputed_cost_usd is not None else (self.claimed_cost_usd or 0.0)
        if item == "verdict":
            return self.summary_status
        if item == "failed_stage":
            return self.failed_stage
        if item == "mode":
            return self.mode
        d = self.to_dict()
        if item in d:
            return d[item]
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the record into a JSON-compatible dictionary."""
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "mode_name": self.mode_name,
            "original_query": self.original_query,
            "query_id": self.query_id,
            "expected_outcome": self.expected_outcome,
            "task_outcome": self.task_outcome,
            "timestamp_utc": self.timestamp_utc,
            "requirement_source": self.requirement_source,
            "problem_type": self.problem_type,
            "requirements": self.requirements,
            "execution_path": self.execution_path,
            "provider": self.provider,
            "model": self.model,
            "solver_name": self.solver_name,
            "solver_seed": self.solver_seed,
            "original_response": self.original_response,
            "normalized_allocation": self.normalized_allocation,
            "claimed_cost_usd": self.claimed_cost_usd,
            "normalization_status": self.normalization_status.value if isinstance(self.normalization_status, NormalizationStatus) else str(self.normalization_status),
            "normalization_errors": self.normalization_errors,
            "extracted_evidence": [e.to_dict() for e in self.extracted_evidence],
            "feasibility": self.feasibility.value if isinstance(self.feasibility, FeasibilityStatus) else str(self.feasibility),
            "optimality_status": self.optimality_status.value if isinstance(self.optimality_status, OptimalityStatus) else str(self.optimality_status),
            "recomputed_cost_usd": self.recomputed_cost_usd,
            "cost_delta_usd": self.cost_delta_usd,
            "cost_error_pct": self.cost_error_pct,
            "budget_headroom_usd": self.budget_headroom_usd,
            "violations": self.violations,
            "audit_events": [a.to_dict() for a in self.audit_events],
            "summary_status": self.summary_status,
            "checked_against_human_ground_truth": self.checked_against_human_ground_truth,
            "is_mock": self.is_mock,
            "timing_ms": {
                "parsing_ms": round(self.parsing_ms, 2),
                "solving_ms": round(self.solving_ms, 2),
                "verification_ms": round(self.verification_ms, 2),
                "explanation_ms": round(self.explanation_ms, 2),
                "total_duration_ms": round(self.total_duration_ms, 2),
            },
            "explanation": {
                "source": self.explanation_source.value if isinstance(self.explanation_source, ExplanationSource) else str(self.explanation_source),
                "status": self.explanation_status,
                "text": self.explanation_text,
                "provider_response_id": self.provider_response_id,
                "elapsed_seconds": self.explanation_elapsed_seconds,
            },
        }

    def to_summary_row(self) -> Dict[str, Any]:
        """Provides a flat summary row for comparative tables and CSVs."""
        claimed = f"${self.claimed_cost_usd:,.2f}" if self.claimed_cost_usd is not None else "—"
        recomputed = f"${self.recomputed_cost_usd:,.2f}" if self.recomputed_cost_usd is not None else "—"
        cost_err = f"{self.cost_error_pct:.1f}%" if self.cost_error_pct is not None else "—"
        
        norm_s = self.normalization_status.value if isinstance(self.normalization_status, NormalizationStatus) else str(self.normalization_status)
        if norm_s in ["UNPARSEABLE", "NORMALIZATION_FAILURE", "MALFORMED_OUTPUT", "AMBIGUOUS"]:
            task_outcome_val = "UNGRADED-UNPARSEABLE"
        elif self.task_outcome:
            task_outcome_val = self.task_outcome
        elif not self.expected_outcome or self.expected_outcome in ["NOT_GRADED", "—"]:
            task_outcome_val = "NOT_GRADED"
        elif self.feasibility == FeasibilityStatus.PASS:
            task_outcome_val = "SUCCESS"
        else:
            task_outcome_val = "FAILURE"

        strict_succ = 1 if task_outcome_val == "SUCCESS" else 0
        lenient_succ = "" if task_outcome_val == "UNGRADED-UNPARSEABLE" else (1 if task_outcome_val == "SUCCESS" else 0)

        return {
            "mode": f"Mode {self.mode}",
            "mode_name": self.mode_name,
            "query_id": self.query_id or "—",
            "expected_outcome": self.expected_outcome or "—",
            "execution_path": self.execution_path,
            "normalization_status": norm_s,
            "claimed_cost": claimed,
            "recomputed_cost": recomputed,
            "cost_error": cost_err,
            "feasibility": self.feasibility.value if isinstance(self.feasibility, FeasibilityStatus) else str(self.feasibility),
            "task_outcome": task_outcome_val,
            "strict_task_success": strict_succ,
            "lenient_task_success": lenient_succ,
            "optimality_status": self.optimality_status.value if isinstance(self.optimality_status, OptimalityStatus) else str(self.optimality_status),
            "verdict": self.summary_status,
            "is_mock": self.is_mock,
            "core_ms": round(self.parsing_ms + self.solving_ms + self.verification_ms, 1),
            "explanation_ms": round(self.explanation_ms, 1),
            "total_ms": round(self.total_duration_ms, 1),
            "explanation_source": self.explanation_source.value if isinstance(self.explanation_source, ExplanationSource) else str(self.explanation_source),
        }
