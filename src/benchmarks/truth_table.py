"""Four-Mode Truth Table and Task Success Evaluation Engine for Neurasym.

Evaluates matched four-mode benchmark runs according to pre-declared task success criteria:
- Task Success (1) vs Task Failure (0):
  * For standard solvable queries: Task success requires a feasible, verified allocation matching requirements.
  * For ambiguous queries: Task success requires emitting CLARIFICATION_REQUIRED (not guessing an arbitrary plan).
  * For unsupported/out-of-domain queries: Task success requires emitting TASK_INCOMPATIBLE / UNSUPPORTED.
  * For provably infeasible queries: Task success requires reporting SOLVER_INFEASIBLE / INFEASIBLE.
- Preserves all 16 binary patterns from '0000' to '1111' (Mode 1, Mode 2, Mode 3, Mode 4).
- Keeps missing runs and pending grading as UNKNOWN/PENDING (never forces arbitrary binary classification).
- Records provider/infrastructure failures separately from reasoning failures.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from src.verifiers.canonical_record import (
    CanonicalExecutionRecord,
    FeasibilityStatus,
    NormalizationStatus,
)


@dataclass
class TaskEvaluationResult:
    """Outcome of evaluating a single mode's execution against expected task criteria."""
    task_pass: Optional[int]  # 1 = success, 0 = failure, None = unknown / pending
    is_feasibility_pass: bool = False
    is_reasoning_failure: bool = False
    is_provider_failure: bool = False
    is_normalization_failure: bool = False
    is_task_outcome_appropriate: bool = False
    failure_category: str = "NONE"  # "NONE", "REASONING_ERROR", "PROVIDER_ERROR", "NORMALIZATION_ERROR", "UNRESOLVED"
    evaluation_reason: str = ""


class TruthTableEngine:
    """Computes, validates, and aggregates the 16-pattern Four-Mode Truth Table."""

    ALL_PATTERNS: List[str] = [
        "".join(bits) for bits in itertools.product("01", repeat=4)
    ]  # '0000' through '1111'

    @classmethod
    def evaluate_mode_task_success(
        cls,
        record: Optional[CanonicalExecutionRecord],
        expected_outcome: str = "FEASIBLE",
    ) -> TaskEvaluationResult:
        """Evaluates whether a single execution record satisfies expected task criteria.
        
        expected_outcome can be:
        - "FEASIBLE": Standard solvable problem. Task success requires verified feasible plan.
        - "CLARIFICATION_REQUIRED": Ambiguous requirements. Task success requires asking for clarification.
        - "UNSUPPORTED": Out-of-domain task. Task success requires rejecting as unsupported.
        - "INFEASIBLE": Over-constrained or impossible task. Task success requires reporting infeasibility.
        """
        if record is None:
            return TaskEvaluationResult(
                task_pass=None,
                is_feasibility_pass=False,
                is_reasoning_failure=False,
                is_provider_failure=False,
                is_normalization_failure=False,
                is_task_outcome_appropriate=False,
                failure_category="UNRESOLVED",
                evaluation_reason="Run record is missing or pending evaluation.",
            )

        exp_clean = expected_outcome.strip().upper()
        norm_stat = record.normalization_status
        feas_stat = record.feasibility

        # Check for provider / network / credential failures
        if norm_stat in [NormalizationStatus.API_FAILURE, NormalizationStatus.MALFORMED_OUTPUT]:
            return TaskEvaluationResult(
                task_pass=0,
                is_feasibility_pass=False,
                is_reasoning_failure=False,
                is_provider_failure=True,
                is_normalization_failure=(norm_stat == NormalizationStatus.MALFORMED_OUTPUT),
                is_task_outcome_appropriate=False,
                failure_category="PROVIDER_ERROR" if norm_stat == NormalizationStatus.API_FAILURE else "NORMALIZATION_ERROR",
                evaluation_reason=f"Provider/Execution failed: {norm_stat.value} ({'; '.join(record.normalization_errors or ['Unknown provider failure'])})",
            )

        # Expected: FEASIBLE
        if exp_clean in ["FEASIBLE", "OPTIMAL", "SOLVABLE", "VALID"]:
            if feas_stat == FeasibilityStatus.PASS and norm_stat in [NormalizationStatus.SUCCESS, NormalizationStatus.NEEDS_REVIEW]:
                return TaskEvaluationResult(
                    task_pass=1,
                    is_feasibility_pass=True,
                    is_reasoning_failure=False,
                    is_provider_failure=False,
                    is_normalization_failure=False,
                    is_task_outcome_appropriate=True,
                    failure_category="NONE",
                    evaluation_reason="Allocation verified mathematically feasible against constraints.",
                )
            else:
                reason = "; ".join(record.violations) if record.violations else f"Infeasible allocation (feasibility={feas_stat.value}, norm={norm_stat.value})"
                return TaskEvaluationResult(
                    task_pass=0,
                    is_feasibility_pass=False,
                    is_reasoning_failure=True,
                    is_provider_failure=False,
                    is_normalization_failure=(norm_stat != NormalizationStatus.SUCCESS),
                    is_task_outcome_appropriate=False,
                    failure_category="REASONING_ERROR",
                    evaluation_reason=reason,
                )

        # Expected: CLARIFICATION_REQUIRED
        elif exp_clean in ["CLARIFICATION_REQUIRED", "AMBIGUOUS", "NEEDS_CLARIFICATION"]:
            if norm_stat == NormalizationStatus.CLARIFICATION_REQUIRED:
                return TaskEvaluationResult(
                    task_pass=1,
                    is_feasibility_pass=False,
                    is_reasoning_failure=False,
                    is_provider_failure=False,
                    is_normalization_failure=False,
                    is_task_outcome_appropriate=True,
                    failure_category="NONE",
                    evaluation_reason="Correctly requested clarification for ambiguous requirements.",
                )
            else:
                return TaskEvaluationResult(
                    task_pass=0,
                    is_feasibility_pass=(feas_stat == FeasibilityStatus.PASS),
                    is_reasoning_failure=True,
                    is_provider_failure=False,
                    is_normalization_failure=False,
                    is_task_outcome_appropriate=False,
                    failure_category="REASONING_ERROR",
                    evaluation_reason="Failed to ask for clarification on ambiguous query (guessed unvalidated plan).",
                )

        # Expected: UNSUPPORTED
        elif exp_clean in ["UNSUPPORTED", "OUT_OF_DOMAIN", "TASK_INCOMPATIBLE"]:
            if norm_stat == NormalizationStatus.TASK_INCOMPATIBLE:
                return TaskEvaluationResult(
                    task_pass=1,
                    is_feasibility_pass=False,
                    is_reasoning_failure=False,
                    is_provider_failure=False,
                    is_normalization_failure=False,
                    is_task_outcome_appropriate=True,
                    failure_category="NONE",
                    evaluation_reason="Correctly identified query as unsupported / out-of-archetype.",
                )
            else:
                return TaskEvaluationResult(
                    task_pass=0,
                    is_feasibility_pass=(feas_stat == FeasibilityStatus.PASS),
                    is_reasoning_failure=True,
                    is_provider_failure=False,
                    is_normalization_failure=False,
                    is_task_outcome_appropriate=False,
                    failure_category="REASONING_ERROR",
                    evaluation_reason="Failed to reject unsupported query (attempted invalid mapping).",
                )

        # Expected: INFEASIBLE
        elif exp_clean in ["INFEASIBLE", "OVER_CONSTRAINED", "BUDGET_EXCEEDED"]:
            if norm_stat == NormalizationStatus.SOLVER_INFEASIBLE or feas_stat in [FeasibilityStatus.INFEASIBLE, FeasibilityStatus.FAIL]:
                # If solver or model proved infeasibility without fabricating a passing plan
                return TaskEvaluationResult(
                    task_pass=1,
                    is_feasibility_pass=False,
                    is_reasoning_failure=False,
                    is_provider_failure=False,
                    is_normalization_failure=False,
                    is_task_outcome_appropriate=True,
                    failure_category="NONE",
                    evaluation_reason="Correctly detected mathematical infeasibility.",
                )
            else:
                return TaskEvaluationResult(
                    task_pass=0,
                    is_feasibility_pass=True,
                    is_reasoning_failure=True,
                    is_provider_failure=False,
                    is_normalization_failure=False,
                    is_task_outcome_appropriate=False,
                    failure_category="REASONING_ERROR",
                    evaluation_reason="Fabricated solution for mathematically impossible problem.",
                )

        # Unknown expected outcome
        return TaskEvaluationResult(
            task_pass=1 if feas_stat == FeasibilityStatus.PASS else 0,
            is_feasibility_pass=(feas_stat == FeasibilityStatus.PASS),
            is_reasoning_failure=(feas_stat != FeasibilityStatus.PASS),
            is_provider_failure=False,
            is_normalization_failure=False,
            is_task_outcome_appropriate=True,
            failure_category="NONE" if feas_stat == FeasibilityStatus.PASS else "REASONING_ERROR",
            evaluation_reason="Evaluated against standard feasibility constraints.",
        )

    @classmethod
    def compute_query_pattern(
        cls,
        mode_records: Dict[int, Optional[CanonicalExecutionRecord]],
        expected_outcome: str = "FEASIBLE",
    ) -> Dict[str, Any]:
        """Computes the 4-mode binary pattern and reasons for a single matched query trial."""
        evaluations = {}
        pass_bits = []
        is_complete = True

        for m in [1, 2, 3, 4]:
            rec = mode_records.get(m)
            eval_res = cls.evaluate_mode_task_success(rec, expected_outcome=expected_outcome)
            evaluations[f"mode{m}"] = eval_res
            if eval_res.task_pass is None:
                pass_bits.append("?")
                is_complete = False
            else:
                pass_bits.append(str(eval_res.task_pass))

        pattern_str = "".join(pass_bits)

        return {
            "pattern": pattern_str,
            "is_complete": is_complete,
            "mode1_pass": evaluations["mode1"].task_pass,
            "mode2_pass": evaluations["mode2"].task_pass,
            "mode3_pass": evaluations["mode3"].task_pass,
            "mode4_pass": evaluations["mode4"].task_pass,
            "mode1_reason": evaluations["mode1"].evaluation_reason,
            "mode2_reason": evaluations["mode2"].evaluation_reason,
            "mode3_reason": evaluations["mode3"].evaluation_reason,
            "mode4_reason": evaluations["mode4"].evaluation_reason,
            "mode1_category": evaluations["mode1"].failure_category,
            "mode2_category": evaluations["mode2"].failure_category,
            "mode3_category": evaluations["mode3"].failure_category,
            "mode4_category": evaluations["mode4"].failure_category,
        }

    @classmethod
    def aggregate_truth_table(
        cls,
        query_patterns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregates outcome patterns across all evaluated queries into a complete 16-state truth table."""
        pattern_counts: Dict[str, int] = {p: 0 for p in cls.ALL_PATTERNS}
        incomplete_count = 0
        total_queries = len(query_patterns)

        for qp in query_patterns:
            pat = qp.get("pattern", "????")
            if pat in pattern_counts:
                pattern_counts[pat] += 1
            else:
                incomplete_count += 1

        # Mode success totals
        mode_success_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for qp in query_patterns:
            for m in [1, 2, 3, 4]:
                if qp.get(f"mode{m}_pass") == 1:
                    mode_success_counts[m] += 1

        return {
            "total_queries_evaluated": total_queries,
            "complete_4way_evaluations": total_queries - incomplete_count,
            "incomplete_evaluations": incomplete_count,
            "pattern_counts": pattern_counts,
            "mode_success_counts": mode_success_counts,
            "mode_success_rates": {
                f"mode_{m}": round((mode_success_counts[m] / max(1, total_queries)) * 100.0, 2)
                for m in [1, 2, 3, 4]
            },
        }
