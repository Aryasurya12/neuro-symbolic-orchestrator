"""Canonical Execution Verdict Explainer for Neurasym.

Builds single-source-of-truth plain English explanations for every execution mode's
results without hardcoded superiority claims or human intervention.

Produces standardized verdicts, headlines, real-number explanations, impact rationale,
and actionable inspection pointers across CLI, Terminal Comparative Runner, and Streamlit Dashboard.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union

# Exact 9 canonical labels required by Neurasym evaluation protocol
LABEL_CORRECT = "CORRECT"
LABEL_CORRECT_BUT_COSTLIER = "CORRECT_BUT_COSTLIER"
LABEL_CORRECT_REFUSAL = "CORRECT_REFUSAL"
LABEL_WRONG_PLAN = "WRONG_PLAN"
LABEL_WRONG_OUTCOME = "WRONG_OUTCOME"
LABEL_RIGHT_ANSWER_WRONG_READING = "RIGHT_ANSWER_WRONG_READING"
LABEL_UNPARSEABLE = "UNPARSEABLE"
LABEL_PROVIDER_FAILURE = "PROVIDER_FAILURE"
LABEL_NOT_GRADED = "NOT_GRADED"

ALL_LABELS = [
    LABEL_CORRECT,
    LABEL_CORRECT_BUT_COSTLIER,
    LABEL_CORRECT_REFUSAL,
    LABEL_WRONG_PLAN,
    LABEL_WRONG_OUTCOME,
    LABEL_RIGHT_ANSWER_WRONG_READING,
    LABEL_UNPARSEABLE,
    LABEL_PROVIDER_FAILURE,
    LABEL_NOT_GRADED,
]

# High-contrast, colorblind-friendly sequential palette
LABEL_COLORS: Dict[str, str] = {
    LABEL_CORRECT: "#65A31C",                     # Lime/Green - Verified Optimal
    LABEL_CORRECT_BUT_COSTLIER: "#FFA600",        # Warm Amber - Feasible but sub-optimal cost
    LABEL_CORRECT_REFUSAL: "#008162",             # Pine Teal - Appropriate refusal/clarification
    LABEL_WRONG_PLAN: "#F5365C",                  # Crimson Red - Invariant/constraint violation
    LABEL_WRONG_OUTCOME: "#E04848",               # Coral Red - High-level outcome mismatch
    LABEL_RIGHT_ANSWER_WRONG_READING: "#B1AA00",  # Olive Gold - Lucky match with misread inputs
    LABEL_UNPARSEABLE: "#8CA0B4",                 # Slate Gray - Unstructured/unparsable prose
    LABEL_PROVIDER_FAILURE: "#9BA6C4",            # Steel - Infrastructure/429/timeout/truncation
    LABEL_NOT_GRADED: "#5A6789",                  # Dim Blue-Gray - No key available
}

LABEL_TOOLTIPS: Dict[str, str] = {
    LABEL_CORRECT: "Plan meets every keyed requirement and the cost is within tolerance of the independent optimum.",
    LABEL_CORRECT_BUT_COSTLIER: "Plan is valid but costs more than the optimum (shows both figures and the % gap).",
    LABEL_CORRECT_REFUSAL: "The key expected infeasible / unsupported / clarification / conflict, and the mode said so.",
    LABEL_WRONG_PLAN: "Plan violates a keyed requirement (names the violated requirement).",
    LABEL_WRONG_OUTCOME: "Said infeasible when feasible, guessed when it should have asked, or planned an unsupported task.",
    LABEL_RIGHT_ANSWER_WRONG_READING: "(Modes 3/4) Outcome matches but an extracted field differs from the key; answer is right only by luck.",
    LABEL_UNPARSEABLE: "Prose could not be turned into a checkable plan; NOT counted as a wrong answer.",
    LABEL_PROVIDER_FAILURE: "Timeout, 429, empty response, or truncation; not a reasoning failure.",
    LABEL_NOT_GRADED: "No key exists for this query.",
}

LABEL_LEGEND: Dict[str, str] = {
    LABEL_CORRECT: "Plan meets every keyed requirement and the cost is within tolerance of the independent optimum.",
    LABEL_CORRECT_BUT_COSTLIER: "Plan is valid but costs more than the optimum (shows both figures and the % gap).",
    LABEL_CORRECT_REFUSAL: "The key expected infeasible / unsupported / clarification / conflict, and the mode said so.",
    LABEL_WRONG_PLAN: "Plan violates a keyed requirement (names the violated requirement).",
    LABEL_WRONG_OUTCOME: "Said infeasible when feasible, guessed when it should have asked, or planned an unsupported task.",
    LABEL_RIGHT_ANSWER_WRONG_READING: "(Modes 3/4) Outcome matches but an extracted field differs from the key (list the fields); the answer is right only by luck.",
    LABEL_UNPARSEABLE: "Prose could not be turned into a checkable plan; NOT counted as a wrong answer.",
    LABEL_PROVIDER_FAILURE: "Timeout, 429, empty response, or truncation; not a reasoning failure.",
    LABEL_NOT_GRADED: "No key exists for this query.",
}

DISCLAIMER_RULE = (
    "Task success, allocation validity, interpretation accuracy, cost vs optimal and speed "
    "are separate measurements. No single score combines them."
)


def _get_val(obj: Any, key_name: str, default: Any = None) -> Any:
    """Safely extracts a field from a dataclass, pydantic model, or dictionary."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key_name, default)
    if hasattr(obj, key_name):
        return getattr(obj, key_name, default)
    return default


def _format_cost(val: Optional[float]) -> str:
    """Formats a float cost in USD."""
    if val is None:
        return "N/A"
    return f"${val:,.2f}"


def _detect_provider_failure(record: Any) -> Tuple[bool, str]:
    """Detects whether an execution failed due to provider infrastructure/API issues."""
    norm_status = str(_get_val(record, "normalization_status", "")).upper()
    opt_status = str(_get_val(record, "optimality_status", "")).upper()
    summary_status = str(_get_val(record, "summary_status", "")).lower()
    violations = _get_val(record, "violations", []) or []
    norm_errors = _get_val(record, "normalization_errors", []) or []
    
    all_err_strings = [summary_status] + [str(v).lower() for v in violations] + [str(e).lower() for e in norm_errors]
    combined_errors = " ".join(all_err_strings)

    if norm_status in ["API_FAILURE", "NORMALIZATIONSTATUS.API_FAILURE"]:
        if "429" in combined_errors or "quota" in combined_errors:
            return True, "Provider HTTP 429 Rate Limit / Daily Quota Exhaustion"
        if "timeout" in combined_errors or "deadline" in combined_errors:
            return True, "Provider API Request Timeout"
        return True, "Provider API Connection / Infrastructure Failure"

    if norm_status in ["TRUNCATION_FAILURE", "NORMALIZATIONSTATUS.TRUNCATION_FAILURE"] or "TRUNCATION" in opt_status:
        return True, "Provider Generation Truncation (Max Tokens Exceeded)"

    if "429" in combined_errors or "quota" in combined_errors or "daily_quota_exhausted" in combined_errors:
        return True, "Provider HTTP 429 Rate Limit / Daily Quota Exhaustion"
    if "timeout" in combined_errors or "deadline" in combined_errors:
        return True, "Provider API Request Timeout"
    if "missing credentials" in combined_errors or "missing_credentials" in combined_errors:
        return True, "Missing Provider API Credentials"

    return False, ""


def _detect_unparseable(record: Any) -> Tuple[bool, str]:
    """Detects whether an LLM produced unparseable output not convertible into a checkable plan."""
    norm_status = str(_get_val(record, "normalization_status", "")).upper()
    feasibility = str(_get_val(record, "feasibility", "")).upper()
    summary_status = str(_get_val(record, "summary_status", ""))
    norm_errors = _get_val(record, "normalization_errors", []) or []
    
    err_desc = "; ".join(str(e) for e in norm_errors) if norm_errors else summary_status

    if norm_status in [
        "UNPARSEABLE",
        "NORMALIZATION_FAILURE",
        "MALFORMED_OUTPUT",
        "AMBIGUOUS",
        "NORMALIZATIONSTATUS.UNPARSEABLE",
        "NORMALIZATIONSTATUS.NORMALIZATION_FAILURE",
        "NORMALIZATIONSTATUS.MALFORMED_OUTPUT",
        "NORMALIZATIONSTATUS.AMBIGUOUS",
    ]:
        return True, err_desc or "No structured SKU instances found in output text"

    return False, ""


def _check_parameter_mismatches(record_reqs: Dict[str, Any], key: Any) -> List[Tuple[str, Any, Any, str]]:
    """Compares extracted requirements against key requirements to find discrepancies.
    
    Returns list of tuples: (field_name, key_value, extracted_value, notes)
    """
    mismatches = []
    fields = [
        ("budget_max_usd", "budget", "$", True),
        ("required_vcpus", "vCPUs", "", False),
        ("required_ram_gb", "RAM", " GB", False),
        ("latency_max_ms", "latency", " ms", False),
        ("sla_availability_pct", "SLA", "%", False),
        ("target_bandwidth_mbps", "bandwidth", " Mbps", False),
        ("max_cpu_pct", "max CPU", "%", False),
    ]

    for field_key, display_name, unit, is_prefix in fields:
        k_val = _get_val(key, field_key)
        if k_val is None:
            continue

        r_val = record_reqs.get(field_key) if isinstance(record_reqs, dict) else None

        def _fmt(v: Any) -> str:
            if isinstance(v, (int, float)):
                if is_prefix:
                    return f"${v:,.2f}".rstrip("0").rstrip(".")
                return f"{v:g}{unit}"
            return f"{v}{unit}" if not is_prefix else f"${v}"

        k_str = _fmt(k_val)
        if r_val is None or r_val == "not stated":
            mismatches.append((display_name, k_val, "not stated", f"key had {k_str} {display_name}, mode did not state it"))
            continue

        try:
            # Float comparison with small tolerance for rounding
            if abs(float(k_val) - float(r_val)) > 0.001:
                r_str = _fmt(r_val)
                mismatches.append((display_name, k_val, r_val, f"key had {k_str} {display_name}, mode used {r_str}"))
        except (ValueError, TypeError):
            if str(k_val) != str(r_val):
                mismatches.append((display_name, k_val, r_val, f"key had {k_str} {display_name}, mode used {r_val}"))

    return mismatches


def explain_run(record: Any, key: Optional[Any] = None) -> Dict[str, Any]:
    """Canonical verdict explanation function for single-run evaluation records.

    Builds dynamic headlines, factual numbers, and objective justifications from real
    record attributes and manifest keys.

    Args:
        record: CanonicalExecutionRecord or dict containing mode execution metadata.
        key: ManifestQueryItem, dict, or None containing ground-truth expectations.

    Returns:
        Dict containing:
            label: One of the 9 canonical strings
            headline: One short sentence
            what_happened: 1-2 plain sentences with REAL numbers from the run
            why_it_matters: 1 plain sentence on impact
            check_this: Optional pointer or None
            color: Color hex code
            tooltip: One-line explanation tooltip
            is_mock: Boolean indicating if offline mock data was used
    """
    mode_num = _get_val(record, "mode", 0)
    mode_name = _get_val(record, "mode_name") or f"Mode {mode_num}"
    is_mock = bool(_get_val(record, "is_mock", False)) or str(_get_val(record, "provider", "")).lower() == "mock" or "mock" in str(_get_val(record, "summary_status", "")).lower()
    
    feasibility = str(_get_val(record, "feasibility", "")).upper()
    if "PASS" in feasibility:
        feasibility_clean = "PASS"
    elif "NOT_EVALUABLE" in feasibility or "NOT_EVALUATED" in feasibility:
        feasibility_clean = "NOT_EVALUABLE"
    elif "INFEASIBLE" in feasibility:
        feasibility_clean = "INFEASIBLE"
    else:
        feasibility_clean = "FAIL"

    norm_status = str(_get_val(record, "normalization_status", "")).upper()
    opt_status = str(_get_val(record, "optimality_status", ""))
    summary_status = str(_get_val(record, "summary_status", ""))
    violations = _get_val(record, "violations", []) or []
    norm_errors = _get_val(record, "normalization_errors", []) or []
    requirements = _get_val(record, "requirements", {}) or {}
    claimed_cost = _get_val(record, "claimed_cost_usd")
    recomputed_cost = _get_val(record, "recomputed_cost_usd")
    actual_cost = recomputed_cost if recomputed_cost is not None else claimed_cost

    # Helper to build model vs key parameter comparison note for Modes 1 & 2
    def _build_model_param_note() -> str:
        if mode_num not in [1, 2] or not key:
            return ""
        mismatches = _check_parameter_mismatches(requirements, key)
        if not mismatches:
            return ""
        m_parts = []
        for name, k_val, r_val, note in mismatches:
            k_disp = f"${k_val:,.2f}" if name == "budget" else (f"{k_val:g}" if isinstance(k_val, (int, float)) else str(k_val))
            if r_val == "not stated" or r_val is None:
                r_disp = "not stated"
            elif name == "budget" and isinstance(r_val, (int, float)):
                r_disp = f"${r_val:,.2f}"
            elif isinstance(r_val, (int, float)):
                r_disp = f"{r_val:g}"
            else:
                r_disp = str(r_val)
            m_parts.append(f"it answered for {name} {r_disp}; you asked {k_disp}")
        return " (" + "; ".join(m_parts) + ")"

    model_param_note = _build_model_param_note()

    # -------------------------------------------------------------------------
    # 1. NOT_GRADED: When no manifest key exists
    # -------------------------------------------------------------------------
    if key is None:
        cost_str = f" at {_format_cost(actual_cost)}" if actual_cost is not None else ""
        return {
            "label": LABEL_NOT_GRADED,
            "headline": f"{mode_name} completed without an evaluation ground-truth key.",
            "what_happened": (
                f"{mode_name} executed{cost_str} and reported status '{summary_status or norm_status}'. "
                f"No ground-truth manifest record was matched for this query."
            ),
            "why_it_matters": "Without a reference key, correctness, constraint adherence, and cost optimality cannot be verified.",
            "check_this": "Add this query to the benchmark manifest with expected parameters to enable automatic grading.",
            "color": LABEL_COLORS[LABEL_NOT_GRADED],
            "tooltip": LABEL_TOOLTIPS[LABEL_NOT_GRADED],
            "is_mock": is_mock,
        }

    key_outcome = str(_get_val(key, "expected_outcome", "NOT_GRADED")).upper()
    key_opt_cost = _get_val(key, "expected_optimal_cost_usd")
    key_opt_source = _get_val(key, "optimum_source")
    key_notes = _get_val(key, "key_notes") or _get_val(key, "notes", "")
    key_vcpus = _get_val(key, "required_vcpus")
    key_ram = _get_val(key, "required_ram_gb")
    key_budget = _get_val(key, "budget_max_usd")
    key_latency = _get_val(key, "latency_max_ms")
    key_sla = _get_val(key, "sla_availability_pct")
    key_bw = _get_val(key, "target_bandwidth_mbps")

    src_str = f" via {key_opt_source}" if key_opt_source else ""

    if key_outcome in ["NOT_GRADED", "NONE", "—", ""]:
        return {
            "label": LABEL_NOT_GRADED,
            "headline": f"{mode_name} executed on an unkeyed query.",
            "what_happened": f"{mode_name} finished with status '{summary_status or norm_status}' and cost {_format_cost(actual_cost)}.",
            "why_it_matters": "Ground-truth grading is omitted because no expected outcome is defined in the manifest.",
            "check_this": None,
            "color": LABEL_COLORS[LABEL_NOT_GRADED],
            "tooltip": LABEL_TOOLTIPS[LABEL_NOT_GRADED],
            "is_mock": is_mock,
        }

    # -------------------------------------------------------------------------
    # 2. PROVIDER_FAILURE: Infrastructure, 429, Timeout, Truncation
    # -------------------------------------------------------------------------
    is_prov_fail, prov_reason = _detect_provider_failure(record)
    if is_prov_fail:
        return {
            "label": LABEL_PROVIDER_FAILURE,
            "headline": f"{mode_name} encountered an external provider infrastructure failure.",
            "what_happened": (
                f"{mode_name} could not complete inference because the upstream provider halted with: {prov_reason}."
            ),
            "why_it_matters": "This is an external API or network failure, not an algorithmic or reasoning error.",
            "check_this": "Check provider API credentials, rate limits (HTTP 429), or max token allocation.",
            "color": LABEL_COLORS[LABEL_PROVIDER_FAILURE],
            "tooltip": LABEL_TOOLTIPS[LABEL_PROVIDER_FAILURE],
            "is_mock": is_mock,
        }

    # -------------------------------------------------------------------------
    # 3. UNPARSEABLE: Prose could not be structured into checkable plan
    # -------------------------------------------------------------------------
    is_unparseable, unparse_reason = _detect_unparseable(record)
    if is_unparseable and key_outcome == "FEASIBLE":
        return {
            "label": LABEL_UNPARSEABLE,
            "headline": f"{mode_name} generated unparseable prose ({unparse_reason}).",
            "what_happened": (
                f"{mode_name} output could not be mapped to instance SKUs and counts ({unparse_reason}){model_param_note}. "
                f"No mathematical verification could be performed."
            ),
            "why_it_matters": "The response lacked structured SKU definitions, so constraints and costs cannot be verified.",
            "check_this": "Prompt formatting or JSON schema enforcement is required to emit structured allocations.",
            "color": LABEL_COLORS[LABEL_UNPARSEABLE],
            "tooltip": LABEL_TOOLTIPS[LABEL_UNPARSEABLE],
            "is_mock": is_mock,
        }

    # -------------------------------------------------------------------------
    # 4. REFUSAL HANDLING: Key expects Refusal / Clarification / Infeasible / Unsupported
    # -------------------------------------------------------------------------
    if key_outcome in ["CLARIFICATION_REQUIRED", "INFEASIBLE", "CONFLICTING_REQUIREMENTS", "UNSUPPORTED"]:
        # Check if the mode correctly refused / asked clarification / proved infeasible
        mode_refused = False
        refusal_type = ""

        if key_outcome == "CLARIFICATION_REQUIRED":
            if (
                norm_status in ["CLARIFICATION_REQUIRED", "NORMALIZATIONSTATUS.CLARIFICATION_REQUIRED"]
                or "CLARIFICATION" in opt_status.upper()
                or "clarification" in summary_status.lower()
                or any("?" in str(v) for v in violations)
            ):
                mode_refused = True
                refusal_type = "requested clarification for missing parameters"

        elif key_outcome in ["INFEASIBLE", "CONFLICTING_REQUIREMENTS"]:
            if (
                feasibility_clean in ["INFEASIBLE", "FAIL"]
                or "INFEASIBLE" in opt_status.upper()
                or "CONFLICTING" in opt_status.upper()
                or norm_status in ["SOLVER_INFEASIBLE", "PROVEN_INFEASIBLE", "NORMALIZATIONSTATUS.SOLVER_INFEASIBLE"]
                or "infeasible" in summary_status.lower()
            ):
                mode_refused = True
                refusal_type = "proved the constraints are mathematically infeasible"

        elif key_outcome == "UNSUPPORTED":
            if (
                norm_status in ["TASK_INCOMPATIBLE", "NORMALIZATIONSTATUS.TASK_INCOMPATIBLE"]
                or "UNSUPPORTED" in opt_status.upper()
                or "unsupported" in summary_status.lower()
                or feasibility_clean == "NOT_EVALUABLE"
            ):
                mode_refused = True
                refusal_type = "refused the unsupported out-of-domain workload"

        if mode_refused:
            return {
                "label": LABEL_CORRECT_REFUSAL,
                "headline": f"{mode_name} correctly refused or requested clarification.",
                "what_happened": (
                    f"The key expects {key_outcome}" + (f" ({key_notes})" if key_notes else "") + f". "
                    f"{mode_name} {refusal_type}, matching the expected ground truth."
                ),
                "why_it_matters": "Refusing impossible requests or asking for missing data prevents hallucinations and broken deployments.",
                "check_this": None,
                "color": LABEL_COLORS[LABEL_CORRECT_REFUSAL],
                "tooltip": LABEL_TOOLTIPS[LABEL_CORRECT_REFUSAL],
                "is_mock": is_mock,
            }
        else:
            # Mode fabricated a plan when it should have refused or asked clarification
            return {
                "label": LABEL_WRONG_OUTCOME,
                "headline": f"{mode_name} returned a plan when the key expected {key_outcome}.",
                "what_happened": (
                    f"The key expected {key_outcome}" + (f" ({key_notes})" if key_notes else "") + f", "
                    f"but {mode_name} fabricated an allocation costing {_format_cost(actual_cost)}{model_param_note} instead of refusing."
                ),
                "why_it_matters": "Fabricating numbers or guessing missing constraints leads to ungrounded and risky infrastructure changes.",
                "check_this": f"The query omitted or conflicted on parameters, but {mode_name} proceeded without clarification.",
                "color": LABEL_COLORS[LABEL_WRONG_OUTCOME],
                "tooltip": LABEL_TOOLTIPS[LABEL_WRONG_OUTCOME],
                "is_mock": is_mock,
            }

    # -------------------------------------------------------------------------
    # 5. KEY EXPECTS FEASIBLE: Mode evaluation
    # -------------------------------------------------------------------------
    has_plan = (actual_cost is not None) or bool(_get_val(record, "normalized_allocation")) or bool(_get_val(record, "allocation"))
    
    # If mode refused or failed to find a plan for a feasible query without producing an allocation
    if (feasibility_clean in ["FAIL", "INFEASIBLE", "NOT_EVALUABLE"] and not has_plan) or (not has_plan and feasibility_clean != "PASS"):
        return {
            "label": LABEL_WRONG_OUTCOME,
            "headline": f"{mode_name} failed to find a plan for a feasible request ({summary_status or norm_status}).",
            "what_happened": (
                f"The key expects a feasible plan with budget {_format_cost(key_budget)}, "
                f"but {mode_name} halted with status '{summary_status or norm_status}'{model_param_note}."
            ),
            "why_it_matters": "Valid workloads are rejected or delayed when the system incorrectly claims they cannot be satisfied.",
            "check_this": "Check why the parser or solver failed to find a valid SKU combination.",
            "color": LABEL_COLORS[LABEL_WRONG_OUTCOME],
            "tooltip": LABEL_TOOLTIPS[LABEL_WRONG_OUTCOME],
            "is_mock": is_mock,
        }

    # If mode returned a plan, but independent verification failed
    if feasibility_clean != "PASS" or violations:
        violation_str = ", ".join(str(v) for v in violations[:2]) if violations else "Constraint check failure"
        return {
            "label": LABEL_WRONG_PLAN,
            "headline": f"{mode_name} returned an invalid plan violating requirement: {violation_str}.",
            "what_happened": (
                f"{mode_name} returned a plan costing {_format_cost(actual_cost)}{model_param_note}, "
                f"but independent verification rejected it: {violation_str}."
            ),
            "why_it_matters": "Deploying an invalid allocation violates SLA thresholds, causes outages, or breaches budgets.",
            "check_this": f"Violated checks: {violation_str}.",
            "color": LABEL_COLORS[LABEL_WRONG_PLAN],
            "tooltip": LABEL_TOOLTIPS[LABEL_WRONG_PLAN],
            "is_mock": is_mock,
        }

    # -------------------------------------------------------------------------
    # 6. VERIFIED FEASIBLE PLAN: Check for Parameter Misreading (Modes 3/4)
    # -------------------------------------------------------------------------
    mismatches = _check_parameter_mismatches(requirements, key)
    if mismatches and mode_num in [3, 4]:
        mismatch_descs = [m[3] for m in mismatches]
        mismatch_summary = "; ".join(mismatch_descs)
        first_mismatched = mismatches[0][0]
        return {
            "label": LABEL_RIGHT_ANSWER_WRONG_READING,
            "headline": f"{mode_name} matched outcome by luck, but misread {first_mismatched} ({mismatches[0][3]}).",
            "what_happened": (
                f"{mode_name} ended with a valid plan at {_format_cost(actual_cost)}, but its extracted parameters "
                f"differed from your key ({mismatch_summary}). The answer matched only by luck under alternate limits."
            ),
            "why_it_matters": "The plan happened to be feasible, but the system misread user intent and used unintended constraints.",
            "check_this": f"{mode_name} never saw the exact {first_mismatched} specified in your query.",
            "color": LABEL_COLORS[LABEL_RIGHT_ANSWER_WRONG_READING],
            "tooltip": LABEL_TOOLTIPS[LABEL_RIGHT_ANSWER_WRONG_READING],
            "is_mock": is_mock,
        }

    # -------------------------------------------------------------------------
    # 7. COST COMPARISON AGAINST INDEPENDENT OPTIMUM
    # -------------------------------------------------------------------------
    if key_opt_cost is not None and actual_cost is not None:
        cost_diff = actual_cost - key_opt_cost
        tolerance = max(0.50, 0.01 * key_opt_cost)

        # Flag if optimum in key differs from catalog recomputation by more than $0.01
        opt_diff_abs = abs(cost_diff)
        opt_flag_str = ""
        check_flag = None
        if opt_diff_abs > 0.01 and opt_diff_abs <= tolerance:
            opt_flag_str = f" [Note: Key optimum {_format_cost(key_opt_cost)} differs from catalog recomputed optimum {_format_cost(actual_cost)} by ${opt_diff_abs:.2f}]"
            check_flag = f"Key optimum {_format_cost(key_opt_cost)} differs from catalog recomputation by ${opt_diff_abs:.2f} (> $0.01 threshold)."

        if cost_diff > tolerance:
            pct_gap = (cost_diff / key_opt_cost) * 100.0
            return {
                "label": LABEL_CORRECT_BUT_COSTLIER,
                "headline": f"{mode_name} returned a valid plan, but costs {pct_gap:.1f}% more than the optimum ({_format_cost(actual_cost)} vs {_format_cost(key_opt_cost)}).",
                "what_happened": (
                    f"{mode_name} returned a valid plan costing {_format_cost(actual_cost)}{model_param_note}, "
                    f"but the cheapest possible plan costs {_format_cost(key_opt_cost)}{src_str} ({pct_gap:.1f}% more). "
                    f"It meets the requirement but wastes money."
                ),
                "why_it_matters": "The plan meets all technical requirements but results in unnecessary cloud infrastructure spend.",
                "check_this": f"A cheaper SKU combination exists saving {_format_cost(cost_diff)}/month.",
                "color": LABEL_COLORS[LABEL_CORRECT_BUT_COSTLIER],
                "tooltip": LABEL_TOOLTIPS[LABEL_CORRECT_BUT_COSTLIER],
                "is_mock": is_mock,
            }
        else:
            # Within tolerance of independent optimum
            delta_abs = abs(cost_diff)
            req_summary = []
            if key_vcpus:
                req_summary.append(f"{key_vcpus} vCPUs")
            if key_ram:
                req_summary.append(f"{key_ram:.0f} GB")
            if key_budget:
                req_summary.append(f"${key_budget:,.0f}")
            req_str = ", ".join(req_summary) if req_summary else "your constraints"

            return {
                "label": LABEL_CORRECT,
                "headline": f"{mode_name} returned a verified optimal plan meeting all requirements.",
                "what_happened": (
                    f"{mode_name} read your request as {req_str} (matches your key) and returned a plan at {_format_cost(actual_cost)}{model_param_note}. "
                    f"That is within ${delta_abs:.2f} of the cheapest possible plan ({_format_cost(key_opt_cost)}{src_str}) and was verified against the catalog.{opt_flag_str}"
                ),
                "why_it_matters": "The plan is mathematically verified to satisfy all resource constraints at minimum cost.",
                "check_this": check_flag,
                "color": LABEL_COLORS[LABEL_CORRECT],
                "tooltip": LABEL_TOOLTIPS[LABEL_CORRECT],
                "is_mock": is_mock,
            }

    # Default fallback for valid plan without explicit cost key
    return {
        "label": LABEL_CORRECT,
        "headline": f"{mode_name} returned a verified plan meeting all requirements.",
        "what_happened": (
            f"{mode_name} returned a verified plan at {_format_cost(actual_cost)}{model_param_note} "
            f"satisfying all specified constraints."
        ),
        "why_it_matters": "The allocation plan satisfies all specified constraints without violations.",
        "check_this": None,
        "color": LABEL_COLORS[LABEL_CORRECT],
        "tooltip": LABEL_TOOLTIPS[LABEL_CORRECT],
        "is_mock": is_mock,
    }


def format_plain_english_box(verdict: Dict[str, Any], width: int = 86) -> str:
    """Formats a single mode verdict into an ASCII 'IN PLAIN ENGLISH' box.
    
    Args:
        verdict: Dict returned by explain_run().
        width: Width of the terminal box in characters.
        
    Returns:
        Formatted multi-line string.
    """
    label = verdict.get("label", "UNKNOWN")
    headline = verdict.get("headline", "")
    what_happened = verdict.get("what_happened", "")
    why_it_matters = verdict.get("why_it_matters", "")
    check_this = verdict.get("check_this")
    is_mock = verdict.get("is_mock", False)

    inner_w = width - 4
    lines = []
    lines.append("+" + "-" * (width - 2) + "+")
    if is_mock:
        lines.append(f"| IN PLAIN ENGLISH: [{label}] [MOCK DATA]".ljust(width - 1) + "|")
        lines.append("| [MOCK DATA - OFFLINE BENCHMARK FIXTURE - ZERO LIVE CALLS]".ljust(width - 1) + "|")
    else:
        lines.append(f"| IN PLAIN ENGLISH: [{label}]".ljust(width - 1) + "|")
    lines.append("|" + " " * (width - 2) + "|")
    
    # Headline
    hl_wrapped = textwrap.wrap(f"Headline     : {headline}", width=inner_w)
    for line in hl_wrapped:
        lines.append(f"| {line}".ljust(width - 1) + "|")

    # What Happened
    wh_wrapped = textwrap.wrap(f"What Happened: {what_happened}", width=inner_w)
    for line in wh_wrapped:
        lines.append(f"| {line}".ljust(width - 1) + "|")

    # Why It Matters
    wm_wrapped = textwrap.wrap(f"Why It Matters: {why_it_matters}", width=inner_w)
    for line in wm_wrapped:
        lines.append(f"| {line}".ljust(width - 1) + "|")

    # Check This (optional)
    if check_this:
        ct_wrapped = textwrap.wrap(f"Check This   : {check_this}", width=inner_w)
        for line in ct_wrapped:
            lines.append(f"| {line}".ljust(width - 1) + "|")

    lines.append("+" + "-" * (width - 2) + "+")
    return "\n".join(lines)


def format_plain_english_summary(
    records: List[Any],
    key: Optional[Any] = None,
    width: int = 86,
) -> str:
    """Formats an end-of-run comparative summary, legend, and caveats list.

    Strictly avoids printing overall winners, scores, or rankings.
    Computes success rates and interpretation accuracy per mode across queries, never pooled.

    Args:
        records: List of CanonicalExecutionRecord or dict items from the run.
        key: Ground-truth key if available.
        width: Width in characters.

    Returns:
        Formatted multi-line summary string.
    """
    has_mock = any(
        bool(_get_val(r, "is_mock", False)) or str(_get_val(r, "provider", "")).lower() == "mock"
        for r in records
    )
    lines = []
    lines.append("=" * width)
    if has_mock:
        lines.append("  INDEPENDENT VERDICT EVALUATION SUMMARY (PLAIN ENGLISH) [MOCK DATA]")
    else:
        lines.append("  INDEPENDENT VERDICT EVALUATION SUMMARY (PLAIN ENGLISH)")
    lines.append("=" * width)
    lines.append("")

    verdicts = []
    careful_items = []

    # Group records by mode to ensure per-mode metric calculation (never pooled)
    from collections import OrderedDict
    mode_groups: Dict[int, List[Tuple[Any, Dict[str, Any]]]] = OrderedDict()

    for r in records:
        v = explain_run(r, key=key)
        verdicts.append((r, v))
        mode_num = _get_val(r, "mode", 0)
        mode_name = _get_val(r, "mode_name") or f"Mode {mode_num}"
        lbl = v["label"]
        hl = v["headline"]
        lines.append(f"  • {mode_name:<24} : [{lbl:<26}] {hl}")

        if mode_num not in mode_groups:
            mode_groups[mode_num] = []
        mode_groups[mode_num].append((r, v))

        # Collect caveat items for "Things to be careful about"
        if lbl == LABEL_PROVIDER_FAILURE:
            careful_items.append(f"{mode_name}: Provider infrastructure failure (HTTP 429 rate limits or timeout).")
        elif lbl == LABEL_UNPARSEABLE:
            careful_items.append(f"{mode_name}: Output was unparseable prose and could not be verified.")
        elif lbl == LABEL_RIGHT_ANSWER_WRONG_READING:
            careful_items.append(f"{mode_name}: Matched output by coincidence; parsed parameters differed from input query.")
        elif lbl == LABEL_CORRECT_BUT_COSTLIER:
            careful_items.append(f"{mode_name}: Plan is valid but more expensive than the independent optimum.")
        elif lbl == LABEL_NOT_GRADED:
            careful_items.append(f"{mode_name}: Query is ungraded (no reference ground-truth key).")

    is_graded = bool(key and str(_get_val(key, "expected_outcome", "")).upper() not in ["NOT_GRADED", "NONE", "—", ""])
    if is_graded:
        metric_lines = []
        for mode_num, m_records in mode_groups.items():
            first_r, _ = m_records[0]
            mode_name = _get_val(first_r, "mode_name") or f"Mode {mode_num}"
            
            n_total = 0
            n_success = 0
            n_unparse = 0
            n_interp_correct = 0

            for r, v in m_records:
                task_out = _get_val(r, "task_outcome")
                lbl = v["label"]
                if not task_out:
                    if lbl == LABEL_NOT_GRADED:
                        task_out = "NOT_GRADED"
                    elif lbl == LABEL_UNPARSEABLE:
                        task_out = "UNGRADED-UNPARSEABLE"
                    elif lbl in [LABEL_CORRECT, LABEL_CORRECT_REFUSAL, LABEL_CORRECT_BUT_COSTLIER, LABEL_RIGHT_ANSWER_WRONG_READING]:
                        task_out = "SUCCESS"
                    else:
                        task_out = "FAILURE"

                if task_out == "NOT_GRADED":
                    continue

                n_total += 1
                if task_out == "SUCCESS":
                    n_success += 1
                elif task_out == "UNGRADED-UNPARSEABLE":
                    n_unparse += 1

                reqs = _get_val(r, "requirements", {})
                mismatches = _check_parameter_mismatches(reqs, key)
                if not mismatches and lbl != LABEL_RIGHT_ANSWER_WRONG_READING:
                    n_interp_correct += 1

            if n_total == 0:
                continue

            strict_pct = (n_success / n_total) * 100.0
            lenient_denom = n_total - n_unparse
            if lenient_denom > 0:
                lenient_pct = (n_success / lenient_denom) * 100.0
                lenient_str = f"{lenient_pct:.1f}% ({n_success}/{lenient_denom}, {n_unparse} unparseable excluded)"
            else:
                lenient_str = f"N/A (0/0, {n_unparse} unparseable excluded)"

            interp_pct = (n_interp_correct / n_total) * 100.0

            metric_lines.append(f"    • {mode_name} (n={n_total}):")
            metric_lines.append(f"        - Task Success (Strict)   : {strict_pct:.1f}% ({n_success}/{n_total}) [Unparseable = No usable answer]")
            metric_lines.append(f"        - Task Success (Lenient)  : {lenient_str}")
            metric_lines.append(f"        - Interpretation Accuracy : {interp_pct:.1f}% ({n_interp_correct}/{n_total})")

        if metric_lines:
            lines.append("")
            lines.append("-" * width)
            lines.append("  BENCHMARK METRICS BY EXECUTION MODE (NEVER POOLED ACROSS MODES):")
            lines.append("-" * width)
            lines.extend(metric_lines)

    lines.append("")
    lines.append("-" * width)
    lines.append("  LABEL REFERENCE & EVALUATION LEGEND:")
    lines.append("-" * width)
    for lbl, desc in LABEL_LEGEND.items():
        lines.append(f"    [{lbl:<26}] : {desc}")

    lines.append("")
    lines.append("-" * width)
    lines.append("  THINGS TO BE CAREFUL ABOUT IN THIS RUN:")
    lines.append("-" * width)
    if careful_items:
        for item in careful_items:
            lines.append(f"    ! {item}")
    else:
        lines.append("    ✓ No unparseable outputs, provider failures, or parameter misreadings detected.")

    lines.append("")
    lines.append("-" * width)
    lines.append(f"  NOTE: {DISCLAIMER_RULE}")
    lines.append("=" * width)

    return "\n".join(lines)


def find_manifest_entry(query_input: str) -> Optional[Dict[str, Any]]:
    """Looks up human-written query specifications from manifest files."""
    import json
    import os
    import re

    manifest_paths = ["data/final_query_manifest.json", "data/development_query_manifest.json"]
    cleaned = query_input.strip()

    # Extract ID prefix like Q1, Q01, 1, 01, Q01_...
    m_qid = re.match(r"^Q?(\d+)(?:_.*)?$", cleaned, re.IGNORECASE)
    target_prefix = f"Q{int(m_qid.group(1)):02d}_" if m_qid else None

    for manifest_path in manifest_paths:
        if not os.path.exists(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            continue

        items = manifest_data.get("queries", []) if isinstance(manifest_data, dict) else manifest_data
        if not isinstance(items, list):
            continue

        # 1. Number or Q-ID prefix match (e.g. "1" -> Q01_, "Q1" -> Q01_, "Q01", "Q01_VM_...")
        if target_prefix:
            for item in items:
                if isinstance(item, dict) and item.get("query_id", "").upper().startswith(target_prefix):
                    return item

        # 2. Exact query_id match (case-insensitive)
        for item in items:
            if isinstance(item, dict) and item.get("query_id", "").lower() == cleaned.lower():
                return item

        # 3. Exact query_text match
        for item in items:
            if isinstance(item, dict) and item.get("query_text", "").strip().lower() == cleaned.lower():
                return item

        # 4. Substring match
        for item in items:
            if isinstance(item, dict):
                q_text = item.get("query_text", "").strip().lower()
                if q_text and (q_text in cleaned.lower() or cleaned.lower() in q_text):
                    return item

    # 5. Token overlap / semantic match for rephrased queries
    query_tokens = set(re.findall(r"\w+", cleaned.lower()))
    best_match = None
    best_overlap = 0.0

    for manifest_path in manifest_paths:
        if not os.path.exists(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            continue

        items = manifest_data.get("queries", []) if isinstance(manifest_data, dict) else manifest_data
        if not isinstance(items, list):
            continue

        for item in items:
            if isinstance(item, dict):
                q_text = item.get("query_text", "").strip().lower()
                if q_text:
                    item_tokens = set(re.findall(r"\w+", q_text))
                    if item_tokens and query_tokens:
                        intersection = item_tokens.intersection(query_tokens)
                        union = item_tokens.union(query_tokens)
                        jaccard = len(intersection) / len(union)
                        if jaccard > 0.4 and jaccard > best_overlap:
                            best_overlap = jaccard
                            best_match = item

    if best_match:
        return best_match

    return None

