"""Terminal Stage Trace for Neuro-Symbolic Cloud Optimization.

Executes ONE query (or batch) through the REAL current pipeline and prints every
stage's actual internal state AS IT HAPPENS — including exactly where and why it fails,
if it fails. Calls the real functions verified in Step 4 without mocks or reconstruction.

CLI Usage:
    python -m src.proof.stage_trace --query "Deploy 8 vCPUs and 16GB RAM for under $300 on AWS" --mode 4
    python -m src.proof.stage_trace --query "Continuous dynamic scaling with target CPU 70% under $1500" --mode 3
    python -m src.proof.stage_trace --queries-file data/diagnostic_queries.json --mode 4 --failures-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 stdout encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config.settings import settings
from src.optimizers.raw_symbolic_runner import run_symbolic_rule_based
from src.orchestrator.service import NeuroSymbolicOrchestrator
from src.semantic.carm_matcher import CARMMatcher
from src.semantic.explainer import FinOpsExplainer
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.scope_parser import SCOPEParser
from src.verifiers.independent_checker import IndependentChecker
from templates.Continuous_PSO_Dynamic_Scaling import solve_pso_continuous_scaling
from templates.Graph_SMT_Z3_MultiRegion_Placement import (
    REGIONS_GRAPH,
    calculate_composite_sla,
    solve_z3_graph_disaster_recovery,
)
from templates.ILP_VM_Knapsack_Allocation import solve_ilp_vm_knapsack


def trace_stage_1_parsing(
    parser: SCOPEParser, query_text: str, silent: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """STAGE 1: Rule-Based Parsing (SCOPEParser).

    Inspects actual regex field extraction and returns (success, extracted_params, failure_reason).
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("--- STAGE 1: Rule-Based Parsing (SCOPEParser) ---")
    log("  Attempting field extraction...")

    # Regex matches inspection
    # 1. vCPUs
    vcpu_match = re.search(
        r"(?:(\d+)\s*(?:vcpus?|cores?|v-cpu)|(?:vcpus?|cores?|v-cpu)\s*(?:>=|<=|:|:=|=)?\s*(\d+))",
        query_text,
        re.IGNORECASE,
    )
    if vcpu_match:
        val_vcpu = int(vcpu_match.group(1) or vcpu_match.group(2))
        log(f'  vcpus:   matched "{vcpu_match.group(0)}" -> {val_vcpu}')
    else:
        log('  vcpus:   NO MATCH (tried patterns: r"(\\d+)\\s*(?:vcpus?|cores?|v-cpu)", r"vcpus?\\s*(?:>=|:)\\s*(\\d+)")')

    # 2. RAM
    ram_match = re.search(
        r"(?:(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)\s*(?:ram|memory)?|(?:ram|memory)\s*(?:>=|<=|:|:=|=)?\s*(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)?)",
        query_text,
        re.IGNORECASE,
    )
    if ram_match:
        val_ram = float(ram_match.group(1) or ram_match.group(2))
        log(f'  ram_gb:  matched "{ram_match.group(0)}" -> {val_ram}')
    else:
        log('  ram_gb:  NO MATCH (tried patterns: r"(\\d+(?:\\.\\d+)?)\\s*(?:gb|gigabytes?)", r"ram\\s*(?:>=|:)\\s*(\\d+)")')

    # 3. Budget (INR / USD)
    inr_patterns = [
        r"(?:₹|rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:inr|rupees?|rs\b)",
    ]
    usd_patterns = [
        r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:usd|dollars?)",
        r"(?:budget|cost|price|spend|limit|under|cap)\s*(?:of|max|limit|under|is|to)?\s*[:=]?\s*\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)",
    ]
    budget_matched = False
    for pat in inr_patterns:
        m = re.search(pat, query_text, re.IGNORECASE)
        if m:
            val_inr = float(m.group(1).replace(",", ""))
            val_usd = round(val_inr / settings.USD_TO_INR_RATE, 2)
            log(f'  budget:  matched "{m.group(0)}" -> {val_inr} INR (~${val_usd} USD)')
            budget_matched = True
            break
    if not budget_matched:
        for pat in usd_patterns:
            m = re.search(pat, query_text, re.IGNORECASE)
            if m:
                val_usd = float(m.group(1).replace(",", ""))
                log(f'  budget:  matched "{m.group(0)}" -> ${val_usd:.2f} USD')
                budget_matched = True
                break
    if not budget_matched:
        log('  budget:  NO MATCH (tried patterns: ["$...", "INR/₹...", "under/budget $..."])')

    # 4. Provider
    prov_match = re.search(r"\b(AWS|Azure|GCP)\b", query_text, re.IGNORECASE)
    if prov_match:
        log(f'  provider: matched "{prov_match.group(0)}" -> "{prov_match.group(1).lower()}"')
    else:
        log('  provider: NO MATCH (tried pattern: r"\\b(AWS|Azure|GCP)\\b")')

    # 5. Latency & SLA
    lat_match = re.search(r"(?:max\s+|under\s+|latency\s+(?:of\s+|under\s+)?|\b)(\d+(?:\.\d+)?)\s*ms", query_text, re.IGNORECASE)
    if lat_match:
        log(f'  latency: matched "{lat_match.group(0)}" -> {float(lat_match.group(1))} ms')
    sla_match = re.search(r"(9\d(?:\.\d+)?)\s*%", query_text, re.IGNORECASE)
    if sla_match:
        log(f'  sla:     matched "{sla_match.group(0)}" -> {float(sla_match.group(1))}%')

    # Domain Intent Check
    has_intent = parser.has_cloud_intent(query_text)
    if not has_intent:
        reason = "RESULT: PARSER_FAILED -- no archetype could be constructed from this input. Stopping here, as Mode 3/4 would."
        log(f"  {reason}")
        return False, None, reason

    # Extract actual parameters
    params = parser.extract_parameters(query_text)

    # Missing field detection
    missing_fields = []
    if "budget_max_usd" not in params:
        missing_fields.append("budget_max_usd (using default $500.00)")
    if not vcpu_match and "required_vcpus" not in params:
        missing_fields.append("required_vcpus")
    if not ram_match and "required_ram_gb" not in params:
        missing_fields.append("required_ram_gb")

    if missing_fields:
        log(f"  RESULT: PARTIAL CONTRACT ({len(missing_fields)} field(s) defaulted/missing: {', '.join(missing_fields)})")
    else:
        log("  RESULT: FULL CONTRACT EXTRACTED")

    return True, params, None


def trace_stage_2_archetype_matching(
    matcher: CARMMatcher, query_text: str, parser: SCOPEParser, silent: bool = False
) -> Tuple[bool, Optional[str], Optional[str]]:
    """STAGE 2: Archetype Matching (CARM).

    Computes Jaccard similarity across all registered templates.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 2: Archetype Matching (CARM) ---")
    extracted_constraints = parser.extract_constraints_from_text(query_text)

    if not extracted_constraints:
        log("  Extracted Constraints: NONE (empty set)")
        log("  ILP_VM_Allocation                  : score 0.00")
        log("  PSO_Continuous_Scaling             : score 0.00")
        log("  Z3_Graph_Disaster_Recovery         : score 0.00")
        reason = "RESULT: UNSUPPORTED_ARCHETYPE -- no matching constraint tokens found. Stopping here."
        log(f"  {reason}")
        return False, None, reason

    scores = {}
    for archetype, data in matcher.TEMPLATE_INDEX.items():
        score = matcher.compute_jaccard_score(
            extracted_constraints, data["constraints"]  # type: ignore[arg-type]
        )
        scores[archetype] = round(score, 4)

    # Print individual scores
    for arch, sc in scores.items():
        log(f"  {arch:<35}: score {sc:.2f}")

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_arch, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score

    if best_score <= 0.0:
        reason = "RESULT: UNSUPPORTED_ARCHETYPE -- no archetype cleared matching threshold (> 0.0). Stopping here."
        log(f"  {reason}")
        return False, None, reason

    log(f"  -> SELECTED: {best_arch} (margin: {margin:.2f})")
    return True, best_arch, None


def trace_stage_3_contract_validation(
    archetype: str, params: Dict[str, Any], silent: bool = False
) -> Tuple[bool, Optional[CloudOptimizationContract], Optional[str]]:
    """STAGE 3: Contract Validation.

    Validates parameters against Pydantic V2 CloudOptimizationContract.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 3: Contract Validation ---")
    try:
        contract = CloudOptimizationContract(
            problem_type=archetype,  # type: ignore[arg-type]
            cloud_providers=params.get("cloud_providers", ["AWS"]),
            budget_max_usd=params.get("budget_max_usd", settings.DEFAULT_BUDGET_USD),
            service_count=params.get("service_count", 1),
            required_vcpus=params.get("required_vcpus", 1),
            required_ram_gb=params.get("required_ram_gb", 1.0),
            latency_max_ms=params.get("latency_max_ms", 100.0),
            sla_availability_pct=params.get("sla_availability_pct", 99.9),
            metadata=params.get("metadata", {}),
        )
        log("  Pydantic validation: PASS")
        return True, contract, None
    except Exception as exc:
        log("  Pydantic validation: FAIL")
        log(f"  Validation Error: {exc}")
        return False, None, f"Pydantic validation failed: {exc}"


def trace_stage_4_solver_execution(
    contract: CloudOptimizationContract, mode: int, silent: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """STAGE 4: Solver Dispatch & Execution.

    Dispatches to HiGHS MILP, Continuous PSO, or Z3 SMT Graph solver.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 4: Solver Dispatch & Execution ---")
    problem = contract.problem_type

    if problem == "ILP_VM_Allocation":
        log("  Routed to: solve_ilp_vm_knapsack (HiGHS MILP)")
        log(
            f"  Constraints: vCPUs >= {contract.required_vcpus}, RAM >= {contract.required_ram_gb}GB, "
            f"Budget <= ${contract.budget_max_usd:.2f}, Provider = {contract.cloud_providers}"
        )
        try:
            prov_list = contract.cloud_providers if contract.cloud_providers else ["AWS"]
            res = solve_ilp_vm_knapsack(
                required_vcpus=contract.required_vcpus,
                required_ram_gb=contract.required_ram_gb,
                budget_max_usd=contract.budget_max_usd,
                target_providers=prov_list,
            )
            log(f"  Raw solver output: {res}")
            return True, res, None
        except Exception as exc:
            log(f"  [Solver Exception]: {type(exc).__name__}: {exc}")
            return False, None, f"Solver exception: {type(exc).__name__}: {exc}"

    elif problem == "PSO_Continuous_Scaling":
        log("  Routed to: solve_pso_continuous_scaling (Continuous PSO)")
        log(
            f"  Bounds: Bandwidth in [100, 1000] Mbps, Replicas in [1.0, 16.0], "
            f"Target CPU = 70.0%, Budget <= ${contract.budget_max_usd:.2f}"
        )
        try:
            prov_list = contract.cloud_providers if contract.cloud_providers else ["AWS"]
            res = solve_pso_continuous_scaling(
                budget_max_usd=contract.budget_max_usd,
                target_cpu_pct=70.0,
                target_providers=prov_list,
            )
            log(f"  Raw solver output: {res}")
            return True, res, None
        except Exception as exc:
            log(f"  [Solver Exception]: {type(exc).__name__}: {exc}")
            return False, None, f"Solver exception: {type(exc).__name__}: {exc}"

    elif problem == "Z3_Graph_Disaster_Recovery":
        log("  Routed to: solve_z3_graph_disaster_recovery (Z3 SMT Graph)")
        log(
            f"  SMT Clauses: Distinct(Region_A, Region_B), Latency(A, B) <= {contract.latency_max_ms}ms, "
            f"Composite_SLA(A, B) >= {contract.sla_availability_pct}%, Cost(A, B) <= ${contract.budget_max_usd:.2f}"
        )
        try:
            prov_list = contract.cloud_providers if contract.cloud_providers else ["AWS", "GCP"]
            res = solve_z3_graph_disaster_recovery(
                sla_pct=contract.sla_availability_pct,
                max_latency_ms=contract.latency_max_ms,
                budget_max_usd=contract.budget_max_usd,
                target_providers=prov_list,
            )
            log(f"  Raw solver output: {res}")
            return True, res, None
        except Exception as exc:
            log(f"  [Solver Exception]: {type(exc).__name__}: {exc}")
            return False, None, f"Solver exception: {type(exc).__name__}: {exc}"

    else:
        reason = f"Unknown problem type: {problem}"
        log(f"  [Solver Error]: {reason}")
        return False, None, reason


def trace_stage_5_independent_verification(
    contract: CloudOptimizationContract,
    solver_result: Dict[str, Any],
    silent: bool = False,
    stage_header: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """STAGE 5 (or 4 in LLM mode): Independent Verification (IndependentChecker).

    Verifies catalog SKUs, physical constraint feasibility, and optimality claims.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    hdr = stage_header or "--- STAGE 5: Independent Verification (IndependentChecker) ---"
    log(f"\n{hdr}")
    check = IndependentChecker.verify_solution(contract, solver_result)

    # 1. Catalog consistency
    cat_pass = check.get("catalog_consistent", False)
    cost_info = check.get("cost_accuracy", {})
    rep_cost = cost_info.get("reported_cost_usd", 0.0)
    calc_cost = cost_info.get("calculated_catalog_cost_usd", 0.0)

    # Explicit Safety Guard: Never claim PASS if reported vs verified differs by > $0.05
    if abs(rep_cost - calc_cost) > 0.05:
        cat_pass = False

    cat_status = "PASS" if cat_pass else "FAIL"
    log(f"  Catalog consistency: {cat_status} (Reported: ${rep_cost:.2f}, Catalog verified: ${calc_cost:.2f})")

    # 2. Feasibility check with recomputed numbers
    feas_pass = check.get("feasible_against_contract", False)
    feas_status = "PASS" if feas_pass else "FAIL"
    recomp = check.get("recomputed_metrics", {})
    problem = contract.problem_type

    if problem == "ILP_VM_Allocation":
        rc_vcpu = recomp.get("vcpus", 0)
        rc_ram = recomp.get("ram_gb", 0.0)
        rc_cost = recomp.get("monthly_cost_usd", recomp.get("total_monthly_cost_usd", 0.0))
        log(
            f"  Feasibility check: {feas_status} ("
            f"vCPU: {rc_vcpu} >= {contract.required_vcpus}, "
            f"RAM: {rc_ram:.1f}GB >= {contract.required_ram_gb:.1f}GB, "
            f"Cost: ${rc_cost:.2f} <= ${contract.budget_max_usd:.2f})"
        )
    elif problem == "PSO_Continuous_Scaling":
        rc_cpu = recomp.get("modeled_cpu_pct", 0.0)
        rc_cost = recomp.get("monthly_cost_usd", recomp.get("total_monthly_cost_usd", 0.0))
        detail = f"Recomputed CPU: {rc_cpu:.1f}% vs ceiling 70.0%, Cost: ${rc_cost:.2f} <= ${contract.budget_max_usd:.2f}"
        if rc_cpu > 70.0 or rc_cost > contract.budget_max_usd:
            log(f"  Feasibility check: {feas_status} ({detail} -> VIOLATION)")
        else:
            log(f"  Feasibility check: {feas_status} ({detail})")
    elif problem == "Z3_Graph_Disaster_Recovery":
        rc_lat = recomp.get("latency_ms", recomp.get("inter_region_latency_ms", 0.0))
        rc_sla = recomp.get("composite_sla_pct", 0.0)
        rc_cost = recomp.get("monthly_cost_usd", recomp.get("total_monthly_cost_usd", 0.0))
        log(
            f"  Feasibility check: {feas_status} ("
            f"Latency: {rc_lat:.1f}ms <= {contract.latency_max_ms:.1f}ms, "
            f"SLA: {rc_sla:.4f}% >= {contract.sla_availability_pct}%, "
            f"Cost: ${rc_cost:.2f} <= ${contract.budget_max_usd:.2f})"
        )
    else:
        log(f"  Feasibility check: {feas_status}")

    # Violations if any
    violations = check.get("violations", [])
    if violations:
        for v in violations:
            log(f"    -> VIOLATION: {v}")

    # 3. Optimality verdict
    opt_verdict = check.get("optimality_verdict", "N/A")
    log(f'  Optimality verdict: "{opt_verdict}"')

    # 4. Final summary verdict
    final_verdict = check.get("summary_status", "Execution complete")
    log(f"  FINAL VERDICT: {final_verdict}")

    return feas_pass, check


def trace_stage_6_explanation(
    contract: CloudOptimizationContract,
    solver_result: Dict[str, Any],
    mode: int,
    silent: bool = False,
) -> None:
    """STAGE 6: Explanation Generation (Mode 4 Only).

    Transforms solver output into structured FinOps explanations.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 6: Explanation Generation (only for Mode 4) ---")
    if mode != 4:
        log("  [SKIPPED: Mode 3 does not generate natural-language explanations]")
        return

    problem = contract.problem_type
    if problem == "ILP_VM_Allocation":
        consumed_fields = ["allocated_vms", "total_monthly_cost_usd", "total_vcpus", "total_ram_gb"]
    elif problem == "PSO_Continuous_Scaling":
        consumed_fields = ["optimal_bandwidth_mbps", "recommended_replicas", "estimated_hourly_cost_usd"]
    elif problem == "Z3_Graph_Disaster_Recovery":
        consumed_fields = ["primary_region", "secondary_region", "inter_region_latency_ms", "achieved_sla_pct"]
    else:
        consumed_fields = list(solver_result.keys())

    log(f"  Fields consumed from solver output: {consumed_fields}")
    try:
        explanation = FinOpsExplainer.generate_report(
            contract=contract,
            solver_result=solver_result,
            enable_llm_explainer=False,
        )
        clean_exp = " ".join(explanation.split())
        snippet = clean_exp[:100] + ("..." if len(clean_exp) > 100 else "")
        log(f'  Generated explanation: "{snippet}"')
    except Exception as exc:
        log(f"  [Explanation Error]: {exc}")


def run_llm_mode_trace(
    query_text: str, mode: int, yes_flag: bool = False, silent: bool = False
) -> Dict[str, Any]:
    """Executes a single live inference query for Mode 1 (Pure LLM) or Mode 2 (Structured LLM).

    Reuses the real execute_dashboard_llm_request function, validates via IndependentChecker,
    logs every call to results/live_api_call_log.jsonl, and enforces the confirmation safety gate.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    # 1. Interactive Confirmation Safety Gate
    if not yes_flag and not silent:
        sys.stdout.write(
            "This will make 1 LIVE API call to OpenRouter and consume 1 of your daily quota. Continue? [y/N] "
        )
        sys.stdout.flush()
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled by user. No API call made.")
            return {
                "query": query_text,
                "mode": mode,
                "passed": False,
                "verdict": "Cancelled by user",
                "cost": 0.0,
            }
        if ans not in ["y", "yes"]:
            print("Cancelled by user. No API call made.")
            return {
                "query": query_text,
                "mode": mode,
                "passed": False,
                "verdict": "Cancelled by user",
                "cost": 0.0,
            }

    log(f'\n=== QUERY: "{query_text}" ===\n')

    # Parse ground truth contract for independent verification
    parser = SCOPEParser()
    contract, _, _ = parser.parse_query_to_contract(query_text)

    # Model and Prompt Metadata
    model = os.getenv("OPENROUTER_MODEL") or getattr(
        settings, "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"
    )
    max_tokens = getattr(settings, "LLM_MAX_COMPLETION_TOKENS", 4096)

    if mode == 1:
        prompt_sent = (
            "System: You are a Cloud Solutions Architect. Recommend a concrete cloud VM allocation plan. "
            "State the recommended provider, instance types, quantities, and the exact total monthly cost in USD ($/month). Be concise.\n"
            f"User: Recommend cloud VMs for this request: \"{query_text}\""
        )
    else:
        req_vcpus = int(getattr(contract, "required_vcpus", 4)) if contract else 4
        req_ram = float(getattr(contract, "required_ram_gb", 16.0)) if contract else 16.0
        schema_sample = {
            "cloud_provider": "AWS",
            "instances": [{"sku": "t3.medium", "quantity": 2, "monthly_cost": 60.74}],
            "total_monthly_cost": 60.74,
            "total_vcpus": req_vcpus,
            "total_ram_gb": req_ram,
        }
        prompt_sent = (
            f"System: You are a Cloud Optimization System. Respond ONLY with valid JSON matching this schema: {json.dumps(schema_sample)}. No explanatory text.\n"
            f"User: Optimize allocation for: \"{query_text}\". Respond in JSON."
        )

    # --- STAGE 1: LLM Request ---
    log("--- STAGE 1: LLM Request ---")
    log(f"  Model: {model}")
    prompt_snip = prompt_sent[:200] + ("..." if len(prompt_sent) > 200 else "")
    log(f'  Prompt sent: "{prompt_snip}"')
    log(f"  max_tokens: {max_tokens}")

    # Real LLM Call
    from app import execute_dashboard_llm_request

    t0 = time.perf_counter()
    llm_res = execute_dashboard_llm_request(
        mode_num=mode,
        query=query_text,
        contract=contract,
    )
    elapsed_s = llm_res.get("elapsed_seconds") or round(time.perf_counter() - t0, 2)
    finish_reason = llm_res.get("finish_reason", "unknown")
    raw_content = llm_res.get("content", "")

    # Log every live call made to results/live_api_call_log.jsonl
    from datetime import datetime, timezone

    log_path = Path("results/live_api_call_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            log_entry = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "query": query_text,
                "model": llm_res.get("model", model),
                "status": llm_res.get("status"),
                "finish_reason": finish_reason,
                "elapsed_seconds": elapsed_s,
                "reported_cost_usd": llm_res.get("reported_cost_usd"),
                "error": llm_res.get("error_message"),
            }
            f.write(json.dumps(log_entry) + "\n")
    except Exception as log_err:
        log(f"  [Warning: could not write to live_api_call_log.jsonl: {log_err}]")

    # --- STAGE 2: LLM Response ---
    log("\n--- STAGE 2: LLM Response ---")
    log(f'  finish_reason: "{finish_reason}"')
    raw_snip = " ".join(raw_content.split())
    raw_snip_300 = raw_snip[:300] + ("..." if len(raw_snip) > 300 else "")
    log(f'  Raw response (first ~300 chars): "{raw_snip_300}"')
    if finish_reason == "length":
        log("  WARNING: Generation reached max token limit and was truncated (finish_reason == 'length'). Output is incomplete.")
    if llm_res.get("status") == "daily_quota_exhausted":
        log(f"  [API Error: 429 Daily Limit Exhausted - {llm_res.get('error_message')}]")
    elif llm_res.get("status") in ["api_error", "missing_credentials", "empty_response"]:
        log(f"  [API Error]: {llm_res.get('error_message')}")

    # --- STAGE 3: Extraction ---
    log("\n--- STAGE 3: Extraction ---")
    if mode == 1:
        log("  (Mode 1 only, prose parsing)")
    else:
        log("  (Mode 2 only, JSON schema parsing)")

    extracted_cost = llm_res.get("reported_cost_usd")
    extracted_alloc: List[Dict[str, Any]] = []

    if mode == 1:
        if extracted_cost is not None:
            log(f"  Extracted cost: ${extracted_cost:,.2f}")
        else:
            err_reason = llm_res.get(
                "error_message", "No dollar amount ($XX.XX) found in prose response"
            )
            log(f"  Extracted cost: EXTRACTION_FAILED ({err_reason})")

        # Extract SKUs from prose
        catalog = IndependentChecker.get_sku_catalog()
        for sku_name, sdata in catalog.items():
            pat = r"\b" + re.escape(sku_name) + r"\b"
            if re.search(pat, raw_content, re.IGNORECASE):
                q_match = re.search(
                    r"(\d+)\s*(?:x|\*|instances?|nodes?|vms?)?\s*" + re.escape(sku_name),
                    raw_content,
                    re.IGNORECASE,
                )
                qty = int(q_match.group(1)) if q_match else 1
                extracted_alloc.append({
                    "sku": sku_name,
                    "instance_type": sku_name,
                    "count": qty,
                    "quantity": qty,
                    "provider": sdata.get("provider", "AWS"),
                    "vcpus_per_vm": sdata.get("vcpus", 2),
                    "ram_gb_per_vm": sdata.get("ram_gb", 4.0),
                    "monthly_cost": round(sdata.get("hourly_cost_usd", 0.05) * 730.0 * qty, 2),
                })
        if extracted_alloc:
            log(f"  Extracted allocation: {extracted_alloc}")
        else:
            log("  Extracted allocation: NONE")

    else:
        # Mode 2
        parsed_json = llm_res.get("parsed_json")
        if parsed_json and extracted_cost is not None:
            log(f"  Extracted cost: ${extracted_cost:,.2f}")
            raw_vms = (
                parsed_json.get("allocated_vms")
                or parsed_json.get("instances")
                or []
            )
            if isinstance(raw_vms, list):
                for v in raw_vms:
                    if isinstance(v, dict):
                        extracted_alloc.append(v)
            if extracted_alloc:
                log(f"  Extracted allocation: {extracted_alloc}")
            else:
                log("  Extracted allocation: NONE")
        else:
            err_reason = llm_res.get(
                "error_message", "JSON parsing failed or no valid cost found"
            )
            log(f"  Extracted cost: EXTRACTION_FAILED ({err_reason})")
            log("  Extracted allocation: NONE")

    # --- STAGE 4: Independent Verification (IndependentChecker) ---
    has_usable_alloc = len(extracted_alloc) > 0
    if has_usable_alloc and contract:
        solver_dict = {
            "problem_type": contract.problem_type,
            "status": "feasible" if (extracted_cost is not None and extracted_cost > 0) else "UNKNOWN",
            "total_monthly_cost_usd": extracted_cost or 0.0,
            "allocated_vms": extracted_alloc,
        }
        s4_ok, check_result = trace_stage_5_independent_verification(
            contract,
            solver_dict,
            silent=silent,
            stage_header="--- STAGE 4: Independent Verification (IndependentChecker) ---",
        )
        final_verdict = check_result.get("summary_status", "Complete")
    else:
        log("\n--- STAGE 4: Independent Verification (IndependentChecker) ---")
        log("  Cannot verify: no structured output to check")
        final_verdict = "Rejected (No Structured Allocation Output)"
        log(f"  FINAL VERDICT: {final_verdict}")
        s4_ok = False
        check_result = {"violations": ["No structured allocation output"]}

    # --- LATENCY ---
    log("\n--- LATENCY ---")
    log(f"  Total wall-clock time: {elapsed_s:.2f}s\n")

    return {
        "query": query_text,
        "mode": mode,
        "passed": s4_ok,
        "failed_stage": None if s4_ok else 4,
        "verdict": final_verdict,
        "cost": extracted_cost or 0.0,
    }


def run_pipeline_trace(
    query_text: str, mode: int = 4, yes_flag: bool = False, silent: bool = False
) -> Dict[str, Any]:
    """Runs a single query through the genuine pipeline, printing each stage state as it happens.

    If any stage fails, execution stops immediately and does NOT synthesize later stages.
    """
    if mode in [1, 2]:
        return run_llm_mode_trace(query_text, mode=mode, yes_flag=yes_flag, silent=silent)

    if not silent:
        print(f'\n=== QUERY: "{query_text}" ===\n')

    parser = SCOPEParser()
    matcher = CARMMatcher()

    # STAGE 1
    s1_ok, params, s1_err = trace_stage_1_parsing(parser, query_text, silent=silent)
    if not s1_ok or params is None:
        return {"query": query_text, "mode": mode, "passed": False, "failed_stage": 1, "verdict": s1_err, "cost": 0.0}

    # STAGE 2
    s2_ok, archetype, s2_err = trace_stage_2_archetype_matching(matcher, query_text, parser, silent=silent)
    if not s2_ok or archetype is None:
        return {"query": query_text, "mode": mode, "passed": False, "failed_stage": 2, "verdict": s2_err, "cost": 0.0}

    # STAGE 3
    s3_ok, contract, s3_err = trace_stage_3_contract_validation(archetype, params, silent=silent)
    if not s3_ok or contract is None:
        return {"query": query_text, "mode": mode, "passed": False, "failed_stage": 3, "verdict": s3_err, "cost": 0.0}

    # STAGE 4
    s4_ok, solver_result, s4_err = trace_stage_4_solver_execution(contract, mode, silent=silent)
    if not s4_ok or solver_result is None:
        return {"query": query_text, "mode": mode, "passed": False, "failed_stage": 4, "verdict": s4_err, "cost": 0.0}

    # STAGE 5
    s5_ok, check_result = trace_stage_5_independent_verification(contract, solver_result, silent=silent)

    # STAGE 6
    trace_stage_6_explanation(contract, solver_result, mode, silent=silent)

    final_verdict = check_result.get("summary_status", "Complete")
    passed = s5_ok and not check_result.get("violations", [])
    cost = solver_result.get("total_monthly_cost_usd", solver_result.get("estimated_monthly_cost_usd", 0.0))

    return {
        "query": query_text,
        "mode": mode,
        "passed": passed,
        "failed_stage": None if passed else 5,
        "verdict": final_verdict,
        "cost": cost,
    }


def load_queries_from_file(filepath: str) -> List[Dict[str, Any]]:
    """Loads query objects from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found at: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        queries = []
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                q_text = item.get("query") or item.get("text") or item.get("prompt", "")
                q_id = item.get("id", f"query_{idx+1}")
                queries.append({"id": q_id, "query": q_text, "raw": item})
            elif isinstance(item, str):
                queries.append({"id": f"query_{idx+1}", "query": item, "raw": item})
        return queries
    elif isinstance(data, dict):
        q_list = data.get("queries", [])
        return [{"id": item.get("id", f"query_{idx+1}"), "query": item.get("query", ""), "raw": item} for idx, item in enumerate(q_list)]
    return []


def main() -> None:
    """CLI Entrypoint for Terminal Stage Trace."""
    parser = argparse.ArgumentParser(
        description="Neurasym Terminal Stage Trace: Live Pipeline Inspection"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single natural language query string to trace",
    )
    parser.add_argument(
        "--queries-file",
        "--query-file",
        dest="queries_file",
        type=str,
        default=None,
        help="Path to JSON file containing queries for batch execution",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Pipeline execution mode: 1 (Pure LLM), 2 (Structured LLM), 3 (Symbolic + rule-based), or 4 (Full Neuro-Symbolic, default)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip live API quota confirmation prompt for Mode 1 and Mode 2",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Batch mode: only print full stage traces for queries that fail or produce constraint violations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of queries to run in batch mode",
    )

    args = parser.parse_args()

    # Safety Guard: Mode 1 and Mode 2 cannot be executed in batch mode
    if args.mode in [1, 2]:
        if args.queries_file or not args.query:
            print(
                "Error: Batch execution (--queries-file) is strictly blocked for Mode 1 and Mode 2 to protect API quota. "
                "Mode 1 and Mode 2 must be executed with a single query via --query."
            )
            sys.exit(1)

    if args.query:
        run_pipeline_trace(args.query, mode=args.mode, yes_flag=args.yes, silent=False)
        return

    queries_file = args.queries_file or "data/diagnostic_queries.json"
    try:
        queries = load_queries_from_file(queries_file)
    except FileNotFoundError:
        print(f"Error: Query file '{queries_file}' not found.")
        sys.exit(1)

    if args.limit:
        queries = queries[: args.limit]

    print(f"=== BATCH STAGE TRACE: {len(queries)} queries from '{queries_file}' (Mode {args.mode}) ===")
    if args.failures_only:
        print("  [Mode: --failures-only active. Passing queries will show single-line status.]\n")

    results = []
    for idx, q_item in enumerate(queries, 1):
        q_id = q_item["id"]
        q_text = q_item["query"]

        if args.failures_only:
            res = run_pipeline_trace(q_text, mode=args.mode, yes_flag=args.yes, silent=True)
            if res.get("passed", False):
                cost = res.get("cost", 0.0)
                verdict = res.get("verdict", "Feasible against checked constraints")
                print(f'PASS: [{q_id}] "{q_text}" -> {verdict} (${cost:.2f})')
                results.append(res)
            else:
                res = run_pipeline_trace(q_text, mode=args.mode, yes_flag=args.yes, silent=False)
                results.append(res)
        else:
            res = run_pipeline_trace(q_text, mode=args.mode, yes_flag=args.yes, silent=False)
            results.append(res)

    # Summary
    passed_count = sum(1 for r in results if r.get("passed", False))
    failed_count = len(results) - passed_count
    print(f"\n{'='*70}")
    print(f"BATCH TRACE SUMMARY: Total: {len(results)} | Passed: {passed_count} | Failed: {failed_count}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
