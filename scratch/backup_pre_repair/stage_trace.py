"""Terminal Stage Trace for Neuro-Symbolic Cloud Optimization.

Executes ONE query (or batch) through the REAL current pipeline and prints every
stage's actual internal state AS IT HAPPENS — including exactly where and why it fails,
if it fails. Produces and logs a CanonicalExecutionRecord for every run.

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

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

from config.settings import settings
from src.optimizers.raw_symbolic_runner import run_symbolic_rule_based
from src.semantic.carm_matcher import CARMMatcher
from src.semantic.explainer import FinOpsExplainer
from src.semantic.normalizer import OutputNormalizer
from src.semantic.nvidia_extractor import NVIDIAExtractor
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.scope_parser import SCOPEParser
from src.verifiers.canonical_record import (
    CanonicalExecutionRecord,
    ExplanationSource,
    FeasibilityStatus,
    NormalizationStatus,
    OptimalityStatus,
)
from src.verifiers.independent_checker import IndependentChecker
from templates.Continuous_PSO_Dynamic_Scaling import solve_pso_continuous_scaling
from templates.Graph_SMT_Z3_MultiRegion_Placement import (
    REGIONS_GRAPH,
    solve_z3_graph_disaster_recovery,
)
from templates.ILP_VM_Knapsack_Allocation import (
    _VM_CATALOG,
    solve_ilp_vm_knapsack,
)


def trace_stage_1_parsing(
    parser: SCOPEParser, query_text: str, silent: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """STAGE 1: Lexical Analysis & Rule-Based Parsing (SCOPEParser)."""
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("--- STAGE 1: Lexical Analysis & Rule-Based Parsing (SCOPEParser) ---")
    log("  [1.1 Lexical Entity Extraction]")

    # 1. vCPUs
    vcpu_match = re.search(
        r"(?:(\d+)\s*(?:vcpus?|cores?|v-cpu)|(?:vcpus?|cores?|v-cpu)\s*(?:>=|<=|:|:=|=)?\s*(\d+))",
        query_text,
        re.IGNORECASE,
    )
    if vcpu_match:
        val_vcpu = int(vcpu_match.group(1) or vcpu_match.group(2))
        log(f'    * vCPUs        : MATCH -> "{vcpu_match.group(0)}" => {val_vcpu} cores')
    else:
        log('    * vCPUs        : NO MATCH (defaulting to archetype requirements)')

    # 2. RAM
    ram_match = re.search(
        r"(?:(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)\s*(?:ram|memory)?|(?:ram|memory)\s*(?:>=|<=|:|:=|=)?\s*(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)?)",
        query_text,
        re.IGNORECASE,
    )
    if ram_match:
        val_ram = float(ram_match.group(1) or ram_match.group(2))
        log(f'    * RAM (GB)     : MATCH -> "{ram_match.group(0)}" => {val_ram:.1f} GB')
    else:
        log('    * RAM (GB)     : NO MATCH (defaulting to archetype requirements)')

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
            log(f'    * Budget       : MATCH -> "{m.group(0)}" => ₹{val_inr:,.2f} INR (~${val_usd:,.2f} USD @ 1 USD = {settings.USD_TO_INR_RATE} INR)')
            budget_matched = True
            break
    if not budget_matched:
        for pat in usd_patterns:
            m = re.search(pat, query_text, re.IGNORECASE)
            if m:
                val_usd = float(m.group(1).replace(",", ""))
                log(f'    * Budget       : MATCH -> "{m.group(0)}" => ${val_usd:,.2f} USD')
                budget_matched = True
                break
    if not budget_matched:
        log('    * Budget       : NO MATCH (defaulting to $500.00 USD)')

    # 4. Provider
    prov_match = re.search(r"\b(AWS|Azure|GCP)\b", query_text, re.IGNORECASE)
    if prov_match:
        prov_val = prov_match.group(1).upper()
        log(f'    * Provider     : MATCH -> "{prov_match.group(0)}" => "{prov_val}"')
    else:
        log('    * Provider     : NO MATCH (defaulting to candidate pool)')

    # 5. Latency & SLA
    lat_match = re.search(r"(?:max\s+|under\s+|latency\s+(?:of\s+|under\s+)?|\b)(\d+(?:\.\d+)?)\s*ms", query_text, re.IGNORECASE)
    if lat_match:
        log(f'    * Latency SLA  : MATCH -> "{lat_match.group(0)}" => {float(lat_match.group(1))} ms')
    sla_match = re.search(r"(9\d(?:\.\d+)?)\s*%", query_text, re.IGNORECASE)
    if sla_match:
        log(f'    * Availability : MATCH -> "{sla_match.group(0)}" => {float(sla_match.group(1))}% SLA')

    # 6. Dynamic Scaling / Traffic Keywords
    cpu_target_match = re.search(r"(?:target\s+cpu|cpu\s+target|cpu\s+utilization|target)\s*(?:of|at|is|:)?\s*(\d+(?:\.\d+)?)\s*%", query_text, re.IGNORECASE)
    if cpu_target_match:
        log(f'    * Target CPU   : MATCH -> "{cpu_target_match.group(0)}" => {float(cpu_target_match.group(1))}%')

    bw_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mbps|gbps|bandwidth)", query_text, re.IGNORECASE)
    if bw_match:
        log(f'    * Bandwidth    : MATCH -> "{bw_match.group(0)}" => {float(bw_match.group(1))} Mbps')

    # Domain Intent Check
    has_intent = parser.has_cloud_intent(query_text)
    log("\n  [1.2 Domain Intent & Contract Parameter Assembly]")
    if not has_intent:
        reason = "RESULT: PARSER_FAILED -- no archetype could be constructed from this input."
        log(f"    * Domain Intent: FAIL (no cloud/FinOps keywords detected)")
        log(f"    * {reason}")
        return False, None, reason

    log("    * Domain Intent: PASS (cloud infrastructure/FinOps keywords verified)")
    params = parser.extract_parameters(query_text)

    log("    * Extracted Parameters Table:")
    for k, v in params.items():
        log(f"      - {k:<22}: {v}")

    return True, params, None


def trace_stage_2_archetype_matching(
    matcher: CARMMatcher, query_text: str, parser: SCOPEParser, silent: bool = False
) -> Tuple[bool, Optional[str], Optional[str]]:
    """STAGE 2: Semantic Archetype Matching (CARM)."""
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 2: Semantic Archetype Matching (CARM) ---")
    extracted_constraints = parser.extract_constraints_from_text(query_text)
    log(f"  [2.1 Query Constraint Tokens]: {sorted(list(extracted_constraints)) if extracted_constraints else 'NONE'}")

    if not extracted_constraints:
        reason = "RESULT: UNSUPPORTED_ARCHETYPE -- no matching constraint tokens found."
        log(f"  {reason}")
        return False, None, reason

    scores = {}
    details = {}
    for archetype, data in matcher.TEMPLATE_INDEX.items():
        template_constraints = set(data["constraints"])
        intersection = extracted_constraints.intersection(template_constraints)
        union = extracted_constraints.union(template_constraints)
        score = len(intersection) / len(union) if union else 0.0
        scores[archetype] = round(score, 4)
        details[archetype] = {
            "template": data.get("template", ""),
            "inter_len": len(intersection),
            "union_len": len(union),
        }

    for arch, sc in scores.items():
        dt = details[arch]
        log(f"    * {arch:<28}: score {sc:.2f}  (|Q∩T|={dt['inter_len']}, |Q∪T|={dt['union_len']})")

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_arch, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score

    if best_score <= 0.0:
        reason = "RESULT: UNSUPPORTED_ARCHETYPE -- no archetype cleared matching threshold (> 0.0)."
        log(f"\n  {reason}")
        return False, None, reason

    log(f"\n  [2.3 Routing Decision] -> Selected: {best_arch} (margin: {margin:.2f})")
    return True, best_arch, None


def trace_stage_3_contract_validation(
    archetype: str, params: Dict[str, Any], silent: bool = False
) -> Tuple[bool, Optional[CloudOptimizationContract], Optional[str]]:
    """STAGE 3: Contract Validation & Mathematical Formulation."""
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 3: Contract Validation & Mathematical Formulation ---")
    try:
        contract = CloudOptimizationContract(
            problem_type=archetype,
            cloud_providers=params.get("cloud_providers", ["AWS"]),
            budget_max_usd=params.get("budget_max_usd", settings.DEFAULT_BUDGET_USD),
            service_count=params.get("service_count", 1),
            required_vcpus=params.get("required_vcpus", 1),
            required_ram_gb=params.get("required_ram_gb", 1.0),
            latency_max_ms=params.get("latency_max_ms", 100.0),
            sla_availability_pct=params.get("sla_availability_pct", 99.9),
            metadata=params.get("metadata", {}),
        )
        log(f"    Pydantic validation: PASS ({contract.problem_type})")
        return True, contract, None
    except Exception as exc:
        log(f"    Pydantic validation: FAIL ({exc})")
        return False, None, f"Pydantic validation failed: {exc}"


def trace_stage_4_solver_execution(
    contract: CloudOptimizationContract, mode: int, silent: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """STAGE 4: Solver Dispatch & Execution."""
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 4: Solver Dispatch & Execution ---")
    problem = contract.problem_type

    if problem == "ILP_VM_Allocation":
        log("  [4.1 Solver Engine: HiGHS MILP / Branch-and-Bound]")
        try:
            prov_list = contract.cloud_providers if contract.cloud_providers else ["AWS"]
            res = solve_ilp_vm_knapsack(
                required_vcpus=contract.required_vcpus,
                required_ram_gb=contract.required_ram_gb,
                budget_max_usd=contract.budget_max_usd,
                target_providers=prov_list,
            )
            log(f"    Status: {res.get('status')} | Cost: ${res.get('total_monthly_cost_usd', 0.0):,.2f} USD")
            return True, res, None
        except Exception as exc:
            return False, None, f"Solver exception: {exc}"

    elif problem == "PSO_Continuous_Scaling":
        log("  [4.1 Solver Engine: Continuous Vectorized PSO]")
        try:
            target_bw = (
                contract.target_bandwidth_mbps
                if contract.target_bandwidth_mbps is not None
                else (contract.min_bandwidth_mbps if contract.min_bandwidth_mbps is not None else 100.0)
            )
            res = solve_pso_continuous_scaling(
                bandwidth_min_mbps=float(contract.min_bandwidth_mbps or 100.0),
                bandwidth_max_mbps=float(contract.max_bandwidth_mbps or 1000.0),
                target_bandwidth_mbps=float(target_bw),
                target_cpu_pct=float(contract.target_cpu_pct or 70.0),
                max_cpu_pct=float(contract.max_cpu_pct) if contract.max_cpu_pct is not None else None,
                budget_max_usd=float(contract.budget_max_usd),
            )
            bw = res.get("optimal_bandwidth_mbps", 0.0)
            reps = res.get("recommended_replicas", 1)
            raw_cpu = (bw / (reps * 75.0)) * 100.0 if reps > 0 else 999.0
            log(f"    Status: {res.get('status')} | Bandwidth: {bw:.1f} Mbps | Replicas: {reps} | Modeled CPU: {raw_cpu:.1f}%")
            return True, res, None
        except Exception as exc:
            return False, None, f"Solver exception: {exc}"

    elif problem == "Z3_Graph_Disaster_Recovery":
        log("  [4.1 Solver Engine: Z3 SMT Graph Solver]")
        try:
            prov_list = contract.cloud_providers if contract.cloud_providers else ["AWS", "GCP"]
            res = solve_z3_graph_disaster_recovery(
                sla_pct=contract.sla_availability_pct,
                max_latency_ms=contract.latency_max_ms,
                budget_max_usd=contract.budget_max_usd,
                target_providers=prov_list,
            )
            log(f"    Status: {res.get('status')} | Primary: {res.get('primary_region')} | Secondary: {res.get('secondary_region')}")
            return True, res, None
        except Exception as exc:
            return False, None, f"Solver exception: {exc}"

    return False, None, f"Unknown problem type: {problem}"


def trace_stage_5_independent_verification(
    contract: CloudOptimizationContract,
    solver_result: Dict[str, Any],
    silent: bool = False,
    stage_header: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """STAGE 5: Independent Verification (IndependentChecker)."""
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    hdr = stage_header or "--- STAGE 5: Independent Verification (IndependentChecker) ---"
    log(f"\n{hdr}")
    check = IndependentChecker.verify_solution(contract, solver_result)

    param_table = IndependentChecker.render_parameter_table(check.get("parameter_checks", []))
    log("  [5.1 Parameter-by-Parameter Ground-Truth Audit Table]")
    log(param_table)

    violations = check.get("violations", [])
    log("\n  [5.2 Detected Constraint Violations]")
    if violations:
        for idx, v in enumerate(violations, 1):
            log(f"    -> [VIOLATION #{idx}] {v}")
    else:
        log("    * None (0 constraint violations found)")

    opt_verdict = check.get("optimality_verdict", "N/A")
    final_verdict = check.get("summary_status", "Execution complete")
    feas_pass = bool(check.get("feasible_against_contract", False))

    log(f"\n  [5.3 Verification Verdict]: {final_verdict}")
    log(f"  [5.4 Proof Classification]: {opt_verdict}")

    return feas_pass, check


def trace_stage_6_explanation(
    contract: CloudOptimizationContract,
    solver_result: Dict[str, Any],
    mode: int,
    check_result: Optional[Dict[str, Any]] = None,
    silent: bool = False,
) -> Tuple[str, ExplanationSource]:
    """STAGE 6: Explanation Generation (Mode 4 Only)."""
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 6: Explanation Generation (only for Mode 4) ---")
    if mode != 4:
        log("  [SKIPPED: Mode 3 is pure symbolic and does not generate natural-language explanations]")
        return "Skipped (Pure Symbolic)", ExplanationSource.UNAVAILABLE

    report_text = FinOpsExplainer.generate_report(
        contract=contract,
        solver_result=solver_result,
        check_result=check_result,
        enable_llm_explainer=True,
    )
    log(report_text)
    return report_text, ExplanationSource.LOCAL_TEMPLATE


def run_llm_mode_trace(
    query_text: str, mode: int, yes_flag: bool = False, silent: bool = False
) -> CanonicalExecutionRecord:
    """Executes a single run for Mode 1 (Raw LLM) or Mode 2 (Schema LLM)."""
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log(f'\n=== QUERY: "{query_text}" (Mode {mode}) ===\n')
    t_start = time.perf_counter()

    parser = SCOPEParser()
    matcher = CARMMatcher()
    contract, _, _ = parser.parse_query_to_contract(query_text)
    prob_type = contract.problem_type if contract else "ILP_VM_Allocation"

    model = os.getenv("GROQ_MODEL") or getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    provider = "Groq"
    execution_path = f"Groq API ({model}) -> {'OutputNormalizer (Prose Regex)' if mode==1 else 'OutputNormalizer (JSON Schema)'} -> IndependentChecker"

    from src.semantic.llm_client import execute_dashboard_llm_request

    t0_llm = time.perf_counter()
    llm_res = execute_dashboard_llm_request(mode_num=mode, query=query_text, contract=contract)
    elapsed_llm_ms = (time.perf_counter() - t0_llm) * 1000.0

    raw_content = llm_res.get("content", "")
    log("--- STAGE 1: LLM Response Telemetry ---")
    log(f"  Provider     : {provider} ({model})")
    log(f"  Finish Reason: {llm_res.get('finish_reason', 'unknown')}")
    log(f"  Elapsed Time : {elapsed_llm_ms:.1f} ms")
    log(f"  Content      :\n    {raw_content}")

    # Normalization
    t0_norm = time.perf_counter()
    if mode == 1:
        norm_status, ext_cost, norm_decision, norm_errors, evidence = OutputNormalizer.normalize_mode1_prose(
            raw_text=raw_content,
            problem_type=prob_type,
            contract_data=contract.model_dump() if contract else None,
        )
    else:
        norm_status, ext_cost, norm_decision, norm_errors, evidence = OutputNormalizer.normalize_mode2_json(
            raw_json_or_text=raw_content,
            requested_problem_type=prob_type,
            contract_data=contract.model_dump() if contract else None,
        )
    elapsed_norm_ms = (time.perf_counter() - t0_norm) * 1000.0

    log("\n--- STAGE 2: Normalization & Extraction ---")
    log(f"  Normalization Status: {norm_status.value}")
    log(f"  Extracted Monthly Cost: ${ext_cost:,.2f}" if ext_cost is not None else "  Extracted Monthly Cost: None")
    if norm_errors:
        for err in norm_errors:
            log(f"  Extraction Note: {err}")

    # Independent Verification
    t0_verif = time.perf_counter()
    candidate_dict = dict(norm_decision) if isinstance(norm_decision, dict) else {}
    candidate_dict["total_monthly_cost_usd"] = ext_cost
    candidate_dict["solver"] = "Raw_LLM_Prose" if mode == 1 else "Structured_JSON_LLM"
    candidate_dict["status"] = "feasible" if ext_cost is not None else "UNKNOWN"

    if contract and norm_status in [NormalizationStatus.SUCCESS, NormalizationStatus.NEEDS_REVIEW]:
        feas_pass, check = trace_stage_5_independent_verification(
            contract, candidate_dict, silent=silent, stage_header="--- STAGE 3: Independent Verification (IndependentChecker) ---"
        )
    else:
        check = {
            "feasible_against_contract": False,
            "optimality_verdict": OptimalityStatus.UNVERIFIED.value,
            "summary_status": f"Rejected ({norm_status.value})",
            "violations": norm_errors or ["Failed normalization/task check"],
            "parameter_checks": [],
        }
        feas_pass = False
        log(f"\n--- STAGE 3: Independent Verification: REJECTED ({norm_status.value}) ---")

    elapsed_verif_ms = (time.perf_counter() - t0_verif) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    record = CanonicalExecutionRecord(
        mode=mode,
        mode_name=f"Mode {mode}: {'Raw LLM' if mode==1 else 'Schema LLM'}",
        original_query=query_text,
        requirement_source="parsed_contract",
        problem_type=prob_type,
        requirements=contract.model_dump() if contract else {},
        execution_path=execution_path,
        provider=provider,
        model=model,
        original_response=raw_content,
        normalized_allocation=norm_decision,
        claimed_cost_usd=ext_cost,
        normalization_status=norm_status,
        normalization_errors=norm_errors,
        extracted_evidence=evidence,
        feasibility=FeasibilityStatus.PASS if feas_pass else FeasibilityStatus.FAIL,
        optimality_status=OptimalityStatus.UNVERIFIED,
        recomputed_cost_usd=check.get("cost_accuracy", {}).get("calculated_catalog_cost_usd"),
        cost_delta_usd=check.get("cost_accuracy", {}).get("cost_delta_usd"),
        cost_error_pct=check.get("cost_accuracy", {}).get("cost_error_pct"),
        violations=check.get("violations", []),
        summary_status=check.get("summary_status", "Rejected"),
        parsing_ms=elapsed_llm_ms,
        solving_ms=elapsed_norm_ms,
        verification_ms=elapsed_verif_ms,
        explanation_ms=0.0,
        total_duration_ms=total_ms,
        explanation_source=ExplanationSource.UNAVAILABLE,
        explanation_status="SKIPPED",
    )
    return record


def run_mode4_pipeline_trace(
    query_text: str, yes_flag: bool = False, silent: bool = False, offline: bool = False
) -> CanonicalExecutionRecord:
    """Runs a single query through the Mode 4 Neuro-Symbolic Pipeline with NVIDIA Neural Requirement Interpretation."""
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    t_start = time.perf_counter()
    log(f'\n=== QUERY: "{query_text}" (Mode 4: Neuro-Symbolic) ===\n')

    target_model = os.getenv("NVIDIA_MODEL") or getattr(settings, "NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")

    # STAGE 1: NVIDIA Neural Requirement Interpretation (Zero SCOPE/CARM)
    log("--- STAGE 1: Neural Requirement Interpretation (NVIDIA API) ---")
    log(f"  Provider     : NVIDIA ({target_model})")
    
    # In test/offline mode or when no API key is present without yes_flag, use offline mock
    is_offline = offline or (not yes_flag and not bool(os.getenv("NVIDIA_API_KEY")))
    t0_nvd = time.perf_counter()
    nvd_res = NVIDIAExtractor.extract_contract_from_query(query=query_text, offline=is_offline)
    elapsed_nvd_ms = (time.perf_counter() - t0_nvd) * 1000.0

    log(f"  Outcome      : {nvd_res.outcome.upper()} ({nvd_res.status})")
    log(f"  Latency      : {elapsed_nvd_ms:.1f} ms")

    if not nvd_res.is_executable:
        log("\n  [PIPELINE HALTED BEFORE SOLVER: Neural Interpretation Did Not Yield Executable Contract]")
        if nvd_res.clarification_questions:
            log("  Clarification Questions Required:")
            for q in nvd_res.clarification_questions:
                log(f"    ? {q}")
        if nvd_res.unsupported_reasons:
            log("  Unsupported Workload Reasons:")
            for r in nvd_res.unsupported_reasons:
                log(f"    ! {r}")
        if nvd_res.conflicting_reasons:
            log("  Conflicting Requirements Detected:")
            for r in nvd_res.conflicting_reasons:
                log(f"    ! {r}")
        if nvd_res.error_message:
            log(f"  Extraction Error: {nvd_res.error_message}")

        status_norm_map = {
            "NEEDS_CLARIFICATION": NormalizationStatus.CLARIFICATION_REQUIRED,
            "UNSUPPORTED": NormalizationStatus.TASK_INCOMPATIBLE,
            "CONFLICTING_REQUIREMENTS": NormalizationStatus.NORMALIZATION_FAILURE,
            "MISSING_CREDENTIALS": NormalizationStatus.API_FAILURE,
            "TIMEOUT": NormalizationStatus.API_FAILURE,
            "NETWORK_ERROR": NormalizationStatus.API_FAILURE,
            "MALFORMED_JSON": NormalizationStatus.MALFORMED_OUTPUT,
            "SCHEMA_ERROR": NormalizationStatus.MALFORMED_OUTPUT,
        }
        norm_stat = status_norm_map.get(nvd_res.status, NormalizationStatus.NORMALIZATION_FAILURE)
        violations = (
            nvd_res.clarification_questions
            or nvd_res.unsupported_reasons
            or nvd_res.conflicting_reasons
            or ([nvd_res.error_message] if nvd_res.error_message else ["Neural extraction rejected query"])
        )
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return CanonicalExecutionRecord(
            mode=4,
            mode_name="Mode 4: Neuro-Symbolic",
            original_query=query_text,
            requirement_source="nvidia_neural_contract",
            problem_type=nvd_res.contract.problem_type if nvd_res.contract else "ILP_VM_Allocation",
            requirements=nvd_res.contract.model_dump() if nvd_res.contract else (nvd_res.parsed_json or {}),
            execution_path=f"NVIDIA API ({nvd_res.model}) -> Interpretation Halted ({nvd_res.status})",
            provider="NVIDIA",
            model=nvd_res.model,
            original_response=nvd_res.raw_response,
            normalized_allocation=None,
            claimed_cost_usd=None,
            normalization_status=norm_stat,
            normalization_errors=violations,
            extracted_evidence=nvd_res.extracted_evidence,
            feasibility=FeasibilityStatus.FAIL,
            optimality_status=OptimalityStatus.UNVERIFIED,
            violations=violations,
            summary_status=f"STAGE 1 (PARSER_FAILED / INTERPRETATION_HALTED): {nvd_res.status}",
            failed_stage=1,
            parsing_ms=elapsed_nvd_ms,
            solving_ms=0.0,
            verification_ms=0.0,
            explanation_ms=0.0,
            total_duration_ms=total_ms,
            explanation_source=ExplanationSource.UNAVAILABLE,
            explanation_status="SKIPPED",
        )

    contract = nvd_res.contract

    # STAGE 2: Contract Schema & Archetype Verification
    log("\n--- STAGE 2: Contract Schema & Archetype Verification ---")
    log(f"  * Archetype Dispatched : {contract.problem_type}")
    log(f"  * Stated Budget Ceiling: ${contract.budget_max_usd:,.2f} USD")
    if contract.problem_type == "ILP_VM_Allocation":
        log(f"  * Compute Target       : {contract.required_vcpus} vCPUs, {contract.required_ram_gb:.1f} GB RAM")
    elif contract.problem_type == "PSO_Continuous_Scaling":
        log(f"  * Target Utilization   : {contract.target_cpu_pct or 70.0:.1f}% CPU")
    elif contract.problem_type == "Z3_Graph_Disaster_Recovery":
        log(f"  * DR Bounds            : <= {contract.latency_max_ms:.1f}ms latency, >= {contract.sla_availability_pct:.2f}% SLA")

    # STAGE 3 & 4: Solver Dispatch & Execution
    t0_solv = time.perf_counter()
    s4_ok, solver_result, s4_err = trace_stage_4_solver_execution(contract, mode=4, silent=silent)
    solving_ms = (time.perf_counter() - t0_solv) * 1000.0
    if not s4_ok or solver_result is None:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return CanonicalExecutionRecord(
            mode=4,
            mode_name="Mode 4: Neuro-Symbolic",
            original_query=query_text,
            requirement_source="nvidia_neural_contract",
            problem_type=contract.problem_type,
            requirements=contract.model_dump(),
            normalization_status=NormalizationStatus.SOLVER_INFEASIBLE,
            summary_status=s4_err or "Solver failed",
            failed_stage=4,
            parsing_ms=elapsed_nvd_ms,
            solving_ms=solving_ms,
            total_duration_ms=total_ms,
        )

    # STAGE 5: Independent Verification (Mode-Blind)
    t0_verif = time.perf_counter()
    s5_ok, check_result = trace_stage_5_independent_verification(contract, solver_result, silent=silent)
    verification_ms = (time.perf_counter() - t0_verif) * 1000.0

    # STAGE 6: Explanation Generation
    t0_exp = time.perf_counter()
    exp_text, exp_source = trace_stage_6_explanation(contract, solver_result, 4, check_result=check_result, silent=silent)
    explanation_ms = (time.perf_counter() - t0_exp) * 1000.0

    total_ms = (time.perf_counter() - t_start) * 1000.0

    cost = solver_result.get("total_monthly_cost_usd", solver_result.get("estimated_monthly_cost_usd", 0.0))
    recomp_cost = check_result.get("cost_accuracy", {}).get("calculated_catalog_cost_usd")

    record = CanonicalExecutionRecord(
        mode=4,
        mode_name="Mode 4: Neuro-Symbolic",
        original_query=query_text,
        requirement_source="nvidia_neural_contract",
        problem_type=contract.problem_type,
        requirements=contract.model_dump(),
        execution_path=f"NVIDIA API ({nvd_res.model}) -> Local Solver ({solver_result.get('solver', 'Solver')}) -> IndependentChecker -> " + ("Local Explainer Template" if exp_source == ExplanationSource.LOCAL_TEMPLATE else "NVIDIA Stage 6 Explainer"),
        provider="NVIDIA",
        model=nvd_res.model,
        solver_name=solver_result.get("solver"),
        original_response=solver_result,
        normalized_allocation=solver_result,
        claimed_cost_usd=float(cost) if cost is not None else None,
        normalization_status=NormalizationStatus.SUCCESS,
        extracted_evidence=nvd_res.extracted_evidence,
        feasibility=FeasibilityStatus.PASS if s5_ok else FeasibilityStatus.FAIL,
        optimality_status=OptimalityStatus(check_result.get("optimality_verdict", OptimalityStatus.INFEASIBLE.value)) if check_result.get("optimality_verdict") in [e.value for e in OptimalityStatus] else OptimalityStatus.HEURISTIC_FEASIBLE,
        recomputed_cost_usd=float(recomp_cost) if recomp_cost is not None else None,
        cost_delta_usd=check_result.get("cost_accuracy", {}).get("cost_delta_usd"),
        cost_error_pct=check_result.get("cost_accuracy", {}).get("cost_error_pct"),
        budget_headroom_usd=max(0.0, float(contract.budget_max_usd) - float(cost or 0.0)) if s5_ok else None,
        violations=check_result.get("violations", []),
        summary_status=check_result.get("summary_status", "Execution complete"),
        failed_stage=None if s5_ok else 5,
        parsing_ms=elapsed_nvd_ms,
        solving_ms=solving_ms,
        verification_ms=verification_ms,
        explanation_ms=explanation_ms,
        total_duration_ms=total_ms,
        explanation_source=exp_source,
        explanation_status="SUCCESS" if s5_ok else "SUPPRESSED_DUE_TO_VIOLATION",
        explanation_text=exp_text,
    )
    return record


def run_mode3_pipeline_trace(
    query_text: str, yes_flag: bool = False, silent: bool = False, offline: bool = True
) -> CanonicalExecutionRecord:
    """Runs a single query through the Mode 3 Pure Symbolic Pipeline (SCOPE -> CARM -> Solver -> Checker)."""
    # MODE 3: Pure Symbolic (SCOPE Lexical -> CARM Archetype -> Local Solvers -> IndependentChecker)
    t_start = time.perf_counter()
    if not silent:
        print(f'\n=== QUERY: "{query_text}" (Mode 3: Pure Symbolic) ===\n')

    parser = SCOPEParser()
    matcher = CARMMatcher()

    # STAGE 1: Lexical
    t0 = time.perf_counter()
    s1_ok, params, s1_err = trace_stage_1_parsing(parser, query_text, silent=silent)
    elapsed_p1 = (time.perf_counter() - t0) * 1000.0
    if not s1_ok or params is None:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return CanonicalExecutionRecord(
            mode=3,
            mode_name="Mode 3: Pure Symbolic",
            original_query=query_text,
            normalization_status=NormalizationStatus.NORMALIZATION_FAILURE,
            summary_status=f"STAGE 1 (PARSER_FAILED): {s1_err or 'Lexical parsing failed'}",
            failed_stage=1,
            total_duration_ms=total_ms,
        )

    # STAGE 2: Archetype
    t0 = time.perf_counter()
    s2_ok, archetype, s2_err = trace_stage_2_archetype_matching(matcher, query_text, parser, silent=silent)
    elapsed_p2 = (time.perf_counter() - t0) * 1000.0
    if not s2_ok or archetype is None:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return CanonicalExecutionRecord(
            mode=3,
            mode_name="Mode 3: Pure Symbolic",
            original_query=query_text,
            normalization_status=NormalizationStatus.TASK_INCOMPATIBLE,
            summary_status=f"STAGE 1 (PARSER_FAILED): {s2_err or 'Unsupported archetype'}",
            failed_stage=1,
            total_duration_ms=total_ms,
        )

    # STAGE 3: Contract Validation
    t0 = time.perf_counter()
    s3_ok, contract, s3_err = trace_stage_3_contract_validation(archetype, params, silent=silent)
    elapsed_p3 = (time.perf_counter() - t0) * 1000.0
    if not s3_ok or contract is None:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return CanonicalExecutionRecord(
            mode=3,
            mode_name="Mode 3: Pure Symbolic",
            original_query=query_text,
            problem_type=archetype,
            normalization_status=NormalizationStatus.MALFORMED_OUTPUT,
            summary_status=s3_err or "Contract validation failed",
            failed_stage=3,
            total_duration_ms=total_ms,
        )

    parsing_total_ms = elapsed_p1 + elapsed_p2 + elapsed_p3

    # STAGE 4: Solver
    t0 = time.perf_counter()
    s4_ok, solver_result, s4_err = trace_stage_4_solver_execution(contract, mode=3, silent=silent)
    solving_ms = (time.perf_counter() - t0) * 1000.0
    if not s4_ok or solver_result is None:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return CanonicalExecutionRecord(
            mode=3,
            mode_name="Mode 3: Pure Symbolic",
            original_query=query_text,
            problem_type=archetype,
            requirements=contract.model_dump(),
            normalization_status=NormalizationStatus.SOLVER_INFEASIBLE,
            summary_status=s4_err or "Solver failed",
            failed_stage=4,
            parsing_ms=parsing_total_ms,
            solving_ms=solving_ms,
            total_duration_ms=total_ms,
        )

    # STAGE 5: Verification (Mode-Blind)
    t0 = time.perf_counter()
    s5_ok, check_result = trace_stage_5_independent_verification(contract, solver_result, silent=silent)
    verification_ms = (time.perf_counter() - t0) * 1000.0

    # STAGE 6: Explanation (Mode 3 skips explanation)
    trace_stage_6_explanation(contract, solver_result, mode=3, check_result=check_result, silent=silent)

    total_ms = (time.perf_counter() - t_start) * 1000.0

    cost = solver_result.get("total_monthly_cost_usd", solver_result.get("estimated_monthly_cost_usd", 0.0))
    recomp_cost = check_result.get("cost_accuracy", {}).get("calculated_catalog_cost_usd")

    record = CanonicalExecutionRecord(
        mode=3,
        mode_name="Mode 3: Pure Symbolic",
        original_query=query_text,
        requirement_source="parsed_contract",
        problem_type=contract.problem_type,
        requirements=contract.model_dump(),
        execution_path=f"Local SCOPE -> Local CARM ({solver_result.get('solver', 'Solver')}) -> IndependentChecker",
        solver_name=solver_result.get("solver"),
        original_response=solver_result,
        normalized_allocation=solver_result,
        claimed_cost_usd=float(cost) if cost is not None else None,
        normalization_status=NormalizationStatus.SUCCESS,
        feasibility=FeasibilityStatus.PASS if s5_ok else FeasibilityStatus.FAIL,
        optimality_status=OptimalityStatus(check_result.get("optimality_verdict", OptimalityStatus.INFEASIBLE.value)) if check_result.get("optimality_verdict") in [e.value for e in OptimalityStatus] else OptimalityStatus.HEURISTIC_FEASIBLE,
        recomputed_cost_usd=float(recomp_cost) if recomp_cost is not None else None,
        cost_delta_usd=check_result.get("cost_accuracy", {}).get("cost_delta_usd"),
        cost_error_pct=check_result.get("cost_accuracy", {}).get("cost_error_pct"),
        budget_headroom_usd=max(0.0, float(contract.budget_max_usd) - float(cost or 0.0)) if s5_ok else None,
        violations=check_result.get("violations", []),
        summary_status=check_result.get("summary_status", "Execution complete"),
        failed_stage=None if s5_ok else 5,
        parsing_ms=parsing_total_ms,
        solving_ms=solving_ms,
        verification_ms=verification_ms,
        explanation_ms=0.0,
        total_duration_ms=total_ms,
        explanation_source=ExplanationSource.UNAVAILABLE,
        explanation_status="SKIPPED",
        explanation_text="",
    )
    return record


def run_pipeline_trace(
    query_text: str, mode: int = 4, yes_flag: bool = False, silent: bool = False, offline: bool = False
) -> CanonicalExecutionRecord:
    """Runs a single query through the genuine pipeline and returns a CanonicalExecutionRecord."""
    if mode in [1, 2]:
        return run_llm_mode_trace(query_text, mode=mode, yes_flag=yes_flag, silent=silent)
    elif mode == 3:
        return run_mode3_pipeline_trace(query_text, yes_flag=yes_flag, silent=silent, offline=offline)
    elif mode == 4:
        return run_mode4_pipeline_trace(query_text, yes_flag=yes_flag, silent=silent, offline=offline)
    else:
        raise ValueError(f"Unknown mode {mode}. Expected 1, 2, 3, or 4.")


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
        raw_list = data.get("queries", [])
        return [
            {"id": item.get("id", f"query_{idx+1}"), "query": item.get("query", ""), "raw": item}
            if isinstance(item, dict)
            else {"id": f"query_{idx+1}", "query": str(item), "raw": item}
            for idx, item in enumerate(raw_list)
        ]
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
        help="Pipeline execution mode: 1 (Pure LLM), 2 (Structured LLM), 3 (Symbolic), or 4 (Full Neuro-Symbolic)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip live API confirmation prompt for Mode 1 and Mode 2",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Batch mode: only print full stage traces for queries that fail",
    )

    args = parser.parse_args()

    if args.query:
        record = run_pipeline_trace(args.query, mode=args.mode, yes_flag=args.yes, silent=False)
        print("\n" + "=" * 80)
        print(f"CANONICAL EXECUTION RECORD (Run ID: {record.run_id})")
        print("=" * 80)
        print(f"Mode          : {record.mode_name}")
        print(f"Feasibility   : {record.feasibility.value}")
        print(f"Optimality    : {record.optimality_status.value}")
        print(f"Verdict       : {record.summary_status}")
        print(f"Claimed Cost  : ${record.claimed_cost_usd:,.2f}" if record.claimed_cost_usd is not None else "Claimed Cost  : N/A")
        print(f"Recomputed    : ${record.recomputed_cost_usd:,.2f}" if record.recomputed_cost_usd is not None else "Recomputed    : N/A")
        print(f"Total Latency : {record.total_duration_ms:.1f} ms")
        print("=" * 80 + "\n")
        return

    queries_file = args.queries_file or "data/diagnostic_queries.json"
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data if isinstance(data, list) else data.get("queries", [])

    print(f"=== BATCH STAGE TRACE: {len(queries)} queries (Mode {args.mode}) ===")
    results = []
    for q_item in queries:
        q_text = q_item.get("query", q_item) if isinstance(q_item, dict) else q_item
        rec = run_pipeline_trace(q_text, mode=args.mode, yes_flag=args.yes, silent=args.failures_only)
        results.append(rec)
        if args.failures_only and rec.feasibility == FeasibilityStatus.PASS:
            print(f'PASS: "{q_text}" -> {rec.summary_status}')

    passed = sum(1 for r in results if r.feasibility == FeasibilityStatus.PASS)
    print(f"\nBATCH SUMMARY: Total: {len(results)} | Passed: {passed} | Failed: {len(results)-passed}")


if __name__ == "__main__":
    main()
