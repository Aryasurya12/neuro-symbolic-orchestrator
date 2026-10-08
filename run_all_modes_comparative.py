"""Universal Comparative Runner Script for Neurasym.

Executes all 4 operational paradigm modes sequentially for any user-provided cloud query:
  - Mode 1: Raw LLM (Unstructured Prose via Groq API)
  - Mode 2: Schema-Constrained LLM (JSON Schema via Groq API)
  - Mode 3: Pure Symbolic Pipeline (100% Offline Local Solvers + IndependentChecker)
  - Mode 4: Full Neuro-Symbolic Pipeline (Local Symbolic Solvers + IndependentChecker + NVIDIA API Stage 6 Explainer)

Eliminates contradictory verdicts by generating one canonical per-run result for each mode
evaluated by the identical, mode-blind IndependentChecker.

CLI Usage:
    python run_all_modes_comparative.py
    python run_all_modes_comparative.py --query "Deploy 8 vCPUs and 16GB RAM for under $300 on AWS"
    python run_all_modes_comparative.py --query "Continuous dynamic scaling with target CPU 70% under $1500"
    python run_all_modes_comparative.py --query "Topological DR across us-east-1 and us-west-2 with 99.99% SLA under $850"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

from config.settings import settings
from src.proof.stage_trace import (
    trace_stage_1_parsing,
    trace_stage_2_archetype_matching,
    trace_stage_3_contract_validation,
    trace_stage_4_solver_execution,
    trace_stage_5_independent_verification,
    trace_stage_6_explanation,
)
from src.semantic.carm_matcher import CARMMatcher
from src.semantic.explainer import FinOpsExplainer
from src.semantic.llm_client import execute_dashboard_llm_request
from src.semantic.normalizer import OutputNormalizer
from src.semantic.nvidia_extractor import NVIDIAExtractor
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.scope_parser import SCOPEParser
from src.verifiers.canonical_record import (
    AuditEvent,
    CanonicalExecutionRecord,
    CheckStatus,
    ExplanationSource,
    FeasibilityStatus,
    NormalizationStatus,
    OptimalityStatus,
)
from src.verifiers.independent_checker import IndependentChecker


def _extract_audit_events(parameter_checks: List[Dict[str, Any]]) -> List[AuditEvent]:
    """Helper to reconstruct AuditEvent objects from parameter checks dict list."""
    events = []
    for p in parameter_checks:
        if isinstance(p, dict):
            status_raw = p.get("status", "FAIL")
            if isinstance(status_raw, CheckStatus):
                c_status = status_raw
            else:
                c_status = CheckStatus(status_raw) if status_raw in [e.value for e in CheckStatus] else CheckStatus.FAIL
            events.append(
                AuditEvent(
                    check_id=p.get("check_id", "CHK_GENERAL"),
                    check_name=p.get("check_name", p.get("name", "Audit Check")),
                    observed_value=p.get("observed_value", p.get("measured", "")),
                    required_value=p.get("required_value", p.get("target", "")),
                    operator=p.get("operator", "=="),
                    source=p.get("source", ""),
                    formula=p.get("formula", ""),
                    recomputed_result=p.get("recomputed_result"),
                    input_values=p.get("input_values"),
                    provenance=p.get("provenance", p.get("source", "")),
                    tolerance=p.get("tolerance", ""),
                    signed_margin=p.get("signed_margin", p.get("delta", "")),
                    status=c_status,
                    reason=p.get("reason", ""),
                )
            )
    return events


def print_banner(title: str, subtitle: Optional[str] = None, char: str = "=") -> None:
    width = 88
    print("\n" + char * width)
    print(f" {title}".center(width))
    if subtitle:
        print(f" {subtitle}".center(width))
    print(char * width + "\n")


def execute_mode_1_raw_llm(
    query_text: str,
    contract: Optional[CloudOptimizationContract],
    mock_llm: bool = False,
) -> CanonicalExecutionRecord:
    """MODE 1: Raw Unconstrained LLM (Groq API)."""
    model_name = os.getenv("GROQ_MODEL") or getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    print_banner(
        "MODE 1: RAW UNCONSTRAINED LLM",
        f"Provider: Groq API | Model: {model_name}" + (" [OFFLINE MOCK FIXTURE]" if mock_llm else ""),
        char="=",
    )
    print("  [Stage 1.1: Request Formulation]")
    print(f'    Query: "{query_text}"')
    print("    Prompting Strategy: Free-form Natural Language Prose (No Schema, No Solver)")

    t_start = time.perf_counter()
    if mock_llm:
        if "scaling" in query_text.lower():
            raw_content = "For continuous dynamic scaling with target CPU 70% under $1500, we recommend provisioning 100 Mbps bandwidth with 1 worker replica at an estimated total cost of $53.00/month ($0.08 per Mbps + $45.00 base replica fee)."
        elif "disaster recovery" in query_text.lower() or "dr" in query_text.lower():
            raw_content = "For multi-region disaster recovery between AWS and GCP with 99.99% SLA, we recommend primary in us-east-1 and secondary in us-central1. Total cost is $450.00/month with 42ms cross-region sync."
        else:
            raw_content = "To deploy your workload requiring 8 vCPUs and 16GB RAM on AWS, we recommend 2x t3.xlarge instances. The estimated total monthly cost is $300.21/month with 730 hours of continuous run time."
        llm_res = {"content": raw_content, "finish_reason": "stop", "status": "mock_fixture"}
    else:
        llm_res = execute_dashboard_llm_request(mode_num=1, query=query_text, contract=contract)
    elapsed_llm_ms = (time.perf_counter() - t_start) * 1000.0

    raw_content = llm_res.get("content", "")
    finish_reason = llm_res.get("finish_reason", "unknown")
    status = llm_res.get("status", "unknown")
    attempts = llm_res.get("attempts", [])

    print("\n  [Stage 1.2: " + ("Mock LLM Fixture]" if mock_llm else "Live LLM Response]"))
    print(f"    Status        : {status.upper()}")
    print(f"    Finish Reason : {finish_reason}")
    print(f"    Latency       : {elapsed_llm_ms:.1f} ms ({elapsed_llm_ms/1000.0:.2f}s)")
    if attempts:
        print(f"    Attempts Made : {len(attempts)}")
        for att in attempts:
            print(f"      - Attempt {att.get('attempt')}: max_tokens={att.get('max_tokens')}, finish_reason={att.get('finish_reason')}, status={att.get('status')}")
    print("    Raw Generation Content:")
    print("    " + "-" * 78)
    for line in raw_content.split("\n"):
        print(f"      {line}")
    print("    " + "-" * 78)

    # Stage 1.3: Robust Prose Normalization
    print("\n  [Stage 1.3: Information Extraction & Semantic Normalization]")
    if contract is None:
        try:
            contract = SCOPEParser.parse_query_to_contract(query_text)
        except Exception:
            contract = None
    prob_type = contract.problem_type if contract else "ILP_VM_Allocation"
    t_norm = time.perf_counter()
    norm_status, ext_cost, norm_decision, norm_errors, evidence = OutputNormalizer.normalize_mode1_prose(
        raw_text=raw_content,
        problem_type=prob_type,
        contract_data=contract.model_dump() if contract else None,
        finish_reason=finish_reason,
    )
    elapsed_norm_ms = (time.perf_counter() - t_norm) * 1000.0

    print(f"    Normalization Status : {norm_status.value}")
    if ext_cost is not None:
        print(f"    Extracted Cost Claim : ${ext_cost:,.2f} USD / month")
    else:
        print("    Extracted Cost Claim : NONE (No parseable dollar figure)")
    if norm_errors:
        for err in norm_errors:
            print(f"    Normalization Note   : {err}")

    # Stage 1.4: Independent Mathematical Verification
    print("\n  [Stage 1.4: Independent Mathematical Verification (IndependentChecker)]")
    t_verif = time.perf_counter()
    candidate_dict = dict(norm_decision) if isinstance(norm_decision, dict) else {}
    candidate_dict["total_monthly_cost_usd"] = ext_cost
    candidate_dict["solver"] = "Raw_LLM_Prose"
    candidate_dict["status"] = "feasible" if ext_cost is not None else "UNKNOWN"

    if contract and norm_status in [NormalizationStatus.SUCCESS, NormalizationStatus.NEEDS_REVIEW]:
        feas_pass, check = trace_stage_5_independent_verification(
            contract, candidate_dict, silent=False, stage_header="  --- Mode 1 Parameter-Wise Independent Verification ---"
        )
        opt_verdict_stat = OptimalityStatus(check.get("optimality_verdict", OptimalityStatus.UNVERIFIED.value)) if check.get("optimality_verdict") in [e.value for e in OptimalityStatus] else OptimalityStatus.UNVERIFIED
    else:
        opt_verdict_stat = OptimalityStatus.TRUNCATION_FAILURE if norm_status == NormalizationStatus.TRUNCATION_FAILURE else OptimalityStatus.UNVERIFIED
        check = {
            "feasible_against_contract": False,
            "optimality_verdict": opt_verdict_stat.value,
            "summary_status": f"Rejected ({norm_status.value})",
            "violations": norm_errors or ["Failed extraction/normalization check"],
            "parameter_checks": [],
        }
        feas_pass = False
        print(f"    * Constraint Feasibility   : FAIL")
        print(f"    * FINAL VERDICT            : Rejected ({norm_status.value})")

    elapsed_verif_ms = (time.perf_counter() - t_verif) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    record = CanonicalExecutionRecord(
        mode=1,
        mode_name="Mode 1: Raw LLM",
        original_query=query_text,
        requirement_source="parsed_contract",
        problem_type=prob_type,
        requirements=contract.model_dump() if contract else {},
        execution_path=f"Groq API ({model_name}) -> OutputNormalizer (Prose Regex) -> IndependentChecker",
        provider="Groq",
        model=model_name,
        original_response=raw_content,
        normalized_allocation=norm_decision,
        claimed_cost_usd=ext_cost,
        normalization_status=norm_status,
        normalization_errors=norm_errors,
        extracted_evidence=evidence,
        feasibility=FeasibilityStatus.PASS if feas_pass else FeasibilityStatus.FAIL,
        optimality_status=opt_verdict_stat,
        recomputed_cost_usd=check.get("cost_accuracy", {}).get("calculated_catalog_cost_usd"),
        cost_delta_usd=check.get("cost_accuracy", {}).get("cost_delta_usd"),
        cost_error_pct=check.get("cost_accuracy", {}).get("cost_error_pct"),
        violations=check.get("violations", []),
        audit_events=_extract_audit_events(check.get("parameter_checks", [])),
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


def execute_mode_2_schema_llm(
    query_text: str,
    contract: Optional[CloudOptimizationContract],
    mock_llm: bool = False,
) -> CanonicalExecutionRecord:
    """MODE 2: Schema-Constrained LLM (Groq API)."""
    model_name = os.getenv("GROQ_MODEL") or getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    print_banner(
        "MODE 2: SCHEMA-CONSTRAINED LLM",
        f"Provider: Groq API | Model: {model_name}" + (" [OFFLINE MOCK FIXTURE]" if mock_llm else ""),
        char="=",
    )
    print("  [Stage 2.1: JSON Schema Prompt Formulation]")
    print(f'    Query: "{query_text}"')
    print("    Prompting Strategy: Strict Pydantic JSON Schema Direct Inference")

    t_start = time.perf_counter()
    if mock_llm:
        if "scaling" in query_text.lower():
            parsed_json = {
                "bandwidth_mbps": 100.0,
                "replicas": 1,
                "total_monthly_cost_usd": 53.00,
            }
            raw_content = json.dumps(parsed_json, indent=2)
        elif "disaster recovery" in query_text.lower() or "dr" in query_text.lower():
            parsed_json = {
                "primary_region": "us-east-1",
                "secondary_region": "us-central1",
                "total_monthly_cost_usd": 450.0,
                "latency_ms": 42.0,
                "achieved_sla_pct": 99.99,
            }
            raw_content = json.dumps(parsed_json, indent=2)
        else:
            parsed_json = {
                "allocated_vms": [
                    {"sku": "t3.xlarge", "provider": "AWS", "quantity": 2, "monthly_cost": 300.21, "vcpus": 8, "ram_gb": 16.0}
                ],
                "total_monthly_cost_usd": 300.21,
            }
            raw_content = json.dumps(parsed_json, indent=2)
        llm_res = {"content": raw_content, "parsed_json": parsed_json, "finish_reason": "stop", "status": "mock_fixture"}
    else:
        llm_res = execute_dashboard_llm_request(mode_num=2, query=query_text, contract=contract)
    elapsed_llm_ms = (time.perf_counter() - t_start) * 1000.0

    raw_content = llm_res.get("content", "")
    parsed_json = llm_res.get("parsed_json", {})
    finish_reason = llm_res.get("finish_reason", "unknown")
    status = llm_res.get("status", "unknown")
    attempts = llm_res.get("attempts", [])

    print("\n  [Stage 2.2: " + ("Mock Structured Fixture]" if mock_llm else "Live Structured Response]"))
    print(f"    Status        : {status.upper()}")
    print(f"    Finish Reason : {finish_reason}")
    print(f"    Latency       : {elapsed_llm_ms:.1f} ms ({elapsed_llm_ms/1000.0:.2f}s)")
    if attempts:
        print(f"    Attempts Made : {len(attempts)}")
        for att in attempts:
            print(f"      - Attempt {att.get('attempt')}: max_tokens={att.get('max_tokens')}, finish_reason={att.get('finish_reason')}, status={att.get('status')}")
    print("    Emitted JSON Object:")
    print("    " + "-" * 78)
    if parsed_json:
        print("      " + json.dumps(parsed_json, indent=6).replace("\n", "\n      "))
    else:
        print(f"      {raw_content}")
    print("    " + "-" * 78)

    # Stage 2.3: Robust JSON Schema Normalization
    print("\n  [Stage 2.3: Information Extraction & Task Suitability Check]")
    if contract is None:
        try:
            contract = SCOPEParser.parse_query_to_contract(query_text)
        except Exception:
            contract = None
    prob_type = contract.problem_type if contract else "ILP_VM_Allocation"
    t_norm = time.perf_counter()
    norm_status, ext_cost, norm_decision, norm_errors, evidence = OutputNormalizer.normalize_mode2_json(
        raw_json_or_text=parsed_json or raw_content,
        requested_problem_type=prob_type,
        contract_data=contract.model_dump() if contract else None,
        finish_reason=finish_reason,
    )
    elapsed_norm_ms = (time.perf_counter() - t_norm) * 1000.0

    print(f"    Normalization Status : {norm_status.value}")
    if ext_cost is not None:
        print(f"    Extracted Cost Claim : ${ext_cost:,.2f} USD / month")
    if norm_errors:
        for err in norm_errors:
            print(f"    Normalization Note   : {err}")

    # Stage 2.4: Independent Mathematical Verification
    print("\n  [Stage 2.4: Independent Mathematical Verification (IndependentChecker)]")
    t_verif = time.perf_counter()
    candidate_dict = dict(norm_decision) if isinstance(norm_decision, dict) else {}
    candidate_dict["total_monthly_cost_usd"] = ext_cost
    candidate_dict["solver"] = "Structured_JSON_LLM"
    candidate_dict["status"] = "feasible" if ext_cost is not None else "UNKNOWN"

    if contract and norm_status in [NormalizationStatus.SUCCESS, NormalizationStatus.NEEDS_REVIEW]:
        feas_pass, check = trace_stage_5_independent_verification(
            contract, candidate_dict, silent=False, stage_header="  --- Mode 2 Parameter-Wise Independent Verification ---"
        )
        opt_verdict_stat = OptimalityStatus(check.get("optimality_verdict", OptimalityStatus.UNVERIFIED.value)) if check.get("optimality_verdict") in [e.value for e in OptimalityStatus] else OptimalityStatus.UNVERIFIED
    else:
        opt_verdict_stat = OptimalityStatus.TRUNCATION_FAILURE if norm_status == NormalizationStatus.TRUNCATION_FAILURE else OptimalityStatus.UNVERIFIED
        check = {
            "feasible_against_contract": False,
            "optimality_verdict": opt_verdict_stat.value,
            "summary_status": f"Rejected ({norm_status.value})",
            "violations": norm_errors or ["Failed schema/task validation"],
            "parameter_checks": [],
        }
        feas_pass = False
        print(f"    * Constraint Feasibility   : FAIL")
        print(f"    * FINAL VERDICT            : Rejected ({norm_status.value})")

    elapsed_verif_ms = (time.perf_counter() - t_verif) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    record = CanonicalExecutionRecord(
        mode=2,
        mode_name="Mode 2: Schema LLM",
        original_query=query_text,
        requirement_source="parsed_contract",
        problem_type=prob_type,
        requirements=contract.model_dump() if contract else {},
        execution_path=f"Groq API ({model_name}) -> OutputNormalizer (JSON Schema) -> IndependentChecker",
        provider="Groq",
        model=model_name,
        original_response=raw_content,
        normalized_allocation=norm_decision,
        claimed_cost_usd=ext_cost,
        normalization_status=norm_status,
        normalization_errors=norm_errors,
        extracted_evidence=evidence,
        feasibility=FeasibilityStatus.PASS if feas_pass else FeasibilityStatus.FAIL,
        optimality_status=opt_verdict_stat,
        recomputed_cost_usd=check.get("cost_accuracy", {}).get("calculated_catalog_cost_usd"),
        cost_delta_usd=check.get("cost_accuracy", {}).get("cost_delta_usd"),
        cost_error_pct=check.get("cost_accuracy", {}).get("cost_error_pct"),
        violations=check.get("violations", []),
        audit_events=_extract_audit_events(check.get("parameter_checks", [])),
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


def execute_mode_3_symbolic(
    query_text: str,
) -> Tuple[CanonicalExecutionRecord, CloudOptimizationContract, Dict[str, Any]]:
    """MODE 3: Pure Symbolic Pipeline (100% Offline & Deterministic)."""
    print_banner(
        "MODE 3: PURE SYMBOLIC PIPELINE",
        "100% Offline | Rule-Based Parser (SCOPE) + Solvers (HiGHS/PSO/Z3) + IndependentChecker",
        char="=",
    )
    t_start = time.perf_counter()
    parser = SCOPEParser()
    matcher = CARMMatcher()

    # Stage 1: Parsing
    t0 = time.perf_counter()
    s1_ok, params, s1_err = trace_stage_1_parsing(parser, query_text)
    p1_ms = (time.perf_counter() - t0) * 1000.0

    # Stage 2: Archetype Matching
    t0 = time.perf_counter()
    s2_ok, archetype, s2_err = trace_stage_2_archetype_matching(matcher, query_text, parser)
    p2_ms = (time.perf_counter() - t0) * 1000.0
    if not archetype:
        archetype = "ILP_VM_Allocation"

    # Stage 3: Contract Validation
    t0 = time.perf_counter()
    s3_ok, contract, s3_err = trace_stage_3_contract_validation(archetype, params or {})
    p3_ms = (time.perf_counter() - t0) * 1000.0
    if not contract:
        contract = CloudOptimizationContract(problem_type="ILP_VM_Allocation")

    parsing_total_ms = p1_ms + p2_ms + p3_ms

    # Stage 4: Solver
    t0 = time.perf_counter()
    s4_ok, solver_res, s4_err = trace_stage_4_solver_execution(contract, mode=3)
    solving_ms = (time.perf_counter() - t0) * 1000.0
    if not solver_res:
        solver_res = {"status": "infeasible", "solver": "Fallback_Local"}

    # Stage 5: Independent Verification
    t0 = time.perf_counter()
    s5_ok, check = trace_stage_5_independent_verification(
        contract, solver_res, silent=False, stage_header="  --- Mode 3 Parameter-Wise Independent Verification ---"
    )
    verification_ms = (time.perf_counter() - t0) * 1000.0

    # Stage 6: Mode 3 skips explanation
    trace_stage_6_explanation(contract, solver_res, mode=3)

    total_ms = (time.perf_counter() - t_start) * 1000.0
    total_cost = solver_res.get("total_monthly_cost_usd", solver_res.get("estimated_monthly_cost_usd", 0.0))
    recomp_cost = check.get("cost_accuracy", {}).get("calculated_catalog_cost_usd")

    record = CanonicalExecutionRecord(
        mode=3,
        mode_name="Mode 3: Pure Symbolic",
        original_query=query_text,
        requirement_source="parsed_contract",
        problem_type=contract.problem_type,
        requirements=contract.model_dump(),
        execution_path=f"Local SCOPE -> Local CARM ({solver_res.get('solver', 'Solver')}) -> IndependentChecker",
        solver_name=solver_res.get("solver"),
        original_response=solver_res,
        normalized_allocation=solver_res,
        claimed_cost_usd=float(total_cost) if total_cost is not None else None,
        normalization_status=NormalizationStatus.SUCCESS,
        feasibility=FeasibilityStatus.PASS if s5_ok else FeasibilityStatus.FAIL,
        optimality_status=OptimalityStatus(check.get("optimality_verdict", OptimalityStatus.INFEASIBLE.value)) if check.get("optimality_verdict") in [e.value for e in OptimalityStatus] else OptimalityStatus.HEURISTIC_FEASIBLE,
        recomputed_cost_usd=float(recomp_cost) if recomp_cost is not None else None,
        cost_delta_usd=check.get("cost_accuracy", {}).get("cost_delta_usd"),
        cost_error_pct=check.get("cost_accuracy", {}).get("cost_error_pct"),
        budget_headroom_usd=max(0.0, float(contract.budget_max_usd) - float(total_cost or 0.0)) if s5_ok else None,
        violations=check.get("violations", []),
        audit_events=_extract_audit_events(check.get("parameter_checks", [])),
        summary_status=check.get("summary_status", "Execution complete"),
        parsing_ms=parsing_total_ms,
        solving_ms=solving_ms,
        verification_ms=verification_ms,
        explanation_ms=0.0,
        total_duration_ms=total_ms,
        explanation_source=ExplanationSource.UNAVAILABLE,
        explanation_status="SKIPPED",
    )
    return record, contract, solver_res


def execute_mode_4_neuro_symbolic(
    query_text: str,
    mock_llm: bool = False,
    enable_explanation: bool = True,
) -> CanonicalExecutionRecord:
    """MODE 4: Full Neuro-Symbolic Pipeline (Neural Interpreter + Local Solvers + Explainer)."""
    provider_name, api_key, base_url, target_model, default_timeout = settings.get_mode4_provider_config()
    print_banner(
        "MODE 4: FULL NEURO-SYMBOLIC PIPELINE",
        f"Provider: {provider_name} API | Model: {target_model} (Neural Interpretation)" + (" [OFFLINE MOCK FIXTURE]" if mock_llm else ""),
        char="*",
    )
    t_start = time.perf_counter()

    # Stage 4.1: Neural Requirement Interpretation (Zero SCOPE/CARM pre-filtering)
    print(f"  [Stage 4.1: {provider_name} Neural Workload Requirement Interpretation]")
    print(f'    Query: "{query_text}"')
    
    nvd_res = NVIDIAExtractor.extract_contract_from_query(
        query=query_text,
        offline=mock_llm,
    )
    elapsed_nvd_ms = nvd_res.elapsed_ms
    print(f"    Interpretation Status : {nvd_res.status}")
    print(f"    Interpretation Outcome: {nvd_res.outcome.upper()}")
    print(f"    Extraction Latency    : {elapsed_nvd_ms:.1f} ms")

    if not nvd_res.is_executable:
        print("\n  [PIPELINE HALTED BEFORE SOLVER: Neural Interpretation Did Not Yield Executable Contract]")
        if nvd_res.clarification_questions:
            print("  Clarification Questions Required:")
            for q in nvd_res.clarification_questions:
                print(f"    ? {q}")
        if nvd_res.unsupported_reasons:
            print("  Unsupported Workload Reasons:")
            for r in nvd_res.unsupported_reasons:
                print(f"    ! {r}")
        if nvd_res.conflicting_reasons:
            print("  Conflicting Requirements Detected:")
            for r in nvd_res.conflicting_reasons:
                print(f"    ! {r}")
        if nvd_res.error_message:
            print(f"  Extraction Error: {nvd_res.error_message}")

        status_norm_map = {
            "NEEDS_CLARIFICATION": NormalizationStatus.CLARIFICATION_REQUIRED,
            "UNSUPPORTED": NormalizationStatus.TASK_INCOMPATIBLE,
            "CONFLICTING_REQUIREMENTS": NormalizationStatus.NORMALIZATION_FAILURE,
            "TRUNCATION_FAILURE": NormalizationStatus.TRUNCATION_FAILURE,
            "MISSING_CREDENTIALS": NormalizationStatus.API_FAILURE,
            "TIMEOUT": NormalizationStatus.API_FAILURE,
            "NETWORK_ERROR": NormalizationStatus.API_FAILURE,
            "MALFORMED_JSON": NormalizationStatus.MALFORMED_OUTPUT,
            "SCHEMA_ERROR": NormalizationStatus.MALFORMED_OUTPUT,
        }
        norm_stat = status_norm_map.get(nvd_res.status, NormalizationStatus.NORMALIZATION_FAILURE)
        
        # Determine feasibility and specific outcome status
        if nvd_res.status == "NEEDS_CLARIFICATION":
            feas = FeasibilityStatus.NOT_EVALUATED
            summary_stat = "Clarification required"
            opt_stat = OptimalityStatus.CLARIFICATION_REQUIRED
        elif nvd_res.status == "UNSUPPORTED":
            feas = FeasibilityStatus.NOT_EVALUABLE
            summary_stat = "Unsupported workload domain"
            opt_stat = OptimalityStatus.UNSUPPORTED
        elif nvd_res.status == "CONFLICTING_REQUIREMENTS":
            feas = FeasibilityStatus.FAIL
            summary_stat = "Conflicting requirements"
            opt_stat = OptimalityStatus.CONFLICTING
        elif nvd_res.status == "TRUNCATION_FAILURE":
            feas = FeasibilityStatus.NOT_EVALUATED
            summary_stat = "Provider truncation failure (finish_reason='length')"
            opt_stat = OptimalityStatus.TRUNCATION_FAILURE
        elif nvd_res.status in ["MISSING_CREDENTIALS", "TIMEOUT", "NETWORK_ERROR", "INFRASTRUCTURE_ERROR", "API_FAILURE"]:
            feas = FeasibilityStatus.NOT_EVALUATED
            summary_stat = f"Neural Interpretation: {nvd_res.status}"
            opt_stat = OptimalityStatus.INFRASTRUCTURE_FAILURE
        else:
            feas = FeasibilityStatus.FAIL
            summary_stat = f"Neural Interpretation: {nvd_res.status}"
            opt_stat = OptimalityStatus.INFEASIBLE

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
            requirement_source="neural_contract",
            problem_type=nvd_res.contract.problem_type if nvd_res.contract else "ILP_VM_Allocation",
            requirements=nvd_res.contract.model_dump() if nvd_res.contract else (nvd_res.parsed_json or {}),
            execution_path=f"{nvd_res.provider} API ({nvd_res.model}) -> Interpretation Halted ({nvd_res.status})",
            provider=nvd_res.provider,
            model=nvd_res.model,
            original_response=nvd_res.raw_response,
            normalized_allocation=None,
            claimed_cost_usd=None,
            normalization_status=norm_stat,
            normalization_errors=violations,
            extracted_evidence=nvd_res.extracted_evidence,
            feasibility=feas,
            optimality_status=opt_stat,
            violations=violations,
            summary_status=summary_stat,
            failed_stage=1,
            parsing_ms=elapsed_nvd_ms,
            solving_ms=0.0,
            verification_ms=0.0,
            explanation_ms=0.0,
            total_duration_ms=total_ms,
            explanation_source=ExplanationSource.UNAVAILABLE,
            explanation_status="SKIPPED",
        )

    # Valid Contract
    contract = nvd_res.contract
    print("\n  [Stage 4.2: Validated Neural Contract & Archetype Dispatch]")
    print(f"    * Problem Archetype        : {contract.problem_type}")
    print(f"    * Budget Maximum (USD)     : ${contract.budget_max_usd:,.2f}")
    if contract.problem_type == "ILP_VM_Allocation":
        print(f"    * Compute Requirements     : {contract.required_vcpus} vCPUs, {contract.required_ram_gb:.1f} GB RAM")
    elif contract.problem_type == "PSO_Continuous_Scaling":
        print(f"    * Scaling Target CPU       : {contract.target_cpu_pct or 70.0:.1f}%")
    elif contract.problem_type == "Z3_Graph_Disaster_Recovery":
        print(f"    * DR Constraints           : <= {contract.latency_max_ms:.1f}ms latency, >= {contract.sla_availability_pct:.2f}% SLA")

    # Stage 4.3: Symbolic Solver Execution
    t_solv = time.perf_counter()
    s4_ok, solver_res, s4_err = trace_stage_4_solver_execution(contract, mode=4)
    solving_ms = (time.perf_counter() - t_solv) * 1000.0
    if not solver_res:
        solver_res = {"status": "infeasible", "solver": "Fallback_Local"}

    # Stage 4.4: Mode-Blind Independent Verification (IndependentChecker)
    t_verif = time.perf_counter()
    s5_ok, check = trace_stage_5_independent_verification(
        contract, solver_res, silent=False, stage_header="  --- Mode 4 Parameter-Wise Independent Verification ---"
    )
    verification_ms = (time.perf_counter() - t_verif) * 1000.0

    # Stage 4.5: Natural Language Explanation (Separately Counted & Sourced)
    print("\n  [Stage 4.5: Natural Language Executive Report Generation via " + ("Local Explainer Template]" if (mock_llm or not enable_explanation) else f"{nvd_res.provider} API]"))
    t_exp = time.perf_counter()
    if mock_llm or not enable_explanation:
        exp_text = FinOpsExplainer.generate_report(contract, solver_res, check_result=check, enable_llm_explainer=False)
        exp_source = ExplanationSource.LOCAL_TEMPLATE
        if not s5_ok:
            print("  [DEPLOYMENT ADVICE SUPPRESSED: Mathematical verification detected constraint violation(s)]")
            print("  This allocation CANNOT be safely deployed as-is.")
            for v in check.get("violations", []):
                print(f"    - Violation: {v}")
        else:
            print(f"    * Executive FinOps Report Formulated ({len(exp_text.splitlines())} lines)")
    else:
        exp_text, exp_source = trace_stage_6_explanation(
            contract=contract,
            solver_result=solver_res,
            mode=4,
            check_result=check,
            silent=False,
        )
    explanation_ms = (time.perf_counter() - t_exp) * 1000.0

    total_ms = (time.perf_counter() - t_start) * 1000.0
    total_cost = solver_res.get("total_monthly_cost_usd", solver_res.get("estimated_monthly_cost_usd", 0.0))
    recomp_cost = check.get("cost_accuracy", {}).get("calculated_catalog_cost_usd")

    if not s5_ok:
        opt_verdict_stat = OptimalityStatus.INFEASIBLE
    else:
        raw_opt = check.get("optimality_verdict", OptimalityStatus.INFEASIBLE.value)
        opt_verdict_stat = OptimalityStatus(raw_opt) if raw_opt in [e.value for e in OptimalityStatus] else OptimalityStatus.HEURISTIC_FEASIBLE

    record = CanonicalExecutionRecord(
        mode=4,
        mode_name="Mode 4: Neuro-Symbolic",
        original_query=query_text,
        requirement_source="neural_contract",
        problem_type=contract.problem_type,
        requirements=contract.model_dump(),
        execution_path=f"{nvd_res.provider} API ({nvd_res.model}) -> Local Solver ({solver_res.get('solver', 'Solver')}) -> IndependentChecker" + (f" -> {nvd_res.provider} Explainer" if exp_source == ExplanationSource.LIVE_PROVIDER else " -> Local Explainer Template"),
        provider=nvd_res.provider,
        model=nvd_res.model,
        solver_name=solver_res.get("solver"),
        original_response=solver_res,
        normalized_allocation=solver_res,
        claimed_cost_usd=float(total_cost) if total_cost is not None else None,
        normalization_status=NormalizationStatus.SUCCESS,
        extracted_evidence=nvd_res.extracted_evidence,
        feasibility=FeasibilityStatus.PASS if s5_ok else FeasibilityStatus.FAIL,
        optimality_status=opt_verdict_stat,
        recomputed_cost_usd=float(recomp_cost) if recomp_cost is not None else None,
        cost_delta_usd=check.get("cost_accuracy", {}).get("cost_delta_usd"),
        cost_error_pct=check.get("cost_accuracy", {}).get("cost_error_pct"),
        budget_headroom_usd=max(0.0, float(contract.budget_max_usd) - float(total_cost or 0.0)) if s5_ok else None,
        violations=check.get("violations", []),
        audit_events=_extract_audit_events(check.get("parameter_checks", [])),
        summary_status=check.get("summary_status", "Execution complete"),
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


def print_comparison_table(records: List[CanonicalExecutionRecord]) -> None:
    print_banner("FOUR-WAY PARADIGM COMPARATIVE BENCHMARK SUMMARY", char="=")

    w_mode = 24
    w_exec = 36
    w_norm = 18
    w_cost = 14
    w_recomp = 14
    w_feas = 12
    w_proof = 34
    w_lat = 10

    sep = (
        f"+{'-'*(w_mode+2)}+{'-'*(w_exec+2)}+{'-'*(w_norm+2)}+{'-'*(w_cost+2)}+"
        f"{'-'*(w_recomp+2)}+{'-'*(w_feas+2)}+{'-'*(w_proof+2)}+{'-'*(w_lat+2)}+"
    )
    header = (
        f"| {'Execution Mode':<{w_mode}} | {'Execution Path':<{w_exec}} | {'Normalization':<{w_norm}} | "
        f"{'Claimed Cost':<{w_cost}} | {'Recomputed':<{w_recomp}} | {'Feasibility':<{w_feas}} | "
        f"{'Optimality / Verdict':<{w_proof}} | {'Latency':<{w_lat}} |"
    )

    print(sep)
    print(header)
    print(sep)

    for r in records:
        mode_s = r.mode_name[:w_mode]
        exec_s = r.execution_path[:w_exec]
        norm_s = r.normalization_status.value[:w_norm]
        claim_s = f"${r.claimed_cost_usd:,.2f}" if r.claimed_cost_usd is not None else "—"
        recomp_s = f"${r.recomputed_cost_usd:,.2f}" if r.recomputed_cost_usd is not None else "—"
        feas_s = f"[{r.feasibility.value}]"
        proof_s = r.optimality_status.value[:w_proof]
        lat_s = f"{r.total_duration_ms:.1f}ms" if r.total_duration_ms < 1000 else f"{r.total_duration_ms/1000.0:.2f}s"

        row = (
            f"| {mode_s:<{w_mode}} | {exec_s:<{w_exec}} | {norm_s:<{w_norm}} | "
            f"{claim_s:<{w_cost}} | {recomp_s:<{w_recomp}} | {feas_s:<{w_feas}} | "
            f"{proof_s:<{w_proof}} | {lat_s:<{w_lat}} |"
        )
        print(row)

    print(sep)

    # Dynamic honest conclusion without hardcoded winner banners
    m1 = next((r for r in records if r.mode == 1), None)
    m2 = next((r for r in records if r.mode == 2), None)
    m3 = next((r for r in records if r.mode == 3), None)
    m4 = next((r for r in records if r.mode == 4), None)

    print("\n" + "=" * 88)
    print(" AUDITABLE EXECUTION SUMMARY & VERDICT ALIGNMENT:")
    print("=" * 88)

    if m3 and m4:
        same_feasibility = (m3.feasibility == m4.feasibility)
        same_cost = (m3.recomputed_cost_usd == m4.recomputed_cost_usd)
        print(f"  • Mode 3 vs Mode 4 Independent Execution:")
        print(f"    - Mode 3 Pure Symbolic Verdict   : [{m3.feasibility.value}] ({m3.summary_status}) | Recomputed Cost: ${m3.recomputed_cost_usd or 0.0:,.2f}")
        print(f"    - Mode 4 Neuro-Symbolic Verdict  : [{m4.feasibility.value}] ({m4.summary_status}) | Recomputed Cost: ${m4.recomputed_cost_usd or 0.0:,.2f}")
        if same_feasibility and same_cost:
            print("    -> Aligned: Both pipelines converged on identical feasibility and allocation cost.")
        elif same_feasibility:
            print("    -> Equivalent Feasibility: Both pipelines found feasible solutions with independent allocations.")
        else:
            print(f"    -> Pipeline Divergence: Mode 3 evaluated [{m3.feasibility.value}], Mode 4 evaluated [{m4.feasibility.value}] due to distinct interpretation/solver paths.")

    if m1:
        print(f"  • Mode 1 (Raw LLM) Normalization   : {m1.normalization_status.value} (Claimed: ${m1.claimed_cost_usd or 0.0:.2f})")
    if m2:
        print(f"  • Mode 2 (Schema LLM) Normalization: {m2.normalization_status.value} (Claimed: ${m2.claimed_cost_usd or 0.0:.2f})")

    print("\n  Requirement Provenance Note: Checked against parsed contract from query input.")
    print("=" * 88 + "\n")


def log_canonical_records(query_text: str, records: List[CanonicalExecutionRecord]) -> None:
    """Logs canonical execution records to results/live_exploration_log.jsonl, results.csv, and results.xlsx."""
    timestamp = datetime.now(timezone.utc).isoformat()
    record_dict = {
        "timestamp": timestamp,
        "query": query_text,
        "records": [r.to_dict() for r in records],
        "results": {f"mode{r.mode}": r.to_summary_row() for r in records},
    }

    os.makedirs("results", exist_ok=True)
    with open("results/live_exploration_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record_dict) + "\n")

    # Synchronize to CSV tabular formats
    csv_paths = ["results.csv", "results/results.csv"]
    csv_headers = [
        "timestamp",
        "query",
        "mode",
        "mode_name",
        "execution_path",
        "normalization_status",
        "claimed_cost",
        "recomputed_cost",
        "cost_error",
        "feasibility",
        "optimality_status",
        "verdict",
        "is_mock",
        "core_ms",
        "explanation_ms",
        "total_ms",
        "explanation_source",
    ]

    rows = []
    for r in records:
        srow = r.to_summary_row()
        srow["timestamp"] = timestamp
        srow["query"] = query_text
        rows.append(srow)

    for path in csv_paths:
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    # Synchronize to Excel XLSX workbooks
    try:
        try:
            from results.export_to_xlsx import generate_results_xlsx
        except ImportError:
            from export_to_xlsx import generate_results_xlsx
        generate_results_xlsx("results/live_exploration_log.jsonl", ["results.xlsx", "results/results.xlsx"])
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neurasym Universal 4-Way Comparative Paradigm Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Continuous dynamic scaling with target CPU 70% under $1500",
        help="Natural language cloud optimization or autoscaling query to benchmark.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save comparison benchmark output as JSON.",
    )
    parser.add_argument(
        "--mock-llm",
        "--offline",
        dest="mock_llm",
        action="store_true",
        help="Execute offline comparative run with mocked provider fixtures and zero live API calls.",
    )

    args = parser.parse_args()
    query_text = args.query.strip()
    is_mock = args.mock_llm

    print_banner(
        "NEURASYM NEURO-SYMBOLIC 4-WAY COMPARATIVE BENCHMARK",
        f"Query: \"{query_text}\"" + (" [OFFLINE MOCK MODE]" if is_mock else ""),
        char="#",
    )

    records: List[CanonicalExecutionRecord] = []

    # 1. Mode 3 first to obtain contract & solver result
    m3_rec, contract, solver_res = execute_mode_3_symbolic(query_text)

    # 2. Mode 1
    m1_rec = execute_mode_1_raw_llm(query_text, contract, mock_llm=is_mock)
    records.append(m1_rec)

    # 3. Mode 2
    m2_rec = execute_mode_2_schema_llm(query_text, contract, mock_llm=is_mock)
    records.append(m2_rec)

    # Add Mode 3
    records.append(m3_rec)

    # 4. Mode 4 (Genuine Groq interpretation + solver dispatch)
    m4_rec = execute_mode_4_neuro_symbolic(query_text, mock_llm=is_mock)
    records.append(m4_rec)

    # Print comparative table
    print_comparison_table(records)

    # Log to files
    log_canonical_records(query_text, records)

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "query": query_text,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "records": [r.to_dict() for r in records],
            }, f, indent=2)
        print(f"Benchmark records saved to: {out_path}")


if __name__ == "__main__":
    main()
