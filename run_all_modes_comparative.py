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
from src.reporting.explain_verdict import (
    explain_run,
    format_plain_english_box,
    format_plain_english_summary,
)


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


def extract_stated_parameters_from_prose(
    raw_text: str,
    normalized_allocation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extracts explicit constraint figures and stated parameters directly from Mode 1 prose.
    
    Never reuses Mode 3's parsed contract or defaults.
    If a parameter is not stated in the prose, it is omitted or set to None ('not stated').
    """
    if not raw_text:
        return {}

    # Normalise Unicode hyphens and dashes to standard ASCII '-'
    cleaned_text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]", "-", raw_text)
    cleaned_text = re.sub(r"[\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000]", " ", cleaned_text)
    
    stated: Dict[str, Any] = {}

    # 1. Stated Budget (e.g. "budget $100", "under $100", "budget of $100", "within $100")
    b_match = re.search(
        r"(?:budget|budget\s+cap|budget\s+of|under|limit\s+of|limit\s+is|spend\s+of|within\s+(?:the|your)?|max\s+budget)\s*(?:of|is|:|=|under|capped\s+at)?\s*\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        cleaned_text,
        re.IGNORECASE,
    )
    if not b_match:
        b_match = re.search(r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:budget|cap|ceiling|limit|max)", cleaned_text, re.IGNORECASE)
    if b_match:
        try:
            stated["budget_max_usd"] = float(b_match.group(1).replace(",", ""))
        except (ValueError, TypeError):
            pass

    # 2. Stated vCPUs (e.g. "8 vCPUs", "8 cores", or sum of allocated VMs)
    vcpu_match = re.search(r"(\d+)\s*(?:vcpus?|cores?|virtual\s+cpus?|vcpu\b)", cleaned_text, re.IGNORECASE)
    if vcpu_match:
        try:
            stated["required_vcpus"] = int(vcpu_match.group(1))
        except (ValueError, TypeError):
            pass
    elif normalized_allocation and isinstance(normalized_allocation, dict):
        vms = normalized_allocation.get("allocated_vms", [])
        if vms and isinstance(vms, list):
            tot_vcpu = sum(v.get("vcpus", 0) * v.get("count", 1) for v in vms if isinstance(v, dict))
            if tot_vcpu > 0:
                stated["required_vcpus"] = tot_vcpu

    # 3. Stated RAM (e.g. "16 GB RAM", "16GB", or sum of allocated VMs)
    ram_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)\s*(?:of\s*)?(?:ram|memory)", cleaned_text, re.IGNORECASE)
    if not ram_match:
        ram_match = re.search(r"(?:ram|memory)\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)", cleaned_text, re.IGNORECASE)
    if ram_match:
        try:
            stated["required_ram_gb"] = float(ram_match.group(1))
        except (ValueError, TypeError):
            pass
    elif normalized_allocation and isinstance(normalized_allocation, dict):
        vms = normalized_allocation.get("allocated_vms", [])
        if vms and isinstance(vms, list):
            tot_ram = sum(v.get("ram_gb", 0.0) * v.get("count", 1) for v in vms if isinstance(v, dict))
            if tot_ram > 0:
                stated["required_ram_gb"] = float(tot_ram)

    # 4. Stated Latency (e.g. "42ms", "50 ms latency")
    lat_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)\s*(?:latency|round[\s-]*trip|delay|sync)?", cleaned_text, re.IGNORECASE)
    if not lat_match:
        lat_match = re.search(r"(?:latency|delay|sync\s+latency)\s*(?:of|under|below|<=|<|:)?\s*(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)", cleaned_text, re.IGNORECASE)
    if lat_match:
        try:
            stated["latency_max_ms"] = float(lat_match.group(1))
        except (ValueError, TypeError):
            pass

    # 5. Stated SLA (e.g. "99.99% SLA", "99.9% uptime")
    sla_match = re.search(r"(\d{2}(?:\.\d+)?)\s*%\s*(?:sla|uptime|availability)", cleaned_text, re.IGNORECASE)
    if not sla_match:
        sla_match = re.search(r"(?:sla|uptime|availability)\s*(?:of|:)?\s*(\d{2}(?:\.\d+)?)\s*%", cleaned_text, re.IGNORECASE)
    if sla_match:
        try:
            stated["sla_availability_pct"] = float(sla_match.group(1))
        except (ValueError, TypeError):
            pass

    # 6. Stated Bandwidth (e.g. "100 Mbps", "100Mbps bandwidth")
    bw_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mbps|gbps)\s*(?:bandwidth|traffic|workload|throughput)?", cleaned_text, re.IGNORECASE)
    if bw_match:
        try:
            stated["target_bandwidth_mbps"] = float(bw_match.group(1))
        except (ValueError, TypeError):
            pass

    # 7. Stated Target/Max CPU (e.g. "70% target CPU", "CPU 70%")
    cpu_match = re.search(r"(?:target\s+cpu|target\s+utilization|cpu\s+target|target\s+cpu\s+utilization|cpu)\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*%", cleaned_text, re.IGNORECASE)
    if cpu_match:
        try:
            stated["target_cpu_pct"] = float(cpu_match.group(1))
        except (ValueError, TypeError):
            pass

    return stated


def extract_stated_parameters_from_json(
    parsed_json_or_text: Any,
    raw_content: str = "",
) -> Dict[str, Any]:
    """Extracts explicit constraint figures and stated parameters directly from Mode 2 JSON and reason fields.
    
    Never reuses Mode 3's parsed contract or defaults.
    If a parameter is not stated in the JSON, it is omitted or set to None ('not stated').
    """
    stated: Dict[str, Any] = {}
    parsed: Dict[str, Any] = {}

    if isinstance(parsed_json_or_text, dict):
        parsed = parsed_json_or_text
    elif isinstance(parsed_json_or_text, str):
        try:
            cleaned = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]", "-", parsed_json_or_text)
            f_b = cleaned.find("{")
            l_b = cleaned.rfind("}")
            if f_b != -1 and l_b != -1:
                parsed = json.loads(cleaned[f_b : l_b + 1])
        except Exception:
            parsed = {}

    # Extract from top-level JSON fields
    # Budget
    raw_b = parsed.get("budget_max_usd") or parsed.get("budget") or parsed.get("max_budget") or parsed.get("budget_usd")
    if raw_b is not None:
        try:
            stated["budget_max_usd"] = float(raw_b)
        except (ValueError, TypeError):
            pass

    # vCPUs
    raw_vcpu = parsed.get("required_vcpus") or parsed.get("vcpus") or parsed.get("total_vcpus")
    if raw_vcpu is not None:
        try:
            stated["required_vcpus"] = int(raw_vcpu)
        except (ValueError, TypeError):
            pass
    elif "allocated_vms" in parsed and isinstance(parsed["allocated_vms"], list):
        tot_vcpu = sum(v.get("vcpus", 0) * v.get("quantity", v.get("count", 1)) for v in parsed["allocated_vms"] if isinstance(v, dict))
        if tot_vcpu > 0:
            stated["required_vcpus"] = tot_vcpu

    # RAM
    raw_ram = parsed.get("required_ram_gb") or parsed.get("ram_gb") or parsed.get("total_ram_gb") or parsed.get("ram")
    if raw_ram is not None:
        try:
            stated["required_ram_gb"] = float(raw_ram)
        except (ValueError, TypeError):
            pass
    elif "allocated_vms" in parsed and isinstance(parsed["allocated_vms"], list):
        tot_ram = sum(v.get("ram_gb", 0.0) * v.get("quantity", v.get("count", 1)) for v in parsed["allocated_vms"] if isinstance(v, dict))
        if tot_ram > 0:
            stated["required_ram_gb"] = float(tot_ram)

    # SLA
    raw_sla = parsed.get("sla_availability_pct") or parsed.get("achieved_sla_pct") or parsed.get("sla_pct") or parsed.get("sla") or parsed.get("uptime_pct")
    if raw_sla is not None:
        try:
            stated["sla_availability_pct"] = float(raw_sla)
        except (ValueError, TypeError):
            pass

    # Latency
    raw_lat = parsed.get("latency_max_ms") or parsed.get("latency_ms") or parsed.get("achieved_latency_ms") or parsed.get("latency")
    if raw_lat is not None:
        try:
            stated["latency_max_ms"] = float(raw_lat)
        except (ValueError, TypeError):
            pass

    # Bandwidth
    raw_bw = parsed.get("target_bandwidth_mbps") or parsed.get("bandwidth_mbps") or parsed.get("optimal_bandwidth_mbps") or parsed.get("bandwidth")
    if raw_bw is not None:
        try:
            stated["target_bandwidth_mbps"] = float(raw_bw)
        except (ValueError, TypeError):
            pass

    # Target CPU / Max CPU
    raw_tcpu = parsed.get("target_cpu_pct") or parsed.get("target_cpu")
    if raw_tcpu is not None:
        try:
            stated["target_cpu_pct"] = float(raw_tcpu)
        except (ValueError, TypeError):
            pass
    raw_mcpu = parsed.get("max_cpu_pct") or parsed.get("max_cpu")
    if raw_mcpu is not None:
        try:
            stated["max_cpu_pct"] = float(raw_mcpu)
        except (ValueError, TypeError):
            pass

    # Check reason / explanation / notes fields in JSON if fields are missing
    reason_str = str(parsed.get("reason", "") or parsed.get("explanation", "") or parsed.get("notes", "") or raw_content)
    if reason_str:
        prose_extracted = extract_stated_parameters_from_prose(reason_str)
        for k, v in prose_extracted.items():
            if k not in stated:
                stated[k] = v

    return stated


def execute_mode_1_raw_llm(
    query_text: str,
    contract: Optional[CloudOptimizationContract],
    mock_llm: bool = False,
    manifest_entry: Optional[Dict[str, Any]] = None,
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
            raw_content = "For continuous dynamic scaling with target CPU 70% under $1500, we recommend provisioning 100 Mbps bandwidth with 2 worker replicas at an estimated total cost of $98.00/month ($0.08 per Mbps + $45.00 base replica fee)."
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
            parser = SCOPEParser()
            contract, _, _ = parser.parse_query_to_contract(query_text)
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

    # Extract explicitly stated requirements from Mode 1 prose (never reuse Mode 3 contract)
    stated_requirements_m1 = extract_stated_parameters_from_prose(raw_content, norm_decision)

    if contract is None:
        try:
            contract = CloudOptimizationContract(
                problem_type="ILP_VM_Allocation",
                budget_max_usd=stated_requirements_m1.get("budget_max_usd") or 500.0,
                required_vcpus=stated_requirements_m1.get("required_vcpus") or 1,
                required_ram_gb=stated_requirements_m1.get("required_ram_gb") or 1.0,
                latency_max_ms=stated_requirements_m1.get("latency_max_ms") or 100.0,
                sla_availability_pct=stated_requirements_m1.get("sla_availability_pct") or 99.9,
                target_bandwidth_mbps=stated_requirements_m1.get("target_bandwidth_mbps"),
                target_cpu_pct=stated_requirements_m1.get("target_cpu_pct") or 70.0,
            )
        except Exception:
            contract = None

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

    # Auto-align contract archetype if decision is DR or Scaling and contract defaulted to VM
    if contract and norm_decision and isinstance(norm_decision, dict):
        if norm_decision.get("primary_region") and norm_decision.get("secondary_region"):
            prob_type = "Z3_Graph_Disaster_Recovery"
            contract.problem_type = "Z3_Graph_Disaster_Recovery"
        elif (norm_decision.get("optimal_bandwidth_mbps") is not None or norm_decision.get("bandwidth_mbps") is not None) and (norm_decision.get("recommended_replicas") is not None or norm_decision.get("replicas") is not None):
            prob_type = "PSO_Continuous_Scaling"
            contract.problem_type = "PSO_Continuous_Scaling"
        elif norm_decision.get("allocated_vms"):
            prob_type = "ILP_VM_Allocation"
            contract.problem_type = "ILP_VM_Allocation"

    if contract and norm_status in [NormalizationStatus.SUCCESS, NormalizationStatus.NEEDS_REVIEW]:
        feas_pass, check = trace_stage_5_independent_verification(
            contract, candidate_dict, silent=False, stage_header="  --- Mode 1 Parameter-Wise Independent Verification ---"
        )
        if check.get("violations"):
            feas_pass = False
        if feas_pass:
            key_opt = manifest_entry.get("expected_optimal_cost_usd") if manifest_entry else None
            actual_c = check.get("cost_accuracy", {}).get("calculated_catalog_cost_usd")
            if key_opt is not None and actual_c is not None:
                cost_diff = actual_c - key_opt
                tol = max(0.50, 0.01 * key_opt)
                if cost_diff > tol:
                    pct_gap = (cost_diff / key_opt) * 100.0
                    opt_verdict_stat = f"Not optimal (gap {pct_gap:.1f}%)"
                else:
                    opt_verdict_stat = OptimalityStatus.MATCHES_INDEPENDENT_OPTIMUM.value
            else:
                opt_verdict_stat = OptimalityStatus.MATCHES_INDEPENDENT_OPTIMUM.value
            feas_status_final = FeasibilityStatus.PASS
            summary_status_final = "VALID_PLAN (Feasible against checked constraints)"
        else:
            opt_verdict_stat = OptimalityStatus.UNVERIFIED.value
            feas_status_final = FeasibilityStatus.FAIL
            summary_status_final = f"INVALID_PLAN (Violations: {', '.join(check.get('violations', []))})"
    elif norm_status in [NormalizationStatus.TASK_INCOMPATIBLE, NormalizationStatus.SOLVER_INFEASIBLE]:
        opt_verdict_stat = OptimalityStatus.UNSUPPORTED if norm_status == NormalizationStatus.TASK_INCOMPATIBLE else OptimalityStatus.INFEASIBLE
        check = {
            "feasible_against_contract": False,
            "optimality_verdict": opt_verdict_stat.value if isinstance(opt_verdict_stat, OptimalityStatus) else opt_verdict_stat,
            "summary_status": f"REFUSAL_OR_UNSUPPORTED ({norm_status.value})",
            "violations": norm_errors or ["Model stated workload is unsupported/infeasible"],
            "parameter_checks": [],
        }
        feas_pass = False
        feas_status_final = FeasibilityStatus.NOT_EVALUABLE
        summary_status_final = f"REFUSAL_OR_UNSUPPORTED ({norm_status.value})"
        print(f"    * Constraint Feasibility   : NOT_EVALUABLE (Refusal/Unsupported)")
        print(f"    * FINAL VERDICT            : Refusal / Unsupported ({norm_status.value})")
    else:
        opt_verdict_stat = OptimalityStatus.TRUNCATION_FAILURE if norm_status == NormalizationStatus.TRUNCATION_FAILURE else OptimalityStatus.UNVERIFIED
        check = {
            "feasible_against_contract": False,
            "optimality_verdict": opt_verdict_stat.value if isinstance(opt_verdict_stat, OptimalityStatus) else opt_verdict_stat,
            "summary_status": f"UNPARSEABLE ({norm_status.value})",
            "violations": norm_errors or ["Failed extraction/normalization check"],
            "parameter_checks": [],
        }
        feas_pass = False
        feas_status_final = FeasibilityStatus.NOT_EVALUABLE
        summary_status_final = f"UNPARSEABLE ({norm_status.value})"
        print(f"    * Constraint Feasibility   : NOT_EVALUABLE (No checkable plan)")
        print(f"    * FINAL VERDICT            : UNPARSEABLE ({norm_status.value})")

    elapsed_verif_ms = (time.perf_counter() - t_verif) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    record = CanonicalExecutionRecord(
        mode=1,
        mode_name="Mode 1: Raw LLM",
        original_query=query_text,
        requirement_source="mode1_prose_stated",
        problem_type=prob_type,
        requirements=stated_requirements_m1,
        execution_path=f"Groq API ({model_name}) -> OutputNormalizer (Prose Regex) -> IndependentChecker",
        provider="Groq",
        model=model_name,
        original_response=raw_content,
        normalized_allocation=norm_decision,
        claimed_cost_usd=ext_cost,
        normalization_status=norm_status,
        normalization_errors=norm_errors,
        extracted_evidence=evidence,
        feasibility=feas_status_final,
        optimality_status=opt_verdict_stat,
        recomputed_cost_usd=check.get("cost_accuracy", {}).get("calculated_catalog_cost_usd"),
        cost_delta_usd=check.get("cost_accuracy", {}).get("cost_delta_usd"),
        cost_error_pct=check.get("cost_accuracy", {}).get("cost_error_pct"),
        violations=check.get("violations", []),
        audit_events=_extract_audit_events(check.get("parameter_checks", [])),
        summary_status=summary_status_final,
        parsing_ms=elapsed_llm_ms,
        solving_ms=elapsed_norm_ms,
        verification_ms=elapsed_verif_ms,
        explanation_ms=0.0,
        total_duration_ms=total_ms,
        explanation_source=ExplanationSource.UNAVAILABLE,
        explanation_status="SKIPPED",
    )
    verdict = explain_run(record, key=manifest_entry)
    print("\n" + format_plain_english_box(verdict) + "\n")
    return record


def execute_mode_2_schema_llm(
    query_text: str,
    contract: Optional[CloudOptimizationContract],
    mock_llm: bool = False,
    manifest_entry: Optional[Dict[str, Any]] = None,
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
                "optimal_bandwidth_mbps": 100.0,
                "recommended_replicas": 2,
                "total_monthly_cost_usd": 98.00,
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
            parser = SCOPEParser()
            contract, _, _ = parser.parse_query_to_contract(query_text)
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

    # Extract explicitly stated requirements from Mode 2 JSON (never reuse Mode 3 contract)
    stated_requirements_m2 = extract_stated_parameters_from_json(parsed_json or raw_content, raw_content=raw_content)

    if contract is None:
        try:
            contract = CloudOptimizationContract(
                problem_type="ILP_VM_Allocation",
                budget_max_usd=stated_requirements_m2.get("budget_max_usd") or 500.0,
                required_vcpus=stated_requirements_m2.get("required_vcpus") or 1,
                required_ram_gb=stated_requirements_m2.get("required_ram_gb") or 1.0,
                latency_max_ms=stated_requirements_m2.get("latency_max_ms") or 100.0,
                sla_availability_pct=stated_requirements_m2.get("sla_availability_pct") or 99.9,
                target_bandwidth_mbps=stated_requirements_m2.get("target_bandwidth_mbps"),
                target_cpu_pct=stated_requirements_m2.get("target_cpu_pct") or 70.0,
            )
        except Exception:
            contract = None

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

    # Auto-align contract archetype if decision is DR or Scaling and contract defaulted to VM
    if contract and norm_decision and isinstance(norm_decision, dict):
        if norm_decision.get("primary_region") and norm_decision.get("secondary_region"):
            prob_type = "Z3_Graph_Disaster_Recovery"
            contract.problem_type = "Z3_Graph_Disaster_Recovery"
        elif (norm_decision.get("optimal_bandwidth_mbps") is not None or norm_decision.get("bandwidth_mbps") is not None) and (norm_decision.get("recommended_replicas") is not None or norm_decision.get("replicas") is not None):
            prob_type = "PSO_Continuous_Scaling"
            contract.problem_type = "PSO_Continuous_Scaling"
        elif norm_decision.get("allocated_vms"):
            prob_type = "ILP_VM_Allocation"
            contract.problem_type = "ILP_VM_Allocation"

    if contract and norm_status in [NormalizationStatus.SUCCESS, NormalizationStatus.NEEDS_REVIEW]:
        feas_pass, check = trace_stage_5_independent_verification(
            contract, candidate_dict, silent=False, stage_header="  --- Mode 2 Parameter-Wise Independent Verification ---"
        )
        if check.get("violations"):
            feas_pass = False
        if feas_pass:
            key_opt = manifest_entry.get("expected_optimal_cost_usd") if manifest_entry else None
            actual_c = check.get("cost_accuracy", {}).get("calculated_catalog_cost_usd")
            if key_opt is not None and actual_c is not None:
                cost_diff = actual_c - key_opt
                tol = max(0.50, 0.01 * key_opt)
                if cost_diff > tol:
                    pct_gap = (cost_diff / key_opt) * 100.0
                    opt_verdict_stat = f"Not optimal (gap {pct_gap:.1f}%)"
                else:
                    opt_verdict_stat = OptimalityStatus.MATCHES_INDEPENDENT_OPTIMUM.value
            else:
                opt_verdict_stat = OptimalityStatus.MATCHES_INDEPENDENT_OPTIMUM.value
            feas_status_final = FeasibilityStatus.PASS
            summary_status_final = "VALID_PLAN (Feasible against checked constraints)"
        else:
            opt_verdict_stat = OptimalityStatus.UNVERIFIED.value
            feas_status_final = FeasibilityStatus.FAIL
            summary_status_final = f"INVALID_PLAN (Violations: {', '.join(check.get('violations', []))})"
    else:
        opt_verdict_stat = OptimalityStatus.TRUNCATION_FAILURE if norm_status == NormalizationStatus.TRUNCATION_FAILURE else OptimalityStatus.UNVERIFIED
        check = {
            "feasible_against_contract": False,
            "optimality_verdict": opt_verdict_stat.value if isinstance(opt_verdict_stat, OptimalityStatus) else opt_verdict_stat,
            "summary_status": f"Rejected ({norm_status.value})",
            "violations": norm_errors or ["Failed schema/task validation"],
            "parameter_checks": [],
        }
        feas_pass = False
        feas_status_final = FeasibilityStatus.FAIL
        summary_status_final = f"Rejected ({norm_status.value})"
        print(f"    * Constraint Feasibility   : FAIL")
        print(f"    * FINAL VERDICT            : Rejected ({norm_status.value})")

    elapsed_verif_ms = (time.perf_counter() - t_verif) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    record = CanonicalExecutionRecord(
        mode=2,
        mode_name="Mode 2: Schema LLM",
        original_query=query_text,
        requirement_source="mode2_json_stated",
        problem_type=prob_type,
        requirements=stated_requirements_m2,
        execution_path=f"Groq API ({model_name}) -> OutputNormalizer (JSON Schema) -> IndependentChecker",
        provider="Groq",
        model=model_name,
        original_response=raw_content,
        normalized_allocation=norm_decision,
        claimed_cost_usd=ext_cost,
        normalization_status=norm_status,
        normalization_errors=norm_errors,
        extracted_evidence=evidence,
        feasibility=feas_status_final,
        optimality_status=opt_verdict_stat,
        recomputed_cost_usd=check.get("cost_accuracy", {}).get("calculated_catalog_cost_usd"),
        cost_delta_usd=check.get("cost_accuracy", {}).get("cost_delta_usd"),
        cost_error_pct=check.get("cost_accuracy", {}).get("cost_error_pct"),
        violations=check.get("violations", []),
        audit_events=_extract_audit_events(check.get("parameter_checks", [])),
        summary_status=summary_status_final,
        parsing_ms=elapsed_llm_ms,
        solving_ms=elapsed_norm_ms,
        verification_ms=elapsed_verif_ms,
        explanation_ms=0.0,
        total_duration_ms=total_ms,
        explanation_source=ExplanationSource.UNAVAILABLE,
        explanation_status="SKIPPED",
    )
    verdict = explain_run(record, key=manifest_entry)
    print("\n" + format_plain_english_box(verdict) + "\n")
    return record


def find_manifest_entry(query_input: str) -> Optional[Dict[str, Any]]:
    """Looks up human-written query specifications from manifest files."""
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

    return None


def evaluate_task_outcome(record: CanonicalExecutionRecord, expected_outcome: Optional[str]) -> str:
    """Evaluates task outcome (SUCCESS vs FAILURE vs UNGRADED-UNPARSEABLE vs NOT_GRADED) against manifest."""
    norm_s = record.normalization_status.value if isinstance(record.normalization_status, NormalizationStatus) else str(record.normalization_status)
    if norm_s in ["UNPARSEABLE", "NORMALIZATION_FAILURE", "MALFORMED_OUTPUT", "AMBIGUOUS"]:
        return "UNGRADED-UNPARSEABLE"

    if not expected_outcome or expected_outcome.strip().upper() in ["NOT_GRADED", "NONE", "UNGRADED", "CUSTOM_QUERY", "CUSTOM", "—", "N/A"]:
        return "NOT_GRADED"
    try:
        from src.benchmarks.truth_table import TruthTableEngine
        eval_res = TruthTableEngine.evaluate_mode_task_success(record, expected_outcome=expected_outcome)
        if eval_res.task_pass is None:
            return "NOT_GRADED"
        return "SUCCESS" if eval_res.task_pass == 1 else "FAILURE"
    except Exception:
        exp_clean = (expected_outcome or "FEASIBLE").strip().upper()
        if exp_clean in ["NOT_GRADED", "NONE", "UNGRADED", "—"]:
            return "NOT_GRADED"
        elif exp_clean == "FEASIBLE":
            return "SUCCESS" if record.feasibility == FeasibilityStatus.PASS else "FAILURE"
        elif exp_clean == "CLARIFICATION_REQUIRED":
            is_clar = (
                record.normalization_status == NormalizationStatus.CLARIFICATION_REQUIRED
                or record.optimality_status == OptimalityStatus.CLARIFICATION_REQUIRED
                or "clarification" in (record.summary_status or "").lower()
            )
            return "SUCCESS" if is_clar else "FAILURE"
        elif exp_clean == "UNSUPPORTED":
            is_unsup = (
                record.normalization_status == NormalizationStatus.TASK_INCOMPATIBLE
                or record.optimality_status == OptimalityStatus.UNSUPPORTED
                or "unsupported" in (record.summary_status or "").lower()
                or "parser_failed" in (record.summary_status or "").lower()
            )
            return "SUCCESS" if is_unsup else "FAILURE"
        elif exp_clean == "INFEASIBLE":
            is_infeas = (
                record.normalization_status in [NormalizationStatus.SOLVER_INFEASIBLE, NormalizationStatus.PROVEN_INFEASIBLE]
                or record.optimality_status in [OptimalityStatus.INFEASIBLE, OptimalityStatus.CONFLICTING]
                or "infeasible" in (record.summary_status or "").lower()
            )
            return "SUCCESS" if is_infeas else "FAILURE"
        return "SUCCESS" if record.feasibility == FeasibilityStatus.PASS else "FAILURE"


def execute_mode_3_symbolic(
    query_text: str,
    manifest_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[CanonicalExecutionRecord, Optional[CloudOptimizationContract], Optional[Dict[str, Any]]]:
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
    if not s1_ok or params is None:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        rec = CanonicalExecutionRecord(
            mode=3,
            mode_name="Mode 3: Pure Symbolic",
            original_query=query_text,
            normalization_status=NormalizationStatus.TASK_INCOMPATIBLE,
            optimality_status=OptimalityStatus.UNSUPPORTED,
            feasibility=FeasibilityStatus.NOT_EVALUABLE,
            summary_status=f"STAGE 1 (PARSER_FAILED): {s1_err or 'Lexical parsing failed'}",
            failed_stage=1,
            total_duration_ms=total_ms,
        )
        verdict = explain_run(rec, key=manifest_entry)
        print("\n" + format_plain_english_box(verdict) + "\n")
        return rec, None, None

    # Stage 2: Archetype Matching
    t0 = time.perf_counter()
    s2_ok, archetype, s2_err = trace_stage_2_archetype_matching(matcher, query_text, parser)
    p2_ms = (time.perf_counter() - t0) * 1000.0
    if not s2_ok or archetype is None:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        rec = CanonicalExecutionRecord(
            mode=3,
            mode_name="Mode 3: Pure Symbolic",
            original_query=query_text,
            normalization_status=NormalizationStatus.TASK_INCOMPATIBLE,
            optimality_status=OptimalityStatus.UNSUPPORTED,
            feasibility=FeasibilityStatus.NOT_EVALUABLE,
            summary_status=f"STAGE 1 (PARSER_FAILED): {s2_err or 'Unsupported archetype'}",
            failed_stage=1,
            total_duration_ms=total_ms,
        )
        verdict = explain_run(rec, key=manifest_entry)
        print("\n" + format_plain_english_box(verdict) + "\n")
        return rec, None, None

    # Stage 3: Contract Validation
    t0 = time.perf_counter()
    s3_ok, contract, s3_err = trace_stage_3_contract_validation(archetype, params or {})
    p3_ms = (time.perf_counter() - t0) * 1000.0
    if not s3_ok or contract is None:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        rec = CanonicalExecutionRecord(
            mode=3,
            mode_name="Mode 3: Pure Symbolic",
            original_query=query_text,
            problem_type=archetype,
            normalization_status=NormalizationStatus.MALFORMED_OUTPUT,
            optimality_status=OptimalityStatus.INFEASIBLE,
            feasibility=FeasibilityStatus.FAIL,
            summary_status=s3_err or "Contract validation failed",
            failed_stage=3,
            total_duration_ms=total_ms,
        )
        verdict = explain_run(rec, key=manifest_entry)
        print("\n" + format_plain_english_box(verdict) + "\n")
        return rec, None, None

    parsing_total_ms = p1_ms + p2_ms + p3_ms

    # Stage 4: Solver
    t0 = time.perf_counter()
    s4_ok, solver_res, s4_err = trace_stage_4_solver_execution(contract, mode=3)
    solving_ms = (time.perf_counter() - t0) * 1000.0
    if not s4_ok or solver_res is None:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        rec = CanonicalExecutionRecord(
            mode=3,
            mode_name="Mode 3: Pure Symbolic",
            original_query=query_text,
            problem_type=contract.problem_type,
            requirements=contract.model_dump(),
            normalization_status=NormalizationStatus.SOLVER_INFEASIBLE,
            optimality_status=OptimalityStatus.INFEASIBLE,
            feasibility=FeasibilityStatus.FAIL,
            summary_status=s4_err or "Solver failed / Infeasible",
            failed_stage=4,
            parsing_ms=parsing_total_ms,
            solving_ms=solving_ms,
            total_duration_ms=total_ms,
        )
        verdict = explain_run(rec, key=manifest_entry)
        print("\n" + format_plain_english_box(verdict) + "\n")
        return rec, contract, None

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
    verdict = explain_run(record, key=manifest_entry)
    print("\n" + format_plain_english_box(verdict) + "\n")
    return record, contract, solver_res


def execute_mode_4_neuro_symbolic(
    query_text: str,
    mock_llm: bool = False,
    enable_explanation: bool = True,
    manifest_entry: Optional[Dict[str, Any]] = None,
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

        rec = CanonicalExecutionRecord(
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
        verdict = explain_run(rec, key=manifest_entry)
        print("\n" + format_plain_english_box(verdict) + "\n")
        return rec

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
    t_exp = time.perf_counter()
    explainer_provider, _, _, explainer_model, _ = settings.get_mode4_provider_config()
    if mock_llm or not enable_explanation:
        print("\n  [Stage 4.5: Natural Language Executive Report Generation via Local Explainer Template]")
        print("    * Provider Event: Live explanation API call skipped (offline mock mode)")
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
        try:
            exp_text, exp_source = FinOpsExplainer.generate_report_with_source(
                contract=contract,
                solver_result=solver_res,
                check_result=check,
                enable_llm_explainer=True,
                offline=False,
            )
            if exp_source == ExplanationSource.LIVE_PROVIDER:
                print(f"\n  [Stage 4.5: Natural Language Executive Report Generation via Live Provider API ({explainer_provider})]")
                print(f"    * Provider Event: Live explanation successfully formulated by {explainer_provider}")
            else:
                print("\n  [Stage 4.5: Natural Language Executive Report Generation via Local Explainer Template (Provider Fallback)]")
                print("    * Provider Event: Live provider call unavailable or failed -> Fallback to Local Template logged")
        except Exception as e_exp:
            print("\n  [Stage 4.5: Natural Language Executive Report Generation via Local Explainer Template (Provider Fallback)]")
            print(f"    * Provider Event: Live API call error ({e_exp}) -> Fallback to Local Template logged")
            exp_text = FinOpsExplainer.generate_report(contract, solver_res, check_result=check, enable_llm_explainer=False)
            exp_source = ExplanationSource.LOCAL_TEMPLATE

        if not s5_ok:
            print("  [DEPLOYMENT ADVICE SUPPRESSED: Mathematical verification detected constraint violation(s)]")
            print("  This allocation CANNOT be safely deployed as-is.")
            for v in check.get("violations", []):
                print(f"    - Violation: {v}")
        else:
            print(f"    * Executive FinOps Report Formulated ({len(exp_text.splitlines())} lines)")
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
        execution_path=f"{nvd_res.provider} API ({nvd_res.model}) -> Local Solver ({solver_res.get('solver', 'Solver')}) -> IndependentChecker" + (f" -> {explainer_provider} Explainer" if exp_source == ExplanationSource.LIVE_PROVIDER else " -> Local Explainer Template"),
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
    verdict = explain_run(record, key=manifest_entry)
    print("\n" + format_plain_english_box(verdict) + "\n")
    return record


def print_comparison_table(
    records: List[CanonicalExecutionRecord],
    m3_contract: Optional[CloudOptimizationContract] = None,
    m4_contract: Optional[CloudOptimizationContract] = None,
) -> None:
    print_banner("FOUR-WAY PARADIGM COMPARATIVE BENCHMARK SUMMARY", char="=")

    w_mode = 20
    w_exec = 32
    w_norm = 16
    w_cost = 13
    w_recomp = 13
    w_feas = 15
    w_task = 22
    w_proof = 36
    w_lat = 9

    sep = (
        f"+{'-'*(w_mode+2)}+{'-'*(w_exec+2)}+{'-'*(w_norm+2)}+{'-'*(w_cost+2)}+"
        f"{'-'*(w_recomp+2)}+{'-'*(w_feas+2)}+{'-'*(w_task+2)}+{'-'*(w_proof+2)}+{'-'*(w_lat+2)}+"
    )
    header = (
        f"| {'Execution Mode':<{w_mode}} | {'Execution Path':<{w_exec}} | {'Normalization':<{w_norm}} | "
        f"{'Claimed Cost':<{w_cost}} | {'Recomputed':<{w_recomp}} | {'Alloc Feas.':<{w_feas}} | "
        f"{'Task Outcome':<{w_task}} | {'Optimality / Verdict':<{w_proof}} | {'Latency':<{w_lat}} |"
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
        
        task_out = r.task_outcome or ("NOT_GRADED" if not r.expected_outcome or r.expected_outcome in ["NOT_GRADED", "—"] else ("SUCCESS" if r.feasibility == FeasibilityStatus.PASS else "FAILURE"))
        task_s = f"[{task_out}]"[:w_task]
        raw_proof = r.optimality_status.value if hasattr(r.optimality_status, "value") else str(r.optimality_status)
        proof_s = raw_proof[:w_proof]
        lat_s = f"{r.total_duration_ms:.1f}ms" if r.total_duration_ms < 1000 else f"{r.total_duration_ms/1000.0:.2f}s"

        row = (
            f"| {mode_s:<{w_mode}} | {exec_s:<{w_exec}} | {norm_s:<{w_norm}} | "
            f"{claim_s:<{w_cost}} | {recomp_s:<{w_recomp}} | {feas_s:<{w_feas}} | "
            f"{task_s:<{w_task}} | {proof_s:<{w_proof}} | {lat_s:<{w_lat}} |"
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
        
        # Check interpretation differences between Mode 3 and Mode 4 contracts
        interp_mismatches = []
        if m3_contract and m4_contract:
            if m3_contract.problem_type != m4_contract.problem_type:
                interp_mismatches.append(f"problem type (Mode 3: {m3_contract.problem_type}, Mode 4: {m4_contract.problem_type})")
            if sorted(m3_contract.cloud_providers or []) != sorted(m4_contract.cloud_providers or []):
                interp_mismatches.append(f"providers (Mode 3: {m3_contract.cloud_providers}, Mode 4: {m4_contract.cloud_providers})")
            if (m3_contract.budget_max_usd or 0.0) != (m4_contract.budget_max_usd or 0.0):
                interp_mismatches.append(f"budget (Mode 3: ${m3_contract.budget_max_usd or 0:,.2f}, Mode 4: ${m4_contract.budget_max_usd or 0:,.2f})")
            if m3_contract.required_vcpus != m4_contract.required_vcpus:
                interp_mismatches.append(f"vCPUs (Mode 3: {m3_contract.required_vcpus}, Mode 4: {m4_contract.required_vcpus})")
            if m3_contract.required_ram_gb != m4_contract.required_ram_gb:
                interp_mismatches.append(f"RAM (Mode 3: {m3_contract.required_ram_gb}GB, Mode 4: {m4_contract.required_ram_gb}GB)")
            if m3_contract.latency_max_ms != m4_contract.latency_max_ms:
                interp_mismatches.append(f"latency (Mode 3: {m3_contract.latency_max_ms}ms, Mode 4: {m4_contract.latency_max_ms}ms)")
            if m3_contract.sla_availability_pct != m4_contract.sla_availability_pct:
                interp_mismatches.append(f"SLA (Mode 3: {m3_contract.sla_availability_pct}%, Mode 4: {m4_contract.sla_availability_pct}%)")
            if m3_contract.target_bandwidth_mbps != m4_contract.target_bandwidth_mbps:
                interp_mismatches.append(f"bandwidth (Mode 3: {m3_contract.target_bandwidth_mbps}Mbps, Mode 4: {m4_contract.target_bandwidth_mbps}Mbps)")
            if m3_contract.target_cpu_pct != m4_contract.target_cpu_pct:
                interp_mismatches.append(f"target CPU (Mode 3: {m3_contract.target_cpu_pct}%, Mode 4: {m4_contract.target_cpu_pct}%)")
            if m3_contract.max_cpu_pct != m4_contract.max_cpu_pct:
                interp_mismatches.append(f"max CPU (Mode 3: {m3_contract.max_cpu_pct}%, Mode 4: {m4_contract.max_cpu_pct}%)")
            if m3_contract.primary_region != m4_contract.primary_region:
                interp_mismatches.append(f"primary region (Mode 3: {m3_contract.primary_region}, Mode 4: {m4_contract.primary_region})")
            if m3_contract.secondary_region != m4_contract.secondary_region:
                interp_mismatches.append(f"secondary region (Mode 3: {m3_contract.secondary_region}, Mode 4: {m4_contract.secondary_region})")
        elif m3_contract != m4_contract:
            interp_mismatches.append("contract presence mismatch")

        cost_m3_str = f"${m3.recomputed_cost_usd:,.2f}" if (m3.feasibility == FeasibilityStatus.PASS and m3.recomputed_cost_usd is not None) else "N/A"
        cost_m4_str = f"${m4.recomputed_cost_usd:,.2f}" if (m4.feasibility == FeasibilityStatus.PASS and m4.recomputed_cost_usd is not None) else "N/A"
        print("  • Mode 3 vs Mode 4 Independent Execution:")
        print(f"    - Mode 3 Pure Symbolic Verdict   : Alloc [{m3.feasibility.value}] | Task [{m3.task_outcome}] | Cost: {cost_m3_str}")
        print(f"    - Mode 4 Neuro-Symbolic Verdict  : Alloc [{m4.feasibility.value}] | Task [{m4.task_outcome}] | Cost: {cost_m4_str}")
        
        both_infeasible = (m3.feasibility in [FeasibilityStatus.FAIL, FeasibilityStatus.NOT_EVALUABLE] and m4.feasibility in [FeasibilityStatus.FAIL, FeasibilityStatus.NOT_EVALUABLE])
        both_feasible = (m3.feasibility == FeasibilityStatus.PASS and m4.feasibility == FeasibilityStatus.PASS)

        if both_infeasible:
            if interp_mismatches:
                m_summary = "; ".join(interp_mismatches)
                print(f"    -> Both infeasible with Parameter Mismatches: Both pipelines determined the workload is infeasible, but extracted different parameters ({m_summary}).")
            else:
                print("    -> Both infeasible: Both pipelines parsed identical requirements and determined the workload is infeasible under the specified constraints.")
        elif interp_mismatches and both_feasible and same_cost and (m3.recomputed_cost_usd or 0.0) > 0:
            m_summary = "; ".join(interp_mismatches)
            print(f"    -> Coincidental Cost Match with Interpretation Mismatch: Both pipelines produced ${m3.recomputed_cost_usd:,.2f} plans, but extracted different parameters ({m_summary}). The answer matched only because the same allocation is optimal under both limits.")
        elif interp_mismatches and same_feasibility:
            m_summary = "; ".join(interp_mismatches)
            print(f"    -> Independent Feasible Plans with Parameter Mismatches: ({m_summary}).")
        elif interp_mismatches:
            m_summary = "; ".join(interp_mismatches)
            print(f"    -> Pipeline Divergence with Parameter Mismatches: Mode 3 [{m3.feasibility.value}], Mode 4 [{m4.feasibility.value}] ({m_summary}).")
        elif both_feasible and same_cost and not interp_mismatches:
            print(f"    -> Fully Aligned: Both pipelines parsed identical requirements and converged on identical feasibility and allocation cost (${m3.recomputed_cost_usd or 0.0:,.2f}).")
        elif both_feasible and not interp_mismatches:
            print("    -> Equivalent Feasibility: Both pipelines parsed identical requirements and found feasible solutions with independent allocations.")
        else:
            print(f"    -> Pipeline Divergence: Mode 3 evaluated [{m3.feasibility.value}], Mode 4 evaluated [{m4.feasibility.value}] due to distinct interpretation/solver paths.")

    if m1:
        if m1.normalization_status in [NormalizationStatus.TASK_INCOMPATIBLE, NormalizationStatus.SOLVER_INFEASIBLE]:
            m1_plan_cat = "REFUSAL_OR_UNSUPPORTED"
        elif m1.feasibility == FeasibilityStatus.PASS:
            m1_plan_cat = "VALID_PLAN"
        elif m1.normalization_status in [NormalizationStatus.UNPARSEABLE, NormalizationStatus.NORMALIZATION_FAILURE, NormalizationStatus.AMBIGUOUS, NormalizationStatus.MALFORMED_OUTPUT] or m1.feasibility == FeasibilityStatus.NOT_EVALUABLE:
            m1_plan_cat = "UNPARSEABLE (Not counted as wrong answer)"
        else:
            m1_plan_cat = "INVALID_PLAN"
        print(f"  • Mode 1 (Raw LLM) Classification  : {m1_plan_cat} (Status: {m1.normalization_status.value}, Task: [{m1.task_outcome}], Claimed: ${m1.claimed_cost_usd or 0.0:.2f})")
    if m2:
        print(f"  • Mode 2 (Schema LLM) Normalization: {m2.normalization_status.value} (Task [{m2.task_outcome}], Claimed: ${m2.claimed_cost_usd or 0.0:.2f})")

    print("\n  Requirement Provenance Note: Evaluated against human manifest specification and independent checker.")
    print("=" * 88 + "\n")


def print_manifest_grading_and_mismatch_report(
    manifest_entry: Optional[Dict[str, Any]],
    records: List[CanonicalExecutionRecord],
    m3_contract: Optional[CloudOptimizationContract],
    m4_contract: Optional[CloudOptimizationContract],
) -> None:
    """Grades every mode against the human-written manifest and prints a detailed Mode 3 vs Mode 4 Mismatch Report."""
    print_banner("MANIFEST GROUND-TRUTH GRADING & INTERPRETATION-MISMATCH REPORT", char="=")
    if not manifest_entry:
        print("  [Manifest Reference: Query not explicitly found in manifest files]")
        print("  Task Outcome: NOT_GRADED (Ad-hoc query excluded from manifest pass-rate grading)\n")
        print(format_plain_english_summary(records, key=None))
        return

    q_id = manifest_entry.get("query_id", "CUSTOM_QUERY")
    q_cat = manifest_entry.get("category", "N/A")
    exp_outcome = manifest_entry.get("expected_outcome", "FEASIBLE")
    intended_arch = manifest_entry.get("intended_archetype", "N/A")
    exp_opt = manifest_entry.get("expected_optimal_cost_usd")
    opt_src = manifest_entry.get("optimum_source", "N/A")

    print(f"  • Query ID              : {q_id}")
    print(f"  • Scenario Family       : {manifest_entry.get('scenario_family_id', 'N/A')}")
    print(f"  • Linguistic Class      : {q_cat}")
    print(f"  • Expected Outcome      : {exp_outcome}")
    print(f"  • Intended Archetype    : {intended_arch}")
    if exp_opt is not None:
        print(f"  • Expected Optimal Cost : ${float(exp_opt):,.2f} (Source: {opt_src})")
    print("-" * 88)

    w_field = 24
    w_exp = 18
    w_m3 = 20
    w_m4 = 20
    w_match = 16

    sep = f"+{'-'*(w_field+2)}+{'-'*(w_exp+2)}+{'-'*(w_m3+2)}+{'-'*(w_m4+2)}+{'-'*(w_match+2)}+"
    header = (
        f"| {'Requirement Field':<{w_field}} | {'Manifest Expected':<{w_exp}} | "
        f"{'Mode 3 (Symbolic)':<{w_m3}} | {'Mode 4 (Neural)':<{w_m4}} | {'Interpretation':<{w_match}} |"
    )

    print(sep)
    print(header)
    print(sep)

    fields_to_compare = [
        ("Problem Archetype", "intended_archetype", lambda c: c.problem_type if c else "UNSUPPORTED"),
        ("Cloud Providers", "cloud_providers", lambda c: str(c.cloud_providers) if c and c.cloud_providers else "['AWS']"),
        ("Budget Max (USD)", "budget_max_usd", lambda c: f"${c.budget_max_usd:,.2f}" if c and c.budget_max_usd is not None else "—"),
        ("Required vCPUs", "required_vcpus", lambda c: str(c.required_vcpus) if c and c.required_vcpus is not None else "—"),
        ("Required RAM (GB)", "required_ram_gb", lambda c: f"{c.required_ram_gb:.1f}" if c and c.required_ram_gb is not None else "—"),
        ("Target Bandwidth", "target_bandwidth_mbps", lambda c: f"{c.target_bandwidth_mbps:.1f} Mbps" if c and c.target_bandwidth_mbps is not None else "—"),
        ("Target CPU (%)", "target_cpu_pct", lambda c: f"{c.target_cpu_pct:.1f}%" if c and c.target_cpu_pct is not None else "—"),
        ("Max CPU Ceiling", "max_cpu_pct", lambda c: f"{c.max_cpu_pct:.1f}%" if c and c.max_cpu_pct is not None else "[UNPARSED]"),
        ("Latency Max (ms)", "latency_max_ms", lambda c: f"{c.latency_max_ms:.1f} ms" if c and c.latency_max_ms is not None else "—"),
        ("SLA Availability", "sla_availability_pct", lambda c: f"{c.sla_availability_pct:.2f}%" if c and c.sla_availability_pct is not None else "—"),
    ]

    for label, manifest_key, contract_extractor in fields_to_compare:
        exp_val = manifest_entry.get(manifest_key)
        if manifest_key == "budget_max_usd" and exp_val is not None:
            exp_str = f"${float(exp_val):,.2f}"
        elif manifest_key == "required_ram_gb" and exp_val is not None:
            exp_str = f"{float(exp_val):.1f}"
        elif manifest_key == "target_bandwidth_mbps" and exp_val is not None:
            exp_str = f"{float(exp_val):.1f} Mbps"
        elif manifest_key in ["target_cpu_pct", "max_cpu_pct"] and exp_val is not None:
            exp_str = f"{float(exp_val):.1f}%"
        elif manifest_key == "latency_max_ms" and exp_val is not None:
            exp_str = f"{float(exp_val):.1f} ms"
        elif manifest_key == "sla_availability_pct" and exp_val is not None:
            exp_str = f"{float(exp_val):.2f}%"
        elif exp_val is None:
            exp_str = "—"
        else:
            exp_str = str(exp_val)

        m3_str = contract_extractor(m3_contract)[:w_m3]
        m4_str = contract_extractor(m4_contract)[:w_m4]

        # Agreement classification
        if m3_str == m4_str:
            agree_str = "MATCH (Both)"
        elif m4_str in [exp_str, exp_str.replace(" ", "")]:
            agree_str = "Mode 4 Match"
        elif m3_str in [exp_str, exp_str.replace(" ", "")]:
            agree_str = "Mode 3 Match"
        else:
            agree_str = "Divergent"

        print(
            f"| {label:<{w_field}} | {exp_str:<{w_exp}} | {m3_str:<{w_m3}} | "
            f"{m4_str:<{w_m4}} | {agree_str:<{w_match}} |"
        )

    print(sep)

    # Flag any mode whose verified cost differs from manifest optimum by > $0.01
    if exp_opt is not None:
        for r in records:
            if r.recomputed_cost_usd is not None and r.feasibility == FeasibilityStatus.PASS:
                diff = abs(r.recomputed_cost_usd - float(exp_opt))
                if diff > 0.01:
                    print(f"  ! NOTICE: {r.mode_name} recomputed cost (${r.recomputed_cost_usd:,.2f}) differs from manifest expected optimum (${float(exp_opt):,.2f} via {opt_src}) by ${diff:.2f} (> $0.01 threshold)")

    print()
    print(format_plain_english_summary(records, key=manifest_entry))


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
        "query_id",
        "expected_outcome",
        "mode",
        "mode_name",
        "execution_path",
        "normalization_status",
        "claimed_cost",
        "recomputed_cost",
        "cost_error",
        "feasibility",
        "task_outcome",
        "strict_task_success",
        "lenient_task_success",
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
    query_input = args.query.strip()
    is_mock = args.mock_llm

    # Match against manifest
    manifest_entry = find_manifest_entry(query_input)
    if manifest_entry and ("Q" in query_input or re.match(r"^\d+$", query_input)):
        query_text = manifest_entry.get("query_text", query_input)
    else:
        query_text = query_input

    query_id = manifest_entry.get("query_id") if manifest_entry else None
    expected_outcome = manifest_entry.get("expected_outcome") if manifest_entry else None

    if is_mock:
        print("\n" + "*" * 88)
        print("*" + " MOCK DATA - OFFLINE BENCHMARK MODE ".center(86) + "*")
        print("*" + " Zero live LLM API calls are made; using offline fixtures ".center(86) + "*")
        print("*" * 88 + "\n")

    print_banner(
        "NEURASYM NEURO-SYMBOLIC 4-WAY COMPARATIVE BENCHMARK",
        f"Query: \"{query_text}\"" + (f" [ID: {query_id}]" if query_id else "") + (" [OFFLINE MOCK MODE]" if is_mock else ""),
        char="#",
    )

    records: List[CanonicalExecutionRecord] = []

    # 1. Mode 3 first to obtain contract & solver result
    m3_rec, m3_contract, solver_res = execute_mode_3_symbolic(query_text, manifest_entry=manifest_entry)

    # 2. Mode 1
    m1_rec = execute_mode_1_raw_llm(query_text, m3_contract, mock_llm=is_mock, manifest_entry=manifest_entry)
    records.append(m1_rec)

    # 3. Mode 2
    m2_rec = execute_mode_2_schema_llm(query_text, m3_contract, mock_llm=is_mock, manifest_entry=manifest_entry)
    records.append(m2_rec)

    # Add Mode 3
    records.append(m3_rec)

    # 4. Mode 4 (Genuine Groq interpretation + solver dispatch)
    m4_rec = execute_mode_4_neuro_symbolic(query_text, mock_llm=is_mock, manifest_entry=manifest_entry)
    records.append(m4_rec)

    # Extract m4 contract if present
    m4_contract = None
    if m4_rec.requirements and isinstance(m4_rec.requirements, dict):
        try:
            m4_contract = CloudOptimizationContract(**m4_rec.requirements)
        except Exception:
            m4_contract = None

    # Attach task outcomes and manifest provenance to all records
    for r in records:
        r.query_id = query_id
        r.expected_outcome = expected_outcome or "NOT_GRADED"
        r.task_outcome = evaluate_task_outcome(r, expected_outcome)
        r.is_mock = is_mock

    # Print comparative summary table
    print_comparison_table(records, m3_contract=m3_contract, m4_contract=m4_contract)

    # Print Manifest Grading & Interpretation-Mismatch Report
    print_manifest_grading_and_mismatch_report(manifest_entry, records, m3_contract, m4_contract)

    # Log to files
    log_canonical_records(query_text, records)

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "query": query_text,
                "query_id": query_id,
                "expected_outcome": expected_outcome,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "records": [r.to_dict() for r in records],
            }, f, indent=2)
        print(f"Benchmark records saved to: {out_path}")


if __name__ == "__main__":
    main()
