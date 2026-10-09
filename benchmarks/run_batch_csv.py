"""Resumable CSV Batch Benchmark Runner for Neurasym 4-Way Evaluation.

Evaluates every query in data/final_query_manifest.json through all 4 execution modes:
- Mode 1: Raw Unconstrained LLM (Groq API)
- Mode 2: Schema-Constrained LLM (Groq API)
- Mode 3: Pure Symbolic Pipeline (100% Offline & Deterministic)
- Mode 4: Full Neuro-Symbolic Pipeline (Groq Neural Workload Interpretation + Local Solvers)

Appends results row-by-row to a CSV file with immediate flush & fsync durability,
strict deduplication/resumption, and clean rate-limit handling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.dataset_manifest import ManifestManager
from src.reporting.explain_verdict import explain_run
from src.semantic.schemas import CloudOptimizationContract
from src.verifiers.canonical_record import (
    CanonicalExecutionRecord,
    FeasibilityStatus,
    NormalizationStatus,
    OptimalityStatus,
)

# Reuse real pipeline functions from run_all_modes_comparative
from run_all_modes_comparative import (
    execute_mode_1_raw_llm,
    execute_mode_2_schema_llm,
    execute_mode_3_symbolic,
    execute_mode_4_neuro_symbolic,
    extract_stated_parameters_from_prose,
    extract_stated_parameters_from_json,
    evaluate_task_outcome,
)


CSV_COLUMNS: List[str] = [
    "run_id",
    "timestamp_utc",
    "manifest_sha256",
    "code_commit",
    "is_mock",
    "trial",
    "query_id",
    "split",
    "category",
    "scenario_family",
    "previously_run_in_development",
    "query_text",
    "expected_outcome",
    "expected_archetype",
    "expected_vcpus",
    "expected_ram_gb",
    "expected_budget_usd",
    "expected_optimal_cost_usd",
    "optimum_source",
    "mode",
    "provider",
    "model",
    "parser",
    "solver",
    "execution_status",
    "attempts",
    "finish_reason",
    "normalization_status",
    "mode_outcome",
    "stated_budget_usd",
    "stated_vcpus",
    "stated_ram_gb",
    "stated_latency_ms",
    "stated_sla_pct",
    "interpretation_match",
    "interpretation_mismatch_fields",
    "allocated_summary",
    "allocated_vcpus",
    "allocated_ram_gb",
    "claimed_cost_usd",
    "recomputed_cost_usd",
    "cost_error_pct",
    "optimal_cost_gap_pct",
    "plan_valid",
    "violations",
    "claimed_cost_accurate",
    "proof_status",
    "scope_expansion",
    "task_success",
    "explain_label",
    "explain_headline",
    "parse_ms",
    "solve_ms",
    "llm_ms",
    "explanation_ms",
    "total_ms",
    "prompt_tokens",
    "completion_tokens",
    "raw_response_path",
    "manual_mode1_outcome",
    "manual_mode1_note",
]


def get_git_commit_hash() -> str:
    """Retrieves short git commit hash if available."""
    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_file_sha256(filepath: Path | str) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def format_cell_empty(val: Any) -> str:
    """Formats cell value; returns empty string for None, empty strings or inapplicable values (never 0)."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        if val != val:  # NaN check
            return ""
        return f"{val:.2f}"
    if isinstance(val, int):
        return str(val)
    return str(val).strip()


def summarize_allocation(normalized_alloc: Any, problem_type: str) -> str:
    """Summarizes allocation (e.g. '4x t3.medium (AWS)' or 'us-east-1 + us-central1')."""
    if not normalized_alloc or not isinstance(normalized_alloc, dict):
        return ""

    if "allocated_vms" in normalized_alloc and normalized_alloc["allocated_vms"]:
        vms = normalized_alloc["allocated_vms"]
        parts = []
        for vm in vms:
            if isinstance(vm, dict):
                sku = vm.get("sku", vm.get("instance_type", "VM"))
                qty = vm.get("count", vm.get("quantity", 1))
                prov = vm.get("provider", "")
                prov_str = f" ({prov})" if prov else ""
                parts.append(f"{qty}x {sku}{prov_str}")
        return ", ".join(parts)

    if "primary_region" in normalized_alloc and "secondary_region" in normalized_alloc:
        r1 = normalized_alloc.get("primary_region")
        r2 = normalized_alloc.get("secondary_region")
        if r1 and r2:
            return f"{r1} + {r2}"

    if "replicas" in normalized_alloc:
        reps = normalized_alloc.get("replicas")
        bw = normalized_alloc.get("bandwidth_mbps", normalized_alloc.get("target_bandwidth_mbps"))
        bw_str = f" ({bw} Mbps)" if bw is not None else ""
        return f"{reps} replicas{bw_str}"

    return ""


def check_interpretation_match(
    mode: int,
    stated_params: Dict[str, Any],
    manifest_query: Dict[str, Any],
) -> Tuple[Optional[bool], str]:
    """Compares extracted parameters for Modes 3 and 4 against the manifest ground-truth key."""
    if mode not in (3, 4):
        return None, ""

    mismatches: List[str] = []
    arch = manifest_query.get("intended_archetype", "")

    if arch == "ILP_VM_Allocation":
        k_vcpu = manifest_query.get("required_vcpus")
        s_vcpu = stated_params.get("required_vcpus")
        if k_vcpu is not None and s_vcpu != k_vcpu:
            mismatches.append(f"required_vcpus (expected={k_vcpu}, stated={s_vcpu})")

        k_ram = manifest_query.get("required_ram_gb")
        s_ram = stated_params.get("required_ram_gb")
        if k_ram is not None and (s_ram is None or abs(float(s_ram) - float(k_ram)) > 0.01):
            mismatches.append(f"required_ram_gb (expected={k_ram}, stated={s_ram})")

        k_bud = manifest_query.get("budget_max_usd")
        s_bud = stated_params.get("budget_max_usd")
        if k_bud is not None and (s_bud is None or abs(float(s_bud) - float(k_bud)) > 0.50):
            mismatches.append(f"budget_max_usd (expected={k_bud}, stated={s_bud})")

    elif arch == "Z3_Graph_Disaster_Recovery":
        k_lat = manifest_query.get("latency_max_ms")
        s_lat = stated_params.get("latency_max_ms")
        if k_lat is not None and (s_lat is None or abs(float(s_lat) - float(k_lat)) > 0.1):
            mismatches.append(f"latency_max_ms (expected={k_lat}, stated={s_lat})")

        k_sla = manifest_query.get("sla_availability_pct")
        s_sla = stated_params.get("sla_availability_pct")
        if k_sla is not None and (s_sla is None or abs(float(s_sla) - float(k_sla)) > 0.001):
            mismatches.append(f"sla_availability_pct (expected={k_sla}, stated={s_sla})")

        k_bud = manifest_query.get("budget_max_usd")
        s_bud = stated_params.get("budget_max_usd")
        if k_bud is not None and (s_bud is None or abs(float(s_bud) - float(k_bud)) > 0.50):
            mismatches.append(f"budget_max_usd (expected={k_bud}, stated={s_bud})")

    elif arch == "PSO_Continuous_Scaling":
        k_bw = manifest_query.get("target_bandwidth_mbps")
        s_bw = stated_params.get("target_bandwidth_mbps")
        if k_bw is not None and (s_bw is None or abs(float(s_bw) - float(k_bw)) > 0.1):
            mismatches.append(f"target_bandwidth_mbps (expected={k_bw}, stated={s_bw})")

        k_cpu = manifest_query.get("max_cpu_pct") or manifest_query.get("target_cpu_pct")
        s_cpu = stated_params.get("max_cpu_pct") or stated_params.get("target_cpu_pct")
        if k_cpu is not None and (s_cpu is None or abs(float(s_cpu) - float(k_cpu)) > 0.1):
            mismatches.append(f"max_cpu_pct (expected={k_cpu}, stated={s_cpu})")

    is_match = (len(mismatches) == 0)
    return is_match, "; ".join(mismatches)


def read_existing_csv_state(csv_path: Path) -> Tuple[Optional[str], Set[Tuple[str, int, int]]]:
    """Reads existing CSV to check manifest SHA and return completed (query_id, mode, trial) keys."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None, set()

    manifest_sha: Optional[str] = None
    completed_keys: Set[Tuple[str, int, int]] = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not manifest_sha and row.get("manifest_sha256"):
                manifest_sha = row.get("manifest_sha256")
            
            q_id = row.get("query_id")
            m_str = row.get("mode")
            t_str = row.get("trial")
            status = row.get("execution_status", "OK")

            if q_id and m_str and t_str:
                try:
                    mode_int = int(m_str)
                    trial_int = int(t_str)
                    # Skip re-running unless previous run failed with provider error or rate limit
                    if status not in ("PROVIDER_FAILURE", "RATE_LIMITED", "TIMEOUT"):
                        completed_keys.add((q_id, mode_int, trial_int))
                except ValueError:
                    pass

    return manifest_sha, completed_keys


def run_batch(
    manifest_path: Path | str,
    out_path: Path | str,
    trials: int = 1,
    delay: float = 2.0,
    pilot: Optional[int] = None,
    modes: Optional[List[int]] = None,
    mock_llm: bool = False,
    force_unapproved: bool = False,
    run_id: Optional[str] = None,
) -> None:
    """Executes the resumable batch run across all manifest queries."""
    manifest_file = Path(manifest_path).resolve()
    out_file = Path(out_path).resolve()

    if not manifest_file.exists():
        print(f"[FATAL] Manifest file missing at '{manifest_file}'. Aborting.")
        sys.exit(1)

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # 1. Approval safety gate
    app_status = str(manifest_data.get("approval_status", "UNAPPROVED")).upper()
    is_app = bool(manifest_data.get("is_approved", False))
    if (app_status != "APPROVED" or not is_app) and not force_unapproved and not mock_llm:
        print(f"[REFUSAL] Manifest approval_status is '{app_status}' (is_approved={is_app}).")
        print("  Batch execution is refused on unapproved draft manifests.")
        print("  To run with mock providers for testing, use '--mock'.")
        print("  To override explicitly during development, use '--force-unapproved'.")
        sys.exit(1)

    # 2. Manifest hash verification
    current_manifest_sha = compute_file_sha256(manifest_file)
    existing_sha, completed_keys = read_existing_csv_state(out_file)

    if existing_sha and existing_sha != current_manifest_sha:
        print(f"[REFUSAL] Output CSV '{out_file}' was generated with a different manifest SHA256.")
        print(f"  Existing CSV SHA : {existing_sha}")
        print(f"  Current SHA      : {current_manifest_sha}")
        print("  Cannot safely resume or append. Please specify a new --out path.")
        sys.exit(1)

    queries: List[Dict[str, Any]] = manifest_data.get("queries", [])
    if pilot is not None and pilot > 0:
        queries = queries[:pilot]

    target_modes = sorted(modes) if modes else [1, 2, 3, 4]
    active_run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    code_commit = get_git_commit_hash()

    raw_dir = PROJECT_ROOT / "results" / "raw" / active_run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_file.exists() and out_file.stat().st_size > 0

    print("=" * 96)
    print(" NEURASYM RESUMABLE BATCH BENCHMARK RUNNER ".center(96, "="))
    print("=" * 96)
    print(f"  Run ID           : {active_run_id}")
    print(f"  Manifest Path    : {manifest_file}")
    print(f"  Manifest SHA256  : {current_manifest_sha[:16]}...")
    print(f"  Output CSV       : {out_file}")
    print(f"  Total Queries    : {len(queries)}")
    print(f"  Target Modes     : {target_modes}")
    print(f"  Trials per Query : {trials}")
    print(f"  Inter-Call Delay : {delay:.1f}s")
    print(f"  Mock Mode        : {mock_llm}")
    print(f"  Resumed Keys     : {len(completed_keys)} previously completed")
    print("=" * 96 + "\n")

    with open(out_file, "a", encoding="utf-8", newline="") as csv_f:
        writer = csv.writer(csv_f, quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            writer.writerow(CSV_COLUMNS)
            csv_f.flush()
            os.fsync(csv_f.fileno())

        total_tasks = len(queries) * len(target_modes) * trials
        completed_count = len(completed_keys)

        for q_idx, q in enumerate(queries, 1):
            q_id = q["query_id"]
            q_text = q["query_text"]
            cat = q.get("category", "")
            fam = q.get("scenario_family_id", "")
            seen_dev = q.get("previously_run_in_development", False)
            exp_outcome = q.get("expected_outcome", "")
            exp_arch = q.get("intended_archetype", "")
            exp_vcpu = q.get("required_vcpus")
            exp_ram = q.get("required_ram_gb")
            exp_bud = q.get("budget_max_usd")
            exp_opt_cost = q.get("expected_optimal_cost_usd")
            opt_src = q.get("optimum_source")

            for mode_num in target_modes:
                for trial_num in range(1, trials + 1):
                    tuple_key = (q_id, mode_num, trial_num)

                    if tuple_key in completed_keys:
                        print(f"  [SKIP] Query {q_id} Mode {mode_num} Trial {trial_num} already in CSV.")
                        continue

                    print(f"[{q_idx}/{len(queries)}] Running {q_id} | Mode {mode_num} | Trial {trial_num}...")
                    t_start_trial = time.perf_counter()
                    timestamp_str = datetime.now(timezone.utc).isoformat()

                    raw_rel_path = ""
                    exec_status = "OK"
                    attempts_count = 1
                    finish_reason = "stop"
                    norm_status_str = "SUCCESS"
                    mode_outcome_str = "PLAN"
                    stated_dict: Dict[str, Any] = {}
                    alloc_summary = ""
                    alloc_vcpu: Optional[int] = None
                    alloc_ram: Optional[float] = None
                    claimed_cost: Optional[float] = None
                    recomp_cost: Optional[float] = None
                    cost_err_pct: Optional[float] = None
                    opt_gap_pct: Optional[float] = None
                    plan_valid: bool = False
                    violations_list: List[str] = []
                    claimed_cost_accurate: Optional[bool] = None
                    proof_status_str = "not proven by this mode"
                    scope_expansion_str = ""
                    task_success_val: Optional[int] = None
                    explain_label_str = ""
                    explain_headline_str = ""
                    parse_ms: float = 0.0
                    solve_ms: float = 0.0
                    llm_ms: float = 0.0
                    explanation_ms: float = 0.0
                    total_ms: float = 0.0
                    prompt_tokens: Optional[int] = None
                    completion_tokens: Optional[int] = None
                    provider_str = "Local"
                    model_str = "Local Solvers"
                    parser_str = "SCOPE"
                    solver_str = "HiGHS / Z3 / PSO"

                    try:
                        if mode_num == 1:
                            provider_str = "Groq" if not mock_llm else "MockProvider"
                            model_str = "openai/gpt-oss-120b"
                            parser_str = "Free-form Prose"
                            solver_str = "None (Direct Generation)"

                            rec = execute_mode_1_raw_llm(q_text, contract=None, mock_llm=mock_llm, manifest_entry=q)
                            raw_content = rec.original_response or ""

                            # Save raw response
                            raw_file_name = f"{q_id}_mode1_trial{trial_num}.txt"
                            raw_full_path = raw_dir / raw_file_name
                            raw_to_write = str(raw_content) if raw_content else f"[EMPTY_RESPONSE]: {norm_status_str} (status={exec_status})"
                            with open(raw_full_path, "w", encoding="utf-8") as rf:
                                rf.write(raw_to_write)
                            raw_rel_path = f"results/raw/{active_run_id}/{raw_file_name}"

                            norm_status_str = rec.normalization_status.value if isinstance(rec.normalization_status, NormalizationStatus) else str(rec.normalization_status)
                            stated_dict = extract_stated_parameters_from_prose(str(raw_content), rec.normalized_allocation)
                            claimed_cost = rec.claimed_cost_usd
                            recomp_cost = rec.recomputed_cost_usd
                            cost_err_pct = rec.cost_error_pct
                            violations_list = rec.violations or []
                            plan_valid = (rec.feasibility == FeasibilityStatus.PASS and not violations_list)

                            if norm_status_str in ["SOLVER_INFEASIBLE", "PROVEN_INFEASIBLE"]:
                                mode_outcome_str = "INFEASIBLE"
                            elif norm_status_str == "CLARIFICATION_REQUIRED":
                                mode_outcome_str = "CLARIFICATION"
                            elif norm_status_str in ["TASK_INCOMPATIBLE", "UNSUPPORTED"]:
                                mode_outcome_str = "UNSUPPORTED"
                            elif norm_status_str in ["UNPARSEABLE", "MALFORMED_OUTPUT"]:
                                mode_outcome_str = "UNPARSEABLE"
                            else:
                                mode_outcome_str = "PLAN"

                            alloc_summary = summarize_allocation(rec.normalized_allocation, rec.problem_type or "")
                            if rec.normalized_allocation and isinstance(rec.normalized_allocation, dict):
                                vms = rec.normalized_allocation.get("allocated_vms", [])
                                if vms and isinstance(vms, list):
                                    alloc_vcpu = sum(v.get("vcpus", 0) * v.get("count", 1) for v in vms if isinstance(v, dict))
                                    alloc_ram = sum(v.get("ram_gb", 0.0) * v.get("count", 1) for v in vms if isinstance(v, dict))

                            # Optimality and Proof Status
                            if rec.optimality_status:
                                proof_status_str = str(rec.optimality_status.value if hasattr(rec.optimality_status, "value") else rec.optimality_status)
                            else:
                                proof_status_str = "not proven by this mode"

                            if recomp_cost is not None and exp_opt_cost is not None and exp_opt_cost > 0 and plan_valid:
                                diff = recomp_cost - exp_opt_cost
                                opt_gap_pct = max(0.0, (diff / exp_opt_cost) * 100.0)

                            if cost_err_pct is not None:
                                claimed_cost_accurate = (cost_err_pct <= 0.50 or (claimed_cost is not None and recomp_cost is not None and abs(claimed_cost - recomp_cost) <= 0.50))

                            task_eval = evaluate_task_outcome(rec, exp_outcome)
                            task_success_val = 1 if task_eval == "SUCCESS" else (0 if task_eval == "FAILURE" else None)

                            verdict = explain_run(rec, key=q)
                            explain_label_str = verdict.get("label", "") if isinstance(verdict, dict) else getattr(verdict, "label", "")
                            explain_headline_str = verdict.get("headline", "") if isinstance(verdict, dict) else getattr(verdict, "headline", "")

                            llm_ms = rec.parsing_ms
                            solve_ms = rec.solving_ms
                            total_ms = rec.total_duration_ms

                        elif mode_num == 2:
                            provider_str = "Groq" if not mock_llm else "MockProvider"
                            model_str = "openai/gpt-oss-120b"
                            parser_str = "Pydantic JSON Schema"
                            solver_str = "None (Direct Generation)"

                            rec = execute_mode_2_schema_llm(q_text, contract=None, mock_llm=mock_llm, manifest_entry=q)
                            raw_content = rec.original_response or ""

                            raw_file_name = f"{q_id}_mode2_trial{trial_num}.json"
                            raw_full_path = raw_dir / raw_file_name
                            if isinstance(raw_content, dict):
                                raw_to_write = json.dumps(raw_content, indent=2)
                            elif raw_content:
                                raw_to_write = str(raw_content)
                            else:
                                raw_to_write = json.dumps({"status": "EMPTY_RESPONSE", "normalization_status": str(rec.normalization_status)}, indent=2)
                            with open(raw_full_path, "w", encoding="utf-8") as rf:
                                rf.write(raw_to_write)
                            raw_rel_path = f"results/raw/{active_run_id}/{raw_file_name}"

                            norm_status_str = rec.normalization_status.value if isinstance(rec.normalization_status, NormalizationStatus) else str(rec.normalization_status)
                            stated_dict = extract_stated_parameters_from_json(raw_content, raw_content=str(raw_content))
                            claimed_cost = rec.claimed_cost_usd
                            recomp_cost = rec.recomputed_cost_usd
                            cost_err_pct = rec.cost_error_pct
                            violations_list = rec.violations or []
                            plan_valid = (rec.feasibility == FeasibilityStatus.PASS and not violations_list)

                            if norm_status_str in ["SOLVER_INFEASIBLE", "PROVEN_INFEASIBLE"]:
                                mode_outcome_str = "INFEASIBLE"
                            elif norm_status_str == "CLARIFICATION_REQUIRED":
                                mode_outcome_str = "CLARIFICATION"
                            elif norm_status_str in ["TASK_INCOMPATIBLE", "UNSUPPORTED"]:
                                mode_outcome_str = "UNSUPPORTED"
                            elif norm_status_str in ["UNPARSEABLE", "MALFORMED_OUTPUT"]:
                                mode_outcome_str = "UNPARSEABLE"
                            else:
                                mode_outcome_str = "PLAN"

                            alloc_summary = summarize_allocation(rec.normalized_allocation, rec.problem_type or "")
                            if rec.normalized_allocation and isinstance(rec.normalized_allocation, dict):
                                vms = rec.normalized_allocation.get("allocated_vms", [])
                                if vms and isinstance(vms, list):
                                    alloc_vcpu = sum(v.get("vcpus", 0) * v.get("count", 1) for v in vms if isinstance(v, dict))
                                    alloc_ram = sum(v.get("ram_gb", 0.0) * v.get("count", 1) for v in vms if isinstance(v, dict))

                            if rec.optimality_status:
                                proof_status_str = str(rec.optimality_status.value if hasattr(rec.optimality_status, "value") else rec.optimality_status)
                            else:
                                proof_status_str = "not proven by this mode"

                            if recomp_cost is not None and exp_opt_cost is not None and exp_opt_cost > 0 and plan_valid:
                                diff = recomp_cost - exp_opt_cost
                                opt_gap_pct = max(0.0, (diff / exp_opt_cost) * 100.0)

                            if cost_err_pct is not None:
                                claimed_cost_accurate = (cost_err_pct <= 0.50 or (claimed_cost is not None and recomp_cost is not None and abs(claimed_cost - recomp_cost) <= 0.50))

                            task_eval = evaluate_task_outcome(rec, exp_outcome)
                            task_success_val = 1 if task_eval == "SUCCESS" else (0 if task_eval == "FAILURE" else None)

                            verdict = explain_run(rec, key=q)
                            explain_label_str = verdict.get("label", "") if isinstance(verdict, dict) else getattr(verdict, "label", "")
                            explain_headline_str = verdict.get("headline", "") if isinstance(verdict, dict) else getattr(verdict, "headline", "")

                            llm_ms = rec.parsing_ms
                            solve_ms = rec.solving_ms
                            total_ms = rec.total_duration_ms

                        elif mode_num == 3:
                            provider_str = "Local"
                            model_str = "Deterministic Rule-Based"
                            parser_str = "SCOPE"
                            solver_str = "HiGHS / Z3 / PSO"

                            rec, contract, solver_res = execute_mode_3_symbolic(q_text, manifest_entry=q)
                            norm_status_str = rec.normalization_status.value if isinstance(rec.normalization_status, NormalizationStatus) else str(rec.normalization_status)

                            if contract:
                                stated_dict = contract.model_dump()
                            stated_dict = {k: v for k, v in stated_dict.items() if v is not None}

                            claimed_cost = rec.claimed_cost_usd
                            recomp_cost = rec.recomputed_cost_usd
                            cost_err_pct = rec.cost_error_pct
                            violations_list = rec.violations or []
                            plan_valid = (rec.feasibility == FeasibilityStatus.PASS and not violations_list)

                            if norm_status_str in ["SOLVER_INFEASIBLE", "PROVEN_INFEASIBLE"]:
                                mode_outcome_str = "INFEASIBLE"
                            elif norm_status_str == "CLARIFICATION_REQUIRED":
                                mode_outcome_str = "CLARIFICATION"
                            elif norm_status_str in ["TASK_INCOMPATIBLE", "UNSUPPORTED"]:
                                mode_outcome_str = "UNSUPPORTED"
                            else:
                                mode_outcome_str = "PLAN"

                            alloc_summary = summarize_allocation(rec.normalized_allocation, rec.problem_type or "")
                            if rec.normalized_allocation and isinstance(rec.normalized_allocation, dict):
                                vms = rec.normalized_allocation.get("allocated_vms", [])
                                if vms and isinstance(vms, list):
                                    alloc_vcpu = sum(v.get("vcpus", 0) * v.get("count", 1) for v in vms if isinstance(v, dict))
                                    alloc_ram = sum(v.get("ram_gb", 0.0) * v.get("count", 1) for v in vms if isinstance(v, dict))

                            proof_status_str = str(rec.optimality_status.value if hasattr(rec.optimality_status, "value") else rec.optimality_status)

                            if recomp_cost is not None and exp_opt_cost is not None and exp_opt_cost > 0 and plan_valid:
                                diff = recomp_cost - exp_opt_cost
                                opt_gap_pct = max(0.0, (diff / exp_opt_cost) * 100.0)

                            claimed_cost_accurate = (cost_err_pct is not None and cost_err_pct <= 0.50)

                            task_eval = evaluate_task_outcome(rec, exp_outcome)
                            task_success_val = 1 if task_eval == "SUCCESS" else (0 if task_eval == "FAILURE" else None)

                            verdict = explain_run(rec, key=q)
                            explain_label_str = verdict.get("label", "") if isinstance(verdict, dict) else getattr(verdict, "label", "")
                            explain_headline_str = verdict.get("headline", "") if isinstance(verdict, dict) else getattr(verdict, "headline", "")

                            parse_ms = rec.parsing_ms
                            solve_ms = rec.solving_ms
                            total_ms = rec.total_duration_ms

                        elif mode_num == 4:
                            provider_str = "Groq" if not mock_llm else "MockProvider"
                            model_str = "openai/gpt-oss-120b (Interpreter)"
                            parser_str = "Neural Interpretation"
                            solver_str = "HiGHS / Z3 / PSO (Local)"

                            rec = execute_mode_4_neuro_symbolic(q_text, mock_llm=mock_llm, enable_explanation=True, manifest_entry=q)
                            raw_content = rec.original_response or ""

                            raw_file_name = f"{q_id}_mode4_trial{trial_num}.json"
                            raw_full_path = raw_dir / raw_file_name
                            if isinstance(raw_content, dict):
                                raw_to_write = json.dumps(raw_content, indent=2)
                            elif raw_content:
                                raw_to_write = str(raw_content)
                            else:
                                raw_to_write = json.dumps({"status": "EMPTY_RESPONSE", "normalization_status": str(rec.normalization_status)}, indent=2)
                            with open(raw_full_path, "w", encoding="utf-8") as rf:
                                rf.write(raw_to_write)
                            raw_rel_path = f"results/raw/{active_run_id}/{raw_file_name}"

                            norm_status_str = rec.normalization_status.value if isinstance(rec.normalization_status, NormalizationStatus) else str(rec.normalization_status)
                            if rec.requirements and isinstance(rec.requirements, dict):
                                stated_dict = dict(rec.requirements)

                            claimed_cost = rec.claimed_cost_usd
                            recomp_cost = rec.recomputed_cost_usd
                            cost_err_pct = rec.cost_error_pct
                            violations_list = rec.violations or []
                            plan_valid = (rec.feasibility == FeasibilityStatus.PASS and not violations_list)

                            if norm_status_str in ["SOLVER_INFEASIBLE", "PROVEN_INFEASIBLE"]:
                                mode_outcome_str = "INFEASIBLE"
                            elif norm_status_str == "CLARIFICATION_REQUIRED":
                                mode_outcome_str = "CLARIFICATION"
                            elif norm_status_str in ["TASK_INCOMPATIBLE", "UNSUPPORTED"]:
                                mode_outcome_str = "UNSUPPORTED"
                            else:
                                mode_outcome_str = "PLAN"

                            alloc_summary = summarize_allocation(rec.normalized_allocation, rec.problem_type or "")
                            if rec.normalized_allocation and isinstance(rec.normalized_allocation, dict):
                                vms = rec.normalized_allocation.get("allocated_vms", [])
                                if vms and isinstance(vms, list):
                                    alloc_vcpu = sum(v.get("vcpus", 0) * v.get("count", 1) for v in vms if isinstance(v, dict))
                                    alloc_ram = sum(v.get("ram_gb", 0.0) * v.get("count", 1) for v in vms if isinstance(v, dict))

                            proof_status_str = str(rec.optimality_status.value if hasattr(rec.optimality_status, "value") else rec.optimality_status)

                            if recomp_cost is not None and exp_opt_cost is not None and exp_opt_cost > 0 and plan_valid:
                                diff = recomp_cost - exp_opt_cost
                                opt_gap_pct = max(0.0, (diff / exp_opt_cost) * 100.0)

                            claimed_cost_accurate = (cost_err_pct is not None and cost_err_pct <= 0.50)

                            task_eval = evaluate_task_outcome(rec, exp_outcome)
                            task_success_val = 1 if task_eval == "SUCCESS" else (0 if task_eval == "FAILURE" else None)

                            verdict = explain_run(rec, key=q)
                            explain_label_str = verdict.get("label", "") if isinstance(verdict, dict) else getattr(verdict, "label", "")
                            explain_headline_str = verdict.get("headline", "") if isinstance(verdict, dict) else getattr(verdict, "headline", "")

                            llm_ms = rec.parsing_ms
                            solve_ms = rec.solving_ms
                            explanation_ms = rec.explanation_ms
                            total_ms = rec.total_duration_ms

                    except Exception as exc:
                        err_msg = str(exc)
                        print(f"  [ERROR] Mode {mode_num} execution exception: {err_msg}")
                        if "429" in err_msg or "rate limit" in err_msg.lower():
                            exec_status = "RATE_LIMITED"
                        elif "timeout" in err_msg.lower():
                            exec_status = "TIMEOUT"
                        else:
                            exec_status = "PROVIDER_FAILURE"
                        violations_list = [f"Execution Exception: {err_msg}"]
                        plan_valid = False
                        task_success_val = 0
                        total_ms = (time.perf_counter() - t_start_trial) * 1000.0

                    # Interpretation Match for Modes 3 & 4
                    interp_match, interp_mismatch_fields = check_interpretation_match(mode_num, stated_dict, q)

                    # Log live API calls to results/live_api_call_log.jsonl
                    if not mock_llm and mode_num in (1, 2, 4):
                        log_path = PROJECT_ROOT / "results" / "live_api_call_log.jsonl"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            with open(log_path, "a", encoding="utf-8") as lf:
                                if mode_num in (1, 2):
                                    lf.write(json.dumps({
                                        "timestamp_utc": timestamp_str,
                                        "mode": mode_num,
                                        "query": q_text,
                                        "model": model_str,
                                        "status": "success" if exec_status == "OK" else exec_status.lower(),
                                        "finish_reason": finish_reason,
                                        "elapsed_seconds": round(llm_ms / 1000.0, 2),
                                        "reported_cost_usd": claimed_cost,
                                        "error": None if exec_status == "OK" else "; ".join(violations_list),
                                    }) + "\n")
                                elif mode_num == 4:
                                    # Neural Interpreter call
                                    lf.write(json.dumps({
                                        "timestamp_utc": timestamp_str,
                                        "mode": 4,
                                        "stage": "Stage 4.1 Neural Interpreter",
                                        "query": q_text,
                                        "model": model_str,
                                        "status": "success" if exec_status == "OK" else exec_status.lower(),
                                        "finish_reason": finish_reason,
                                        "elapsed_seconds": round(llm_ms / 1000.0, 2),
                                        "reported_cost_usd": None,
                                        "error": None if exec_status == "OK" else "; ".join(violations_list),
                                    }) + "\n")
                                    # Stage 4.5 FinOps Explainer call
                                    if explanation_ms > 0:
                                        lf.write(json.dumps({
                                            "timestamp_utc": timestamp_str,
                                            "mode": 4,
                                            "stage": "Stage 4.5 FinOps Explainer",
                                            "query": q_text,
                                            "model": model_str,
                                            "status": "success",
                                            "finish_reason": "stop",
                                            "elapsed_seconds": round(explanation_ms / 1000.0, 2),
                                            "reported_cost_usd": claimed_cost,
                                            "error": None,
                                        }) + "\n")
                        except Exception:
                            pass

                    # Invariant Check: plan_valid=false or violations MUST NEVER have task_success=1 on FEASIBLE
                    if (not plan_valid or violations_list) and exp_outcome == "FEASIBLE":
                        task_success_val = 0

                    split_val = "dev" if (seen_dev is True or q.get("split") == "dev") else "eval"

                    row_dict: Dict[str, Any] = {
                        "run_id": active_run_id,
                        "timestamp_utc": timestamp_str,
                        "manifest_sha256": current_manifest_sha,
                        "code_commit": code_commit,
                        "is_mock": mock_llm,
                        "trial": trial_num,
                        "query_id": q_id,
                        "split": split_val,
                        "category": cat,
                        "scenario_family": fam,
                        "previously_run_in_development": seen_dev,
                        "query_text": q_text,
                        "expected_outcome": exp_outcome,
                        "expected_archetype": exp_arch,
                        "expected_vcpus": exp_vcpu,
                        "expected_ram_gb": exp_ram,
                        "expected_budget_usd": exp_bud,
                        "expected_optimal_cost_usd": exp_opt_cost,
                        "optimum_source": opt_src,
                        "mode": mode_num,
                        "provider": provider_str,
                        "model": model_str,
                        "parser": parser_str,
                        "solver": solver_str,
                        "execution_status": exec_status,
                        "attempts": attempts_count,
                        "finish_reason": finish_reason,
                        "normalization_status": norm_status_str,
                        "mode_outcome": mode_outcome_str,
                        "stated_budget_usd": stated_dict.get("budget_max_usd") or stated_dict.get("budget_usd"),
                        "stated_vcpus": stated_dict.get("required_vcpus") or stated_dict.get("vcpus"),
                        "stated_ram_gb": stated_dict.get("required_ram_gb") or stated_dict.get("ram_gb"),
                        "stated_latency_ms": stated_dict.get("latency_max_ms") or stated_dict.get("latency_ms"),
                        "stated_sla_pct": stated_dict.get("sla_availability_pct") or stated_dict.get("sla_pct"),
                        "interpretation_match": interp_match,
                        "interpretation_mismatch_fields": interp_mismatch_fields,
                        "allocated_summary": alloc_summary,
                        "allocated_vcpus": alloc_vcpu,
                        "allocated_ram_gb": alloc_ram,
                        "claimed_cost_usd": claimed_cost,
                        "recomputed_cost_usd": recomp_cost,
                        "cost_error_pct": cost_err_pct,
                        "optimal_cost_gap_pct": opt_gap_pct,
                        "plan_valid": plan_valid,
                        "violations": "; ".join(violations_list),
                        "claimed_cost_accurate": claimed_cost_accurate,
                        "proof_status": proof_status_str,
                        "scope_expansion": scope_expansion_str,
                        "task_success": task_success_val,
                        "explain_label": explain_label_str,
                        "explain_headline": explain_headline_str,
                        "parse_ms": round(parse_ms, 2) if parse_ms > 0 else "",
                        "solve_ms": round(solve_ms, 2) if solve_ms > 0 else "",
                        "llm_ms": round(llm_ms, 2) if llm_ms > 0 else "",
                        "explanation_ms": round(explanation_ms, 2) if explanation_ms > 0 else "",
                        "total_ms": round(total_ms, 2),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "raw_response_path": raw_rel_path,
                        "manual_mode1_outcome": "",
                        "manual_mode1_note": "",
                    }

                    # Format row according to CSV_COLUMNS
                    row_values = [format_cell_empty(row_dict.get(col)) for col in CSV_COLUMNS]

                    # Append, flush, fsync immediately
                    writer.writerow(row_values)
                    csv_f.flush()
                    os.fsync(csv_f.fileno())

                    completed_count += 1
                    completed_keys.add(tuple_key)
                    print(f"    -> Status: {exec_status} | Mode Outcome: {mode_outcome_str} | Task Success: {task_success_val} | Time: {total_ms:.1f}ms")

                    # Handle Rate Limit clean exit
                    if exec_status == "RATE_LIMITED":
                        print("\n[RATE_LIMITED] HTTP 429 received and backoff exhausted. Stopping batch cleanly.")
                        print(f"Results saved durably up to query '{q_id}' mode {mode_num}.")
                        return

                    # Inter-call delay for live LLM requests
                    if not mock_llm and mode_num in (1, 2, 4) and delay > 0:
                        time.sleep(delay)

    # Automatically generate synced Excel spreadsheet (.xlsx)
    try:
        from scripts.export_excel import export_csv_to_excel
        xlsx_file = out_file.with_suffix(".xlsx")
        export_csv_to_excel(csv_path=out_file, manifest_path=manifest_file, excel_path=xlsx_file)
    except Exception as exc:
        print(f"  [NOTE] Excel export skipped: {exc}")

    print("\n" + "=" * 96)
    print(f" BATCH RUN COMPLETED SUCCESSFULLY: {completed_count}/{total_tasks} rows written ".center(96, "="))
    print(f" CSV Output Location   : {out_file}")
    print(f" Excel Output Location : {out_file.with_suffix('.xlsx')}")
    print("=" * 96)


def main():
    parser = argparse.ArgumentParser(description="Neurasym Resumable Batch Benchmark CSV Runner")
    parser.add_argument("--manifest", type=str, default="data/final_query_manifest.json", help="Path to manifest JSON")
    parser.add_argument("--out", type=str, default="results/study_run_01.csv", help="Path to output CSV file")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per query x mode")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between live calls")
    parser.add_argument("--pilot", type=int, default=None, help="Run only the first N queries")
    parser.add_argument("--modes", type=str, default="1,2,3,4", help="Comma-separated modes (e.g. '3,4' or '1,2,3,4')")
    parser.add_argument("--mock", action="store_true", help="Use offline mock provider fixtures")
    parser.add_argument("--force-unapproved", action="store_true", help="Allow running unapproved manifest in dev")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID")

    args = parser.parse_args()

    mode_list = [int(m.strip()) for m in args.modes.split(",") if m.strip().isdigit()]

    run_batch(
        manifest_path=args.manifest,
        out_path=args.out,
        trials=args.trials,
        delay=args.delay,
        pilot=args.pilot,
        modes=mode_list,
        mock_llm=args.mock,
        force_unapproved=args.force_unapproved,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
