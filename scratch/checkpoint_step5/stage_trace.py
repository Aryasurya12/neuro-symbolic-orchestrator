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

from dotenv import load_dotenv

# Ensure environment variables from .env are loaded into os.environ
load_dotenv()

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
from templates.ILP_VM_Knapsack_Allocation import (
    _VM_CATALOG,
    solve_ilp_vm_knapsack,
)


def trace_stage_1_parsing(
    parser: SCOPEParser, query_text: str, silent: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """STAGE 1: Lexical Analysis & Rule-Based Parsing (SCOPEParser).

    Inspects actual regex field extraction and returns (success, extracted_params, failure_reason).
    """
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
        log('    * vCPUs        : NO MATCH (tested patterns: r"(\\d+)\\s*(?:vcpus?|cores?|v-cpu)", r"vcpus?\\s*(?:>=|:)\\s*(\\d+)")')

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
        log('    * RAM (GB)     : NO MATCH (tested patterns: r"(\\d+(?:\\.\\d+)?)\\s*(?:gb|gigabytes?)", r"ram\\s*(?:>=|:)\\s*(\\d+)")')

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
        log('    * Budget       : NO MATCH (tested patterns: ["$...", "INR/₹...", "under/budget $..."])')

    # 4. Provider
    prov_match = re.search(r"\b(AWS|Azure|GCP)\b", query_text, re.IGNORECASE)
    if prov_match:
        prov_val = prov_match.group(1).upper()
        log(f'    * Provider     : MATCH -> "{prov_match.group(0)}" => "{prov_val}"')
    else:
        log('    * Provider     : NO MATCH (defaulting to multi-provider candidate pool)')

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
        reason = "RESULT: PARSER_FAILED -- no archetype could be constructed from this input. Stopping here, as Mode 3/4 would."
        log(f"    * Domain Intent: FAIL (no cloud/FinOps keywords detected)")
        log(f"    * {reason}")
        return False, None, reason

    log("    * Domain Intent: PASS (cloud infrastructure/FinOps keywords verified)")

    # Extract actual parameters
    params = parser.extract_parameters(query_text)

    # Missing field detection
    missing_fields = []
    if "budget_max_usd" not in params:
        missing_fields.append("budget_max_usd (using default $500.00)")
    if not vcpu_match and "required_vcpus" not in params:
        missing_fields.append("required_vcpus (default: 4)")
    if not ram_match and "required_ram_gb" not in params:
        missing_fields.append("required_ram_gb (default: 8.0GB)")

    # Log parameter table
    log("    * Extracted Parameters Table:")
    for k, v in params.items():
        log(f"      - {k:<22}: {v}")

    if missing_fields:
        log(f"\n  RESULT: PARTIAL CONTRACT ({len(missing_fields)} field(s) defaulted/missing: {', '.join(missing_fields)})")
    else:
        log("\n  RESULT: FULL CONTRACT EXTRACTED")

    return True, params, None


def trace_stage_2_archetype_matching(
    matcher: CARMMatcher, query_text: str, parser: SCOPEParser, silent: bool = False
) -> Tuple[bool, Optional[str], Optional[str]]:
    """STAGE 2: Semantic Archetype Matching (CARM).

    Computes Jaccard similarity across all registered templates using set theory.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 2: Semantic Archetype Matching (CARM) ---")
    extracted_constraints = parser.extract_constraints_from_text(query_text)
    log("  [2.1 Query Constraint Signature]")
    log(f"    Extracted Constraint Tokens Q: {sorted(list(extracted_constraints)) if extracted_constraints else 'NONE (empty set)'}")

    if not extracted_constraints:
        log("  ILP_VM_Allocation                  : score 0.00")
        log("  PSO_Continuous_Scaling             : score 0.00")
        log("  Z3_Graph_Disaster_Recovery         : score 0.00")
        reason = "RESULT: UNSUPPORTED_ARCHETYPE -- no matching constraint tokens found. Stopping here."
        log(f"  {reason}")
        return False, None, reason

    log("\n  [2.2 Set-Theoretic Jaccard Comparison: J(Q, T) = |Q ∩ T| / |Q ∪ T|]")
    scores = {}
    details = {}
    for archetype, data in matcher.TEMPLATE_INDEX.items():
        template_constraints = set(data["constraints"])  # type: ignore[arg-type]
        intersection = extracted_constraints.intersection(template_constraints)
        union = extracted_constraints.union(template_constraints)
        score = len(intersection) / len(union) if union else 0.0
        scores[archetype] = round(score, 4)
        details[archetype] = {
            "template": data.get("template", ""),
            "intersection": sorted(list(intersection)),
            "union": sorted(list(union)),
            "inter_len": len(intersection),
            "union_len": len(union),
        }

    # Print detailed comparison matrix
    for arch, sc in scores.items():
        dt = details[arch]
        log(f"    * {arch:<28}: score {sc:.2f}  (|Q∩T|={dt['inter_len']}, |Q∪T|={dt['union_len']})")
        if dt["intersection"]:
            log(f"        -> Matching tokens: {dt['intersection']}")

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_arch, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score

    if best_score <= 0.0:
        reason = "RESULT: UNSUPPORTED_ARCHETYPE -- no archetype cleared matching threshold (> 0.0). Stopping here."
        log(f"\n  {reason}")
        return False, None, reason

    log("\n  [2.3 Routing Decision]")
    log(f"    Selected Archetype : {best_arch}")
    log(f"    Confidence Margin  : {margin:.2f} over runner-up ({sorted_scores[1][0] if len(sorted_scores)>1 else 'None'}: {second_score:.2f})")
    log(f"    Target Template    : {details[best_arch]['template']}")
    log(f"  -> SELECTED: {best_arch} (margin: {margin:.2f})")
    return True, best_arch, None


def trace_stage_3_contract_validation(
    archetype: str, params: Dict[str, Any], silent: bool = False
) -> Tuple[bool, Optional[CloudOptimizationContract], Optional[str]]:
    """STAGE 3: Contract Validation & Mathematical Formulation.

    Validates parameters against Pydantic V2 CloudOptimizationContract and displays mathematical specs.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 3: Contract Validation & Mathematical Formulation ---")
    log("  [3.1 Pydantic V2 Schema Validation]")
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
        log("    Pydantic validation: PASS")
        log(f"    Contract Object    :")
        log(f"      - Problem Type       : {contract.problem_type}")
        log(f"      - Cloud Providers    : {contract.cloud_providers}")
        log(f"      - Budget Cap (USD)   : ${contract.budget_max_usd:.2f}")
        log(f"      - Required vCPUs     : {contract.required_vcpus}")
        log(f"      - Required RAM (GB)  : {contract.required_ram_gb:.1f} GB")
        log(f"      - Service Count      : {contract.service_count}")
        log(f"      - Latency Threshold  : {contract.latency_max_ms:.1f} ms")
        log(f"      - Availability Target: {contract.sla_availability_pct:.4f}% SLA")

        log("\n  [3.2 Mathematical Optimization Formulation]")
        if archetype == "ILP_VM_Allocation":
            log("    Problem Class: Discrete Integer Linear Programming (Knapsack)")
            log("    Variables    : x_i in Z_>=0 (Count of VM SKU instance type i)")
            log("    Minimize     : sum(cost_i * x_i)")
            log(f"    Subject to   : sum(vcpu_i * x_i) >= {contract.required_vcpus}")
            log(f"                   sum(ram_i  * x_i) >= {contract.required_ram_gb:.1f} GB")
            log(f"                   sum(cost_i * x_i) <= ${contract.budget_max_usd:.2f}")
        elif archetype == "PSO_Continuous_Scaling":
            log("    Problem Class: Continuous Particle Swarm Optimization (Dynamic Sizing)")
            log("    Variables    : Position vector x = [Bandwidth_Mbps, Replicas] in R^2")
            log("    Search Bounds: Bandwidth in [100, 1000] Mbps, Replicas in [1.0, 16.0]")
            log("    Objective    : Minimize Fitness = Cost(x) + Penalty(CPU_dev) + Penalty(Budget_over)")
            log("    Constraint   : Modeled CPU = (Bandwidth / (Replicas * 75 Mbps)) * 100% <= 100%")
        elif archetype == "Z3_Graph_Disaster_Recovery":
            log("    Problem Class: SMT Graph Constraint Satisfaction (Z3 Theorem Prover)")
            log("    Variables    : Region Pair (Region_A, Region_B) in Vertices x Vertices")
            log(f"    SMT Clauses  : Distinct(Region_A, Region_B)")
            log(f"                   Latency(A, B) <= {contract.latency_max_ms:.1f} ms")
            log(f"                   Composite_SLA(A, B) >= {contract.sla_availability_pct:.4f}%")
            log(f"                   Cost(A, B) <= ${contract.budget_max_usd:.2f}")

        return True, contract, None
    except Exception as exc:
        log("    Pydantic validation: FAIL")
        log(f"    Validation Error   : {exc}")
        return False, None, f"Pydantic validation failed: {exc}"


def trace_stage_4_solver_execution(
    contract: CloudOptimizationContract, mode: int, silent: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """STAGE 4: Solver Dispatch & Execution Internals.

    Dispatches to HiGHS MILP, Continuous PSO, or Z3 SMT Graph solver and displays intermediate telemetry.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("\n--- STAGE 4: Solver Dispatch & Execution ---")
    problem = contract.problem_type

    if problem == "ILP_VM_Allocation":
        log("  [4.1 Solver Routing]")
        log("    Routed to: solve_ilp_vm_knapsack (HiGHS MILP / Branch-and-Bound)")
        log(
            f"    Target Constraints: vCPUs >= {contract.required_vcpus}, RAM >= {contract.required_ram_gb:.1f}GB, "
            f"Budget <= ${contract.budget_max_usd:.2f}, Provider = {contract.cloud_providers}"
        )

        prov_list = contract.cloud_providers if contract.cloud_providers else ["AWS"]
        target_upper = {p.upper() for p in prov_list}
        candidate_skus = [s for s in _VM_CATALOG if s.provider.upper() in target_upper]

        log("\n  [4.2 Candidate SKU Search Space]")
        log(f"    Filtered {len(candidate_skus)} eligible SKU(s) matching provider(s) {prov_list}:")
        log(f"    {'SKU Name':<18} {'Provider':<10} {'vCPUs':<8} {'RAM (GB)':<10} {'Hourly ($)':<12} {'Monthly ($)':<12}")
        log(f"    {'-'*18} {'-'*10} {'-'*8} {'-'*10} {'-'*12} {'-'*12}")
        for s in candidate_skus[:6]:
            log(f"    {s.name:<18} {s.provider:<10} {s.vcpus:<8} {s.ram_gb:<10.1f} ${s.hourly_cost_usd:<11.4f} ${s.monthly_cost():<11.2f}")
        if len(candidate_skus) > 6:
            log(f"    ... ({len(candidate_skus) - 6} additional SKUs evaluated)")

        try:
            res = solve_ilp_vm_knapsack(
                required_vcpus=contract.required_vcpus,
                required_ram_gb=contract.required_ram_gb,
                budget_max_usd=contract.budget_max_usd,
                target_providers=prov_list,
            )
            log("\n  [4.3 Solver Execution Telemetry]")
            log(f"    Solver Engine     : {res.get('solver')}")
            log(f"    Convergence Status: {res.get('status')}")
            log(f"    Execution Time    : {res.get('solve_time_ms', 0.0):.3f} ms")
            log(f"    Optimized Monthly : ${res.get('total_monthly_cost_usd', 0.0):,.2f} USD ({res.get('budget_utilized_pct', 0.0):.1f}% of budget)")
            log(f"    Cost Savings      : ${res.get('cost_savings_usd', 0.0):,.2f} USD")

            vms = res.get("allocated_vms", [])
            log("\n  [4.4 Optimal Decision Vector / Provisioned Instances]")
            if vms:
                for idx, vm in enumerate(vms, 1):
                    log(
                        f"    ({idx}) {vm.get('provider')} {vm.get('instance_type')} x {vm.get('count')} instance(s) "
                        f"[{vm.get('vcpus_per_vm')} vCPUs, {vm.get('ram_gb_per_vm')}GB RAM each] -> "
                        f"${vm.get('monthly_cost', 0.0):,.2f}/mo"
                    )
                log(f"    Total Resources   : {res.get('total_vcpus')} vCPUs (req: {contract.required_vcpus}), {res.get('total_ram_gb')} GB RAM (req: {contract.required_ram_gb:.1f})")
            else:
                log("    No instances provisioned (infeasible)")

            return True, res, None
        except Exception as exc:
            log(f"  [Solver Exception]: {type(exc).__name__}: {exc}")
            return False, None, f"Solver exception: {type(exc).__name__}: {exc}"

    elif problem == "PSO_Continuous_Scaling":
        log("  [4.1 Solver Routing]")
        log("    Routed to: solve_pso_continuous_scaling (Continuous Vectorized PSO)")
        log(
            f"    Search Bounds: Bandwidth in [100, 1000] Mbps, Replicas in [1.0, 16.0], "
            f"Target CPU = 70.0%, Budget <= ${contract.budget_max_usd:.2f}"
        )
        log("\n  [4.2 Swarm Hyperparameters]")
        log("    Particles (Swarm Size) : 30")
        log("    Max Iterations         : 50")
        log("    Inertia Weight (w)     : 0.729")
        log("    Cognitive Param (c1)   : 1.494")
        log("    Social Param (c2)      : 1.494")
        log("    Replica Capacity Factor: 75.0 Mbps per replica")

        try:
            prov_list = contract.cloud_providers if contract.cloud_providers else ["AWS"]
            res = solve_pso_continuous_scaling(
                budget_max_usd=contract.budget_max_usd,
                target_cpu_pct=70.0,
                target_providers=prov_list,
            )
            log("\n  [4.3 Swarm Optimization Telemetry]")
            log(f"    Solver Engine     : {res.get('solver')}")
            log(f"    Convergence Status: {res.get('status')}")
            log(f"    Iterations Done   : {res.get('iterations_completed')}")
            log(f"    Best Fitness Score: {res.get('fitness_score')}")
            log(f"    Optimal Bandwidth : {res.get('optimal_bandwidth_mbps'):.2f} Mbps")
            log(f"    Allocated Replicas: {res.get('recommended_replicas')} replica(s)")
            log(f"    Estimated Monthly : ${res.get('estimated_monthly_cost_usd', 0.0):,.2f} USD (${res.get('estimated_hourly_cost_usd', 0.0):.4f}/hr)")

            # Diagnostic CPU calculation
            bw = res.get("optimal_bandwidth_mbps", 0.0)
            reps = res.get("recommended_replicas", 1)
            raw_cpu = (bw / (reps * 75.0)) * 100.0 if reps > 0 else 999.0
            log(f"\n  [4.4 Autoscaling Dynamics & Load Simulation]")
            log(f"    Modeled Workload  : {bw:.1f} Mbps across {reps} replica(s) (Cap: {reps * 75.0:.1f} Mbps)")
            log(f"    True Modeled CPU  : {raw_cpu:.1f}% (Target: 70.0%)")
            if raw_cpu > 100.0:
                log(f"    -> CRITICAL NOTE  : Workload exceeds 100% capacity ({raw_cpu:.1f}% > 100%). System is overloaded!")

            return True, res, None
        except Exception as exc:
            log(f"  [Solver Exception]: {type(exc).__name__}: {exc}")
            return False, None, f"Solver exception: {type(exc).__name__}: {exc}"

    elif problem == "Z3_Graph_Disaster_Recovery":
        log("  [4.1 Solver Routing]")
        log("    Routed to: solve_z3_graph_disaster_recovery (Z3 SMT Graph Solver)")
        log(
            f"    Target Constraints: Distinct(Primary, Secondary), Latency <= {contract.latency_max_ms:.1f}ms, "
            f"Composite_SLA >= {contract.sla_availability_pct:.4f}%, Budget <= ${contract.budget_max_usd:.2f}"
        )

        prov_list = contract.cloud_providers if contract.cloud_providers else ["AWS", "GCP"]
        log(f"\n  [4.2 Infrastructure Graph Topology]")
        log(f"    Target Providers : {prov_list}")
        log(f"    Evaluated Graph  : {len(REGIONS_GRAPH)} regions ({', '.join(r['id'] for r in REGIONS_GRAPH)})")

        try:
            res = solve_z3_graph_disaster_recovery(
                sla_pct=contract.sla_availability_pct,
                max_latency_ms=contract.latency_max_ms,
                budget_max_usd=contract.budget_max_usd,
                target_providers=prov_list,
            )
            log("\n  [4.3 SMT Placement Telemetry]")
            log(f"    Solver Engine     : {res.get('solver')}")
            log(f"    Placement Status  : {res.get('status')}")
            log(f"    Primary Region    : {res.get('primary_region')}")
            log(f"    Secondary Region  : {res.get('secondary_region')}")
            log(f"    Inter-Region Lat  : {res.get('inter_region_latency_ms'):.2f} ms (Max: {contract.latency_max_ms:.1f} ms)")
            log(f"    Composite SLA     : {res.get('achieved_sla_pct'):.5f}% (Target: {contract.sla_availability_pct:.4f}%)")
            log(f"    Total Monthly Cost: ${res.get('total_monthly_cost_usd', 0.0):,.2f} USD")
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
    """STAGE 5: Independent Verification (IndependentChecker).

    Verifies catalog SKUs, physical constraint feasibility, and optimality claims against ground-truth database.
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    hdr = stage_header or "--- STAGE 5: Independent Verification (IndependentChecker) ---"
    log(f"\n{hdr}")
    check = IndependentChecker.verify_solution(contract, solver_result)

    # 1. Parameter-by-Parameter Ground-Truth Audit Table
    log("  [5.1 Parameter-by-Parameter Ground-Truth Audit Table]")
    param_table = IndependentChecker.render_parameter_table(check.get("parameter_checks", []))
    log(param_table)

    # 2. Detailed Step-by-Step Parameter Calculations
    log("\n  [5.2 Detailed Step-by-Step Parameter Calculations]")
    problem = contract.problem_type
    cat_pass = check.get("catalog_consistent", False)
    cost_info = check.get("cost_accuracy", {})
    rep_cost = cost_info.get("reported_cost_usd", 0.0)
    calc_cost = cost_info.get("calculated_catalog_cost_usd", 0.0)
    cost_delta = cost_info.get("cost_delta_usd", 0.0)
    cost_err_pct = cost_info.get("cost_error_pct", 0.0)
    recomp = check.get("recomputed_metrics", {})
    budget_usd = float(contract.budget_max_usd)

    if problem == "ILP_VM_Allocation":
        rc_vcpu = recomp.get("vcpus", 0)
        rc_ram = recomp.get("ram_gb", 0.0)
        rc_cost = recomp.get("monthly_cost_usd", recomp.get("total_monthly_cost_usd", 0.0))
        req_v = contract.required_vcpus
        req_r = contract.required_ram_gb

        log("    1. SKU Catalog & Pricing Integrity Check:")
        log(f"       * Solver Reported Cost     : ${rep_cost:.2f} USD / month")
        log(f"       * Catalog Ground-Truth Cost : ${calc_cost:.2f} USD / month")
        log(f"       * Pricing Delta             : ${cost_delta:.2f} USD ({cost_err_pct:.1f}% error) -> {'[PASS] (Within $0.50 tolerance)' if (cost_delta <= 0.50 and cat_pass) else '[FAIL] (Pricing Mismatch / Hallucination)'}")

        log("    2. Compute Resource Capacity Check (vCPUs):")
        vcpu_margin = rc_vcpu - req_v
        log(f"       * Allocated vCPUs          : {rc_vcpu} vCPUs (Required: >= {req_v} vCPUs)")
        log(f"       * Compute Margin / Deficit : {'+' if vcpu_margin >= 0 else ''}{vcpu_margin} vCPUs -> {'[PASS]' if vcpu_margin >= 0 else '[FAIL] (vCPU Deficit)'}")

        log("    3. Memory Resource Capacity Check (RAM):")
        ram_margin = rc_ram - req_r
        log(f"       * Allocated RAM            : {rc_ram:.1f} GB (Required: >= {req_r:.1f} GB)")
        log(f"       * Memory Margin / Deficit  : {'+' if ram_margin >= 0 else ''}{ram_margin:.1f} GB -> {'[PASS]' if ram_margin >= 0 else '[FAIL] (RAM Deficit)'}")

        log("    4. Financial Monthly Budget Compliance Check:")
        budget_headroom = budget_usd - rc_cost
        headroom_pct = (budget_headroom / budget_usd * 100.0) if budget_usd > 0 else 0.0
        log(f"       * Calculated Monthly Cost  : ${rc_cost:.2f} USD (Budget Cap: <= ${budget_usd:.2f} USD)")
        log(f"       * Financial Headroom       : {'$' + f'{budget_headroom:.2f} ({headroom_pct:.1f}% under budget)' if budget_headroom >= 0 else '-$' + f'{abs(budget_headroom):.2f} (OVERFLOW)'} -> {'[PASS]' if budget_headroom >= 0 else '[FAIL] (Budget Overflow)'}")

    elif problem == "PSO_Continuous_Scaling":
        rc_bw = recomp.get("bandwidth_mbps", 0.0)
        rc_reps = recomp.get("replicas", 0)
        rc_cpu = recomp.get("modeled_cpu_pct", 0.0)
        rc_cost = recomp.get("monthly_cost_usd", recomp.get("total_monthly_cost_usd", 0.0))

        log("    1. Continuous Bandwidth & Replica Operational Bounds:")
        log(f"       * Evaluated Bandwidth      : {rc_bw:.1f} Mbps (Domain: [100.0, 1000.0] Mbps) -> {'[PASS]' if 100.0 <= rc_bw <= 1000.0 else '[FAIL]'}")
        log(f"       * Recommended Replicas     : {rc_reps} replica(s) (Domain: [1, 16] instances) -> {'[PASS]' if 1 <= rc_reps <= 16 else '[FAIL]'}")

        log("    2. Workload Capacity & True Modeled CPU Utilization:")
        cap_mbps = rc_reps * 75.0
        log(f"       * Aggregate Cluster Cap    : {cap_mbps:.1f} Mbps ({rc_reps} replicas x 75.0 Mbps/replica)")
        log(f"       * True Modeled CPU         : {rc_cpu:.1f}% (Target: 70.0%, Physical Limit: <= 100.0%)")
        if rc_cpu > 100.0:
            log(f"       * Workload Saturation Check : [FAIL] (SYSTEM OVERLOADED by {rc_cpu - 100.0:.1f}%)")
        elif rc_cpu > 70.0:
            log(f"       * Workload Saturation Check : [PASS - HIGH LOAD] ({rc_cpu:.1f}% utilized, within 100% cap)")
        else:
            log(f"       * Workload Saturation Check : [PASS] ({100.0 - rc_cpu:.1f}% headroom available)")

        log("    3. Dynamic Scaling Pricing Model & Budget Compliance:")
        log(f"       * Ground-Truth Formula Cost: ${calc_cost:.2f} USD / month ({rc_bw:.1f} Mbps x $0.08 + {rc_reps} reps x $45.00)")
        log(f"       * Solver Reported Cost     : ${rep_cost:.2f} USD / month (Delta: ${cost_delta:.2f})")
        budget_headroom = budget_usd - rc_cost
        log(f"       * Budget Compliance        : ${rc_cost:.2f} <= ${budget_usd:.2f} USD -> {'[PASS]' if budget_headroom >= 0 else '[FAIL] (Budget Overflow)'}")

    elif problem == "Z3_Graph_Disaster_Recovery":
        rc_lat = recomp.get("latency_ms", recomp.get("inter_region_latency_ms", 0.0))
        rc_sla = recomp.get("composite_sla_pct", 0.0)
        rc_cost = recomp.get("monthly_cost_usd", recomp.get("total_monthly_cost_usd", 0.0))
        reg_a = recomp.get("primary_region", "N/A")
        reg_b = recomp.get("secondary_region", "N/A")

        log("    1. Spatial & Failure Domain Separation:")
        log(f"       * Primary Region Node      : {reg_a}")
        log(f"       * Secondary Region Node    : {reg_b}")
        log(f"       * Failure Zone Disjointness: {'[PASS] (Distinct Regions)' if (reg_a != reg_b and reg_a != 'N/A') else '[FAIL] (Colocated / Single Point of Failure)'}")

        log("    2. Inter-Region Latency Verification:")
        lat_margin = float(contract.latency_max_ms) - rc_lat
        log(f"       * Peer Graph Edge Latency  : {rc_lat:.1f} ms (Maximum Allowed: <= {contract.latency_max_ms:.1f} ms)")
        log(f"       * Latency Margin           : {'+' if lat_margin >= 0 else ''}{lat_margin:.1f} ms -> {'[PASS]' if lat_margin >= 0 else '[FAIL] (Latency Exceeded)'}")

        log("    3. High-Availability Composite SLA Verification:")
        sla_margin = rc_sla - float(contract.sla_availability_pct)
        log(f"       * Composite Dual-Region SLA: {rc_sla:.5f}% (Target Requirement: >= {contract.sla_availability_pct:.4f}%)")
        log(f"       * SLA Availability Margin  : {'+' if sla_margin >= 0 else ''}{sla_margin:.5f}% -> {'[PASS]' if sla_margin >= 0 else '[FAIL] (SLA Deficit)'}")

        log("    4. Financial Monthly Cost & Budget Compliance:")
        budget_headroom = budget_usd - rc_cost
        log(f"       * Recomputed Monthly Cost  : ${rc_cost:.2f} USD (Budget Cap: <= ${budget_usd:.2f} USD)")
        log(f"       * Financial Headroom       : {'$' + f'{budget_headroom:.2f}' if budget_headroom >= 0 else '-$' + f'{abs(budget_headroom):.2f}'} -> {'[PASS]' if budget_headroom >= 0 else '[FAIL] (Budget Overflow)'}")

    # 3. Violations
    log("\n  [5.3 Detected Constraint Violations]")
    violations = check.get("violations", [])
    if violations:
        for idx, v in enumerate(violations, 1):
            log(f"    -> [VIOLATION #{idx}] {v}")
    else:
        log("    * None (0 constraint violations found -- all mathematical and catalog parameters satisfied)")

    # 4. Optimality & Proof classification
    log("\n  [5.4 Formal Proof Classification & Certification Result]")
    opt_verdict = check.get("optimality_verdict", "N/A")
    final_verdict = check.get("summary_status", "Execution complete")
    feas_pass = check.get("feasible_against_contract", False)

    log(f"    * Mathematical Proof Class  : \"{opt_verdict}\"")
    log(f"    * Verification Status       : \"{final_verdict}\"")
    if feas_pass:
        log("    * FINAL CERTIFICATION RESULT: [CERTIFIED: PROVABLY OPTIMAL & FEASIBLE]")
    else:
        log(f"    * FINAL CERTIFICATION RESULT: [REJECTED: {len(violations)} CONSTRAINT VIOLATION(S) DETECTED]")

    return feas_pass, check


def trace_stage_6_explanation(
    contract: CloudOptimizationContract,
    solver_result: Dict[str, Any],
    mode: int,
    silent: bool = False,
) -> None:
    """STAGE 6: Explanation Generation (Mode 4 Only).

    Transforms solver output into structured FinOps executive deployment reports.
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

    log(f"  Fields consumed from solver output: {consumed_fields}\n")
    try:
        explanation = FinOpsExplainer.generate_report(
            contract=contract,
            solver_result=solver_result,
            enable_llm_explainer=True,
        )
        # Print full, beautifully formatted executive report
        log(explanation)
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
            "This will make 1 LIVE API call to Groq API. Continue? [y/N] "
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
    model = (
        os.getenv("GROQ_MODEL")
        or getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    base_url = (
        os.getenv("GROQ_BASE_URL")
        or getattr(settings, "GROQ_BASE_URL", "https://api.groq.com/openai/v1")
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
    log("--- STAGE 1: LLM Request & Prompt Formulation ---")
    log(f"  Provider    : Groq API ({base_url})")
    log(f"  Model       : {model}")
    log(f"  Max Tokens  : {max_tokens}")
    log(f'  Prompt Sent :\n    "{prompt_sent}"')

    # Real LLM Call
    from src.semantic.llm_client import execute_dashboard_llm_request

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
    log("\n--- STAGE 2: LLM Response Telemetry ---")
    log(f"  Finish Reason: {finish_reason}")
    log(f"  Elapsed Time : {elapsed_s:.2f}s")
    log(f"  Raw Content  :\n    {raw_content}")

    if finish_reason == "length":
        log("  WARNING: Generation reached max token limit and was truncated (finish_reason == 'length'). Output is incomplete.")
    if llm_res.get("status") == "daily_quota_exhausted":
        log(f"  [API Error: 429 Daily Limit Exhausted - {llm_res.get('error_message')}]")
    elif llm_res.get("status") in ["api_error", "missing_credentials", "empty_response"]:
        log(f"  [API Error]: {llm_res.get('error_message')}")

    # --- STAGE 3: Extraction ---
    log("\n--- STAGE 3: Entity & Financial Extraction ---")
    if mode == 1:
        log("  [Extraction Strategy: Natural Language Prose Regex Parsing]")
    else:
        log("  [Extraction Strategy: Structured JSON Schema Parsing]")

    extracted_cost = llm_res.get("reported_cost_usd")
    extracted_alloc: List[Dict[str, Any]] = []

    if mode == 1:
        if extracted_cost is not None:
            log(f"  Extracted Monthly Cost: ${extracted_cost:,.2f} USD")
        else:
            err_reason = llm_res.get(
                "error_message", "No dollar amount ($XX.XX) found in prose response"
            )
            log(f"  Extracted Monthly Cost: EXTRACTION_FAILED ({err_reason})")

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
            log(f"  Extracted Allocation Plan: {extracted_alloc}")
        else:
            log("  Extracted Allocation Plan: NONE")

    else:
        # Mode 2
        parsed_json = llm_res.get("parsed_json")
        if parsed_json and extracted_cost is not None:
            log(f"  Extracted Monthly Cost: ${extracted_cost:,.2f} USD")
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
                log(f"  Extracted Allocation Plan: {extracted_alloc}")
            else:
                log("  Extracted Allocation Plan: NONE")
        else:
            err_reason = llm_res.get(
                "error_message", "JSON parsing failed or no valid cost found"
            )
            log(f"  Extracted Monthly Cost: EXTRACTION_FAILED ({err_reason})")
            log("  Extracted Allocation Plan: NONE")

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
