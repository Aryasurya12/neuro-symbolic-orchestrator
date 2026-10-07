"""Symbolic Solver Runner for Mode 3 Benchmark Evaluation.

Executes genuine traditional mathematical solvers (SciPy HiGHS MILP, Z3 SMT,
Continuous PSO) utilizing local rule-based parsing (SCOPE regex/CARM matcher)
to obtain structured optimization parameters from natural language inputs.

Declared correction: Traditional solvers strictly operate on structured inputs.
Mode 3 uses local rule-based parsing as the baseline input interface rather
than artificial exceptions.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from src.semantic.schemas import CloudOptimizationContract
from src.semantic.scope_parser import SCOPEParser
from src.verifiers.independent_checker import IndependentChecker
from templates.Continuous_PSO_Dynamic_Scaling import solve_pso_continuous_scaling
from templates.Graph_SMT_Z3_MultiRegion_Placement import solve_z3_graph_disaster_recovery
from templates.ILP_VM_Knapsack_Allocation import solve_ilp_vm_knapsack


def run_symbolic_rule_based(
    query_text: str,
    contract: Optional[Union[CloudOptimizationContract, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Executes Mode 3 (Symbolic + rule-based parsing).

    Traditional mathematical solvers require structured numerical matrices or
    relational constraint formulas. Mode 3 extracts these parameters using the
    local deterministic rule-based parser (SCOPEParser regex & CARM matcher)
    without any LLM calls, then routes to the appropriate traditional solver template.

    Args:
        query_text: Natural language user query string.
        contract: Optional pre-parsed contract. If None, parses locally via SCOPEParser.

    Returns:
        Structured result dictionary containing solver allocations, timing boundaries,
        and independent verification checks.
    """
    t_start = time.perf_counter()
    parse_latency_ms = 0.0

    # 1. Local Rule-Based Parsing Boundary
    parsed_contract: CloudOptimizationContract
    if contract is not None:
        if isinstance(contract, dict):
            c_dict = dict(contract)
            if "problem_type" not in c_dict:
                c_dict["problem_type"] = "ILP_VM_Allocation"
            parsed_contract = CloudOptimizationContract(**c_dict)
        else:
            parsed_contract = contract
    else:
        t_p0 = time.perf_counter()
        try:
            parser = SCOPEParser()
            if not parser.has_cloud_intent(query_text):
                # Query lacks cloud optimization intent
                parse_latency_ms = (time.perf_counter() - t_p0) * 1000.0
                total_latency_ms = (time.perf_counter() - t_start) * 1000.0
                return {
                    "mode": "Mode 3: Symbolic + rule-based parsing",
                    "mode_name": "Symbolic + rule-based parsing",
                    "status": "UNSUPPORTED_INTENT",
                    "solver_name": "None",
                    "nlu_success": False,
                    "is_feasible": False,
                    "catalog_cost": 0.0,
                    "allocated_vcpu": 0,
                    "allocated_ram": 0.0,
                    "is_provably_optimal": False,
                    "violation_rate_pct": 100.0,
                    "budget_overflow_usd": 0.0,
                    "error_message": f"Rule-based parser identified no cloud optimization intent in: '{query_text}'",
                    "parse_latency_ms": round(parse_latency_ms, 2),
                    "solve_latency_ms": 0.0,
                    "latency_ms": round(total_latency_ms, 2),
                    "total_monthly_cost_usd": 0.0,
                    "allocated_vms": [],
                    "contract": None,
                    "solver_res": {},
                    "independent_check": {
                        "structure_valid": False,
                        "catalog_consistent": False,
                        "feasible_against_contract": False,
                        "interpretation_correct": "Not independently evaluated",
                        "optimality_verdict": "Infeasible",
                        "summary_status": "No cloud intent detected",
                        "violations": ["Query did not match any known cloud optimization rules."],
                        "cost_accuracy": {"reported_cost_usd": 0.0, "calculated_catalog_cost_usd": 0.0, "cost_delta_usd": 0.0, "cost_error_pct": 0.0},
                    },
                }

            parsed_contract, _, _ = parser.parse_query_to_contract(query_text)
            parse_latency_ms = (time.perf_counter() - t_p0) * 1000.0
        except Exception as exc:
            parse_latency_ms = (time.perf_counter() - t_p0) * 1000.0
            total_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "mode": "Mode 3: Symbolic + rule-based parsing",
                "mode_name": "Symbolic + rule-based parsing",
                "status": "PARSING_ERROR",
                "solver_name": "None",
                "nlu_success": False,
                "is_feasible": False,
                "catalog_cost": 0.0,
                "allocated_vcpu": 0,
                "allocated_ram": 0.0,
                "is_provably_optimal": False,
                "violation_rate_pct": 100.0,
                "budget_overflow_usd": 0.0,
                "error_message": f"Rule-based parser failed: {str(exc)}",
                "parse_latency_ms": round(parse_latency_ms, 2),
                "solve_latency_ms": 0.0,
                "latency_ms": round(total_latency_ms, 2),
                "total_monthly_cost_usd": 0.0,
                "allocated_vms": [],
                "contract": None,
                "solver_res": {},
                "independent_check": {
                    "structure_valid": False,
                    "catalog_consistent": False,
                    "feasible_against_contract": False,
                    "interpretation_correct": "Not independently evaluated",
                    "optimality_verdict": "Infeasible",
                    "summary_status": "Execution error",
                    "violations": [f"Parsing error: {str(exc)}"],
                    "cost_accuracy": {"reported_cost_usd": 0.0, "calculated_catalog_cost_usd": 0.0, "cost_delta_usd": 0.0, "cost_error_pct": 0.0},
                },
            }

    # 2. Solver Execution Boundary
    problem_type = getattr(parsed_contract, "problem_type", "ILP_VM_Allocation")
    target_providers = getattr(parsed_contract, "cloud_providers", ["AWS"])
    budget_usd = float(getattr(parsed_contract, "budget_max_usd", 500.0))

    t_solve_start = time.perf_counter()
    solver_res: Dict[str, Any]

    if problem_type == "ILP_VM_Allocation":
        req_vcpus = int(getattr(parsed_contract, "required_vcpus", 4))
        req_ram = float(getattr(parsed_contract, "required_ram_gb", 8.0))
        solver_res = solve_ilp_vm_knapsack(
            required_vcpus=req_vcpus,
            required_ram_gb=req_ram,
            budget_max_usd=budget_usd,
            target_providers=list(target_providers),
        )
    elif problem_type == "Z3_Graph_Disaster_Recovery":
        target_sla = float(getattr(parsed_contract, "sla_availability_pct", 99.99))
        max_lat = float(getattr(parsed_contract, "latency_max_ms", 100.0))
        solver_res = solve_z3_graph_disaster_recovery(
            sla_pct=target_sla,
            max_latency_ms=max_lat,
            budget_max_usd=budget_usd,
            target_providers=list(target_providers),
        )
    elif problem_type == "PSO_Continuous_Scaling":
        solver_res = solve_pso_continuous_scaling(
            bandwidth_min_mbps=100.0,
            bandwidth_max_mbps=1000.0,
            target_cpu_pct=70.0,
            budget_max_usd=budget_usd,
            target_providers=list(target_providers),
        )
    else:
        solver_res = {
            "status": "UNSUPPORTED_ARCHETYPE",
            "solver": "None",
            "is_feasible": False,
            "error_message": f"Unsupported archetype: {problem_type}",
            "total_monthly_cost_usd": 0.0,
        }

    solve_latency_ms = (time.perf_counter() - t_solve_start) * 1000.0
    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    # 3. Independent Result Verification
    independent_check = IndependentChecker.verify_solution(parsed_contract, solver_res)

    optimal_cost = float(
        solver_res.get(
            "total_monthly_cost_usd",
            solver_res.get("estimated_monthly_cost_usd", solver_res.get("catalog_cost", 0.0)),
        )
    )

    recomp = independent_check.get("recomputed_metrics", {})
    alloc_vcpu = recomp.get("vcpus", solver_res.get("total_vcpus", 0))
    alloc_ram = recomp.get("ram_gb", solver_res.get("total_ram_gb", 0.0))
    is_opt = "Provably Optimal" in independent_check.get("optimality_verdict", "")

    return {
        "mode": "Mode 3: Symbolic + rule-based parsing",
        "mode_name": "Symbolic + rule-based parsing",
        "status": solver_res.get("status", "UNKNOWN"),
        "solver_name": solver_res.get("solver", solver_res.get("solver_name", "Traditional_Solver")),
        "nlu_success": True,
        "is_feasible": independent_check["feasible_against_contract"],
        "catalog_cost": optimal_cost,
        "total_monthly_cost_usd": optimal_cost,
        "allocated_vms": solver_res.get("allocated_vms", []),
        "allocated_vcpu": alloc_vcpu,
        "allocated_ram": alloc_ram,
        "is_provably_optimal": is_opt,
        "violation_rate_pct": 0.0 if independent_check["feasible_against_contract"] else 100.0,
        "budget_overflow_usd": max(0.0, optimal_cost - budget_usd),
        "parse_latency_ms": round(parse_latency_ms, 2),
        "solve_latency_ms": round(solve_latency_ms, 2),
        "latency_ms": round(total_latency_ms, 2),
        "contract": parsed_contract,
        "solver_res": solver_res,
        "independent_check": independent_check,
        "raw_input": query_text,
        "feasibility_certificate": {"is_feasible": independent_check["feasible_against_contract"]},
        "optimality_certificate": {"is_provably_optimal": is_opt, "status_str": independent_check.get("optimality_verdict", "")},
    }


# Backwards compatibility aliases
def run_pure_symbolic_raw(raw_user_query: str) -> Dict[str, Any]:
    """Compatibility wrapper delegating to run_symbolic_rule_based."""
    return run_symbolic_rule_based(raw_user_query)


def run_pure_symbolic_structured(
    contract: Union[CloudOptimizationContract, Dict[str, Any]],
) -> Dict[str, Any]:
    """Executes traditional solver against structured contract with independent checking."""
    return run_symbolic_rule_based(query_text="", contract=contract)
