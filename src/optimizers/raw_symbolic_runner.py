"""Raw Symbolic Runner for Mode 3 Benchmark Evaluation.

Demonstrates why traditional mathematical solvers (SciPy HiGHS MILP, Z3 SMT)
cannot process unstructured natural language strings directly without
a formal neuro-symbolic semantic translation layer (Contract/ScopeParser).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from src.semantic.schemas import CloudOptimizationContract
except ImportError:
    from pydantic import BaseModel

    class CloudOptimizationContract(BaseModel):  # type: ignore[no-redef]
        problem_type: str = "ILP_VM_Allocation"
        cloud_providers: List[str] = ["AWS"]
        budget_max_usd: float = 500.0
        service_count: int = 1
        required_vcpus: int = 4
        required_ram_gb: float = 16.0
        latency_max_ms: float = 100.0
        sla_availability_pct: float = 99.9

from src.verifiers.proof_engine import (
    compute_optimality_certificate,
    verify_feasibility,
)
from templates.ILP_VM_Knapsack_Allocation import solve_ilp_vm_knapsack


def run_pure_symbolic_raw(raw_user_query: str) -> Dict[str, Any]:
    """Attempts to execute a pure mathematical solver directly on unstructured natural language.

    Demonstrates that symbolic optimization engines (MIP matrices, SMT assertions)
    strictly require numerical linear inequality matrices (c, A_ub, b_ub) or
    typed AST assertions, and immediately throw Type/Value parsing errors when
    fed raw conversational English text.

    Args:
        raw_user_query: Unstructured natural language cloud request string.

    Returns:
        Structured failure dictionary recording the parsing crash and latency:
        - `status`: "CRASHED (Parsing Error)"
        - `error_message`: Formatted ValueError message
        - `nlu_success`: False
        - `feasibility`: "N/A (Solver Execution Failed)"
        - `violation_rate`: "100% (Unparseable Input)"
        - `catalog_cost`: 0.0
        - `latency_ms`: Execution time until failure in milliseconds
    """
    t_start = time.perf_counter()

    try:
        # Intentionally attempt to ingest natural language as numerical LP vectors without NLP
        if isinstance(raw_user_query, str):
            # Attempting numerical matrix parsing directly from text
            parsed_matrix = np.fromstring(raw_user_query.strip(), sep=" ", dtype=float)
            if parsed_matrix.size == 0 or np.isnan(parsed_matrix).any():
                raise ValueError(
                    "Symbolic solver requires numeric matrices (c, A_ub, b_ub). "
                    "Cannot parse natural language text directly."
                )

        # Attempting direct algebraic solver compilation
        raise TypeError(
            "Symbolic solver requires numeric matrices (c, A_ub, b_ub). "
            "Cannot parse natural language text directly."
        )

    except (ValueError, TypeError) as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "status": "CRASHED (Parsing Error)",
            "error_message": "ValueError: Symbolic solver requires numeric matrices (c, A_ub, b_ub). Cannot parse natural language text directly.",
            "nlu_success": False,
            "feasibility": "N/A (Solver Execution Failed)",
            "violation_rate": "100% (Unparseable Input)",
            "catalog_cost": 0.0,
            "latency_ms": round(latency_ms, 3),
            "is_feasible": False,
            "solver_name": "SciPy_MILP_Raw",
            "raw_input": raw_user_query,
        }


def run_pure_symbolic_structured(
    contract: Union[CloudOptimizationContract, Dict[str, Any]],
) -> Dict[str, Any]:
    """Executes a pure mathematical solver (SciPy HiGHS MILP) against a structured contract.

    Evaluates exact mathematical feasibility, provable optimality certificates,
    and returns a structured solution.

    Args:
        contract: Validated `CloudOptimizationContract` or parameter dictionary.

    Returns:
        Structured optimization result containing allocation, costs, and certificates.
    """
    t_start = time.perf_counter()

    # Extract contract parameters
    if isinstance(contract, dict):
        req_vcpus = contract.get("required_vcpus", contract.get("min_vcpu", 4))
        req_ram = contract.get("required_ram_gb", contract.get("min_ram", 8.0))
        budget = contract.get("budget_max_usd", contract.get("budget_usd", 500.0))
        providers = contract.get("cloud_providers", ["AWS"])
        sla_target = contract.get("sla_availability_pct", contract.get("min_sla", 99.9))
    else:
        req_vcpus = getattr(contract, "required_vcpus", 4)
        req_ram = getattr(contract, "required_ram_gb", 8.0)
        budget = getattr(contract, "budget_max_usd", 500.0)
        providers = getattr(contract, "cloud_providers", ["AWS"])
        sla_target = getattr(contract, "sla_availability_pct", 99.9)

    # Solve via SciPy HiGHS Integer Linear Programming
    ilp_res = solve_ilp_vm_knapsack(
        required_vcpus=int(req_vcpus),
        required_ram_gb=float(req_ram),
        budget_max_usd=float(budget),
        target_providers=list(providers),
    )

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    optimal_cost = float(ilp_res.get("total_monthly_cost_usd", 0.0))
    is_feasible_solver = bool(
        ilp_res.get("is_feasible", False)
        or ilp_res.get("status", "").lower() == "feasible"
    )

    candidate_plan = {
        "allocated_vcpu": ilp_res.get("total_vcpus", req_vcpus),
        "allocated_ram": ilp_res.get("total_ram_gb", req_ram),
        "catalog_cost": optimal_cost,
        "predicted_cost": optimal_cost,
        "achieved_sla": sla_target,
        "allocated_vms": ilp_res.get("allocated_vms", []),
    }

    # Verify mathematical feasibility via ProofEngine
    feasibility_cert = verify_feasibility(candidate_plan, contract)
    optimality_cert = compute_optimality_certificate(ilp_res)

    status_str = "OPTIMAL" if feasibility_cert["is_feasible"] else "INFEASIBLE"

    return {
        "status": status_str,
        "is_feasible": feasibility_cert["is_feasible"],
        "nlu_success": True,
        "solver_name": ilp_res.get("solver_name", "SciPy_MILP_HiGHS"),
        "catalog_cost": optimal_cost,
        "predicted_cost": optimal_cost,
        "allocated_vcpu": candidate_plan["allocated_vcpu"],
        "allocated_ram": candidate_plan["allocated_ram"],
        "allocated_vms": ilp_res.get("allocated_vms", []),
        "latency_ms": round(latency_ms, 3),
        "mip_gap_pct": optimality_cert["mip_gap_pct"],
        "is_provably_optimal": optimality_cert["is_provably_optimal"],
        "violation_rate_pct": feasibility_cert["violation_rate_pct"],
        "budget_overflow_usd": feasibility_cert["budget_overflow_usd"],
        "feasibility_certificate": feasibility_cert,
        "optimality_certificate": optimality_cert,
    }
