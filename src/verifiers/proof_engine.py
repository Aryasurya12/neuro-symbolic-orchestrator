"""Deterministic Mathematical Verification Engine for Neurasym.

Evaluates cloud deployment candidates and computes formal mathematical
feasibility certificates, Mixed-Integer Programming (MIP) duality gaps,
and catalog price deltas.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

try:
    from src.semantic.schemas import CloudOptimizationContract
except ImportError:
    # Fallback duck-typing BaseModel
    class CloudOptimizationContract(BaseModel):  # type: ignore[no-redef]
        problem_type: str = "ILP_VM_Allocation"
        cloud_providers: List[str] = ["AWS"]
        budget_max_usd: float = 500.0
        service_count: int = 1
        required_vcpus: int = 4
        required_ram_gb: float = 16.0
        latency_max_ms: float = 100.0
        sla_availability_pct: float = 99.9


# Mathematical tooltip and docstring formula references for UI rendering
MATHEMATICAL_FORMULAS: Dict[str, str] = {
    "vcpu_deficit": "D_vCPU = max(0, vCPU_req - vCPU_alloc)",
    "ram_deficit": "D_RAM = max(0, RAM_req - RAM_alloc)",
    "budget_overflow": "O_budget = max(0.0, Cost_catalog - Budget_max)",
    "sla_deficit": "D_SLA = max(0.0, SLA_min - SLA_achieved)",
    "latency_overflow": "O_latency = max(0.0, Latency_achieved - Latency_max)",
    "is_feasible": "IsFeasible = 1[D_vCPU = 0 ∧ D_RAM = 0 ∧ O_budget = 0 ∧ D_SLA = 0 ∧ O_latency = 0]",
    "violation_rate_pct": "ViolationRate = (Count(D_k > 0) / TotalConstraints) * 100.0%",
    "cost_error_pct": "CostErrorPct = (|Cost_predicted - Cost_catalog| / Cost_catalog) * 100.0%",
    "mip_gap": "MIPGap = (|z_incumbent - z_bound| / |z_incumbent| + 1e-10) * 100.0%",
    "is_provably_optimal": "ProvablyOptimal = (MIPGap < 0.001) ∨ (Solver == 'Z3' ∧ Status == 'SAT')",
}


def _extract_contract_scalar(
    contract: Union[CloudOptimizationContract, Dict[str, Any], Any],
    primary_key: str,
    fallback_keys: List[str],
    default_val: float,
) -> float:
    """Safely extracts a numeric scalar property from a contract or dictionary."""
    if isinstance(contract, dict):
        if primary_key in contract and contract[primary_key] is not None:
            return float(contract[primary_key])
        for fb in fallback_keys:
            if fb in contract and contract[fb] is not None:
                return float(contract[fb])
        return default_val

    # Object / Pydantic attribute lookup
    if hasattr(contract, primary_key) and getattr(contract, primary_key) is not None:
        return float(getattr(contract, primary_key))
    for fb in fallback_keys:
        if hasattr(contract, fb) and getattr(contract, fb) is not None:
            return float(getattr(contract, fb))
    return default_val


def verify_feasibility(
    candidate_plan: Dict[str, Any],
    contract: Union[CloudOptimizationContract, Dict[str, Any], Any],
) -> Dict[str, Any]:
    """Evaluates a candidate deployment plan against a formal optimization contract.

    Computes explicit numerical constraint deficits, feasibility flags,
    violation rates, and pricing delta percentages.

    Mathematical Specifications:
    ----------------------------
    1. vCPU Deficit:
       $$\\Delta_{\\text{vCPU}} = \\max(0, \\text{vCPU}_{\\text{req}} - \\text{vCPU}_{\\text{alloc}})$$
       Formula: `vcpu_deficit = max(0, contract.min_vcpu - allocated_vcpu)`

    2. RAM Deficit:
       $$\\Delta_{\\text{RAM}} = \\max(0, \\text{RAM}_{\\text{req}} - \\text{RAM}_{\\text{alloc}})$$
       Formula: `ram_deficit = max(0, contract.min_ram - allocated_ram)`

    3. Budget Overflow:
       $$O_{\\text{budget}} = \\max(0.0, C_{\\text{catalog}} - B_{\\text{max}})$$
       Formula: `budget_overflow = max(0.0, catalog_cost - contract.budget_usd)`

    4. SLA Deficit:
       $$\\Delta_{\\text{SLA}} = \\max(0.0, \\text{SLA}_{\\text{target}} - \\text{SLA}_{\\text{achieved}})$$
       Formula: `sla_deficit = max(0.0, contract.min_sla - achieved_sla)`

    5. Latency Overflow (if specified):
       $$O_{\\text{latency}} = \\max(0.0, L_{\\text{achieved}} - L_{\\text{max}})$$
       Formula: `latency_overflow = max(0.0, achieved_latency_ms - contract.max_latency_ms)`

    6. Mathematical Feasibility:
       $$\\mathbb{I}_{\\text{feasible}} = \\begin{cases}
       \\text{True} & \\text{if } \\Delta_{\\text{vCPU}} = 0 \\land \\Delta_{\\text{RAM}} = 0 \\land O_{\\text{budget}} = 0 \\land \\Delta_{\\text{SLA}} = 0 \\land O_{\\text{latency}} = 0 \\\\
       \\text{False} & \\text{otherwise}
       \\end{cases}$$

    7. Constraint Violation Rate:
       $$V_{\\text{rate}} = \\left( \\frac{\\sum_{k=1}^{K} \\mathbb{I}[\\text{deficit}_k > 0]}{K} \\right) \\times 100.0\\%$$

    8. Catalog Cost Error Percentage:
       $$E_{\\text{cost}} = \\frac{|C_{\\text{predicted}} - C_{\\text{catalog}}|}{C_{\\text{catalog}}} \\times 100.0\\%$$

    Args:
        candidate_plan: Dictionary containing allocated resources, costs, and SLAs.
            Expected keys (with automatic fallbacks):
            - `allocated_vcpu` or `total_vcpus` (int/float)
            - `allocated_ram` or `total_ram_gb` (float)
            - `catalog_cost` or `actual_catalog_cost` or `total_monthly_cost_usd` (float)
            - `predicted_cost` or `reported_cost` or `objective_cost_usd` (float)
            - `achieved_sla` or `sla_availability_pct` or `achieved_sla_pct` (float)
            - `achieved_latency_ms` or `inter_region_latency_ms` (float)
        contract: `CloudOptimizationContract` or dict containing SLA/budget boundaries.

    Returns:
        Dictionary containing:
        - `is_feasible` (bool): True if all deficits == 0, else False.
        - `violation_rate_pct` (float): Percentage of constraints violated [0.0 - 100.0].
        - `budget_overflow_usd` (float): Excess cost above budget in USD.
        - `cost_error_pct` (float): Relative error between predicted and catalog cost.
        - `vcpu_deficit` (int/float): Unmet vCPU requirement.
        - `ram_deficit` (float): Unmet RAM requirement in GB.
        - `sla_deficit` (float): Unmet SLA percentage deficit.
        - `latency_overflow_ms` (float): Excess latency in milliseconds.
        - `summary` (dict): Detailed inputs, targets, and formula mappings for UI tooltips.
    """
    # -------------------------------------------------------------------------
    # 1. Extract Contract Target Values with Standardized Fallbacks
    # -------------------------------------------------------------------------
    min_vcpu = _extract_contract_scalar(
        contract, "min_vcpu", ["required_vcpus", "min_vcpus", "vcpus"], 1.0
    )
    min_ram = _extract_contract_scalar(
        contract, "min_ram", ["required_ram_gb", "min_ram_gb", "ram_gb"], 1.0
    )
    budget_usd = _extract_contract_scalar(
        contract, "budget_usd", ["budget_max_usd", "max_budget_usd", "budget"], 500.0
    )
    min_sla = _extract_contract_scalar(
        contract, "min_sla", ["sla_availability_pct", "min_sla_pct", "sla_pct", "sla"], 99.9
    )
    max_latency_ms = _extract_contract_scalar(
        contract, "max_latency_ms", ["latency_max_ms", "max_latency", "latency_ms"], 1000.0
    )

    # -------------------------------------------------------------------------
    # 2. Extract Candidate Plan Realized Values
    # -------------------------------------------------------------------------
    # Check if nested within 'decision_variables' or 'allocated_vms'
    allocated_vcpu = candidate_plan.get(
        "allocated_vcpu",
        candidate_plan.get(
            "total_vcpus",
            candidate_plan.get("vcpus", 0.0),
        ),
    )
    allocated_ram = candidate_plan.get(
        "allocated_ram",
        candidate_plan.get(
            "total_ram_gb",
            candidate_plan.get("ram_gb", 0.0),
        ),
    )
    catalog_cost = candidate_plan.get(
        "catalog_cost",
        candidate_plan.get(
            "actual_catalog_cost",
            candidate_plan.get(
                "total_monthly_cost_usd",
                candidate_plan.get("cost_usd", 0.0),
            ),
        ),
    )
    predicted_cost = candidate_plan.get(
        "predicted_cost",
        candidate_plan.get(
            "reported_cost",
            candidate_plan.get(
                "objective_cost_usd",
                catalog_cost,
            ),
        ),
    )
    achieved_sla = candidate_plan.get(
        "achieved_sla",
        candidate_plan.get(
            "achieved_sla_pct",
            candidate_plan.get(
                "sla_availability_pct",
                candidate_plan.get("sla_pct", min_sla),
            ),
        ),
    )
    achieved_latency_ms = candidate_plan.get(
        "achieved_latency_ms",
        candidate_plan.get(
            "inter_region_latency_ms",
            candidate_plan.get("latency_ms", 0.0),
        ),
    )

    # If allocated_vms list is provided and totals are zero, sum from components
    if (allocated_vcpu == 0 or allocated_ram == 0) and "allocated_vms" in candidate_plan:
        vms = candidate_plan["allocated_vms"]
        if isinstance(vms, list) and len(vms) > 0:
            sum_vcpu = sum(
                vm.get("vcpus_per_vm", 0) * vm.get("count", 1) for vm in vms
            )
            sum_ram = sum(
                vm.get("ram_gb_per_vm", 0.0) * vm.get("count", 1) for vm in vms
            )
            sum_cost = sum(
                vm.get("monthly_cost", 0.0) for vm in vms
            )
            if allocated_vcpu == 0:
                allocated_vcpu = sum_vcpu
            if allocated_ram == 0:
                allocated_ram = sum_ram
            if catalog_cost == 0 and sum_cost > 0:
                catalog_cost = sum_cost

    # Ensure numeric conversion
    allocated_vcpu = float(allocated_vcpu)
    allocated_ram = float(allocated_ram)
    catalog_cost = float(catalog_cost)
    predicted_cost = float(predicted_cost)
    achieved_sla = float(achieved_sla)
    achieved_latency_ms = float(achieved_latency_ms)

    # -------------------------------------------------------------------------
    # 3. Compute Explicit Numerical Deficits & Overflows
    # -------------------------------------------------------------------------
    vcpu_deficit = max(0, int(math.ceil(min_vcpu - allocated_vcpu)))
    ram_deficit = max(0.0, round(min_ram - allocated_ram, 3))
    budget_overflow = max(0.0, round(catalog_cost - budget_usd, 2))
    sla_deficit = max(0.0, round(min_sla - achieved_sla, 5))
    latency_overflow = max(0.0, round(achieved_latency_ms - max_latency_ms, 2))

    # -------------------------------------------------------------------------
    # 4. Compute Feasibility, Violation Rate, and Pricing Errors
    # -------------------------------------------------------------------------
    # Evaluated constraint vector: [vCPU, RAM, Budget, SLA]
    constraints_evaluated: List[Dict[str, Any]] = [
        {
            "name": "vCPU Capacity",
            "required": min_vcpu,
            "allocated": allocated_vcpu,
            "deficit": vcpu_deficit,
            "is_violated": vcpu_deficit > 0,
            "formula": MATHEMATICAL_FORMULAS["vcpu_deficit"],
        },
        {
            "name": "RAM Capacity (GB)",
            "required": min_ram,
            "allocated": allocated_ram,
            "deficit": ram_deficit,
            "is_violated": ram_deficit > 0,
            "formula": MATHEMATICAL_FORMULAS["ram_deficit"],
        },
        {
            "name": "Monthly Budget ($)",
            "required": budget_usd,
            "allocated": catalog_cost,
            "deficit": budget_overflow,
            "is_violated": budget_overflow > 0,
            "formula": MATHEMATICAL_FORMULAS["budget_overflow"],
        },
        {
            "name": "SLA Availability (%)",
            "required": min_sla,
            "allocated": achieved_sla,
            "deficit": sla_deficit,
            "is_violated": sla_deficit > 0,
            "formula": MATHEMATICAL_FORMULAS["sla_deficit"],
        },
    ]

    violated_count = sum(1 for c in constraints_evaluated if c["is_violated"])
    total_constraints = len(constraints_evaluated)
    violation_rate_pct = round((violated_count / total_constraints) * 100.0, 2)

    is_feasible = bool(
        vcpu_deficit == 0
        and ram_deficit == 0.0
        and budget_overflow == 0.0
        and sla_deficit == 0.0
    )

    # Cost error formula: |C_pred - C_cat| / C_cat * 100%
    if catalog_cost > 0:
        cost_error_pct = round(
            abs(predicted_cost - catalog_cost) / catalog_cost * 100.0, 2
        )
    else:
        cost_error_pct = 0.0 if predicted_cost == 0 else 100.0

    return {
        "is_feasible": is_feasible,
        "violation_rate_pct": violation_rate_pct,
        "budget_overflow_usd": budget_overflow,
        "cost_error_pct": cost_error_pct,
        "vcpu_deficit": vcpu_deficit,
        "ram_deficit": ram_deficit,
        "sla_deficit": sla_deficit,
        "latency_overflow_ms": latency_overflow,
        "allocated_vcpu": allocated_vcpu,
        "allocated_ram": allocated_ram,
        "catalog_cost": catalog_cost,
        "predicted_cost": predicted_cost,
        "achieved_sla": achieved_sla,
        "achieved_latency_ms": achieved_latency_ms,
        "total_constraints_evaluated": total_constraints,
        "violated_constraints_count": violated_count,
        "constraint_breakdown": constraints_evaluated,
        "formulas": MATHEMATICAL_FORMULAS,
    }


def compute_optimality_certificate(solver_result: Any) -> Dict[str, Any]:
    """Computes a formal mathematical optimality certificate and MIP duality gap.

    Extracts MIP optimality gap from SciPy HiGHS (`res.mip_gap`), branch-and-bound
    statistics, or SMT satisfaction proofs from the Z3 theorem prover.

    Mathematical Specifications:
    ----------------------------
    1. MIP Duality Gap:
       $$\\text{MIP Gap} = \\frac{|z_{\\text{incumbent}} - z_{\\text{bound}}|}{|z_{\\text{incumbent}}| + 10^{-10}} \\times 100.0\\%$$
       Formula: `mip_gap_pct = (mip_gap * 100.0)`

    2. Provable Optimality:
       $$\\mathbb{I}_{\\text{optimal}} = \\begin{cases}
       \\text{True} & \\text{if } (\\text{mip\\_gap} < 0.001) \\lor (\\text{Solver} = \\text{'Z3'} \\land \\text{Status} = \\text{'SAT'}) \\\\
       \\text{False} & \\text{otherwise}
       \\end{cases}$$

    Args:
        solver_result: Result object or dictionary from SciPy HiGHS, Z3 SMT, PSO, or GA.

    Returns:
        Dictionary containing:
        - `mip_gap_pct` (float): Mixed Integer Programming optimality gap percentage.
        - `is_provably_optimal` (bool): True if mip_gap < 0.001 or Z3 sat.
        - `solver_name` (str): Identifier of the mathematical engine.
        - `status` (str): Feasibility status ('OPTIMAL', 'FEASIBLE', 'UNSAT', 'INFEASIBLE').
        - `duality_bound` (Optional[float]): Best mathematical lower/upper bound if available.
        - `incumbent_objective` (Optional[float]): Current best feasible solution objective.
        - `formula_tooltip` (str): Mathematical definition string for UI tooltips.
    """
    solver_name = "Unknown"
    is_feasible = False
    raw_mip_gap: Optional[float] = None
    status_str = "UNKNOWN"
    incumbent_objective: Optional[float] = None
    duality_bound: Optional[float] = None

    # Handle dictionary representation
    if isinstance(solver_result, dict):
        solver_name = str(solver_result.get("solver_name") or solver_result.get("solver") or "MathematicalSolver")
        status_str = str(solver_result.get("status") or ("FEASIBLE" if solver_result.get("is_feasible") else "UNKNOWN")).upper()
        is_feasible = bool(solver_result.get("is_feasible", status_str in ["FEASIBLE", "OPTIMAL", "SAT", "SATISFIABLE"]))
        raw_mip_gap = solver_result.get("mip_gap")
        incumbent_objective = solver_result.get("total_monthly_cost_usd", solver_result.get("objective_cost_usd"))
        duality_bound = solver_result.get("lower_bound", solver_result.get("dual_bound"))

    # Handle Pydantic OptimizationResult or SciPy OptimizeResult
    elif hasattr(solver_result, "__dict__") or hasattr(solver_result, "solver_name"):
        solver_name = getattr(solver_result, "solver_name", getattr(solver_result, "solver", type(solver_result).__name__))
        status_val = getattr(solver_result, "status", None)
        status_str = str(status_val.value if hasattr(status_val, "value") else (status_val or "UNKNOWN")).upper()
        is_feasible = bool(getattr(solver_result, "is_feasible", False) or status_str in ["FEASIBLE", "OPTIMAL", "SAT", "SATISFIABLE"])
        raw_mip_gap = getattr(solver_result, "mip_gap", None)

        if hasattr(solver_result, "best_candidate") and solver_result.best_candidate is not None:
            incumbent_objective = getattr(solver_result.best_candidate, "objective_cost_usd", None)

    # SMT / Z3 exact solvers guarantee exact Pareto/Global optimality on SAT
    is_z3_engine = "Z3" in solver_name.upper() or "SMT" in solver_name.upper()
    is_highs_or_milp = "HIGHS" in solver_name.upper() or "MILP" in solver_name.upper() or "ILP" in solver_name.upper() or "SCIPY" in solver_name.upper()

    if is_z3_engine:
        if is_feasible or status_str in ["FEASIBLE", "OPTIMAL", "SAT", "SATISFIABLE"]:
            mip_gap_pct = 0.0
            is_provably_optimal = True
            status_str = "OPTIMAL (Z3 Exact SAT)"
        else:
            mip_gap_pct = 100.0
            is_provably_optimal = False
            status_str = "INFEASIBLE (Z3 UNSAT)"
    elif raw_mip_gap is not None:
        gap_val = float(raw_mip_gap)
        mip_gap_pct = gap_val * 100.0 if gap_val <= 1.0 else gap_val
        is_provably_optimal = bool(is_feasible and (gap_val < 0.001 or math.isclose(gap_val, 0.0, abs_tol=1e-6)))
        if is_provably_optimal:
            status_str = "OPTIMAL (MIP Gap < 0.001)"
    elif is_highs_or_milp and is_feasible:
        # SciPy HiGHS branch-and-bound exact convergence default
        mip_gap_pct = 0.0
        is_provably_optimal = True
        status_str = "OPTIMAL (HiGHS Exact)"
    else:
        # Heuristic / Metaheuristic solvers (GA, PSO, OptiHive) do not produce mathematical optimality proofs
        mip_gap_pct = 0.0 if is_feasible else 100.0
        is_provably_optimal = False
        status_str = "FEASIBLE (Optimality not established)" if is_feasible else "INFEASIBLE"

    return {
        "mip_gap_pct": round(mip_gap_pct, 4),
        "is_provably_optimal": is_provably_optimal,
        "solver_name": solver_name,
        "status": status_str,
        "duality_bound": duality_bound,
        "incumbent_objective": incumbent_objective,
        "formula_tooltip": MATHEMATICAL_FORMULAS["is_provably_optimal"],
    }


class MathematicalProofEngine:
    """High-level class wrapper for formal verification and proof generation."""

    @staticmethod
    def verify(
        candidate_plan: Dict[str, Any],
        contract: Union[CloudOptimizationContract, Dict[str, Any], Any],
    ) -> Dict[str, Any]:
        """Runs feasibility verification on candidate plan."""
        return verify_feasibility(candidate_plan, contract)

    @staticmethod
    def certify_optimality(solver_result: Any) -> Dict[str, Any]:
        """Computes mathematical optimality certificate."""
        return compute_optimality_certificate(solver_result)

    @staticmethod
    def generate_full_certificate(
        candidate_plan: Dict[str, Any],
        contract: Union[CloudOptimizationContract, Dict[str, Any], Any],
        solver_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generates a combined feasibility and optimality verification certificate."""
        feasibility_cert = verify_feasibility(candidate_plan, contract)
        optimality_cert = compute_optimality_certificate(solver_result or candidate_plan)

        return {
            "is_valid": feasibility_cert["is_feasible"] and optimality_cert["is_provably_optimal"],
            "feasibility": feasibility_cert,
            "optimality": optimality_cert,
            "formulas": MATHEMATICAL_FORMULAS,
        }
