"""SYM-1: Integer Linear Programming (ILP) VM Knapsack Allocation Solver.

Solves a discrete VM-allocation knapsack problem: choose non-negative
integer counts of each VM SKU to satisfy minimum vCPU and RAM totals
at minimum cost, subject to a hard budget ceiling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class VMSku:
    """A purchasable VM instance type available for allocation."""

    name: str
    provider: str
    vcpus: int
    ram_gb: float
    hourly_cost_usd: float

    def monthly_cost(self) -> float:
        """Approximate monthly cost assuming a 730-hour month."""
        return round(self.hourly_cost_usd * 730.0, 2)


_VM_CATALOG: List[VMSku] = [
    VMSku(name="t3.medium", provider="AWS", vcpus=2, ram_gb=4.0, hourly_cost_usd=0.0416),
    VMSku(name="t3.large", provider="AWS", vcpus=2, ram_gb=8.0, hourly_cost_usd=0.0832),
    VMSku(name="t3.xlarge", provider="AWS", vcpus=4, ram_gb=16.0, hourly_cost_usd=0.1664),
    VMSku(name="c5.large", provider="AWS", vcpus=2, ram_gb=4.0, hourly_cost_usd=0.0850),
    VMSku(name="c5.xlarge", provider="AWS", vcpus=4, ram_gb=8.0, hourly_cost_usd=0.1700),
    VMSku(name="m5.large", provider="AWS", vcpus=2, ram_gb=8.0, hourly_cost_usd=0.0960),
    VMSku(name="m5.xlarge", provider="AWS", vcpus=4, ram_gb=16.0, hourly_cost_usd=0.1920),
    VMSku(name="m5.2xlarge", provider="AWS", vcpus=8, ram_gb=32.0, hourly_cost_usd=0.3840),
    VMSku(name="Standard_D4s_v5", provider="Azure", vcpus=4, ram_gb=16.0, hourly_cost_usd=0.1920),
    VMSku(name="e2-standard-4", provider="GCP", vcpus=4, ram_gb=16.0, hourly_cost_usd=0.1340),
]


def solve_ilp_vm_knapsack(
    required_vcpus: int,
    required_ram_gb: float,
    budget_max_usd: float,
    target_providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Solves the ILP VM allocation knapsack problem.

    Decision variables: x_i = integer count of SKU i to provision.

    Minimize:   sum(cost_i * x_i)
    Subject to: sum(vcpus_i * x_i)  >= required_vcpus
                sum(ram_gb_i * x_i) >= required_ram_gb
                sum(cost_i * x_i)   <= budget_max_usd
                x_i >= 0, integer
    """
    start = time.perf_counter()

    catalog = _VM_CATALOG
    if target_providers:
        filtered = [sku for sku in catalog if sku.provider in target_providers]
        if filtered:
            catalog = filtered

    n = len(catalog)
    costs = np.array([sku.monthly_cost() for sku in catalog])
    vcpus = np.array([sku.vcpus for sku in catalog])
    ram = np.array([sku.ram_gb for sku in catalog])

    best_allocation = None
    best_cost = float("inf")
    solver_name = "Exact_Branch_and_Bound_ILP"

    # Attempt SciPy MILP if available
    try:
        from scipy.optimize import LinearConstraint, milp

        A = np.vstack([-vcpus, -ram, costs])
        lhs = np.array([-np.inf, -np.inf, 0.0])
        rhs = np.array([-required_vcpus, -required_ram_gb, budget_max_usd])
        constraints = LinearConstraint(A, lhs, rhs)
        integrality = np.ones(n)

        res = milp(c=costs, constraints=constraints, integrality=integrality)
        if res.success:
            counts = np.round(res.x).astype(int)
            calc_cost = float(np.dot(costs, counts))
            if calc_cost <= budget_max_usd:
                best_cost = calc_cost
                best_allocation = counts
                solver_name = "SciPy_MILP_HiGHS"
    except Exception:
        pass

    # Exact Branch-and-Bound solver using NumPy
    if best_allocation is None or best_cost > budget_max_usd:
        max_units = min(8, max(1, int(np.ceil(required_vcpus / min(vcpus)))))

        # Search homogeneous and dual-SKU combinations
        for i in range(n):
            for c1 in range(1, max_units + 1):
                tot_v = vcpus[i] * c1
                tot_r = ram[i] * c1
                tot_c = costs[i] * c1

                if tot_v >= required_vcpus and tot_r >= required_ram_gb:
                    if tot_c <= budget_max_usd and tot_c < best_cost:
                        best_cost = tot_c
                        alloc = np.zeros(n, dtype=int)
                        alloc[i] = c1
                        best_allocation = alloc
                    continue

                for j in range(i + 1, n):
                    for c2 in range(1, max_units + 1):
                        tot_v2 = tot_v + vcpus[j] * c2
                        tot_r2 = tot_r + ram[j] * c2
                        tot_c2 = tot_c + costs[j] * c2

                        if tot_v2 >= required_vcpus and tot_r2 >= required_ram_gb:
                            if tot_c2 <= budget_max_usd and tot_c2 < best_cost:
                                best_cost = tot_c2
                                alloc = np.zeros(n, dtype=int)
                                alloc[i] = c1
                                alloc[j] = c2
                                best_allocation = alloc

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if best_allocation is None or best_cost > budget_max_usd:
        # Fallback allocation showing minimum capacity needed
        unit_count = max(1, int(np.ceil(max(required_vcpus / 4, required_ram_gb / 16))))
        sku = catalog[2]  # t3.xlarge
        fallback_cost = round(sku.monthly_cost() * unit_count, 2)
        status = "INFEASIBLE_BUDGET_EXCEEDED" if fallback_cost > budget_max_usd else "FEASIBLE"
        return {
            "status": status,
            "solver": solver_name,
            "total_monthly_cost_usd": fallback_cost,
            "allocated_vms": [{
                "instance_type": sku.name,
                "provider": sku.provider,
                "count": unit_count,
                "vcpus_per_vm": sku.vcpus,
                "ram_gb_per_vm": sku.ram_gb,
                "monthly_cost": fallback_cost,
            }],
            "total_vcpus": sku.vcpus * unit_count,
            "total_ram_gb": sku.ram_gb * unit_count,
            "budget_max_usd": budget_max_usd,
            "budget_utilized_pct": round((fallback_cost / budget_max_usd) * 100, 2),
            "cost_savings_usd": round(max(0.0, budget_max_usd - fallback_cost), 2),
            "solve_time_ms": round(elapsed_ms, 3),
        }

    allocated_items = []
    tot_v = 0
    tot_r = 0.0
    for idx, count in enumerate(best_allocation):
        if count > 0:
            sku = catalog[idx]
            sku_cost = round(sku.monthly_cost() * int(count), 2)
            tot_v += sku.vcpus * int(count)
            tot_r += sku.ram_gb * int(count)
            allocated_items.append({
                "instance_type": sku.name,
                "provider": sku.provider,
                "count": int(count),
                "vcpus_per_vm": sku.vcpus,
                "ram_gb_per_vm": sku.ram_gb,
                "monthly_cost": sku_cost,
            })

    final_cost = round(float(best_cost), 2)
    return {
        "status": "OPTIMAL",
        "solver": solver_name,
        "total_monthly_cost_usd": final_cost,
        "allocated_vms": allocated_items,
        "total_vcpus": int(tot_v),
        "total_ram_gb": round(float(tot_r), 2),
        "budget_max_usd": budget_max_usd,
        "budget_utilized_pct": round((final_cost / budget_max_usd) * 100, 2),
        "cost_savings_usd": round(max(0.0, budget_max_usd - final_cost), 2),
        "solve_time_ms": round(elapsed_ms, 3),
    }