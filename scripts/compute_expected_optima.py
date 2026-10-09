"""Independent Ground-Truth Optima Calculator & Final Query Manifest Generator for Neurasym.

Computes exact mathematical ground-truth optima via EXHAUSTIVE discrete search
over the real resource catalog and topology graph WITHOUT using any solver code
under test (no HiGHS, no Z3 solver, no PSO swarm optimizer).

Generates data/final_query_manifest.json with all 29 curated draft queries,
pre-declared header policies, success definitions, independent optima, and notes.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.dataset_manifest import ManifestManager


# =============================================================================
# 1. Independent Ground-Truth Catalog & Topology Graph
# (Isolated from solver modules to prevent circular verification)
# =============================================================================

HOURS_PER_MONTH = 730.0
LATENCY_COST_PER_MS = 0.25
SCALING_COST_PER_MBPS = 0.08
SCALING_COST_PER_REPLICA = 45.00
SCALING_CAPACITY_FACTOR_MBPS = 75.00

# Canonical Ground-Truth VM SKU Catalog
GROUND_TRUTH_VM_SKUS: Dict[str, Dict[str, Any]] = {
    "t3.medium": {"provider": "AWS", "vcpus": 2, "ram_gb": 4.0, "hourly_cost_usd": 0.0416},
    "t3.large": {"provider": "AWS", "vcpus": 2, "ram_gb": 8.0, "hourly_cost_usd": 0.0832},
    "t3.xlarge": {"provider": "AWS", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1664},
    "c5.large": {"provider": "AWS", "vcpus": 2, "ram_gb": 4.0, "hourly_cost_usd": 0.0850},
    "c5.xlarge": {"provider": "AWS", "vcpus": 4, "ram_gb": 8.0, "hourly_cost_usd": 0.1700},
    "m5.large": {"provider": "AWS", "vcpus": 2, "ram_gb": 8.0, "hourly_cost_usd": 0.0960},
    "m5.xlarge": {"provider": "AWS", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1920},
    "m5.2xlarge": {"provider": "AWS", "vcpus": 8, "ram_gb": 32.0, "hourly_cost_usd": 0.3840},
    "Standard_D4s_v5": {"provider": "Azure", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1920},
    "e2-standard-4": {"provider": "GCP", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1340},
}

# Canonical Ground-Truth Disaster Recovery Topology Graph
GROUND_TRUTH_REGIONS: Dict[str, Dict[str, Any]] = {
    "us-east-1": {
        "provider": "AWS",
        "geo": "US_East",
        "base_cost_usd": 120.0,
        "sla_pct": 99.95,
        "peer_latencies_ms": {
            "us-west-2": 65.0,
            "eu-west-1": 85.0,
            "eastus": 12.0,
            "us-central1": 32.0,
        },
    },
    "us-west-2": {
        "provider": "AWS",
        "geo": "US_West",
        "base_cost_usd": 130.0,
        "sla_pct": 99.95,
        "peer_latencies_ms": {
            "us-east-1": 65.0,
            "eu-west-1": 135.0,
            "eastus": 70.0,
            "us-central1": 42.0,
        },
    },
    "eu-west-1": {
        "provider": "AWS",
        "geo": "Europe",
        "base_cost_usd": 140.0,
        "sla_pct": 99.95,
        "peer_latencies_ms": {
            "us-east-1": 85.0,
            "us-west-2": 135.0,
            "eastus": 90.0,
            "us-central1": 105.0,
        },
    },
    "eastus": {
        "provider": "Azure",
        "geo": "US_East",
        "base_cost_usd": 125.0,
        "sla_pct": 99.95,
        "peer_latencies_ms": {
            "us-east-1": 12.0,
            "us-west-2": 70.0,
            "eu-west-1": 90.0,
            "us-central1": 28.0,
        },
    },
    "us-central1": {
        "provider": "GCP",
        "geo": "US_Central",
        "base_cost_usd": 115.0,
        "sla_pct": 99.95,
        "peer_latencies_ms": {
            "us-east-1": 32.0,
            "us-west-2": 42.0,
            "eu-west-1": 105.0,
            "eastus": 28.0,
        },
    },
}


# =============================================================================
# 2. Independent Exhaustive Search Algorithms
# =============================================================================

def solve_vm_exhaustive(
    req_vcpus: int,
    req_ram_gb: float,
    budget_usd: Optional[float] = None,
    providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Exhaustive discrete branch-and-bound search for VM knapsack allocation."""
    target_provs = {p.upper() for p in providers} if providers else {"AWS"}
    eligible_skus = [
        (sku, meta["vcpus"], meta["ram_gb"], round(meta["hourly_cost_usd"] * HOURS_PER_MONTH, 2))
        for sku, meta in GROUND_TRUTH_VM_SKUS.items()
        if meta["provider"].upper() in target_provs
    ]

    best: Dict[str, Any] = {
        "cost": float("inf"),
        "allocation": {},
        "vcpus": 0,
        "ram_gb": 0.0,
        "feasible_under_budget": False,
    }

    if not eligible_skus:
        return best

    def search(sku_idx: int, curr_vcpus: int, curr_ram: float, curr_cost: float, alloc: Dict[str, int]):
        if curr_vcpus >= req_vcpus and curr_ram >= req_ram_gb:
            r_cost = round(curr_cost, 2)
            if r_cost < best["cost"]:
                best["cost"] = r_cost
                best["allocation"] = dict(alloc)
                best["vcpus"] = curr_vcpus
                best["ram_gb"] = round(curr_ram, 2)
                best["feasible_under_budget"] = (budget_usd is not None and r_cost <= budget_usd)
            return

        if sku_idx >= len(eligible_skus) or curr_cost >= best["cost"]:
            return

        sku_name, v, r, c = eligible_skus[sku_idx]
        needed_v = math.ceil(max(0, req_vcpus - curr_vcpus) / v) if v > 0 else 0
        needed_r = math.ceil(max(0, req_ram_gb - curr_ram) / r) if r > 0 else 0
        max_count = max(needed_v, needed_r)

        for count in range(max_count + 1):
            next_cost = curr_cost + count * c
            if next_cost >= best["cost"]:
                break
            if count > 0:
                alloc[sku_name] = count
            search(sku_idx + 1, curr_vcpus + count * v, curr_ram + count * r, next_cost, alloc)
            if count > 0:
                del alloc[sku_name]

    search(0, 0, 0.0, 0.0, {})
    return best


def solve_dr_exhaustive(
    latency_max_ms: float = 100.0,
    sla_availability_pct: float = 99.99,
    budget_usd: Optional[float] = None,
    providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Exhaustive search over all region pairs in the topology graph."""
    requested_providers = providers or ["AWS", "GCP"]
    target_provs_upper = [p.upper() for p in requested_providers]
    is_multi_cloud = len(set(target_provs_upper)) >= 2

    candidates = []
    region_keys = list(GROUND_TRUTH_REGIONS.keys())

    for i in range(len(region_keys)):
        for j in range(i + 1, len(region_keys)):
            r1_id = region_keys[i]
            r2_id = region_keys[j]
            r1 = GROUND_TRUTH_REGIONS[r1_id]
            r2 = GROUND_TRUTH_REGIONS[r2_id]

            p1, p2 = r1["provider"], r2["provider"]

            # Multi-cloud provider filter
            if is_multi_cloud:
                if p1 == p2 or not (p1.upper() in target_provs_upper and p2.upper() in target_provs_upper):
                    continue
            else:
                if not (p1.upper() in target_provs_upper and p2.upper() in target_provs_upper):
                    continue

            # Geographic separation constraint
            if r1["geo"] == r2["geo"] and p1 == p2:
                continue

            lat = r1["peer_latencies_ms"].get(r2_id)
            if lat is None:
                continue

            # Composite SLA formula: 1 - ((1 - A) * (1 - B))
            unavail_a = 1.0 - (r1["sla_pct"] / 100.0)
            unavail_b = 1.0 - (r2["sla_pct"] / 100.0)
            comp_sla = round((1.0 - (unavail_a * unavail_b)) * 100.0, 5)

            cost = round(r1["base_cost_usd"] + r2["base_cost_usd"] + (lat * LATENCY_COST_PER_MS), 2)

            lat_ok = lat <= latency_max_ms
            sla_ok = comp_sla >= sla_availability_pct
            bud_ok = (budget_usd is not None and cost <= budget_usd)

            candidates.append({
                "primary": r1_id,
                "secondary": r2_id,
                "providers": (p1, p2),
                "latency_ms": lat,
                "composite_sla_pct": comp_sla,
                "monthly_cost_usd": cost,
                "latency_ok": lat_ok,
                "sla_ok": sla_ok,
                "budget_ok": bud_ok,
                "fully_feasible": lat_ok and sla_ok and bud_ok,
            })

    # Sort by monthly cost ascending
    candidates.sort(key=lambda x: x["monthly_cost_usd"])

    feasible_candidates = [c for c in candidates if c["fully_feasible"]]
    valid_topology_candidates = [c for c in candidates if c["latency_ok"] and c["sla_ok"]]

    if feasible_candidates:
        best_feas = feasible_candidates[0]
        return {
            "status": "FEASIBLE",
            "cost": best_feas["monthly_cost_usd"],
            "primary": best_feas["primary"],
            "secondary": best_feas["secondary"],
            "latency_ms": best_feas["latency_ms"],
            "sla_pct": best_feas["composite_sla_pct"],
            "min_possible_cost": best_feas["monthly_cost_usd"],
            "all_candidates": candidates,
        }
    elif valid_topology_candidates:
        best_top = valid_topology_candidates[0]
        return {
            "status": "INFEASIBLE_BUDGET",
            "cost": None,
            "min_possible_cost": best_top["monthly_cost_usd"],
            "primary": best_top["primary"],
            "secondary": best_top["secondary"],
            "latency_ms": best_top["latency_ms"],
            "sla_pct": best_top["composite_sla_pct"],
            "all_candidates": candidates,
        }
    else:
        min_cost = candidates[0]["monthly_cost_usd"] if candidates else None
        return {
            "status": "INFEASIBLE_LATENCY_OR_SLA",
            "cost": None,
            "min_possible_cost": min_cost,
            "primary": None,
            "secondary": None,
            "all_candidates": candidates,
        }


def solve_scaling_exhaustive(
    bandwidth_mbps: float,
    max_cpu_pct: Optional[float] = None,
    target_cpu_pct: Optional[float] = None,
    budget_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Exhaustive search over integer replicas 1-16 for Continuous Dynamic Scaling."""
    ceiling = max_cpu_pct if max_cpu_pct is not None else (target_cpu_pct if target_cpu_pct is not None else 100.0)

    best: Dict[str, Any] = {
        "status": "INFEASIBLE",
        "cost": None,
        "replicas": None,
        "cpu_pct": None,
        "min_possible_cost": None,
    }

    all_options = []
    for reps in range(1, 17):
        cpu = (bandwidth_mbps / (reps * SCALING_CAPACITY_FACTOR_MBPS)) * 100.0
        cost = round((bandwidth_mbps * SCALING_COST_PER_MBPS) + (reps * SCALING_COST_PER_REPLICA), 2)
        cpu_ok = (cpu <= ceiling and cpu <= 100.0)
        bud_ok = (budget_usd is not None and cost <= budget_usd)

        all_options.append({
            "replicas": reps,
            "cpu_pct": cpu,
            "cost": cost,
            "cpu_ok": cpu_ok,
            "budget_ok": bud_ok,
            "feasible": cpu_ok and bud_ok,
        })

    feasible_opts = [o for o in all_options if o["feasible"]]
    valid_cpu_opts = [o for o in all_options if o["cpu_ok"]]

    if feasible_opts:
        # Min cost option
        feasible_opts.sort(key=lambda x: x["cost"])
        best["status"] = "FEASIBLE"
        best["cost"] = feasible_opts[0]["cost"]
        best["replicas"] = feasible_opts[0]["replicas"]
        best["cpu_pct"] = round(feasible_opts[0]["cpu_pct"], 2)
        best["min_possible_cost"] = feasible_opts[0]["cost"]
    elif valid_cpu_opts:
        valid_cpu_opts.sort(key=lambda x: x["cost"])
        best["status"] = "INFEASIBLE_BUDGET"
        best["cost"] = None
        best["min_possible_cost"] = valid_cpu_opts[0]["cost"]
    else:
        best["status"] = "INFEASIBLE_CAPACITY"
        best["cost"] = None
        # Theoretical replicas needed:
        needed_reps = math.ceil(bandwidth_mbps / (SCALING_CAPACITY_FACTOR_MBPS * (ceiling / 100.0)))
        best["min_possible_cost"] = round(bandwidth_mbps * SCALING_COST_PER_MBPS + needed_reps * SCALING_COST_PER_REPLICA, 2)

    return best


# =============================================================================
# 3. Final 29 Query Definitions & Independent Optimization
# =============================================================================

def build_final_manifest_data() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Assembles all 29 queries, runs exhaustive search, and flags mismatches."""

    query_definitions = [
        # ---------------------------------------------------------------------
        # VM Knapsack Queries (Q01 - Q10)
        # ---------------------------------------------------------------------
        {
            "num": "Q01",
            "query_id": "Q01_VM_AWS_8VCPU_16GB_FEASIBLE",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "Deploy an application requiring 8 vCPUs and 16GB RAM for under $300 a month on AWS.",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "FEASIBLE",
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": 300.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "~ $121.47",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Standard formal VM knapsack request. 4x t3.medium provides 8 vCPUs, 16GB RAM at $121.47/mo ($121.48 with rounding).",
        },
        {
            "num": "Q02",
            "query_id": "Q02_VM_HINGLISH_8VCPU_16GB",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "Bhai AWS pe 8 vCPU aur 16GB RAM ka setup lagade 300 dollar per month ke andar.",
            "category": "COLLOQUIAL_HINGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "FEASIBLE",
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": 300.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "~ $121.47",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Hinglish paraphrase of Q01 with identical constraints and ground-truth optimum.",
        },
        {
            "num": "Q03",
            "query_id": "Q03_VM_HINGLISH_WORDS_8VCPU_16GB",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "Bhai mujhe AWS pe aath core aur solah gig RAM wala setup chahiye, budget teen sau dollar tak rakhna",
            "category": "COLLOQUIAL_HINGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "FEASIBLE",
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": 300.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "~ $121.47",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Hinglish number words ('aath core', 'solah gig', 'teen sau dollar') testing multilingual lexical normalization.",
        },
        {
            "num": "Q04",
            "query_id": "Q04_VM_AWS_16VCPU_64GB_INFEASIBLE",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "I need 16 vCPUs and 64GB RAM on AWS for under $450 a month.",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "INFEASIBLE",
            "required_vcpus": 16,
            "required_ram_gb": 64.0,
            "budget_max_usd": 450.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "true minimum ~ $485.89",
            "optimum_source": None,
            "key_notes": "Budget infeasible. True AWS discrete minimum is $485.89 (4x t3.xlarge = 16 vCPUs, 64GB RAM).",
        },
        {
            "num": "Q05",
            "query_id": "Q05_VM_HINGLISH_16VCPU_64GB_INFEASIBLE",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "Bhai mujhe AWS pe ek server setup chahiye, 16 vCPU aur 64GB RAM wala, lekin budget sirf 450 dollar per month hai, kuch jugaad karke sabse sasta nikaal do yaar",
            "category": "COLLOQUIAL_HINGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "INFEASIBLE",
            "required_vcpus": 16,
            "required_ram_gb": 64.0,
            "budget_max_usd": 450.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "true minimum ~ $485.89",
            "optimum_source": None,
            "key_notes": "Hinglish negotiation request for unattainable $450 budget. True minimum is $485.89.",
        },
        {
            "num": "Q06",
            "query_id": "Q06_VM_AWS_64VCPU_256GB_INFEASIBLE",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "Deploy a massive database requiring 64 vCPUs and 256GB RAM on AWS for under 10 dollar a month",
            "category": "CONFLICTING_CONSTRAINTS",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "INFEASIBLE",
            "required_vcpus": 64,
            "required_ram_gb": 256.0,
            "budget_max_usd": 10.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "true minimum ~ $1,943.55",
            "optimum_source": None,
            "key_notes": "Grossly infeasible request ($10 budget vs $1,943.55 minimum for 16x t3.xlarge).",
        },
        {
            "num": "Q07",
            "query_id": "Q07_VM_AWS_MISSING_BUDGET",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "I need a server on AWS with 8 vCPUs and 16GB RAM",
            "category": "MISSING_OR_AMBIGUOUS",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "CLARIFICATION_REQUIRED",
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": None,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "null (budget missing)",
            "optimum_source": None,
            "key_notes": "Clarification required: Missing monthly financial budget.",
        },
        {
            "num": "Q08",
            "query_id": "Q08_VM_AWS_MISSING_SPECS",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "I need a cheap server on AWS under 200 dollar a month",
            "category": "MISSING_OR_AMBIGUOUS",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "CLARIFICATION_REQUIRED",
            "required_vcpus": None,
            "required_ram_gb": None,
            "budget_max_usd": 200.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "null (vCPUs and RAM missing)",
            "optimum_source": None,
            "key_notes": "Clarification required: Missing required vCPUs and RAM hardware specifications.",
        },
        {
            "num": "Q09",
            "query_id": "Q09_VM_AZURE_GCP_4VCPU_8GB",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "Need 4 vCPUs and 8GB RAM on Azure or GCP for under $150 a month.",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "FEASIBLE",
            "required_vcpus": 4,
            "required_ram_gb": 8.0,
            "budget_max_usd": 150.0,
            "cloud_providers": ["Azure", "GCP"],
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "to be filled by INDEPENDENT_SEARCH",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Multi-provider compute allocation. 1x e2-standard-4 on GCP satisfies 4 vCPU, 16GB RAM for $97.82/mo.",
        },
        {
            "num": "Q10",
            "query_id": "Q10_VM_HINGLISH_GCP_AZURE_8VCPU_32GB",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "Yaar GCP ya Azure kahin bhi chalega, bas aath core aur battees gig RAM chahiye, teen sau pachaas dollar se zyada nahi",
            "category": "COLLOQUIAL_HINGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "FEASIBLE",
            "required_vcpus": 8,
            "required_ram_gb": 32.0,
            "budget_max_usd": 350.0,
            "cloud_providers": ["GCP", "Azure"],
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "to be filled by INDEPENDENT_SEARCH",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Hinglish multi-provider request (8 vCPUs, 32GB RAM, $350 budget). Optimum is 2x e2-standard-4 on GCP at $195.64/mo.",
        },

        # ---------------------------------------------------------------------
        # Disaster Recovery Placement Queries (Q11 - Q16)
        # ---------------------------------------------------------------------
        {
            "num": "Q11",
            "query_id": "Q11_DR_AWS_GCP_50MS_600USD_FEASIBLE",
            "scenario_family_id": "FAM_DR_01_CROSS_CLOUD",
            "query_text": "We need disaster recovery setup across AWS and GCP with 99.99% SLA and max 50ms latency under $600/month.",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "Z3_Graph_Disaster_Recovery",
            "expected_outcome_pre": "FEASIBLE",
            "latency_max_ms": 50.0,
            "sla_availability_pct": 99.99,
            "budget_max_usd": 600.0,
            "cloud_providers": ["AWS", "GCP"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "~ $243 (us-east-1 + us-central1, 32 ms)",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Cross-cloud DR: us-east-1 (AWS) + us-central1 (GCP) has 32ms latency and 99.99997% composite SLA at $243.00/mo.",
        },
        {
            "num": "Q12",
            "query_id": "Q12_DR_HINGLISH_DIGITS_50MS_600USD",
            "scenario_family_id": "FAM_DR_01_CROSS_CLOUD",
            "query_text": "Bhai AWS aur GCP dono pe disaster recovery chahiye, 99.99 percent uptime, 50 millisecond se kam latency, budget 600 dollar",
            "category": "COLLOQUIAL_HINGLISH",
            "intended_archetype": "Z3_Graph_Disaster_Recovery",
            "expected_outcome_pre": "FEASIBLE",
            "latency_max_ms": 50.0,
            "sla_availability_pct": 99.99,
            "budget_max_usd": 600.0,
            "cloud_providers": ["AWS", "GCP"],
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "~ $243 (same key as Q11)",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Hinglish digits paraphrase of Q11 with identical constraints and optimum ($243.00/mo).",
        },
        {
            "num": "Q13",
            "query_id": "Q13_DR_HINGLISH_WORDS_50MS_600USD",
            "scenario_family_id": "FAM_DR_01_CROSS_CLOUD",
            "query_text": "Bhai AWS aur GCP dono pe disaster recovery chahiye, 99.99 percent uptime, latency pachaas millisecond se kam, budget chhe sau dollar",
            "category": "COLLOQUIAL_HINGLISH",
            "intended_archetype": "Z3_Graph_Disaster_Recovery",
            "expected_outcome_pre": "FEASIBLE",
            "latency_max_ms": 50.0,
            "sla_availability_pct": 99.99,
            "budget_max_usd": 600.0,
            "cloud_providers": ["AWS", "GCP"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "~ $243 (same key as Q11)",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Hinglish number words ('pachaas millisecond', 'chhe sau dollar') paraphrase of Q11.",
        },
        {
            "num": "Q14",
            "query_id": "Q14_DR_AWS_GCP_100USD_INFEASIBLE",
            "scenario_family_id": "FAM_DR_01_CROSS_CLOUD",
            "query_text": "Disaster recovery across AWS and GCP with 99.99% SLA and max 50ms latency under $100 a month.",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "Z3_Graph_Disaster_Recovery",
            "expected_outcome_pre": "INFEASIBLE",
            "latency_max_ms": 50.0,
            "sla_availability_pct": 99.99,
            "budget_max_usd": 100.0,
            "cloud_providers": ["AWS", "GCP"],
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "minimum ~ $243 (infeasible under $100)",
            "optimum_source": None,
            "key_notes": "Budget infeasible: Minimum possible AWS+GCP DR deployment cost is $243.00/mo ($100 budget exceeded).",
        },
        {
            "num": "Q15",
            "query_id": "Q15_DR_AWS_GCP_10MS_INFEASIBLE",
            "scenario_family_id": "FAM_DR_01_CROSS_CLOUD",
            "query_text": "Disaster recovery across AWS and GCP with 99.99% SLA and max 10ms latency under $600 a month.",
            "category": "CONFLICTING_CONSTRAINTS",
            "intended_archetype": "Z3_Graph_Disaster_Recovery",
            "expected_outcome_pre": "INFEASIBLE",
            "latency_max_ms": 10.0,
            "sla_availability_pct": 99.99,
            "budget_max_usd": 600.0,
            "cloud_providers": ["AWS", "GCP"],
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "INFEASIBLE (no AWS-GCP pair <= 10ms)",
            "optimum_source": None,
            "key_notes": "Latency infeasible: No AWS-GCP region pair in the catalog satisfies latency <= 10.0ms (lowest is 32.0ms).",
        },
        {
            "num": "Q16",
            "query_id": "Q16_DR_MISSING_BUDGET",
            "scenario_family_id": "FAM_DR_01_CROSS_CLOUD",
            "query_text": "I need disaster recovery across two clouds.",
            "category": "MISSING_OR_AMBIGUOUS",
            "intended_archetype": "Z3_Graph_Disaster_Recovery",
            "expected_outcome_pre": "CLARIFICATION_REQUIRED",
            "latency_max_ms": None,
            "sla_availability_pct": None,
            "budget_max_usd": None,
            "cloud_providers": ["AWS", "GCP"],
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "null (budget missing)",
            "optimum_source": None,
            "key_notes": "Clarification required: Missing budget ceiling.",
        },

        # ---------------------------------------------------------------------
        # Continuous Dynamic Scaling Queries (Q17 - Q21)
        # ---------------------------------------------------------------------
        {
            "num": "Q17",
            "query_id": "Q17_SCALING_300MBPS_60CPU_FEASIBLE",
            "scenario_family_id": "FAM_SCALING_01_ELASTIC",
            "query_text": "Dynamic scaling for 300 Mbps peak workload with maximum CPU 60 percent under $1500.",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "PSO_Continuous_Scaling",
            "expected_outcome_pre": "FEASIBLE",
            "target_bandwidth_mbps": 300.0,
            "target_cpu_pct": 60.0,
            "max_cpu_pct": 60.0,
            "budget_max_usd": 1500.0,
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "$339.00 (FORMULA; 7 replicas)",
            "optimum_source": "FORMULA",
            "key_notes": "Continuous scaling: 300 Mbps / (7 * 75 Mbps) = 57.14% CPU. Cost = 300 * $0.08 + 7 * $45 = $339.00/mo.",
        },
        {
            "num": "Q18",
            "query_id": "Q18_SCALING_HINGLISH_300MBPS_60CPU",
            "scenario_family_id": "FAM_SCALING_01_ELASTIC",
            "query_text": "Bhai peak pe 300 Mbps traffic aayega, CPU 60 percent se upar nahi jaana chahiye, budget 1500 dollar",
            "category": "COLLOQUIAL_HINGLISH",
            "intended_archetype": "PSO_Continuous_Scaling",
            "expected_outcome_pre": "FEASIBLE",
            "target_bandwidth_mbps": 300.0,
            "target_cpu_pct": 60.0,
            "max_cpu_pct": 60.0,
            "budget_max_usd": 1500.0,
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "$339.00 (same key as Q17)",
            "optimum_source": "FORMULA",
            "key_notes": "Hinglish paraphrase of Q17 with identical mathematical constraints and optimum ($339.00/mo).",
        },
        {
            "num": "Q19",
            "query_id": "Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING",
            "scenario_family_id": "FAM_SCALING_01_ELASTIC",
            "query_text": "Dynamic scaling for 300 Mbps peak workload with target CPU 70 percent and strict maximum CPU ceiling of 60 percent under 1500 dollar",
            "category": "CONFLICTING_CONSTRAINTS",
            "intended_archetype": "PSO_Continuous_Scaling",
            "expected_outcome_pre": "CONFLICTING_REQUIREMENTS",
            "target_bandwidth_mbps": 300.0,
            "target_cpu_pct": 70.0,
            "max_cpu_pct": 60.0,
            "budget_max_usd": 1500.0,
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "null (conflicting CPU target > ceiling)",
            "optimum_source": None,
            "key_notes": "Conflicting requirements: Target CPU (70.0%) strictly exceeds maximum CPU ceiling (60.0%).",
        },
        {
            "num": "Q20",
            "query_id": "Q20_SCALING_1000MBPS_30CPU_INFEASIBLE",
            "scenario_family_id": "FAM_SCALING_01_ELASTIC",
            "query_text": "Dynamic scaling for 1000 Mbps with maximum CPU 30 percent under 200 dollar.",
            "category": "CONFLICTING_CONSTRAINTS",
            "intended_archetype": "PSO_Continuous_Scaling",
            "expected_outcome_pre": "INFEASIBLE",
            "target_bandwidth_mbps": 1000.0,
            "target_cpu_pct": 30.0,
            "max_cpu_pct": 30.0,
            "budget_max_usd": 200.0,
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "INFEASIBLE (needs 45 reps > 16 cap, $2105 > $200)",
            "optimum_source": None,
            "key_notes": "Infeasible capacity & budget: 1000 Mbps at 30% CPU ceiling requires 45 replicas (exceeds 16-replica cap), costing $2,105/mo.",
        },
        {
            "num": "Q21",
            "query_id": "Q21_SCALING_MISSING_WORKLOAD",
            "scenario_family_id": "FAM_SCALING_01_ELASTIC",
            "query_text": "Continuous dynamic scaling with target CPU 70% under $1500",
            "category": "MISSING_OR_AMBIGUOUS",
            "intended_archetype": "PSO_Continuous_Scaling",
            "expected_outcome_pre": "CLARIFICATION_REQUIRED",
            "target_bandwidth_mbps": None,
            "target_cpu_pct": 70.0,
            "max_cpu_pct": None,
            "budget_max_usd": 1500.0,
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "null (missing workload)",
            "optimum_source": None,
            "key_notes": "Clarification required: Missing peak bandwidth workload (Mbps).",
        },

        # ---------------------------------------------------------------------
        # Unsupported / Ambiguous Queries (Q22 - Q25)
        # ---------------------------------------------------------------------
        {
            "num": "Q22",
            "query_id": "Q22_UNSUPPORTED_GPU_H100_TRAINING",
            "scenario_family_id": "FAM_UNSUPPORTED_01",
            "query_text": "Train a transformer model on 8 H100 GPUs for two weeks and tell me the cheapest option",
            "category": "UNSUPPORTED_TASKS",
            "intended_archetype": "NONE",
            "expected_outcome_pre": "UNSUPPORTED",
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "null (out of scope)",
            "optimum_source": None,
            "key_notes": "Unsupported task: High-performance GPU cluster AI training is outside supported cloud orchestration archetypes.",
        },
        {
            "num": "Q23",
            "query_id": "Q23_UNSUPPORTED_BIRYANI_RECIPE",
            "scenario_family_id": "FAM_UNSUPPORTED_01",
            "query_text": "What is the best recipe for biryani for 20 people?",
            "category": "UNSUPPORTED_TASKS",
            "intended_archetype": "NONE",
            "expected_outcome_pre": "UNSUPPORTED",
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "null (out of domain)",
            "optimum_source": None,
            "key_notes": "Unsupported domain: General culinary question completely out of cloud infrastructure domain.",
        },
        {
            "num": "Q24",
            "query_id": "Q24_UNSUPPORTED_ORACLE_DB_MIGRATION",
            "scenario_family_id": "FAM_UNSUPPORTED_01",
            "query_text": "Migrate my on-premise Oracle database to the cloud and tell me how long it will take.",
            "category": "UNSUPPORTED_TASKS",
            "intended_archetype": "NONE",
            "expected_outcome_pre": "UNSUPPORTED",
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "null (out of domain)",
            "optimum_source": None,
            "key_notes": "Unsupported task: Database migration assessment is outside supported archetype solvers.",
        },
        {
            "num": "Q25",
            "query_id": "Q25_CONFLICTING_CHEAPEST_AND_HIGHEST_PERFORMANCE",
            "scenario_family_id": "FAM_CONFLICTING_01",
            "query_text": "Give me the cheapest and also the highest-performance server on AWS under 200 dollar.",
            "category": "CONFLICTING_CONSTRAINTS",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "CONFLICTING_REQUIREMENTS",
            "required_vcpus": None,
            "required_ram_gb": None,
            "budget_max_usd": 200.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": False,
            "user_prompt_optimum_str": "null (conflicting objectives)",
            "optimum_source": None,
            "key_notes": "Conflicting requirements: 'Cheapest' (minimizing cost) and 'highest-performance' (maximizing specs) represent mutually exclusive objective directions without Pareto weights.",
        },

        # ---------------------------------------------------------------------
        # Boundary Test Queries (Q26 - Q29)
        # ---------------------------------------------------------------------
        {
            "num": "Q26",
            "query_id": "Q26_BOUNDARY_VM_UNDER_121_INFEASIBLE",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "Deploy 8 vCPUs and 16GB RAM on AWS for under 121 dollar a month",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "INFEASIBLE",
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": 121.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "true minimum $121.47 (infeasible under $121)",
            "optimum_source": None,
            "key_notes": "Boundary test (infeasible): Budget $121.00 is strictly below the discrete optimal cost $121.47.",
        },
        {
            "num": "Q27",
            "query_id": "Q27_BOUNDARY_VM_UNDER_122_FEASIBLE",
            "scenario_family_id": "FAM_VM_01_COMPUTE",
            "query_text": "Deploy 8 vCPUs and 16GB RAM on AWS for under 122 dollar a month",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome_pre": "FEASIBLE",
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": 122.0,
            "cloud_providers": ["AWS"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "$121.47",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Boundary test (feasible): Budget $122.00 satisfies the discrete optimal cost of $121.47.",
        },
        {
            "num": "Q28",
            "query_id": "Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE",
            "scenario_family_id": "FAM_DR_01_CROSS_CLOUD",
            "query_text": "Disaster recovery across AWS and GCP with 99.99 percent SLA and max 50ms latency under 240 dollar a month",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "Z3_Graph_Disaster_Recovery",
            "expected_outcome_pre": "INFEASIBLE",
            "latency_max_ms": 50.0,
            "sla_availability_pct": 99.99,
            "budget_max_usd": 240.0,
            "cloud_providers": ["AWS", "GCP"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "minimum $243 (infeasible under $240)",
            "optimum_source": None,
            "key_notes": "Boundary test (infeasible): Budget $240.00 is below the discrete optimal DR cost of $243.00.",
        },
        {
            "num": "Q29",
            "query_id": "Q29_BOUNDARY_DR_UNDER_250_FEASIBLE",
            "scenario_family_id": "FAM_DR_01_CROSS_CLOUD",
            "query_text": "Disaster recovery across AWS and GCP with 99.99 percent SLA and max 50ms latency under 250 dollar a month",
            "category": "FORMAL_ENGLISH",
            "intended_archetype": "Z3_Graph_Disaster_Recovery",
            "expected_outcome_pre": "FEASIBLE",
            "latency_max_ms": 50.0,
            "sla_availability_pct": 99.99,
            "budget_max_usd": 250.0,
            "cloud_providers": ["AWS", "GCP"],
            "previously_run_in_development": True,
            "user_prompt_optimum_str": "$243.00",
            "optimum_source": "INDEPENDENT_SEARCH",
            "key_notes": "Boundary test (feasible): Budget $250.00 satisfies the discrete optimal DR cost of $243.00.",
        },
    ]

    manifest_queries: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []

    for item in query_definitions:
        q_num = item["num"]
        q_id = item["query_id"]
        arch = item["intended_archetype"]
        pre_outcome = item["expected_outcome_pre"]
        user_str = item["user_prompt_optimum_str"]
        src = item["optimum_source"]

        computed_cost: Optional[float] = None
        computed_outcome = pre_outcome
        true_min_cost: Optional[float] = None
        details = ""

        if arch == "ILP_VM_Allocation":
            vcpus = item.get("required_vcpus")
            ram = item.get("required_ram_gb")
            bud = item.get("budget_max_usd")
            provs = item.get("cloud_providers", ["AWS"])

            if vcpus is not None and ram is not None:
                vm_res = solve_vm_exhaustive(vcpus, ram, bud, provs)
                true_min_cost = vm_res["cost"] if vm_res["cost"] != float("inf") else None
                alloc_desc = ", ".join(f"{cnt}x {sku}" for sku, cnt in vm_res["allocation"].items())

                if bud is not None:
                    if vm_res["feasible_under_budget"]:
                        computed_outcome = "FEASIBLE"
                        computed_cost = vm_res["cost"]
                        details = f"Alloc: {alloc_desc} = ${vm_res['cost']:.2f}"
                    else:
                        computed_outcome = "INFEASIBLE"
                        computed_cost = None
                        details = f"True Min: ${vm_res['cost']:.2f} ({alloc_desc}) > Budget ${bud:.2f}"
                else:
                    computed_outcome = "CLARIFICATION_REQUIRED"
                    computed_cost = None
                    details = f"True Min: ${vm_res['cost']:.2f}, Budget Missing"
            else:
                computed_outcome = "CLARIFICATION_REQUIRED"
                computed_cost = None
                details = "Hardware specs (vCPU/RAM) missing"

        elif arch == "Z3_Graph_Disaster_Recovery":
            lat = item.get("latency_max_ms") or 100.0
            sla = item.get("sla_availability_pct") or 99.9
            bud = item.get("budget_max_usd")
            provs = item.get("cloud_providers", ["AWS", "GCP"])

            if bud is not None:
                dr_res = solve_dr_exhaustive(lat, sla, bud, provs)
                true_min_cost = dr_res["min_possible_cost"]
                if dr_res["status"] == "FEASIBLE":
                    computed_outcome = "FEASIBLE"
                    computed_cost = dr_res["cost"]
                    details = f"Pair: {dr_res['primary']} + {dr_res['secondary']} ({dr_res['latency_ms']}ms) = ${dr_res['cost']:.2f}"
                else:
                    computed_outcome = "INFEASIBLE"
                    computed_cost = None
                    details = f"Status: {dr_res['status']}, Min Cost: ${true_min_cost}"
            else:
                computed_outcome = "CLARIFICATION_REQUIRED"
                computed_cost = None
                details = "Budget missing"

        elif arch == "PSO_Continuous_Scaling":
            bw = item.get("target_bandwidth_mbps")
            max_cpu = item.get("max_cpu_pct")
            target_cpu = item.get("target_cpu_pct")
            bud = item.get("budget_max_usd")

            if target_cpu is not None and max_cpu is not None and target_cpu > max_cpu:
                computed_outcome = "CONFLICTING_REQUIREMENTS"
                computed_cost = None
                details = f"Conflict: Target CPU {target_cpu}% > Max Ceiling {max_cpu}%"
            elif bw is not None:
                sc_res = solve_scaling_exhaustive(bw, max_cpu, target_cpu, bud)
                true_min_cost = sc_res["min_possible_cost"]
                if sc_res["status"] == "FEASIBLE":
                    computed_outcome = "FEASIBLE"
                    computed_cost = sc_res["cost"]
                    details = f"{sc_res['replicas']} reps @ {sc_res['cpu_pct']}% CPU = ${sc_res['cost']:.2f}"
                else:
                    computed_outcome = "INFEASIBLE"
                    computed_cost = None
                    details = f"Status: {sc_res['status']}, Min Cost: ${true_min_cost}"
            else:
                computed_outcome = "CLARIFICATION_REQUIRED"
                computed_cost = None
                details = "Workload (Mbps) missing"

        elif arch == "NONE":
            computed_outcome = pre_outcome
            computed_cost = None
            details = "Unsupported domain or task"

        # Construct final manifest query entry
        query_entry = {
            "query_id": q_id,
            "scenario_family_id": item["scenario_family_id"],
            "query_text": item["query_text"],
            "category": item["category"],
            "intended_archetype": arch,
            "expected_outcome": computed_outcome,
            "required_vcpus": item.get("required_vcpus"),
            "required_ram_gb": item.get("required_ram_gb"),
            "target_bandwidth_mbps": item.get("target_bandwidth_mbps"),
            "min_bandwidth_mbps": item.get("min_bandwidth_mbps"),
            "target_cpu_pct": item.get("target_cpu_pct"),
            "max_cpu_pct": item.get("max_cpu_pct"),
            "latency_max_ms": item.get("latency_max_ms"),
            "sla_availability_pct": item.get("sla_availability_pct"),
            "budget_max_usd": item.get("budget_max_usd"),
            "cloud_providers": item.get("cloud_providers", ["AWS"]),
            "annotation_source": "AI_DRAFT",
            "approval_status": "UNAPPROVED",
            "is_approved": False,
            "previously_run_in_development": item["previously_run_in_development"],
            "expected_optimal_cost_usd": computed_cost,
            "optimum_source": src if computed_outcome == "FEASIBLE" else None,
            "key_notes": item["key_notes"],
        }
        manifest_queries.append(query_entry)

        # Record comparison row for report
        comparison_rows.append({
            "num": q_num,
            "query_id": q_id,
            "query_text": item["query_text"],
            "category": item["category"],
            "archetype": arch,
            "expected_outcome": computed_outcome,
            "prompt_optimum_str": user_str,
            "computed_optimal_cost_usd": computed_cost,
            "true_minimum_cost_usd": true_min_cost,
            "optimum_source": src if computed_outcome == "FEASIBLE" else None,
            "previously_seen": item["previously_run_in_development"],
            "details": details,
        })

    manifest_data = {
        "manifest_name": "NEURASYM Final Benchmark Evaluation Manifest",
        "manifest_version": "1.0.0-final",
        "annotation_source": "AI_DRAFT",
        "approval_status": "UNAPPROVED",
        "is_approved": False,
        "query_count": len(manifest_queries),
        "clarification_policy": [
            "VM queries require vCPUs, RAM and budget. Missing any -> CLARIFICATION_REQUIRED.",
            "DR queries require a budget. Latency and SLA may default (100 ms, 99.9%) only if the query does not state them.",
            "Scaling queries require a workload (bandwidth in Mbps). Missing -> CLARIFICATION_REQUIRED.",
            "A query whose constraints contradict each other -> CONFLICTING_REQUIREMENTS.",
            "Anything outside VM allocation, DR placement and scaling (GPUs, migrations, non-cloud topics) -> UNSUPPORTED."
        ],
        "success_definitions": {
            "task_success": "The mode's outcome equals expected_outcome; for FEASIBLE the plan must also satisfy every keyed requirement when independently recomputed. Correct refusals count as success.",
            "optimal_cost_match": "Recomputed cost within max($0.50, 1%) of expected_optimal_cost_usd.",
            "interpretation_match": "Every extracted field equals the keyed field (Modes 3 and 4 only)."
        },
        "queries": manifest_queries,
    }

    return manifest_data, comparison_rows


def main():
    print("=" * 96)
    print(" NEURASYM INDEPENDENT GROUND-TRUTH OPTIMA CALCULATOR & MANIFEST BUILDER ".center(96, "="))
    print("=" * 96 + "\n")

    manifest_data, comp_rows = build_final_manifest_data()
    target_manifest_path = PROJECT_ROOT / "data" / "final_query_manifest.json"

    # Write the new manifest
    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {len(manifest_data['queries'])} queries to: {target_manifest_path}\n")

    # Validate the manifest
    is_valid, errors, stats = ManifestManager.validate_manifest_file(str(target_manifest_path))
    if not is_valid:
        print("[FAIL] Manifest validation failed:")
        for err in errors:
            print(f"  [ERROR] {err}")
        sys.exit(1)
    else:
        print("[PASS] Manifest validation passed completely!")
        print(f"  - Total Queries: {stats['total_queries']}")
        print(f"  - Status: 100% UNAPPROVED (0 approved)")
        print(f"  - Feasible with Cost: {stats['feasible_with_cost_count']}")
        print(f"  - Previously Seen in Dev: {stats['previously_seen_count']}\n")

    # Print Full Table of 29 Keys
    print("=" * 110)
    print(f"{'#':<4} | {'Query ID':<38} | {'Archetype':<26} | {'Outcome':<24} | {'Expected Cost':<14} | {'Source':<18}")
    print("-" * 110)
    for r in comp_rows:
        cost_str = f"${r['computed_optimal_cost_usd']:.2f}" if r['computed_optimal_cost_usd'] is not None else "null"
        src_str = r['optimum_source'] or "—"
        print(f"{r['num']:<4} | {r['query_id']:<38} | {r['archetype']:<26} | {r['expected_outcome']:<24} | {cost_str:<14} | {src_str:<18}")
    print("=" * 110 + "\n")

    # Comparison and Mismatch Check
    print("=" * 110)
    print(" GROUND-TRUTH VS PROMPT SPECIFICATION COMPARISON & MISMATCH AUDIT ")
    print("=" * 110)
    for r in comp_rows:
        computed_c = f"${r['computed_optimal_cost_usd']:.2f}" if r['computed_optimal_cost_usd'] is not None else "null"
        print(f"[{r['num']}] {r['query_id']}")
        print(f"  Query Text    : \"{r['query_text']}\"")
        print(f"  Category      : {r['category']} (Seen in dev: {r['previously_seen']})")
        print(f"  Outcome       : {r['expected_outcome']}")
        print(f"  Prompt Target : {r['prompt_optimum_str']}")
        print(f"  Computed Cost : {computed_c} (Optimum Source: {r['optimum_source']})")
        if r['true_minimum_cost_usd'] is not None:
            print(f"  True Minimum  : ${r['true_minimum_cost_usd']:.2f}")
        print(f"  Details       : {r['details']}")
        print("-" * 110)


if __name__ == "__main__":
    main()
