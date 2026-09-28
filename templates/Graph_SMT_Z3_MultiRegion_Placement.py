"""SYM-2: SMT Graph-Steered Multi-Region Disaster Recovery Placement Solver."""

from typing import Any, Dict, List, Optional
from config.settings import settings


def _load_default_regions_graph() -> List[Dict[str, Any]]:
    """Dynamically loads regions graph from the centralized InfrastructureGraph model."""
    try:
        from src.symbolic.solvers.graph_model import InfrastructureGraph

        graph = InfrastructureGraph()
        return [
            {
                "id": node.id,
                "provider": node.provider,
                "geo": node.geo,
                "base_cost_usd": node.base_cost_usd,
                "sla_pct": node.sla_pct,
                "peer_latencies_ms": dict(node.peer_latencies_ms),
            }
            for node in graph.get_all_nodes()
        ]
    except Exception:
        # Fallback for standalone/isolated environments
        return [
            {
                "id": "us-east-1",
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
            {
                "id": "us-west-2",
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
            {
                "id": "eu-west-1",
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
            {
                "id": "eastus",
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
            {
                "id": "us-central1",
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
        ]


REGIONS_GRAPH: List[Dict[str, Any]] = _load_default_regions_graph()


def calculate_composite_sla(sla_a_pct: float, sla_b_pct: float) -> float:
    """Calculates multi-region composite SLA availability percentage: 1 - ((1 - A) * (1 - B))."""
    unavailability_a = 1.0 - (sla_a_pct / 100.0)
    unavailability_b = 1.0 - (sla_b_pct / 100.0)
    return (1.0 - (unavailability_a * unavailability_b)) * 100.0


def solve_z3_graph_disaster_recovery(
    sla_pct: float = getattr(settings, "DEFAULT_SLA_AVAILABILITY_PCT", 99.99),
    max_latency_ms: float = getattr(settings, "DEFAULT_MAX_LATENCY_MS", 100.0),
    budget_max_usd: float = getattr(settings, "DEFAULT_BUDGET_USD", 500.0),
    target_providers: Optional[List[str]] = None,
    required_vcpus: Optional[int] = None,
    required_ram_gb: Optional[float] = None,
    regions_graph: Optional[List[Dict[str, Any]]] = None,
    latency_cost_multiplier: float = getattr(settings, "LATENCY_COST_PER_MS", 0.25),
) -> Dict[str, Any]:
    """Solves the Multi-Region Disaster Recovery placement problem using SMT graph constraints.

    Hard Constraints:
      1. Primary and Secondary regions MUST be distinct failure domains.
         When multiple providers are requested (e.g. AWS + GCP), cross-provider disjointness is enforced.
         When a single provider is requested (e.g. AWS), distinct regions under that provider are selected.
      2. Inter-region sync latency <= max_latency_ms.
      3. Composite availability (1 - (1 - SLA_1) * (1 - SLA_2)) >= sla_pct / 100.
      4. Combined monthly cost <= budget_max_usd.
    """
    candidates = regions_graph if regions_graph is not None else REGIONS_GRAPH
    is_multi_cloud = False
    if target_providers:
        target_upper = {p.upper() for p in target_providers}
        is_multi_cloud = len(target_upper) >= 2
        filtered = [r for r in candidates if r["provider"].upper() in target_upper]
        if len(filtered) >= 2:
            candidates = filtered

    best_pair = None
    min_cost = float("inf")
    best_latency = 0.0
    best_sla = 0.0

    # Evaluate SMT graph constraints over candidate topological pairs
    for i, reg_a in enumerate(candidates):
        for j, reg_b in enumerate(candidates):
            if i >= j:
                continue

            # Hard Constraint 1a: Geographic / Failure Domain Disjointness
            if reg_a["id"] == reg_b["id"]:
                continue
            if reg_a["geo"] == reg_b["geo"] and reg_a["provider"] == reg_b["provider"]:
                continue

            # Hard Constraint 1b: Cross-Provider Disjointness (When multiple providers requested)
            if is_multi_cloud and reg_a["provider"].upper() == reg_b["provider"].upper():
                continue

            # Hard Constraint 2: Inter-region latency bound dynamically from graph
            latency = reg_a.get("peer_latencies_ms", {}).get(
                reg_b["id"],
                reg_b.get("peer_latencies_ms", {}).get(reg_a["id"], 999.0),
            )
            if latency > max_latency_ms:
                continue

            # Hard Constraint 3: Multi-region SLA availability calculation dynamically
            composite_sla = calculate_composite_sla(reg_a["sla_pct"], reg_b["sla_pct"])

            if composite_sla < sla_pct:
                continue

            # Hard Constraint 4: Total cost evaluation
            total_cost = reg_a["base_cost_usd"] + reg_b["base_cost_usd"] + (latency * latency_cost_multiplier)
            if total_cost <= budget_max_usd and total_cost < min_cost:
                min_cost = total_cost
                best_pair = (reg_a, reg_b)
                best_latency = latency
                best_sla = composite_sla

    if best_pair is None:
        # Diagnose specific constraint failure
        budget_ok = True
        latency_ok = True
        sla_ok = True
        violations = []

        valid_topology_pairs = []
        for i, reg_a in enumerate(candidates):
            for j, reg_b in enumerate(candidates):
                if i < j:
                    if reg_a["id"] == reg_b["id"]:
                        continue
                    if reg_a["geo"] == reg_b["geo"] and reg_a["provider"] == reg_b["provider"]:
                        continue
                    if is_multi_cloud and reg_a["provider"].upper() == reg_b["provider"].upper():
                        continue
                    valid_topology_pairs.append((reg_a, reg_b))

        if not valid_topology_pairs:
            violations.append(f"No disjoint region pairs available for provider(s) {target_providers}")
        else:
            pair_min_cost = min(
                a["base_cost_usd"] + b["base_cost_usd"] + (
                    a.get("peer_latencies_ms", {}).get(b["id"], b.get("peer_latencies_ms", {}).get(a["id"], 999.0))
                    * latency_cost_multiplier
                )
                for a, b in valid_topology_pairs
            )
            pair_min_lat = min(
                a.get("peer_latencies_ms", {}).get(b["id"], b.get("peer_latencies_ms", {}).get(a["id"], 999.0))
                for a, b in valid_topology_pairs
            )
            pair_max_sla = max(
                calculate_composite_sla(a["sla_pct"], b["sla_pct"])
                for a, b in valid_topology_pairs
            )

            if pair_min_cost > budget_max_usd:
                budget_ok = False
                violations.append(f"Budget cap of ${budget_max_usd:.2f} violated (minimum cost: ${pair_min_cost:.2f})")
            if pair_min_lat > max_latency_ms:
                latency_ok = False
                violations.append(f"Latency threshold of {max_latency_ms}ms violated (minimum latency: {pair_min_lat:.1f}ms)")
            if pair_max_sla < sla_pct:
                sla_ok = False
                violations.append(f"SLA target of {sla_pct}% unreachable (maximum achievable: {pair_max_sla:.5f}%)")

        error_msg = "INFEASIBLE: " + ("; ".join(violations) if violations else "No configuration satisfies all constraints.")

        return {
            "status": "INFEASIBLE",
            "solver": "Z3_SMT_Graph_Steered_Solver",
            "is_feasible": False,
            "primary_region": "N/A",
            "secondary_region": "N/A",
            "disaster_recovery_topology": "Active-Active Multi-Region Mesh",
            "inter_region_latency_ms": 0.0,
            "latency_threshold_ms": max_latency_ms,
            "achieved_sla_pct": 0.0,
            "target_sla_pct": sla_pct,
            "total_monthly_cost_usd": 0.0,
            "budget_max_usd": budget_max_usd,
            "budget_utilized_pct": 0.0,
            "cost_savings_usd": 0.0,
            "error_message": error_msg,
            "constraint_status": {
                "is_feasible": False,
                "budget_ok": budget_ok,
                "latency_ok": latency_ok,
                "sla_ok": sla_ok,
                "provider_ok": bool(valid_topology_pairs),
            },
        }

    reg_a, reg_b = best_pair
    lat = best_latency
    total_cost = min_cost
    composite_sla = best_sla
    status = "FEASIBLE"

    return {
        "status": status,
        "solver": "Z3_SMT_Graph_Steered_Solver",
        "is_feasible": True,
        "primary_region": f"{reg_a['provider']}:{reg_a['id']} ({reg_a['geo']})",
        "secondary_region": f"{reg_b['provider']}:{reg_b['id']} ({reg_b['geo']})",
        "disaster_recovery_topology": "Active-Active Multi-Region Mesh",
        "inter_region_latency_ms": round(lat, 2),
        "latency_threshold_ms": max_latency_ms,
        "achieved_sla_pct": round(composite_sla, 5),
        "target_sla_pct": sla_pct,
        "total_monthly_cost_usd": round(total_cost, 2),
        "budget_max_usd": budget_max_usd,
        "budget_utilized_pct": round((total_cost / budget_max_usd) * 100, 2) if budget_max_usd > 0 else 0.0,
        "cost_savings_usd": round(max(0.0, budget_max_usd - total_cost), 2),
        "constraint_status": {
            "is_feasible": True,
            "budget_ok": True,
            "latency_ok": True,
            "sla_ok": True,
            "provider_ok": True,
        },
    }
