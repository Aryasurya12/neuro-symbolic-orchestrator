"""Z3 SMT & MaxSMT Multi-Region Placement Solver with SAGE-GNN Soft Constraint Integration."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import z3

from src.optimizers.sage_gnn_adapter import (
    ComponentNode,
    SAGEGNNBenchmarkTopology,
    SageGNNConstraintAdapter,
    VMNode,
)


def solve_sage_gnn_placement(
    topology: SAGEGNNBenchmarkTopology,
    weight_scale: int = 100,
    use_gnn_soft_constraints: bool = True,
    timeout_ms: int = 5000,
) -> Dict[str, Any]:
    """Solves the SAGE-GNN component-to-VM cloud deployment placement problem using Z3 MaxSMT.

    Enforces:
      Hard Constraints (Strict mathematical guarantees via opt.add()):
        1. Exact Assignment: Each component is placed on exactly 1 VM.
        2. Resource Capacity: Total vCPUs and RAM assigned to each VM cannot exceed capacity.
        3. Anti-Affinity: Specified component pairs cannot share the same VM host/domain.
        4. Latency SLA: Inter-component network latency cannot exceed max_latency_ms (e.g. 50ms).
        5. Budget Ceiling: Combined monthly cost of all active VMs <= budget_max_usd.

      Soft Constraints (MaxSMT neural preferences via adapter.inject_soft_constraints()):
        - SAGE-GNN RGCN component-to-VM classification probabilities p(c, v) converted
          to integer weights w = int(p * weight_scale) and added via opt.add_soft().

      Objective:
        - Cost-optimal financial deployment minimizing total monthly cloud spend.

    Returns:
        Structured dictionary with placement mapping, cost, latency, feasibility, and diagnostics.
    """
    start_time = time.perf_counter()

    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    components = topology.components
    vm_catalog = topology.vm_catalog
    comp_names = list(components.keys())
    vm_ids = list(vm_catalog.keys())

    # -------------------------------------------------------------------------
    # 1. Decision Variables
    # -------------------------------------------------------------------------
    # placement_vars[(c, v)] is True iff component c is placed on VM v
    placement_vars: Dict[Tuple[str, str], z3.BoolRef] = {
        (c, v): z3.Bool(f"place_{c}_{v}")
        for c in comp_names
        for v in vm_ids
    }

    # vm_active[v] is True iff at least one component is placed on VM v
    vm_active_vars: Dict[str, z3.BoolRef] = {
        v: z3.Bool(f"active_{v}")
        for v in vm_ids
    }

    # -------------------------------------------------------------------------
    # 2. Hard Constraint 1: Exact Assignment (Each component on exactly 1 VM)
    # -------------------------------------------------------------------------
    for c in comp_names:
        # At least one VM
        opt.add(z3.Or([placement_vars[(c, v)] for v in vm_ids]))
        # At most one VM (pairwise disjoint)
        for i, v1 in enumerate(vm_ids):
            for j, v2 in enumerate(vm_ids):
                if i < j:
                    opt.add(z3.Not(z3.And(placement_vars[(c, v1)], placement_vars[(c, v2)])))

    # Link vm_active_vars to component placements
    for v in vm_ids:
        comp_placed_on_v = [placement_vars[(c, v)] for c in comp_names]
        opt.add(vm_active_vars[v] == z3.Or(comp_placed_on_v))

    # -------------------------------------------------------------------------
    # 3. Hard Constraint 2: Resource Capacity (vCPUs and RAM per VM)
    # -------------------------------------------------------------------------
    for v in vm_ids:
        vm_spec = vm_catalog[v]
        vcpu_terms = [
            z3.If(placement_vars[(c, v)], components[c].required_vcpus, 0)
            for c in comp_names
        ]
        ram_terms = [
            z3.If(placement_vars[(c, v)], z3.RealVal(components[c].required_ram_gb), z3.RealVal(0.0))
            for c in comp_names
        ]
        opt.add(z3.Sum(vcpu_terms) <= vm_spec.vcpus)
        opt.add(z3.Sum(ram_terms) <= z3.RealVal(vm_spec.ram_gb))

    # -------------------------------------------------------------------------
    # 4. Hard Constraint 3: Anti-Affinity Rules
    # -------------------------------------------------------------------------
    for c1, c2 in topology.anti_affinity_pairs:
        if c1 in components and c2 in components:
            for v in vm_ids:
                # c1 and c2 cannot both be placed on VM v
                opt.add(z3.Not(z3.And(placement_vars[(c1, v)], placement_vars[(c2, v)])))

    # -------------------------------------------------------------------------
    # 5. Hard Constraint 4: Inter-Component Network Latency (< max_latency_ms)
    # -------------------------------------------------------------------------
    # For any pair of communicating components, inter-region latency cannot exceed max_latency_ms
    for i, c1 in enumerate(comp_names):
        for j, c2 in enumerate(comp_names):
            if i < j:
                for v1 in vm_ids:
                    for v2 in vm_ids:
                        reg1 = vm_catalog[v1].region
                        reg2 = vm_catalog[v2].region
                        lat = topology.get_latency(reg1, reg2)
                        if lat > topology.max_latency_ms:
                            # Cannot place c1 on v1 and c2 on v2 simultaneously
                            opt.add(z3.Not(z3.And(placement_vars[(c1, v1)], placement_vars[(c2, v2)])))

    # -------------------------------------------------------------------------
    # 6. Hard Constraint 5: Monthly Budget Cap ($USD)
    # -------------------------------------------------------------------------
    cost_terms = [
        z3.If(vm_active_vars[v], z3.RealVal(vm_catalog[v].monthly_cost_usd), z3.RealVal(0.0))
        for v in vm_ids
    ]
    total_cost_expr = z3.Sum(cost_terms)
    opt.add(total_cost_expr <= z3.RealVal(topology.budget_max_usd))

    # -------------------------------------------------------------------------
    # 7. Soft Constraints: SAGE-GNN RGCN Placement Probabilities
    # -------------------------------------------------------------------------
    injected_soft_assertions = []
    if use_gnn_soft_constraints and topology.gnn_probabilities:
        adapter = SageGNNConstraintAdapter(weight_scale=weight_scale)
        injected_soft_assertions = adapter.inject_soft_constraints(
            opt, placement_vars, topology.gnn_probabilities, weight_scale=weight_scale
        )

    # -------------------------------------------------------------------------
    # 8. Optimization Objective
    # -------------------------------------------------------------------------
    opt.minimize(total_cost_expr)

    # -------------------------------------------------------------------------
    # 9. Solve and Extract Result
    # -------------------------------------------------------------------------
    check_status = opt.check()
    solve_duration_sec = time.perf_counter() - start_time
    solve_duration_ms = round(solve_duration_sec * 1000.0, 2)

    if check_status == z3.sat:
        model = opt.model()

        # Extract component placement mapping
        placements: Dict[str, Dict[str, Any]] = {}
        active_vms: Dict[str, Dict[str, Any]] = {}
        active_regions: set = set()
        total_vcpu_allocated = 0
        total_ram_allocated = 0.0

        for c in comp_names:
            for v in vm_ids:
                if z3.is_true(model.eval(placement_vars[(c, v)])):
                    vm_info = vm_catalog[v]
                    gnn_prob = topology.gnn_probabilities.get((c, v), 0.0)
                    placements[c] = {
                        "component": c,
                        "tier": components[c].tier,
                        "required_vcpus": components[c].required_vcpus,
                        "required_ram_gb": components[c].required_ram_gb,
                        "vm_id": v,
                        "provider": vm_info.provider,
                        "region": vm_info.region,
                        "instance_family": vm_info.instance_family,
                        "vm_vcpus": vm_info.vcpus,
                        "vm_ram_gb": vm_info.ram_gb,
                        "monthly_cost_usd": vm_info.monthly_cost_usd,
                        "gnn_prediction_prob": gnn_prob,
                    }
                    active_regions.add(vm_info.region)

                    if v not in active_vms:
                        active_vms[v] = {
                            "vm_id": v,
                            "provider": vm_info.provider,
                            "region": vm_info.region,
                            "instance_family": vm_info.instance_family,
                            "monthly_cost_usd": vm_info.monthly_cost_usd,
                            "vcpus_capacity": vm_info.vcpus,
                            "ram_capacity_gb": vm_info.ram_gb,
                            "hosted_components": [],
                            "used_vcpus": 0,
                            "used_ram_gb": 0.0,
                        }
                    active_vms[v]["hosted_components"].append(c)
                    active_vms[v]["used_vcpus"] += components[c].required_vcpus
                    active_vms[v]["used_ram_gb"] += components[c].required_ram_gb
                    total_vcpu_allocated += components[c].required_vcpus
                    total_ram_allocated += components[c].required_ram_gb

        actual_monthly_cost = round(sum(v["monthly_cost_usd"] for v in active_vms.values()), 2)
        budget_utilization_pct = round((actual_monthly_cost / topology.budget_max_usd) * 100.0, 1) if topology.budget_max_usd > 0 else 0.0

        # Calculate inter-component maximum latency in the resulting placement
        max_observed_latency = 0.0
        for i, c1 in enumerate(comp_names):
            for j, c2 in enumerate(comp_names):
                if i < j:
                    reg1 = placements[c1]["region"]
                    reg2 = placements[c2]["region"]
                    lat = topology.get_latency(reg1, reg2)
                    if lat > max_observed_latency:
                        max_observed_latency = lat

        # Evaluate GNN alignment rate
        gnn_top_matches = 0
        total_soft_weight_achieved = 0
        for c in comp_names:
            chosen_v = placements[c]["vm_id"]
            # Find VM with max GNN prediction probability for component c
            c_probs = {v: topology.gnn_probabilities.get((c, v), 0.0) for v in vm_ids}
            best_gnn_vm = max(c_probs, key=c_probs.get) if c_probs else None
            if chosen_v == best_gnn_vm:
                gnn_top_matches += 1
            prob = topology.gnn_probabilities.get((c, chosen_v), 0.0)
            total_soft_weight_achieved += int(prob * weight_scale)

        gnn_alignment_pct = round((gnn_top_matches / len(comp_names)) * 100.0, 1) if comp_names else 0.0

        return {
            "status": "FEASIBLE",
            "is_feasible": True,
            "topology_name": topology.name,
            "description": topology.description,
            "solver": "Z3_MaxSMT_SAGE_GNN_Optimizer",
            "runtime_ms": solve_duration_ms,
            "runtime_sec": round(solve_duration_sec, 4),
            "total_monthly_cost_usd": actual_monthly_cost,
            "budget_max_usd": topology.budget_max_usd,
            "budget_utilization_pct": budget_utilization_pct,
            "cost_savings_usd": round(max(0.0, topology.budget_max_usd - actual_monthly_cost), 2),
            "active_vm_count": len(active_vms),
            "active_vms": list(active_vms.values()),
            "component_placements": placements,
            "active_regions": list(active_regions),
            "max_inter_component_latency_ms": max_observed_latency,
            "latency_sla_threshold_ms": topology.max_latency_ms,
            "anti_affinity_verified": True,
            "gnn_soft_constraints_active": use_gnn_soft_constraints,
            "soft_assertions_count": len(injected_soft_assertions),
            "gnn_alignment_pct": gnn_alignment_pct,
            "soft_weight_achieved": total_soft_weight_achieved,
            "constraint_status": {
                "is_feasible": True,
                "budget_ok": actual_monthly_cost <= topology.budget_max_usd,
                "latency_ok": max_observed_latency <= topology.max_latency_ms,
                "anti_affinity_ok": True,
                "capacity_ok": True,
            },
        }

    else:
        # UNSAT Diagnosis
        return {
            "status": "INFEASIBLE",
            "is_feasible": False,
            "topology_name": topology.name,
            "description": topology.description,
            "solver": "Z3_MaxSMT_SAGE_GNN_Optimizer",
            "runtime_ms": solve_duration_ms,
            "runtime_sec": round(solve_duration_sec, 4),
            "total_monthly_cost_usd": 0.0,
            "budget_max_usd": topology.budget_max_usd,
            "budget_utilization_pct": 0.0,
            "cost_savings_usd": 0.0,
            "active_vm_count": 0,
            "active_vms": [],
            "component_placements": {},
            "active_regions": [],
            "max_inter_component_latency_ms": 0.0,
            "latency_sla_threshold_ms": topology.max_latency_ms,
            "anti_affinity_verified": False,
            "gnn_soft_constraints_active": use_gnn_soft_constraints,
            "soft_assertions_count": len(injected_soft_assertions),
            "gnn_alignment_pct": 0.0,
            "soft_weight_achieved": 0,
            "error_message": f"INFEASIBLE: Constraints for topology '{topology.name}' could not be simultaneously satisfied.",
            "constraint_status": {
                "is_feasible": False,
                "budget_ok": False,
                "latency_ok": False,
                "anti_affinity_ok": False,
                "capacity_ok": False,
            },
        }


def run_all_sage_gnn_benchmarks(weight_scale: int = 100) -> Dict[str, Dict[str, Any]]:
    """Runs the Z3 MaxSMT placement optimization on all 3 SAGE-GNN (IJCNN 2024) benchmark topologies."""
    dataset = SageGNNConstraintAdapter.load_benchmark_dataset()
    results: Dict[str, Dict[str, Any]] = {}

    for name, topology in dataset.items():
        res = solve_sage_gnn_placement(
            topology=topology,
            weight_scale=weight_scale,
            use_gnn_soft_constraints=True,
        )
        results[name] = res

    return results
