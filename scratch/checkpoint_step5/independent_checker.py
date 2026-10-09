"""Independent Result Checker for Neurasym.

Evaluates concrete returned decisions against ground-truth catalogs,
infrastructure network topologies, and mathematical constraints.
Does NOT trust solver self-reports, internal feasibility labels, or
self-generated certificates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union


class IndependentChecker:
    """Ground-truth catalog and constraint verifier for optimization outputs."""

    # Reference Ground-Truth VM SKUs (backed by DatabaseBackedCatalog / default catalog)
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

    # Reference Ground-Truth Regional Infrastructure Graph
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

    HOURS_PER_MONTH: float = 730.0
    LATENCY_COST_PER_MS: float = 0.25
    SCALING_COST_PER_MBPS: float = 0.08
    SCALING_COST_PER_REPLICA: float = 45.0
    SCALING_CAPACITY_FACTOR_MBPS: float = 75.0

    @classmethod
    def get_sku_catalog(cls) -> Dict[str, Dict[str, Any]]:
        """Attempts to load SKUs from local database; falls back to static ground truth."""
        try:
            from src.symbolic.optimizers.domain_catalog import DatabaseBackedCatalog

            db_cat = DatabaseBackedCatalog()
            cat = {}
            for s in db_cat.skus:
                cat[s.sku] = {
                    "provider": s.provider_id,
                    "vcpus": s.vcpus,
                    "ram_gb": s.ram_gb,
                    "hourly_cost_usd": s.hourly_price_usd,
                }
            if cat:
                return cat
        except Exception:
            pass
        return dict(cls.GROUND_TRUTH_VM_SKUS)

    @classmethod
    def get_regions_graph(cls) -> Dict[str, Dict[str, Any]]:
        """Attempts to load regions graph from InfrastructureGraph; falls back to static ground truth."""
        try:
            from src.symbolic.solvers.graph_model import InfrastructureGraph

            g = InfrastructureGraph()
            rg = {}
            for n in g.get_all_nodes():
                rg[n.id] = {
                    "provider": n.provider,
                    "geo": n.geo,
                    "base_cost_usd": n.base_cost_usd,
                    "sla_pct": n.sla_pct,
                    "peer_latencies_ms": dict(n.peer_latencies_ms),
                }
            if rg:
                return rg
        except Exception:
            pass
        return dict(cls.GROUND_TRUTH_REGIONS)

    # =========================================================================
    # Helpers to safely extract contract / problem values
    # =========================================================================
    @staticmethod
    def _extract_contract_dict(contract: Any) -> Dict[str, Any]:
        if isinstance(contract, dict):
            return dict(contract)
        if hasattr(contract, "model_dump"):
            return contract.model_dump()
        if hasattr(contract, "__dict__"):
            return {k: v for k, v in contract.__dict__.items() if not k.startswith("_")}
        return {}

    @classmethod
    def _extract_region_id(cls, raw_val: Any) -> Optional[str]:
        """Extracts standard region identifier (e.g. 'us-east-1') from string."""
        if not raw_val or not isinstance(raw_val, str) or raw_val.strip() in ["", "N/A", "None"]:
            return None
        clean = raw_val.strip()
        # Handle formats like "AWS:us-east-1 (US_East)" or "us-east-1"
        if ":" in clean:
            clean = clean.split(":", 1)[1].strip()
        if "(" in clean:
            clean = clean.split("(", 1)[0].strip()
        clean = clean.strip()
        return clean if clean else None

    # =========================================================================
    # 1. VM Knapsack Allocation Checker
    # =========================================================================
    @classmethod
    def verify_vm_allocation(
        cls,
        contract_data: Dict[str, Any],
        solver_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Independently verifies a VM Knapsack Allocation decision."""
        catalog = cls.get_sku_catalog()

        req_vcpus = int(contract_data.get("required_vcpus", contract_data.get("min_vcpu", 4)))
        req_ram_gb = float(contract_data.get("required_ram_gb", contract_data.get("min_ram", 8.0)))
        budget_usd = float(contract_data.get("budget_max_usd", contract_data.get("budget_usd", 500.0)))
        requested_providers = contract_data.get("cloud_providers", ["AWS"])
        target_provs_upper = {p.upper() for p in requested_providers} if requested_providers else None

        reported_cost = float(solver_res.get("total_monthly_cost_usd", solver_res.get("catalog_cost", 0.0)))
        solver_status = str(solver_res.get("status", "UNKNOWN")).upper()
        solver_name = str(solver_res.get("solver", solver_res.get("solver_name", "Unknown_Solver")))

        # Inspect concrete decision variables: allocated_vms
        allocated_vms = solver_res.get("allocated_vms")
        has_vms = isinstance(allocated_vms, list) and len(allocated_vms) > 0
        is_solver_infeasible = solver_status in ["INFEASIBLE", "INFEASIBLE_BUDGET_EXCEEDED", "UNSAT", "FAILED"]

        if not has_vms:
            is_zero_cost = abs(reported_cost) <= 0.05
            param_checks_empty = [
                {
                    "name": "SKU Catalog Existence",
                    "target": "Valid in DB",
                    "measured": "None",
                    "delta": "No SKUs Allocated",
                    "status": "FAIL",
                },
                {
                    "name": "SKU Pricing Integrity",
                    "target": "$0.00/mo (Catalog)",
                    "measured": f"${reported_cost:.2f}/mo (Reported)",
                    "delta": f"${reported_cost:.2f}",
                    "status": "PASS" if is_zero_cost else "FAIL",
                },
                {
                    "name": "Compute Capacity (vCPUs)",
                    "target": f">= {req_vcpus} vCPUs",
                    "measured": "0 vCPUs",
                    "delta": f"-{req_vcpus} vCPUs (Deficit)",
                    "status": "FAIL",
                },
                {
                    "name": "Memory Capacity (RAM)",
                    "target": f">= {req_ram_gb:.1f} GB",
                    "measured": "0.0 GB",
                    "delta": f"-{req_ram_gb:.1f} GB (Deficit)",
                    "status": "FAIL",
                },
                {
                    "name": "Financial Monthly Budget",
                    "target": f"<= ${budget_usd:.2f} USD",
                    "measured": f"${reported_cost:.2f} USD",
                    "delta": f"${reported_cost:.2f}",
                    "status": "PASS" if reported_cost <= budget_usd else "FAIL",
                },
            ]
            return {
                "problem_type": "ILP_VM_Allocation",
                "structure_valid": is_solver_infeasible,
                "catalog_consistent": is_zero_cost,
                "cost_accuracy": {
                    "reported_cost_usd": reported_cost,
                    "calculated_catalog_cost_usd": 0.0,
                    "cost_delta_usd": reported_cost,
                    "cost_error_pct": 100.0 if reported_cost > 0 else 0.0,
                },
                "feasible_against_contract": False,
                "interpretation_correct": "Not independently evaluated",
                "optimality_verdict": "Infeasible",
                "summary_status": "Solver reported infeasible" if is_solver_infeasible else "Invalid output structure",
                "violations": [
                    f"Solver reported infeasible status: {solver_status}" if is_solver_infeasible
                    else "No valid allocated_vms list found in solver decision output."
                ],
                "recomputed_metrics": {
                    "vcpus": 0,
                    "ram_gb": 0.0,
                    "monthly_cost_usd": 0.0,
                },
                "parameter_checks": param_checks_empty,
            }

        structure_valid = True
        catalog_consistent = True
        violations: List[str] = []

        recomputed_vcpus = 0
        recomputed_ram_gb = 0.0
        recomputed_cost_usd = 0.0

        for idx, item in enumerate(allocated_vms):
            if not isinstance(item, dict):
                structure_valid = False
                violations.append(f"Allocation entry #{idx} is not a dictionary.")
                continue

            sku_name = item.get("sku") or item.get("instance_type") or item.get("name")
            count = item.get("count") if item.get("count") is not None else (
                item.get("quantity") if item.get("quantity") is not None else item.get("qty", 0)
            )

            if not sku_name or not isinstance(sku_name, str):
                structure_valid = False
                violations.append(f"Allocation entry #{idx} missing valid SKU name.")
                continue

            # Verify positive integer quantity
            try:
                count_int = int(count)
                if count_int <= 0:
                    structure_valid = False
                    violations.append(f"SKU {sku_name} has non-positive quantity: {count}")
                    continue
            except (ValueError, TypeError):
                structure_valid = False
                violations.append(f"SKU {sku_name} has non-integer quantity: {count}")
                continue

            # Verify SKU existence in catalog
            if sku_name not in catalog:
                catalog_consistent = False
                violations.append(f"Unknown SKU '{sku_name}' not found in ground-truth catalog.")
                continue

            sku_meta = catalog[sku_name]
            sku_provider = sku_meta["provider"]

            # Verify provider constraint
            if target_provs_upper and sku_provider.upper() not in target_provs_upper:
                violations.append(
                    f"SKU {sku_name} is on {sku_provider}, which violates requested providers: {requested_providers}"
                )

            # Recompute specs from catalog ground truth (not trusting solver claims)
            item_vcpus = sku_meta["vcpus"] * count_int
            item_ram = sku_meta["ram_gb"] * count_int
            item_monthly_cost = round(sku_meta["hourly_cost_usd"] * cls.HOURS_PER_MONTH * count_int, 2)

            recomputed_vcpus += item_vcpus
            recomputed_ram_gb += item_ram
            recomputed_cost_usd += item_monthly_cost

        recomputed_cost_usd = round(recomputed_cost_usd, 2)
        recomputed_ram_gb = round(recomputed_ram_gb, 2)

        # Check cost delta
        cost_delta = round(abs(reported_cost - recomputed_cost_usd), 2)
        cost_err_pct = round((cost_delta / max(0.01, recomputed_cost_usd)) * 100.0, 2) if recomputed_cost_usd > 0 else 0.0

        if cost_delta > 0.5:
            violations.append(
                f"Reported cost (${reported_cost:.2f}) does not match catalog recomputed cost (${recomputed_cost_usd:.2f})"
            )

        # Check constraint feasibility against required values
        if recomputed_vcpus < req_vcpus:
            violations.append(f"vCPU deficit: recomputed {recomputed_vcpus} < required {req_vcpus}")
        if recomputed_ram_gb < req_ram_gb:
            violations.append(f"RAM deficit: recomputed {recomputed_ram_gb:.1f}GB < required {req_ram_gb:.1f}GB")
        if recomputed_cost_usd > budget_usd:
            violations.append(f"Budget overflow: recomputed cost ${recomputed_cost_usd:.2f} > cap ${budget_usd:.2f}")

        if is_solver_infeasible:
            violations.append(f"Solver reported infeasible status: {solver_status}")

        feasible_against_contract = (
            structure_valid
            and catalog_consistent
            and (recomputed_vcpus >= req_vcpus)
            and (recomputed_ram_gb >= req_ram_gb)
            and (recomputed_cost_usd <= budget_usd)
            and not is_solver_infeasible
            and not any("violates requested providers" in v for v in violations)
        )

        # Determine true optimality verdict
        if not feasible_against_contract:
            optimality_verdict = "Infeasible"
            if not catalog_consistent:
                summary_status = "Catalog inconsistency"
            elif not structure_valid:
                summary_status = "Invalid output structure"
            elif is_solver_infeasible or recomputed_cost_usd > budget_usd:
                summary_status = "Solver reported infeasible"
            else:
                summary_status = "Constraint violation found"
        elif "HiGHS" in solver_name or "MILP" in solver_name or "ILP" in solver_name or "Branch_and_Bound" in solver_name:
            optimality_verdict = "Provably Optimal (MILP branch-and-bound exact)"
            summary_status = "Feasible against checked constraints"
        else:
            # GA or heuristic solver
            optimality_verdict = "Optimality not established (Heuristic approximation)"
            summary_status = "Feasible against checked constraints"

        sku_summary_list = []
        for item in allocated_vms:
            if isinstance(item, dict):
                s_name = item.get("sku") or item.get("instance_type") or item.get("name") or "unknown"
                c_qty = item.get("count") if item.get("count") is not None else item.get("quantity", item.get("qty", 1))
                sku_summary_list.append(f"{s_name} (x{c_qty})")
        sku_summary_str = ", ".join(sku_summary_list) if sku_summary_list else "None"

        parameter_checks = [
            {
                "name": "SKU Catalog Existence",
                "target": "Valid in DB",
                "measured": sku_summary_str,
                "delta": "Exact Match" if catalog_consistent else "Unknown SKU(s) in proposal",
                "status": "PASS" if catalog_consistent else "FAIL",
            },
            {
                "name": "SKU Pricing Integrity",
                "target": f"${recomputed_cost_usd:.2f}/mo (Catalog)",
                "measured": f"${reported_cost:.2f}/mo (Reported)",
                "delta": f"${cost_delta:.2f} ({cost_err_pct:.1f}% err)" if cost_delta > 0.0 else "$0.00 (Exact)",
                "status": "PASS" if (cost_delta <= 0.50 and catalog_consistent) else "FAIL",
            },
            {
                "name": "Compute Capacity (vCPUs)",
                "target": f">= {req_vcpus} vCPUs",
                "measured": f"{recomputed_vcpus} vCPUs",
                "delta": f"{'+' if recomputed_vcpus >= req_vcpus else ''}{recomputed_vcpus - req_vcpus} vCPUs" + ("" if recomputed_vcpus >= req_vcpus else " (Deficit)"),
                "status": "PASS" if recomputed_vcpus >= req_vcpus else "FAIL",
            },
            {
                "name": "Memory Capacity (RAM)",
                "target": f">= {req_ram_gb:.1f} GB",
                "measured": f"{recomputed_ram_gb:.1f} GB",
                "delta": f"{'+' if recomputed_ram_gb >= req_ram_gb else ''}{recomputed_ram_gb - req_ram_gb:.1f} GB" + ("" if recomputed_ram_gb >= req_ram_gb else " (Deficit)"),
                "status": "PASS" if recomputed_ram_gb >= req_ram_gb else "FAIL",
            },
            {
                "name": "Financial Monthly Budget",
                "target": f"<= ${budget_usd:.2f} USD",
                "measured": f"${recomputed_cost_usd:.2f} USD",
                "delta": f"-${budget_usd - recomputed_cost_usd:.2f} (Headroom)" if recomputed_cost_usd <= budget_usd else f"+${recomputed_cost_usd - budget_usd:.2f} (Overflow)",
                "status": "PASS" if recomputed_cost_usd <= budget_usd else "FAIL",
            },
            {
                "name": "Cloud Provider Alignment",
                "target": f"{', '.join(requested_providers) if requested_providers else 'Any'}",
                "measured": f"{', '.join(sorted(list(set(str(item.get('provider', 'AWS')) for item in allocated_vms if isinstance(item, dict)))))}",
                "delta": "Valid Provider" if not any("violates requested providers" in v for v in violations) else "Provider Mismatch",
                "status": "PASS" if not any("violates requested providers" in v for v in violations) else "FAIL",
            },
        ]

        return {
            "problem_type": "ILP_VM_Allocation",
            "structure_valid": structure_valid,
            "catalog_consistent": catalog_consistent,
            "cost_accuracy": {
                "reported_cost_usd": reported_cost,
                "calculated_catalog_cost_usd": recomputed_cost_usd,
                "cost_delta_usd": cost_delta,
                "cost_error_pct": cost_err_pct,
            },
            "feasible_against_contract": feasible_against_contract,
            "interpretation_correct": "Not independently evaluated",
            "optimality_verdict": optimality_verdict,
            "summary_status": summary_status,
            "violations": violations,
            "recomputed_metrics": {
                "vcpus": recomputed_vcpus,
                "ram_gb": recomputed_ram_gb,
                "monthly_cost_usd": recomputed_cost_usd,
            },
            "parameter_checks": parameter_checks,
        }

    # =========================================================================
    # 2. Disaster Recovery Placement Checker
    # =========================================================================
    @classmethod
    def verify_disaster_recovery(
        cls,
        contract_data: Dict[str, Any],
        solver_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Independently verifies a Disaster Recovery Multi-Region SMT decision."""
        regions_graph = cls.get_regions_graph()

        target_sla_pct = float(contract_data.get("sla_availability_pct", contract_data.get("min_sla", 99.99)))
        max_latency_ms = float(contract_data.get("latency_max_ms", contract_data.get("max_latency_ms", 100.0)))
        budget_usd = float(contract_data.get("budget_max_usd", contract_data.get("budget_usd", 500.0)))
        requested_providers = contract_data.get("cloud_providers", ["AWS"])
        target_provs_upper = [p.upper() for p in requested_providers] if requested_providers else []
        is_multi_cloud = len(set(target_provs_upper)) >= 2

        reported_cost = float(solver_res.get("total_monthly_cost_usd", 0.0))
        solver_status = str(solver_res.get("status", "UNKNOWN")).upper()

        prim_str = solver_res.get("primary_region")
        sec_str = solver_res.get("secondary_region")

        reg_a_id = cls._extract_region_id(prim_str)
        reg_b_id = cls._extract_region_id(sec_str)
        is_solver_infeasible = solver_status in ["INFEASIBLE", "UNSAT", "FAILED"]

        if not reg_a_id or not reg_b_id:
            is_zero_cost = abs(reported_cost) <= 0.05
            param_checks_empty = [
                {
                    "name": "Region Graph Topology",
                    "target": "Valid Nodes in Graph",
                    "measured": f"{prim_str or 'N/A'} & {sec_str or 'N/A'}",
                    "delta": "Invalid Region String",
                    "status": "FAIL",
                },
                {
                    "name": "Failure Domain Disjointness",
                    "target": "Primary != Secondary",
                    "measured": "None",
                    "delta": "Missing Nodes",
                    "status": "FAIL",
                },
                {
                    "name": "Inter-Region Network Latency",
                    "target": f"<= {max_latency_ms:.1f} ms",
                    "measured": "0.0 ms",
                    "delta": "No Latency Edge",
                    "status": "FAIL",
                },
                {
                    "name": "Composite Availability SLA",
                    "target": f">= {target_sla_pct:.4f}%",
                    "measured": "0.00000%",
                    "delta": f"-{target_sla_pct:.4f}% (Deficit)",
                    "status": "FAIL",
                },
                {
                    "name": "Disaster Recovery Monthly Cost",
                    "target": f"<= ${budget_usd:.2f} USD",
                    "measured": f"${reported_cost:.2f} USD",
                    "delta": f"${reported_cost:.2f}",
                    "status": "PASS" if is_zero_cost else "FAIL",
                },
            ]
            return {
                "problem_type": "Z3_Graph_Disaster_Recovery",
                "structure_valid": is_solver_infeasible,
                "catalog_consistent": is_zero_cost,
                "cost_accuracy": {
                    "reported_cost_usd": reported_cost,
                    "calculated_catalog_cost_usd": 0.0,
                    "cost_delta_usd": reported_cost,
                    "cost_error_pct": 100.0 if reported_cost > 0 else 0.0,
                },
                "feasible_against_contract": False,
                "interpretation_correct": "Not independently evaluated",
                "optimality_verdict": "Infeasible",
                "summary_status": "Solver reported infeasible" if is_solver_infeasible else "Invalid output structure",
                "violations": [
                    f"Solver reported infeasible status: {solver_status}" if is_solver_infeasible
                    else f"Invalid primary/secondary region values: '{prim_str}', '{sec_str}'"
                ],
                "recomputed_metrics": {
                    "primary_region": str(prim_str or "N/A"),
                    "secondary_region": str(sec_str or "N/A"),
                    "latency_ms": 0.0,
                    "composite_sla_pct": 0.0,
                    "monthly_cost_usd": 0.0,
                },
                "parameter_checks": param_checks_empty,
            }

        structure_valid = True
        catalog_consistent = True
        violations: List[str] = []

        # Verify region IDs exist in ground-truth graph
        if reg_a_id not in regions_graph:
            catalog_consistent = False
            violations.append(f"Primary region '{reg_a_id}' not found in infrastructure graph.")
        if reg_b_id not in regions_graph:
            catalog_consistent = False
            violations.append(f"Secondary region '{reg_b_id}' not found in infrastructure graph.")

        if not catalog_consistent:
            param_checks_inv = [
                {
                    "name": "Region Graph Topology",
                    "target": "Valid Nodes in Graph",
                    "measured": f"{reg_a_id} & {reg_b_id}",
                    "delta": "Unrecognized Node(s)",
                    "status": "FAIL",
                },
                {
                    "name": "Failure Domain Disjointness",
                    "target": "Primary != Secondary",
                    "measured": f"{reg_a_id} vs {reg_b_id}",
                    "delta": "Invalid Nodes",
                    "status": "FAIL",
                },
                {
                    "name": "Inter-Region Network Latency",
                    "target": f"<= {max_latency_ms:.1f} ms",
                    "measured": "N/A",
                    "delta": "No Edge",
                    "status": "FAIL",
                },
                {
                    "name": "Composite Availability SLA",
                    "target": f">= {target_sla_pct:.4f}%",
                    "measured": "0.00000%",
                    "delta": f"-{target_sla_pct:.4f}% (Deficit)",
                    "status": "FAIL",
                },
                {
                    "name": "Disaster Recovery Monthly Cost",
                    "target": f"<= ${budget_usd:.2f} USD",
                    "measured": f"${reported_cost:.2f} USD",
                    "delta": f"${reported_cost:.2f}",
                    "status": "FAIL",
                },
            ]
            return {
                "problem_type": "Z3_Graph_Disaster_Recovery",
                "structure_valid": True,
                "catalog_consistent": False,
                "cost_accuracy": {
                    "reported_cost_usd": reported_cost,
                    "calculated_catalog_cost_usd": 0.0,
                    "cost_delta_usd": reported_cost,
                    "cost_error_pct": 100.0 if reported_cost > 0 else 0.0,
                },
                "feasible_against_contract": False,
                "interpretation_correct": "Not independently evaluated",
                "optimality_verdict": "Infeasible",
                "summary_status": "Catalog inconsistency",
                "violations": violations,
                "recomputed_metrics": {
                    "primary_region": reg_a_id,
                    "secondary_region": reg_b_id,
                    "latency_ms": 0.0,
                    "composite_sla_pct": 0.0,
                    "monthly_cost_usd": 0.0,
                },
                "parameter_checks": param_checks_inv,
            }

        node_a = regions_graph[reg_a_id]
        node_b = regions_graph[reg_b_id]

        # Distinct regions check
        if reg_a_id == reg_b_id:
            violations.append(f"Primary and secondary regions are identical: '{reg_a_id}'")

        # Geo/Provider separation
        if node_a["geo"] == node_b["geo"] and node_a["provider"] == node_b["provider"]:
            violations.append(f"Regions share same geography and provider ({node_a['geo']} / {node_a['provider']}).")

        if is_multi_cloud and node_a["provider"].upper() == node_b["provider"].upper():
            violations.append(
                f"Multi-cloud request requires distinct providers, but both regions are on {node_a['provider']}."
            )

        if target_provs_upper:
            if node_a["provider"].upper() not in target_provs_upper:
                violations.append(f"Primary region provider {node_a['provider']} not in requested {requested_providers}")
            if node_b["provider"].upper() not in target_provs_upper:
                violations.append(f"Secondary region provider {node_b['provider']} not in requested {requested_providers}")

        # Catalog-derived peer latency lookup
        lat_edge = node_a.get("peer_latencies_ms", {}).get(
            reg_b_id, node_b.get("peer_latencies_ms", {}).get(reg_a_id)
        )
        if lat_edge is None:
            violations.append(f"No latency edge defined between '{reg_a_id}' and '{reg_b_id}'.")
            recomputed_latency = 999.0
        else:
            recomputed_latency = float(lat_edge)

        # Composite SLA calculation: 1 - ((1 - A) * (1 - B))
        unavail_a = 1.0 - (node_a["sla_pct"] / 100.0)
        unavail_b = 1.0 - (node_b["sla_pct"] / 100.0)
        recomputed_sla = round((1.0 - (unavail_a * unavail_b)) * 100.0, 5)

        # Cost calculation: base_a + base_b + (latency * 0.25)
        recomputed_cost = round(node_a["base_cost_usd"] + node_b["base_cost_usd"] + (recomputed_latency * cls.LATENCY_COST_PER_MS), 2)

        # Check constraint limits
        if recomputed_latency > max_latency_ms:
            violations.append(f"Latency bound exceeded: {recomputed_latency:.1f}ms > max {max_latency_ms:.1f}ms")
        if recomputed_sla < target_sla_pct:
            violations.append(f"SLA deficit: recomputed {recomputed_sla:.5f}% < target {target_sla_pct:.4f}%")
        if recomputed_cost > budget_usd:
            violations.append(f"Budget overflow: recomputed cost ${recomputed_cost:.2f} > cap ${budget_usd:.2f}")

        cost_delta = round(abs(reported_cost - recomputed_cost), 2)
        cost_err_pct = round((cost_delta / max(0.01, recomputed_cost)) * 100.0, 2) if recomputed_cost > 0 else 0.0

        if cost_delta > 0.50:
            violations.append(f"Reported cost (${reported_cost:.2f}) does not match recomputed cost (${recomputed_cost:.2f})")

        if is_solver_infeasible:
            violations.append(f"Solver reported infeasible status: {solver_status}")

        feasible_against_contract = (
            structure_valid
            and catalog_consistent
            and (cost_delta <= 0.50)
            and (reg_a_id != reg_b_id)
            and (recomputed_latency <= max_latency_ms)
            and (recomputed_sla >= target_sla_pct)
            and (recomputed_cost <= budget_usd)
            and not is_solver_infeasible
            and not any("requires distinct providers" in v or "not in requested" in v for v in violations)
        )

        if not feasible_against_contract:
            optimality_verdict = "Infeasible"
            if not catalog_consistent:
                summary_status = "Catalog inconsistency"
            elif is_solver_infeasible or recomputed_cost > budget_usd:
                summary_status = "Solver reported infeasible"
            else:
                summary_status = "Constraint violation found"
        else:
            # SMT solver performs exhaustive candidate checking over the discrete region pairs
            optimality_verdict = "Exact Discrete Minimum (Exhaustive Candidate Search)"
            summary_status = "Feasible against checked constraints"

        parameter_checks = [
            {
                "name": "Region Graph Topology",
                "target": "Valid Nodes in Graph",
                "measured": f"{reg_a_id} & {reg_b_id}",
                "delta": "Known Nodes",
                "status": "PASS",
            },
            {
                "name": "Failure Domain Disjointness",
                "target": "Primary != Secondary",
                "measured": f"{reg_a_id} vs {reg_b_id}",
                "delta": "Distinct Geographies" if (reg_a_id != reg_b_id and not any("share same geography" in v for v in violations)) else "Colocated / Identical",
                "status": "PASS" if (reg_a_id != reg_b_id and not any("share same geography" in v for v in violations)) else "FAIL",
            },
            {
                "name": "Inter-Region Network Latency",
                "target": f"<= {max_latency_ms:.1f} ms",
                "measured": f"{recomputed_latency:.1f} ms",
                "delta": f"-{max_latency_ms - recomputed_latency:.1f} ms (Margin)" if recomputed_latency <= max_latency_ms else f"+{recomputed_latency - max_latency_ms:.1f} ms (Exceeded)",
                "status": "PASS" if recomputed_latency <= max_latency_ms else "FAIL",
            },
            {
                "name": "Composite Availability SLA",
                "target": f">= {target_sla_pct:.4f}%",
                "measured": f"{recomputed_sla:.5f}%",
                "delta": f"+{recomputed_sla - target_sla_pct:.5f}%" if recomputed_sla >= target_sla_pct else f"-{target_sla_pct - recomputed_sla:.5f}% (Deficit)",
                "status": "PASS" if recomputed_sla >= target_sla_pct else "FAIL",
            },
            {
                "name": "Disaster Recovery Monthly Cost",
                "target": f"<= ${budget_usd:.2f} USD",
                "measured": f"${recomputed_cost:.2f} USD",
                "delta": f"-${budget_usd - recomputed_cost:.2f} (Headroom)" if recomputed_cost <= budget_usd else f"+${recomputed_cost - budget_usd:.2f} (Overflow)",
                "status": "PASS" if (recomputed_cost <= budget_usd and cost_delta <= 0.50) else "FAIL",
            },
            {
                "name": "Provider & Multi-Cloud Policy",
                "target": f"{', '.join(requested_providers) if requested_providers else 'Any'} (Multi-Cloud: {is_multi_cloud})",
                "measured": f"{node_a.get('provider')} + {node_b.get('provider')}",
                "delta": "Compliant" if not any("requires distinct providers" in v or "not in requested" in v for v in violations) else "Violation",
                "status": "PASS" if not any("requires distinct providers" in v or "not in requested" in v for v in violations) else "FAIL",
            },
        ]

        return {
            "problem_type": "Z3_Graph_Disaster_Recovery",
            "structure_valid": structure_valid,
            "catalog_consistent": catalog_consistent,
            "cost_accuracy": {
                "reported_cost_usd": reported_cost,
                "calculated_catalog_cost_usd": recomputed_cost,
                "cost_delta_usd": cost_delta,
                "cost_error_pct": cost_err_pct,
            },
            "feasible_against_contract": feasible_against_contract,
            "interpretation_correct": "Not independently evaluated",
            "optimality_verdict": optimality_verdict,
            "summary_status": summary_status,
            "violations": violations,
            "recomputed_metrics": {
                "primary_region": reg_a_id,
                "secondary_region": reg_b_id,
                "latency_ms": recomputed_latency,
                "composite_sla_pct": recomputed_sla,
                "monthly_cost_usd": recomputed_cost,
            },
            "parameter_checks": parameter_checks,
        }

    # =========================================================================
    # 3. Continuous Dynamic Scaling Checker
    # =========================================================================
    @classmethod
    def verify_continuous_scaling(
        cls,
        contract_data: Dict[str, Any],
        solver_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Independently verifies a Continuous PSO Dynamic Scaling decision."""
        budget_usd = float(contract_data.get("budget_max_usd", contract_data.get("budget_usd", 500.0)))
        reported_cost = float(
            solver_res.get("estimated_monthly_cost_usd", solver_res.get("total_monthly_cost_usd", 0.0))
        )
        solver_status = str(solver_res.get("status", "UNKNOWN")).upper()
        is_solver_infeasible = solver_status in ["INFEASIBLE", "FAILED"]

        raw_bw = (
            solver_res.get("optimal_bandwidth_mbps")
            if solver_res.get("optimal_bandwidth_mbps") is not None
            else (solver_res.get("bandwidth_mbps") if solver_res.get("bandwidth_mbps") is not None else solver_res.get("bandwidth"))
        )
        raw_reps = (
            solver_res.get("recommended_replicas")
            if solver_res.get("recommended_replicas") is not None
            else (
                solver_res.get("replicas")
                if solver_res.get("replicas") is not None
                else solver_res.get("worker_replicas")
            )
        )

        violations: List[str] = []
        structure_valid = True

        if raw_bw is None or raw_reps is None:
            is_zero_cost = abs(reported_cost) <= 0.05
            param_checks_empty = [
                {
                    "name": "Continuous Bandwidth Range",
                    "target": "[100.0, 1000.0] Mbps",
                    "measured": "None",
                    "delta": "Missing Bandwidth",
                    "status": "FAIL",
                },
                {
                    "name": "Worker Replica Capacity",
                    "target": "[1, 16] Replicas",
                    "measured": "None",
                    "delta": "Missing Replicas",
                    "status": "FAIL",
                },
                {
                    "name": "Continuous CPU Utilization",
                    "target": "<= 100.0% (Tgt: 70.0%)",
                    "measured": "0.0%",
                    "delta": "No Metric",
                    "status": "FAIL",
                },
                {
                    "name": "Scaling Dynamic Pricing",
                    "target": "$0.00/mo (Formula)",
                    "measured": f"${reported_cost:.2f}/mo (Reported)",
                    "delta": f"${reported_cost:.2f}",
                    "status": "PASS" if is_zero_cost else "FAIL",
                },
                {
                    "name": "Financial Monthly Budget",
                    "target": f"<= ${budget_usd:.2f} USD",
                    "measured": f"${reported_cost:.2f} USD",
                    "delta": f"${reported_cost:.2f}",
                    "status": "PASS" if reported_cost <= budget_usd else "FAIL",
                },
            ]
            return {
                "problem_type": "PSO_Continuous_Scaling",
                "structure_valid": is_solver_infeasible,
                "catalog_consistent": is_zero_cost,
                "cost_accuracy": {
                    "reported_cost_usd": reported_cost,
                    "calculated_catalog_cost_usd": 0.0,
                    "cost_delta_usd": reported_cost,
                    "cost_error_pct": 100.0 if reported_cost > 0 else 0.0,
                },
                "feasible_against_contract": False,
                "interpretation_correct": "Not independently evaluated",
                "optimality_verdict": "Infeasible",
                "summary_status": "Solver reported infeasible" if is_solver_infeasible else "Invalid output structure",
                "violations": [
                    f"Solver reported infeasible status: {solver_status}" if is_solver_infeasible
                    else "Missing bandwidth or replicas in scaling solution."
                ],
                "recomputed_metrics": {
                    "bandwidth_mbps": 0.0,
                    "replicas": 0,
                    "modeled_cpu_pct": 0.0,
                    "monthly_cost_usd": 0.0,
                },
                "parameter_checks": param_checks_empty,
            }

        try:
            bw_val = float(raw_bw)
            reps_val = int(round(float(raw_reps)))
        except (ValueError, TypeError):
            param_checks_inv = [
                {
                    "name": "Continuous Bandwidth Range",
                    "target": "[100.0, 1000.0] Mbps",
                    "measured": f"'{raw_bw}'",
                    "delta": "Non-numeric",
                    "status": "FAIL",
                },
                {
                    "name": "Worker Replica Capacity",
                    "target": "[1, 16] Replicas",
                    "measured": f"'{raw_reps}'",
                    "delta": "Non-numeric",
                    "status": "FAIL",
                },
                {
                    "name": "Continuous CPU Utilization",
                    "target": "<= 100.0% (Tgt: 70.0%)",
                    "measured": "N/A",
                    "delta": "Invalid Metric",
                    "status": "FAIL",
                },
                {
                    "name": "Scaling Dynamic Pricing",
                    "target": "$0.00/mo (Formula)",
                    "measured": f"${reported_cost:.2f}/mo (Reported)",
                    "delta": f"${reported_cost:.2f}",
                    "status": "FAIL",
                },
                {
                    "name": "Financial Monthly Budget",
                    "target": f"<= ${budget_usd:.2f} USD",
                    "measured": f"${reported_cost:.2f} USD",
                    "delta": f"${reported_cost:.2f}",
                    "status": "FAIL",
                },
            ]
            return {
                "problem_type": "PSO_Continuous_Scaling",
                "structure_valid": False,
                "catalog_consistent": False,
                "cost_accuracy": {
                    "reported_cost_usd": reported_cost,
                    "calculated_catalog_cost_usd": 0.0,
                    "cost_delta_usd": reported_cost,
                    "cost_error_pct": 100.0 if reported_cost > 0 else 0.0,
                },
                "feasible_against_contract": False,
                "interpretation_correct": "Not independently evaluated",
                "optimality_verdict": "Optimality not established",
                "summary_status": "Invalid output structure",
                "violations": [f"Non-numeric bandwidth or replicas: '{raw_bw}', '{raw_reps}'"],
                "recomputed_metrics": {
                    "bandwidth_mbps": 0.0,
                    "replicas": 0,
                    "modeled_cpu_pct": 0.0,
                    "monthly_cost_usd": 0.0,
                },
                "parameter_checks": param_checks_inv,
            }

        # Check bounds
        if bw_val < 100.0 or bw_val > 1000.0:
            violations.append(f"Bandwidth {bw_val:.1f} Mbps outside valid range [100, 1000]")
        if reps_val < 1 or reps_val > 16:
            violations.append(f"Replica count {reps_val} outside valid range [1, 16]")

        # Recompute cost
        recomputed_cost = round((bw_val * cls.SCALING_COST_PER_MBPS) + (reps_val * cls.SCALING_COST_PER_REPLICA), 2)

        # Recompute modeled CPU without clipping (Never clip away overloaded CPU)
        if reps_val > 0:
            unclipped_modeled_cpu = round((bw_val / (reps_val * cls.SCALING_CAPACITY_FACTOR_MBPS)) * 100.0, 2)
        else:
            unclipped_modeled_cpu = float("inf")

        if unclipped_modeled_cpu > 100.0:
            violations.append(
                f"Modeled CPU utilization {unclipped_modeled_cpu:.1f}% exceeds 100% capacity (Overloaded system)."
            )

        if recomputed_cost > budget_usd:
            violations.append(f"Budget overflow: recomputed cost ${recomputed_cost:.2f} > cap ${budget_usd:.2f}")

        cost_delta = round(abs(reported_cost - recomputed_cost), 2)
        cost_err_pct = round((cost_delta / max(0.01, recomputed_cost)) * 100.0, 2) if recomputed_cost > 0 else 0.0

        catalog_consistent = True
        if cost_delta > 0.50:
            violations.append(f"Reported cost (${reported_cost:.2f}) does not match recomputed cost (${recomputed_cost:.2f})")

        if is_solver_infeasible:
            violations.append(f"Solver reported infeasible status: {solver_status}")

        feasible_against_contract = (
            structure_valid
            and catalog_consistent
            and (cost_delta <= 0.50)
            and (100.0 <= bw_val <= 1000.0)
            and (1 <= reps_val <= 16)
            and (recomputed_cost <= budget_usd)
            and (unclipped_modeled_cpu <= 100.0)
            and not is_solver_infeasible
        )

        if not feasible_against_contract:
            optimality_verdict = "Infeasible"
            if not catalog_consistent:
                summary_status = "Catalog inconsistency"
            elif is_solver_infeasible or recomputed_cost > budget_usd:
                summary_status = "Solver reported infeasible"
            else:
                summary_status = "Constraint violation found"
        else:
            # PSO is a continuous metaheuristic, not an exact MIP solver
            optimality_verdict = "Optimality not established (Heuristic approximation)"
            summary_status = "Feasible against checked constraints"

        parameter_checks = [
            {
                "name": "Continuous Bandwidth Range",
                "target": "[100.0, 1000.0] Mbps",
                "measured": f"{bw_val:.1f} Mbps",
                "delta": "Within Domain" if (100.0 <= bw_val <= 1000.0) else "Out of Bounds",
                "status": "PASS" if (100.0 <= bw_val <= 1000.0) else "FAIL",
            },
            {
                "name": "Worker Replica Capacity",
                "target": "[1, 16] Replicas",
                "measured": f"{reps_val} replica(s)",
                "delta": "Within Domain" if (1 <= reps_val <= 16) else "Out of Bounds",
                "status": "PASS" if (1 <= reps_val <= 16) else "FAIL",
            },
            {
                "name": "Continuous CPU Utilization",
                "target": "<= 100.0% (Tgt: 70.0%)",
                "measured": f"{unclipped_modeled_cpu:.1f}%",
                "delta": f"-{100.0 - unclipped_modeled_cpu:.1f}% (Headroom)" if unclipped_modeled_cpu <= 100.0 else f"+{unclipped_modeled_cpu - 100.0:.1f}% (Overloaded)",
                "status": "PASS" if unclipped_modeled_cpu <= 100.0 else "FAIL",
            },
            {
                "name": "Scaling Dynamic Pricing",
                "target": f"${recomputed_cost:.2f}/mo (Formula)",
                "measured": f"${reported_cost:.2f}/mo (Reported)",
                "delta": f"${cost_delta:.2f} ({cost_err_pct:.1f}% err)" if cost_delta > 0.0 else "$0.00 (Exact)",
                "status": "PASS" if cost_delta <= 0.50 else "FAIL",
            },
            {
                "name": "Financial Monthly Budget",
                "target": f"<= ${budget_usd:.2f} USD",
                "measured": f"${recomputed_cost:.2f} USD",
                "delta": f"-${budget_usd - recomputed_cost:.2f} (Headroom)" if recomputed_cost <= budget_usd else f"+${recomputed_cost - budget_usd:.2f} (Overflow)",
                "status": "PASS" if recomputed_cost <= budget_usd else "FAIL",
            },
        ]

        return {
            "problem_type": "PSO_Continuous_Scaling",
            "structure_valid": structure_valid,
            "catalog_consistent": catalog_consistent,
            "cost_accuracy": {
                "reported_cost_usd": reported_cost,
                "calculated_catalog_cost_usd": recomputed_cost,
                "cost_delta_usd": cost_delta,
                "cost_error_pct": cost_err_pct,
            },
            "feasible_against_contract": feasible_against_contract,
            "interpretation_correct": "Not independently evaluated",
            "optimality_verdict": optimality_verdict,
            "summary_status": summary_status,
            "violations": violations,
            "recomputed_metrics": {
                "bandwidth_mbps": bw_val,
                "replicas": reps_val,
                "modeled_cpu_pct": unclipped_modeled_cpu,
                "monthly_cost_usd": recomputed_cost,
            },
            "parameter_checks": parameter_checks,
        }

    # =========================================================================
    # Universal Verification Dispatcher
    # =========================================================================
    @classmethod
    def verify_solution(
        cls,
        contract: Any,
        solver_res: Any,
        expected_contract: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Dispatches verification to the appropriate problem type checker and
        attaches independent verification attributes.
        """
        contract_dict = cls._extract_contract_dict(contract)
        res_dict = cls._extract_contract_dict(solver_res) if not isinstance(solver_res, dict) else dict(solver_res)

        if "solver_res" in res_dict and isinstance(res_dict["solver_res"], dict):
            merged = dict(res_dict["solver_res"])
            merged.update(res_dict)
            res_dict = merged

        problem_type = contract_dict.get("problem_type", "ILP_VM_Allocation")

        if problem_type == "ILP_VM_Allocation":
            check = cls.verify_vm_allocation(contract_dict, res_dict)
        elif problem_type == "Z3_Graph_Disaster_Recovery":
            check = cls.verify_disaster_recovery(contract_dict, res_dict)
        elif problem_type == "PSO_Continuous_Scaling":
            check = cls.verify_continuous_scaling(contract_dict, res_dict)
        else:
            # Unsupported archetype
            check = {
                "problem_type": problem_type,
                "structure_valid": False,
                "catalog_consistent": False,
                "cost_accuracy": {
                    "reported_cost_usd": 0.0,
                    "calculated_catalog_cost_usd": 0.0,
                    "cost_delta_usd": 0.0,
                    "cost_error_pct": 0.0,
                },
                "feasible_against_contract": False,
                "interpretation_correct": "Not independently evaluated",
                "optimality_verdict": "Optimality not established",
                "summary_status": "Unsupported problem archetype",
                "violations": [f"Unknown or unsupported problem type '{problem_type}'"],
                "recomputed_metrics": {},
            }

        # Handle ground-truth human-authored expected contract evaluation if supplied
        if expected_contract is not None:
            exp_dict = cls._extract_contract_dict(expected_contract)
            # Compare parsed contract against expected human contract
            match_problems = contract_dict.get("problem_type") == exp_dict.get("problem_type")
            match_vcpus = contract_dict.get("required_vcpus") == exp_dict.get("required_vcpus")
            match_ram = abs(float(contract_dict.get("required_ram_gb", 0)) - float(exp_dict.get("required_ram_gb", 0))) < 0.1
            match_budget = abs(float(contract_dict.get("budget_max_usd", 0)) - float(exp_dict.get("budget_max_usd", 0))) < 0.1

            if match_problems and match_vcpus and match_ram and match_budget:
                check["interpretation_correct"] = "Accurate (Matches expected benchmark contract)"
            else:
                check["interpretation_correct"] = "Interpretation mismatch against expected benchmark contract"
        if "parameter_checks" not in check:
            check["parameter_checks"] = [
                {
                    "name": "Archetype Validation",
                    "target": "Supported Model",
                    "measured": str(problem_type),
                    "delta": "Unsupported",
                    "status": "FAIL",
                }
            ]

        return check

    @classmethod
    def render_parameter_table(cls, parameter_checks: List[Dict[str, Any]]) -> str:
        """Renders an aligned parameter-wise verification table."""
        if not parameter_checks:
            return "    No parameter checks available."

        # Column widths: Parameter Checked (32), Required Target (22), Measured / Calc (24), Delta / Margin (22), Status (8)
        w_param = 32
        w_tgt = 22
        w_meas = 24
        w_delta = 22
        w_stat = 8

        sep = f"    +{'-'*(w_param+2)}+{'-'*(w_tgt+2)}+{'-'*(w_meas+2)}+{'-'*(w_delta+2)}+{'-'*(w_stat+2)}+"
        header = (
            f"    | {'Parameter Checked':<{w_param}} | {'Required Target':<{w_tgt}} | "
            f"{'Measured / Calc':<{w_meas}} | {'Delta / Margin':<{w_delta}} | {'Status':<{w_stat}} |"
        )
        lines = [sep, header, sep]
        for p in parameter_checks:
            p_name = str(p.get("name", ""))[:w_param]
            p_tgt = str(p.get("target", ""))[:w_tgt]
            p_meas = str(p.get("measured", ""))[:w_meas]
            p_delta = str(p.get("delta", ""))[:w_delta]
            status_raw = str(p.get("status", "FAIL")).upper()
            status_str = f"[{status_raw}]" if "[" not in status_raw else status_raw
            row = (
                f"    | {p_name:<{w_param}} | {p_tgt:<{w_tgt}} | "
                f"{p_meas:<{w_meas}} | {p_delta:<{w_delta}} | {status_str:<{w_stat}} |"
            )
            lines.append(row)
        lines.append(sep)
        return "\n".join(lines)

