from src.semantic.schemas import CloudOptimizationContract
from .models import SymbolicOptimizationRequest, OptimizationResult

def from_contract(contract: CloudOptimizationContract) -> SymbolicOptimizationRequest:
    """
    Adapter to convert the semantic CloudOptimizationContract into a SymbolicOptimizationRequest.
    This preserves the Part-A integration boundary.
    """
    return SymbolicOptimizationRequest(
        problem_type=contract.problem_type,
        cloud_providers=contract.cloud_providers,
        budget_max_usd=contract.budget_max_usd,
        service_count=contract.service_count,
        required_vcpus=contract.required_vcpus,
        required_ram_gb=contract.required_ram_gb,
        latency_max_ms=contract.latency_max_ms,
        sla_availability_pct=contract.sla_availability_pct,
    )

def to_explainer_dict(result: OptimizationResult, contract: CloudOptimizationContract) -> dict:
    """
    Adapter to convert the Symbolic OptimizationResult into the unstructured 
    dictionary format expected by the FinOpsExplainer.
    """
    if not result.is_feasible or not result.best_candidate:
        out = {
            "status": "Infeasible",
            "solver": result.solver_name,
            "error_message": result.error_message
        }
        if result.best_candidate and result.best_candidate.constraint_status:
            out["constraint_status"] = result.best_candidate.constraint_status.model_dump()
        return out

    cand = result.best_candidate
    decision_vars = cand.decision_variables
    problem = contract.problem_type

    res = {
        "status": "Feasible",
        "solver": result.solver_name,
        "total_monthly_cost_usd": cand.objective_cost_usd,
    }

    if problem == "ILP_VM_Allocation":
        # ILP decision variables: {"AWS-t3.medium": 2, ...}
        from src.symbolic.data.repository import CloudDataRepository
        repo = CloudDataRepository()
        catalog = repo.get_skus()
        
        allocated_vms = []
        total_vcpus = 0
        total_ram = 0.0
        
        for sku_name, count in decision_vars.items():
            if count > 0:
                # Find SKU in catalog
                sku = next((s for s in catalog if s.id == sku_name), None)
                if sku:
                    monthly_cost = sku.hourly_price_usd * 730 * count
                    allocated_vms.append({
                        "provider": sku.provider_id,
                        "instance_type": sku.sku,
                        "count": count,
                        "vcpus_per_vm": sku.vcpus,
                        "ram_gb_per_vm": sku.ram_gb,
                        "monthly_cost": monthly_cost
                    })
                    total_vcpus += sku.vcpus * count
                    total_ram += sku.ram_gb * count
        
        res["allocated_vms"] = allocated_vms
        res["total_vcpus"] = total_vcpus
        res["total_ram_gb"] = total_ram

    elif problem == "PSO_Continuous_Scaling":
        bw = decision_vars.get("bandwidth_mbps", 0.0)
        reps = decision_vars.get("replicas", 1.0)
        # Reconstruct target CPU (from objective logic)
        simulated_cpu = min(max((bw / (reps * 75.0)) * 100.0, 10.0), 99.0) if reps > 0 else 0.0
        
        res["optimal_bandwidth_mbps"] = bw
        res["recommended_replicas"] = round(reps)
        res["target_cpu_utilization_pct"] = simulated_cpu
        res["estimated_hourly_cost_usd"] = cand.objective_cost_usd / 730.0

    elif problem == "Z3_Graph_Disaster_Recovery":
        res["primary_region"] = decision_vars.get("primary_region", "N/A")
        res["secondary_region"] = decision_vars.get("secondary_region", "N/A")
        res["inter_region_latency_ms"] = decision_vars.get("latency_ms", 0.0)
        res["achieved_sla_pct"] = decision_vars.get("achieved_sla", 99.99)
        res["disaster_recovery_topology"] = "Active-Active Mesh"

    return res
