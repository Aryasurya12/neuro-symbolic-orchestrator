from src.semantic.schemas import CloudOptimizationContract
from .models import SymbolicOptimizationRequest

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
