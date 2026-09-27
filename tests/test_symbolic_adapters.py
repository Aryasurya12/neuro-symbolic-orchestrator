import pytest
from src.semantic.schemas import CloudOptimizationContract
from src.symbolic.adapters import from_contract

def test_from_contract():
    contract = CloudOptimizationContract(
        problem_type="ILP_VM_Allocation",
        cloud_providers=["AWS", "GCP"],
        budget_max_usd=500.0,
        service_count=3,
        required_vcpus=8,
        required_ram_gb=16.0,
        latency_max_ms=20.0,
        sla_availability_pct=99.95
    )
    req = from_contract(contract)
    assert req.problem_type == "ILP_VM_Allocation"
    assert req.cloud_providers == ["AWS", "GCP"]
    assert req.budget_max_usd == 500.0
    assert req.required_vcpus == 8
    assert req.required_ram_gb == 16.0
    assert req.latency_max_ms == 20.0
    assert req.sla_availability_pct == 99.95
