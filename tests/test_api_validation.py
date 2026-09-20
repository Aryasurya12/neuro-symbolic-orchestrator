import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_optimize_missing_fields():
    # Missing required_vcpus and latency_max_ms
    payload = {
        "problem_type": "ILP_VM_Allocation",
        "cloud_providers": ["AWS"],
        "budget_max_usd": 300.0,
        "service_count": 1,
        "required_ram_gb": 16.0,
        "sla_availability_pct": 99.0
    }
    
    response = client.post("/optimize", json=payload)
    
    assert response.status_code == 422
    data = response.json()
    # Pydantic 422 returns an array of details
    assert "detail" in data
    assert any(err["loc"] == ["body", "required_vcpus"] for err in data["detail"])

def test_optimize_infeasible_request():
    # Impossible budget
    payload = {
        "problem_type": "Test",
        "cloud_providers": ["AWS"],
        "budget_max_usd": 1.0,
        "service_count": 1,
        "required_vcpus": 128,
        "required_ram_gb": 512.0,
        "latency_max_ms": 1.0,
        "sla_availability_pct": 99.9
    }
    
    response = client.post("/optimize", json=payload)
    
    # Should be 200 OK, but return is_feasible: false
    assert response.status_code == 200
    data = response.json()
    assert not data["is_feasible"]
    assert data["best_candidate"] is None
