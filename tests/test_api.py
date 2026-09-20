import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_optimize_valid_request():
    payload = {
        "problem_type": "ILP_VM_Allocation",
        "cloud_providers": ["AWS", "Azure"],
        "budget_max_usd": 300.0,
        "service_count": 1,
        "required_vcpus": 4,
        "required_ram_gb": 16.0,
        "latency_max_ms": 50.0,
        "sla_availability_pct": 99.0
    }
    
    response = client.post("/optimize", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "is_feasible" in data
    assert "solver_name" in data
    assert "metrics" in data
    assert "runtime_ms" in data["metrics"]
    
    if data["is_feasible"]:
        assert data["best_candidate"] is not None
        assert "decision_variables" in data["best_candidate"]
        assert "objective_cost_usd" in data["best_candidate"]
