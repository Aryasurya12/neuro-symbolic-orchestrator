import pytest
from fastapi.testclient import TestClient
from src.api.app import app

def test_websocket_flow():
    client = TestClient(app)
    
    payload = {
        "problem_type": "Test",
        "cloud_providers": ["AWS", "Azure"],
        "budget_max_usd": 300.0,
        "service_count": 1,
        "required_vcpus": 4,
        "required_ram_gb": 16.0,
        "latency_max_ms": 50.0,
        "sla_availability_pct": 99.0
    }
    
    with client.websocket_connect("/ws/optimize") as websocket:
        websocket.send_json(payload)
        
        # 1. Started event
        data = websocket.receive_json()
        assert data["event"] == "started"
        assert "run_id" in data
        run_id = data["run_id"]
        
        events_received = []
        
        # Collect events until completed or error
        while True:
            data = websocket.receive_json()
            events_received.append(data["event"])
            assert data["run_id"] == run_id
            
            if data["event"] in ("completed", "error"):
                if data["event"] == "completed":
                    assert "result" in data
                    assert "is_feasible" in data["result"]
                break
                
        # We should have seen progress and solver events
        assert "progress" in events_received
        assert "solver_started" in events_received
        assert "solver_completed" in events_received

def test_websocket_concurrency():
    """
    Test that two concurrent websocket connections don't interfere.
    Since TestClient's websocket is somewhat synchronous, we test them sequentially here
    but rely on the backend async isolation.
    """
    client = TestClient(app)
    
    payload1 = {
        "problem_type": "Test",
        "cloud_providers": ["AWS"],
        "budget_max_usd": 1000.0,
        "service_count": 1,
        "required_vcpus": 2,
        "required_ram_gb": 4.0,
        "latency_max_ms": 50.0,
        "sla_availability_pct": 99.0
    }
    
    payload2 = {
        "problem_type": "Test",
        "cloud_providers": ["Azure"],
        "budget_max_usd": 500.0,
        "service_count": 1,
        "required_vcpus": 4,
        "required_ram_gb": 8.0,
        "latency_max_ms": 50.0,
        "sla_availability_pct": 99.0
    }
    
    # Normally this would be asyncio.gather, but starlette's TestClient websocket is synchronous.
    # The isolation is guaranteed by `run_id` and service thread scoping.
    with client.websocket_connect("/ws/optimize") as ws1, client.websocket_connect("/ws/optimize") as ws2:
        ws1.send_json(payload1)
        ws2.send_json(payload2)
        
        started1 = ws1.receive_json()
        started2 = ws2.receive_json()
        
        assert started1["run_id"] != started2["run_id"]
        
        # drain ws1
        while True:
            data = ws1.receive_json()
            if data["event"] == "completed":
                break
                
        # drain ws2
        while True:
            data = ws2.receive_json()
            if data["event"] == "completed":
                break
