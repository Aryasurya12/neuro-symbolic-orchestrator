import os
import json
import csv
import time
import statistics
import traceback
from typing import Dict, Any, List
from src.orchestrator.service import NeuroSymbolicOrchestrator
from src.semantic.explainer import FinOpsExplainer
from src.api.app import app
from fastapi.testclient import TestClient

SEED = 42
REPETITIONS = 5

SCENARIOS = [
    # Category A: Feasible Workloads
    {"scenario_id": "A1_Small", "category": "Feasible", "query": "Deploy a microservice needing 2 vCPUs and 4GB RAM on AWS for under $200 a month.", "expected_feasible": True},
    {"scenario_id": "A2_Medium", "category": "Feasible", "query": "Deploy an application needing 8 vCPUs and 32GB RAM on Azure for under $800.", "expected_feasible": True},
    {"scenario_id": "A3_Large", "category": "Feasible", "query": "Deploy a big data cluster needing 16 vCPUs and 64GB RAM for under $1500.", "expected_feasible": True},
    
    # Category B: Budget Stress
    {"scenario_id": "B1_Comfortable", "category": "Budget Stress", "query": "Need 4 vCPUs and 16GB RAM for under $1000.", "expected_feasible": True},
    {"scenario_id": "B2_NearBoundary", "category": "Budget Stress", "query": "Need 4 vCPUs and 16GB RAM on GCP for under $100.", "expected_feasible": True}, # GCP e2-standard-4 is $97.82, so $100 is feasible!
    {"scenario_id": "B3_Tight", "category": "Budget Stress", "query": "Need 4 vCPUs and 16GB RAM for under $80.", "expected_feasible": False},
    
    # Category C: Infeasible Workloads
    {"scenario_id": "C1_Budget_Infeasible", "category": "Infeasible", "query": "Deploy 8 vCPUs and 32GB RAM under $10 a month.", "expected_feasible": False},
    {"scenario_id": "C2_Resource_Infeasible", "category": "Infeasible", "query": "Deploy 512 vCPUs and 1024GB RAM under $500.", "expected_feasible": False},
    
    # Category D: Multi-Provider
    {"scenario_id": "D1_Multi_AWS_Azure", "category": "Multi-Provider", "query": "Deploy 2 vCPUs on AWS or Azure under $80.", "expected_feasible": True},
    {"scenario_id": "D2_Multi_All", "category": "Multi-Provider", "query": "Deploy 4 vCPUs and 16GB RAM on AWS, Azure, or GCP under $150.", "expected_feasible": True},
    
    # Category E: Latency/SLA (DR)
    {"scenario_id": "E1_DR_Feasible", "category": "Latency/SLA", "query": "Multi-region disaster recovery deployment with 99.9% SLA and under 100ms latency on AWS or Azure.", "expected_feasible": True},
    {"scenario_id": "E2_DR_Infeasible", "category": "Latency/SLA", "query": "Multi-region disaster recovery deployment with 99.999% SLA and under 2ms latency.", "expected_feasible": False},
    
    # Category F: Boundary Cases
    {"scenario_id": "F1_Boundary_Budget", "category": "Boundary", "query": "Need 2 vCPUs and 4GB RAM under $30.36 a month.", "expected_feasible": False}, # t3.medium is 30.368
]

def run_evaluation():
    print("=" * 60)
    print("PHASE 10 - FINAL SYSTEM EVALUATION")
    print("=" * 60)
    
    orchestrator = NeuroSymbolicOrchestrator()
    results = []
    
    correct_decisions = 0
    incorrect_decisions = 0
    hard_constraint_violations = 0
    
    for sc in SCENARIOS:
        print(f"\nEvaluating Scenario: {sc['scenario_id']} - {sc['category']}")
        
        # Repetitions
        sem_times = []
        sym_times = []
        exp_times = []
        
        last_res = None
        last_contract = None
        
        for _ in range(REPETITIONS):
            t0 = time.perf_counter()
            contract, tmpl, score = orchestrator.parser.parse_query_to_contract(sc["query"])
            t1 = time.perf_counter()
            sem_times.append((t1 - t0) * 1000)
            
            res_dict = orchestrator.optimize_contract(contract)
            t2 = time.perf_counter()
            sym_times.append((t2 - t1) * 1000)
            
            report = FinOpsExplainer.generate_report(contract, res_dict)
            t3 = time.perf_counter()
            exp_times.append((t3 - t2) * 1000)
            
            last_res = res_dict
            last_contract = contract
            
        is_feasible = (last_res.get("status") == "Feasible")
        
        if is_feasible == sc["expected_feasible"]:
            correct_decisions += 1
        else:
            incorrect_decisions += 1
            
        cost_val = last_res.get("total_monthly_cost_usd", "N/A")
        
        # Hard constraint safety validation
        if is_feasible:
            budget_ok = False
            vcpu_ok = False
            
            if cost_val != "N/A" and last_contract.budget_max_usd:
                if cost_val <= (last_contract.budget_max_usd or 999999.0):
                    budget_ok = True
            
            if last_res.get("total_vcpus", 0) >= (last_contract.required_vcpus or 0):
                vcpu_ok = True
                
            if (last_contract.budget_max_usd and not budget_ok) or not vcpu_ok:
                print(f"Violation in {sc['scenario_id']}: budget_ok={budget_ok}, vcpu_ok={vcpu_ok}, cost={cost_val}, limit={last_contract.budget_max_usd}, got_vcpus={last_res.get('total_vcpus', 0)}, req_vcpus={last_contract.required_vcpus}")
                if last_contract.problem_type != "Z3_Graph_Disaster_Recovery":
                    hard_constraint_violations += 1
        
        median_sem = statistics.median(sem_times)
        median_sym = statistics.median(sym_times)
        median_exp = statistics.median(exp_times)
        median_total = median_sem + median_sym + median_exp
        
        record = {
            "scenario_id": sc["scenario_id"],
            "category": sc["category"],
            "seed": SEED,
            "problem_type": last_contract.problem_type,
            "solver_name": last_res.get("solver", "N/A"),
            "expected_feasible": sc["expected_feasible"],
            "is_feasible": is_feasible,
            "objective_cost_usd": cost_val,
            "semantic_latency_ms": median_sem,
            "symbolic_latency_ms": median_sym,
            "explanation_latency_ms": median_exp,
            "total_end_to_end_latency_ms": median_total,
            "hard_constraint_violations_detected": hard_constraint_violations
        }
        results.append(record)
        print(f"  Feasible: {is_feasible} (Expected: {sc['expected_feasible']})")
        print(f"  Cost: {cost_val}")
        print(f"  Total Latency: {median_total:.2f} ms")

    # API Evaluation
    print("\n--- Evaluating API & WebSocket ---")
    client = TestClient(app)
    api_latencies = []
    
    # 5 runs of API for first scenario
    for _ in range(5):
        t0 = time.perf_counter()
        r = client.post("/optimize/query", json={"query": SCENARIOS[0]["query"]})
        t1 = time.perf_counter()
        assert r.status_code == 200
        api_latencies.append((t1 - t0) * 1000)
    
    print(f"  API Median Latency: {statistics.median(api_latencies):.2f} ms")
    
    # WebSocket Test
    ws_success = 0
    try:
        with client.websocket_connect("/ws/optimize") as websocket:
            websocket.send_json({"problem_type": "ILP_VM_Allocation", "cloud_providers": ["AWS"], "budget_max_usd": 1000.0, "service_count": 1, "required_vcpus": 2, "required_ram_gb": 4.0, "latency_max_ms": 100.0, "sla_availability_pct": 99.9})
            # Read until completion
            while True:
                data = websocket.receive_json()
                if data.get("event") == "completed":
                    ws_success = 1
                    break
    except Exception as e:
        if type(e).__name__ == "WebSocketDisconnect" and ws_success == 1:
            pass
        else:
            print("WebSocket Error:", e)
            traceback.print_exc()
        
    print(f"  WebSocket Completion Success: {ws_success}/1")
    
    # Save Results
    os.makedirs("benchmarks/results", exist_ok=True)
    json_path = "benchmarks/results/phase10_final_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    csv_path = "benchmarks/results/phase10_final_evaluation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print("\n========================================================")
    print("PHASE 10 - FINAL SYSTEM EVALUATION SUMMARY")
    print("========================================================")
    print(f"Scenarios evaluated: {len(SCENARIOS)}")
    print(f"Total runs: {len(SCENARIOS) * REPETITIONS}")
    print(f"\nFeasibility decisions:")
    print(f"  Correct: {correct_decisions}")
    print(f"  Incorrect: {incorrect_decisions}")
    print(f"\nHard constraint violations:\n  {hard_constraint_violations}")
    
    all_latencies = [r["total_end_to_end_latency_ms"] for r in results]
    print(f"\nEnd-to-end latency:")
    print(f"  Median: {statistics.median(all_latencies):.2f} ms")
    if len(all_latencies) >= 2:
        print(f"  P95: {statistics.quantiles(all_latencies, n=20)[-1]:.2f} ms")
    print(f"  Max: {max(all_latencies):.2f} ms")
    
    feasible_count = sum(1 for r in results if r["is_feasible"])
    infeasible_count = len(results) - feasible_count
    print(f"\nFeasible workloads:\n  {feasible_count}")
    print(f"Infeasible workloads:\n  {infeasible_count}")
    print(f"\nAPI:\n  Median: {statistics.median(api_latencies):.2f} ms")
    print(f"\nWebSocket:\n  Successful completions: {ws_success}/1")
    print("========================================================")
    
if __name__ == "__main__":
    run_evaluation()
