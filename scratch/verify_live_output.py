import sys
import os

sys.path.insert(0, os.path.abspath("."))

from app import execute_live_pipeline
from src.optimizers.raw_symbolic_runner import run_symbolic_rule_based
from src.verifiers.independent_checker import IndependentChecker

query = "128 vCPUs / 512GB RAM / 4 H100 GPUs / 99.999% SLA / under $50 budget"

print("=================================================================")
print("LIVE EXECUTION VERIFICATION ON DELIBERATELY INFEASIBLE QUERY")
print(f"Query: {query}")
print("=================================================================")

res = execute_live_pipeline(query)

print("\n[PIPELINE OUTPUT SUMMARY]")
print(f"Problem Type:        {res['problem_type']}")
print(f"Budget:              ${res['budget_usd']:.2f}")
print(f"Optimal Cost:        ${res['optimal_cost_usd']:.2f}")
print(f"Is Feasible Flag:    {res['is_feasible']}")
print(f"Allocations List:    {res['allocations']}")
print(f"Allocations Count:   {len(res['allocations'])}")
print(f"Explanation:         {res['explanation']}")
print(f"Solver Engine:       {res['solver_engine']}")
print(f"Solve Latency:       {res['solve_latency_ms']:.2f}ms")
print(f"Total Pipeline Lat:  {res['total_latency_ms']:.2f}ms")

print("\n[INDEPENDENT CHECKER VERIFICATION (MODE 4)]")
m4_check = res["m4_check"]
print(f"Summary Status:      {m4_check['summary_status']}")
print(f"Feasible:            {m4_check['feasible_against_contract']}")
print(f"Optimality Verdict:  {m4_check['optimality_verdict']}")
print(f"Violations Count:    {len(m4_check['violations'])}")
for i, v in enumerate(m4_check['violations'], 1):
    print(f"  Violation {i}:      {v}")

print("\n[MODE 3 EXECUTION]")
m3_res = res["m3_result"]
print(f"Mode 3 Status:       {m3_res.get('status')}")
print(f"Mode 3 Latency:      {m3_res.get('latency_ms'):.2f}ms")
print(f"Mode 3 Solve Lat:    {m3_res.get('solve_latency_ms'):.2f}ms")
print(f"Mode 3 Parse Lat:    {m3_res.get('parse_latency_ms'):.2f}ms")

print("\n[4-WAY BENCHMARK DATA (MODE 3 & MODE 4)]")
for mode in res["bench_data"]:
    if "Mode 3" in mode["mode"] or "Mode 4" in mode["mode"]:
        print(f"--- {mode['mode']} ---")
        print(f"  Reported Cost:     {mode['reported_cost']}")
        print(f"  Math Status:       {mode['math']}")
        print(f"  Latency:           {mode['latency']}")
        print(f"  Violations:        {mode['violations']}")
        print(f"  Is Feasible:       {mode['is_feasible']}")

print("\nVERIFICATION COMPLETE.")
