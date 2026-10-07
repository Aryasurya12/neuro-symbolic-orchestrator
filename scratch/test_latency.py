import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

from src.optimizers.raw_symbolic_runner import run_symbolic_rule_based
from app import execute_live_pipeline

query = "128 vCPUs / 512GB RAM / 4 H100 GPUs / 99.999% SLA / under $50 budget"

print("=" * 60)
print(f"QUERY: {query}")
print("=" * 60)

print("\n--- Testing Mode 3 (run_symbolic_rule_based) 3 times ---")
for i in range(1, 4):
    t0 = time.perf_counter()
    m3_res = run_symbolic_rule_based(query)
    t_total = (time.perf_counter() - t0) * 1000.0
    print(f"Run {i}: total={t_total:.2f}ms | reported_latency={m3_res.get('latency_ms', 0.0):.2f}ms | solve_latency={m3_res.get('solve_latency_ms', 0.0):.2f}ms | parse_latency={m3_res.get('parse_latency_ms', 0.0):.2f}ms | status={m3_res.get('status')}")

print("\n--- Testing Mode 4 (execute_live_pipeline) 3 times ---")
for i in range(1, 4):
    t0 = time.perf_counter()
    m4_res = execute_live_pipeline(query)
    t_total = (time.perf_counter() - t0) * 1000.0
    print(f"Run {i}: total={t_total:.2f}ms | total_pipeline_ms={m4_res.get('total_latency_ms', 0.0):.2f}ms | solve_latency={m4_res.get('solve_latency_ms', 0.0):.2f}ms | parse_latency={m4_res.get('parse_latency_ms', 0.0):.2f}ms | explanation_latency={m4_res.get('explanation_latency_ms', 0.0):.2f}ms")
