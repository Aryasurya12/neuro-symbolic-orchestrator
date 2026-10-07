import time
from app import execute_live_pipeline

query = '128 vCPUs / 512GB RAM / 4 H100 GPUs / 99.999% SLA / under $50 budget'

for i in range(3):
    res = execute_live_pipeline(query)
    m3 = res['m3_result']
    print(f"=== Run {i+1} ===")
    print(f"Mode 3 total latency_ms: {m3.get('latency_ms', 0.0):.2f} (parse: {m3.get('parse_latency_ms', 0.0):.2f}ms, solve: {m3.get('solve_latency_ms', 0.0):.2f}ms)")
    print(f"Mode 4 total latency_ms: {res.get('total_latency_ms', 0.0):.2f} (parse: {res.get('parse_latency_ms', 0.0):.2f}ms, solve: {res.get('solve_latency_ms', 0.0):.2f}ms, explain: {res.get('explanation_latency_ms', 0.0):.2f}ms)")
    print(f"Allocations count: {len(res.get('allocations', []))}")
