import sys
import os

sys.path.insert(0, os.path.abspath("."))

from app import execute_live_pipeline

query = "128 vCPUs / 512GB RAM / 4 H100 GPUs / 99.999% SLA / under $50 budget"

print("--- Running execute_live_pipeline ---")
res = execute_live_pipeline(query)

print("problem_type:", res["problem_type"])
print("budget_usd:", res["budget_usd"])
print("optimal_cost_usd:", res["optimal_cost_usd"])
print("savings_usd:", res["savings_usd"])
print("savings_pct:", res["savings_pct"])
print("allocations:", res["allocations"])
print("explanation:", res["explanation"])
print("m4_check:", res["m4_check"])
