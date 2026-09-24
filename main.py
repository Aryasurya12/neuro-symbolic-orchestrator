"""Main entrypoint and interactive CLI driver for neurasym (FinOps Neuro-Symbolic Orchestrator)."""

import sys

# Ensure UTF-8 stdout encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.semantic.scope_parser import parse_fallback_nemotron
from src.semantic.explainer import FinOpsExplainer
from templates.ILP_VM_Knapsack_Allocation import solve_ilp_vm_knapsack
from templates.Continuous_PSO_Dynamic_Scaling import solve_pso_continuous_scaling
from templates.Graph_SMT_Z3_MultiRegion_Placement import solve_z3_graph_disaster_recovery


def main() -> None:
    print("============================================================")
    print("🚀 neurasym FinOps Orchestrator (Interactive CLI Mode)")
    print("Type 'exit' or 'quit' to terminate.")
    print("============================================================\n")

    while True:
        try:
            user_query = input("\n💬 Enter Cloud Request: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Orchestrator. Bye!")
            break

        if user_query.strip().lower() in ["exit", "quit"]:
            print("Exiting Orchestrator. Bye!")
            break

        if not user_query.strip():
            continue

        try:
            # Forced OpenRouter LLM Execution
            print("🧠 [FORCED LLM TEST] Running via OpenRouter NVIDIA Nemotron...")
            contract = parse_fallback_nemotron(user_query)

            # 2. Dynamic Routing to Matched Solver Template
            if contract.problem_type == "PSO_Continuous_Scaling":
                solver_res = solve_pso_continuous_scaling(
                    bandwidth_min_mbps=100.0,
                    bandwidth_max_mbps=1000.0,
                    target_cpu_pct=70.0,
                    budget_max_usd=contract.budget_max_usd,
                )
            elif contract.problem_type == "Z3_Graph_Disaster_Recovery":
                solver_res = solve_z3_graph_disaster_recovery(
                    sla_pct=contract.sla_availability_pct,
                    max_latency_ms=contract.latency_max_ms,
                    budget_max_usd=contract.budget_max_usd,
                    target_providers=contract.cloud_providers,
                )
            else:
                # Default: ILP VM Knapsack Allocation
                solver_res = solve_ilp_vm_knapsack(
                    required_vcpus=contract.required_vcpus,
                    required_ram_gb=contract.required_ram_gb,
                    budget_max_usd=contract.budget_max_usd,
                    target_providers=contract.cloud_providers,
                )


            # 3. Generate Stage 6 Executive Dual-Currency Report ($ & ₹)
            report = FinOpsExplainer.generate_report(contract, solver_res)
            print("\n" + report)

        except Exception as e:
            print(f"❌ Error processing query: {e}")


if __name__ == "__main__":
    main()
