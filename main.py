"""Main entrypoint and interactive CLI driver for neurasym (FinOps Neuro-Symbolic Orchestrator & 4-Way Analysis)."""

import argparse
import sys

# Ensure UTF-8 stdout encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from benchmarks.run_4way_benchmark import FourWayBenchmarker, print_comparison_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="neurasym: FinOps Neuro-Symbolic Orchestrator & 4-Way Comparative Analysis"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Optional dynamic cloud request query for 4-way comparative analysis.",
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run 4-way comparative benchmark on representative enterprise workloads.",
    )
    parser.add_argument(
        "--diagnose", "-d",
        action="store_true",
        help="Run diagnostic test harness on LLM modes and Neurasym orchestrator.",
    )
    parser.add_argument(
        "--sage-gnn", "-s",
        action="store_true",
        help="Run SAGE-GNN (IJCNN 2024) MaxSMT graph placement benchmark across all 3 cloud topologies.",
    )
    args = parser.parse_args()

    # SAGE-GNN Benchmark mode
    if args.sage_gnn:
        from src.optimizers.z3_smt_solver import run_all_sage_gnn_benchmarks
        print("\n🚀 Executing SAGE-GNN (IJCNN 2024) MaxSMT Benchmark Suite across 3 Cloud Topologies...\n")
        sage_results = run_all_sage_gnn_benchmarks()
        print("============================================================================================")
        print("🌐 SAGE-GNN (IJCNN 2024) Graph Neural Network MaxSMT Benchmark Evaluation")
        print("============================================================================================")
        for name, res in sage_results.items():
            status_icon = "✅" if res["is_feasible"] else "❌"
            print(f"\n📌 Topology: {name} {status_icon}")
            print(f"   Description: {res.get('description', '')}")
            print(f"   Status: {res['status']} | Solver: {res['solver']} | Runtime: {res['runtime_ms']:.2f} ms")
            print(f"   Monthly Spend: ${res['total_monthly_cost_usd']:,.2f} / ${res['budget_max_usd']:,.2f} Budget ({res['budget_utilization_pct']:.1f}% utilized)")
            print(f"   Savings: ${res['cost_savings_usd']:,.2f} | Max Latency: {res['max_inter_component_latency_ms']:.1f}ms (< {res['latency_sla_threshold_ms']:.0f}ms SLA)")
            print(f"   Neural Alignment: {res['gnn_alignment_pct']:.1f}% GNN edge predictions satisfied")
            print("   Component Placements:")
            for c, p in res.get("component_placements", {}).items():
                print(f"     • {c:18s} -> {p['vm_id']:22s} [{p['provider']}:{p['region']}] (GNN Prob: {p['gnn_prediction_prob']:.2f}, {p['required_vcpus']} vCPU / {p['required_ram_gb']:.0f}GB RAM)")
        print("\n============================================================================================\n")
        return

    # Diagnostics mode
    if args.diagnose:
        from diagnose_llm_modes import main as run_diagnostics
        run_diagnostics()
        return

    benchmarker = FourWayBenchmarker()

    # Benchmark mode
    if args.benchmark:
        default_query = (
            args.query or "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
        )
        print(f"\n🚀 Running 4-Way Comparative Benchmark for query: \"{default_query}\"\n")
        results = benchmarker.run_all(default_query)
        print_comparison_table(results, default_query)
        return

    # Direct query mode
    if args.query:
        print(f"\n🚀 Running 4-Way Comparative Benchmark for query: \"{args.query}\"\n")
        results = benchmarker.run_all(args.query)
        print_comparison_table(results, args.query)
        return

    # Interactive CLI Mode
    print("============================================================================================")
    print("🚀 neurasym FinOps Orchestrator & 4-Way Comparative Analysis (Interactive Mode)")
    print("Type 'exit' or 'quit' to terminate.")
    print("============================================================================================\n")

    while True:
        try:
            user_query = input("\n💬 Enter Cloud Request for 4-Way Analysis: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Orchestrator. Bye!")
            break

        if user_query.strip().lower() in ["exit", "quit"]:
            print("Exiting Orchestrator. Bye!")
            break

        if not user_query.strip():
            continue

        try:
            results = benchmarker.run_all(user_query)
            print_comparison_table(results, user_query)

        except Exception as e:
            print(f"❌ Error running 4-way comparative analysis: {e}")


if __name__ == "__main__":
    main()
