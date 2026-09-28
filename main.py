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
    args = parser.parse_args()

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
