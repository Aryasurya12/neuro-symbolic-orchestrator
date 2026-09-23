"""Main entrypoint and interactive CLI driver for neurasym (FinOps Neuro-Symbolic Orchestrator)."""

import sys

# Ensure UTF-8 stdout encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.orchestrator.service import NeuroSymbolicOrchestrator

def main() -> None:
    orchestrator = NeuroSymbolicOrchestrator()

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
            report = orchestrator.process_query(user_query)
            print("\n" + report)

        except Exception as e:
            print(f"❌ Error processing query: {e}")


if __name__ == "__main__":
    main()
