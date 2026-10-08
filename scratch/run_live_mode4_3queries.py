"""Live execution script for 3 queries in Mode 4 via Groq."""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from config.settings import settings
from src.proof.stage_trace import run_mode4_pipeline_trace

QUERIES = [
    "Deploy 8 vCPUs and 16GB RAM for under $300 on AWS",
    "Dynamic scaling for 300 Mbps peak workload with target CPU 70% under $1500",
    "We need disaster recovery setup across AWS and GCP with 99.99% SLA and max 50ms latency under $600/month",
]

def main():
    print("=" * 80)
    print("NEURASYM LIVE MODE 4 EXECUTION (GROQ NEURAL INTERPRETATION)")
    provider, api_key, base_url, model, timeout = settings.get_mode4_provider_config()
    print(f"Active Provider : {provider}")
    print(f"Active Model    : {model}")
    print(f"Base URL        : {base_url}")
    print(f"Key Present     : {'YES (Loaded)' if bool(api_key) else 'NO (Missing)'}")
    print("=" * 80)

    for idx, q in enumerate(QUERIES, 1):
        print(f"\n\n################################################################################")
        print(f"QUERY #{idx}: \"{q}\"")
        print(f"################################################################################\n")

        rec = run_mode4_pipeline_trace(
            query_text=q,
            yes_flag=True,
            silent=False,
            offline=False,
        )

        print("\n--- EXTRACTED NEURAL CONTRACT (RAW & VALIDATED) ---")
        print(json.dumps(rec.requirements, indent=2))

        print("\n--- CANONICAL EXECUTION RECORD SUMMARY ---")
        print(f"Run ID              : {rec.run_id}")
        print(f"Execution Path      : {rec.execution_path}")
        print(f"Problem Type        : {rec.problem_type}")
        print(f"Solver Used         : {rec.solver_name}")
        print(f"Claimed Cost (USD)  : {rec.claimed_cost_usd}")
        print(f"Recomputed Cost     : {rec.recomputed_cost_usd}")
        print(f"Feasibility Verdict : {rec.feasibility.value}")
        print(f"Optimality Verdict  : {rec.optimality_status.value}")
        print(f"Summary Status      : {rec.summary_status}")
        print(f"Parsing Time (ms)   : {rec.parsing_ms:.2f}")
        print(f"Solving Time (ms)   : {rec.solving_ms:.2f}")
        print(f"Total Time (ms)     : {rec.total_duration_ms:.2f}")
        print(f"Violations Count    : {len(rec.violations)}")
        if rec.violations:
            for v in rec.violations:
                print(f"  * {v}")

        # Save individual query record
        out_dir = PROJECT_ROOT / "results" / "live_mode4_groq_3queries"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"query_{idx}_canonical_record.json", "w", encoding="utf-8") as f:
            json.dump(rec.to_dict(), f, indent=2)

if __name__ == "__main__":
    main()
