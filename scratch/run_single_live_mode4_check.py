"""Single controlled live Mode 4 check for Neurasym.

Executes exactly ONE live neural extraction call via OpenRouter (nvidia/nemotron-3.5-lightning:free),
disabling Stage 6 LLM explainer, retries, and loops.
Outputs and saves the complete canonical execution record and audit artifacts.
"""

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
from src.verifiers.canonical_record import CanonicalExecutionRecord

def main():
    query = "Deploy 8 vCPUs and 16GB RAM for under $300 a month on AWS"
    print("=" * 80)
    print("NEURASYM CONTROLLED SINGLE LIVE MODE 4 EXECUTION CHECK")
    print(f"Target Query: \"{query}\"")
    print("=" * 80)

    provider_name, api_key, base_url, target_model, timeout = settings.get_mode4_provider_config()
    print(f"Configured Provider : {provider_name}")
    print(f"Configured Model    : {target_model}")
    print(f"Configured Base URL : {base_url}")
    print(f"Key Present         : {'YES (Loaded)' if bool(api_key) else 'NO (Missing)'}")
    print("-" * 80)

    if not api_key:
        print("ABORTING LIVE CALL: Missing API credentials for Mode 4.")
        return

    # Execute exactly ONE live call with explanations and retries disabled
    t0 = time.perf_counter()
    record = run_mode4_pipeline_trace(
        query_text=query,
        yes_flag=True,
        silent=False,
        offline=False,
    )
    total_elapsed = time.perf_counter() - t0

    # Save artifact
    output_dir = PROJECT_ROOT / "results" / "live_mode4_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "canonical_record.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)

    print("\n" + "=" * 80)
    print("CANONICAL RECORD SUMMARY:")
    print(f"Run ID              : {record.run_id}")
    print(f"Mode                : {record.mode_name}")
    print(f"Provider            : {record.provider}")
    print(f"Model               : {record.model}")
    print(f"Problem Archetype   : {record.problem_type}")
    print(f"Neural Requirements : {json.dumps(record.requirements, indent=2)}")
    print(f"Extraction Latency  : {record.parsing_ms:.2f} ms")
    print(f"Solver Name         : {record.solver_name}")
    print(f"Solving Latency     : {record.solving_ms:.2f} ms")
    print(f"Verification Verdict: {record.summary_status}")
    print(f"Feasibility Status  : {record.feasibility.value}")
    print(f"Optimality Status   : {record.optimality_status.value}")
    print(f"Claimed / Recomputed: ${record.claimed_cost_usd} / ${record.recomputed_cost_usd}")
    print(f"Audit Violations    : {len(record.violations)} detected")
    print(f"Artifact Saved to   : {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
