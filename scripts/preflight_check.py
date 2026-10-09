"""Pre-flight verification script for Neurasym live study run."""

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from config.settings import settings

def main():
    manifest_path = PROJECT_ROOT / "data" / "final_query_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    queries = manifest_data.get("queries", [])
    T = len(queries)
    
    dev_queries = []
    eval_queries = []
    query_ids = []
    
    for q in queries:
        qid = q["query_id"]
        query_ids.append(qid)
        is_dev = q.get("previously_run_in_development", False) or q.get("split") == "dev"
        if is_dev:
            dev_queries.append(qid)
        else:
            eval_queries.append(qid)

    print("=== PRE-FLIGHT CHECK 1: MANIFEST ENTRIES ===")
    print(f"Total manifest entries (T): {T}")
    print(f"Count of split = 'dev'     : {len(dev_queries)}")
    print(f"Count of split = 'eval'    : {len(eval_queries)}")
    print(f"List of all query_ids ({len(query_ids)} total):")
    for i, qid in enumerate(query_ids, 1):
        s_tag = "dev" if qid in dev_queries else "eval"
        print(f"  {i:2d}. [{s_tag:4s}] {qid}")

    print("\n=== PRE-FLIGHT CHECK 2: LIVE PROVIDERS ===")
    groq_api_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", "")
    key_present = bool(groq_api_key and len(groq_api_key.strip()) > 5)
    key_masked = f"{groq_api_key[:6]}...{groq_api_key[-4:]}" if key_present else "NOT FOUND"
    print(f"GROQ_API_KEY present : {key_present} ({key_masked})")
    
    model = os.getenv("GROQ_MODEL") or getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    print(f"Configured Model     : {model}")
    
    # Providers for Mode 1, 2, 4
    m1_prov = "Groq API"
    m1_model = model
    m2_prov = "Groq API"
    m2_model = model
    
    p_name, p_key, p_url, p_model, p_to = settings.get_mode4_provider_config()
    m4_prov = f"{p_name} API"
    m4_model = p_model
    
    print(f"Resolved Mode 1 Provider : {m1_prov} ({m1_model})")
    print(f"Resolved Mode 2 Provider : {m2_prov} ({m2_model})")
    print(f"Resolved Mode 3 Provider : Pure Local Symbolic (SCOPE + HiGHS/Z3/PSO)")
    print(f"Resolved Mode 4 Provider : {m4_prov} ({m4_model}) + Local Solvers")

    print("\n=== PRE-FLIGHT CHECK 3: INITIAL GROQ REQUEST COUNT ===")
    log_path = PROJECT_ROOT / "results" / "live_api_call_log.jsonl"
    initial_log_count = 0
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as lf:
            initial_log_count = sum(1 for line in lf if line.strip())
    print(f"Recorded calls in live_api_call_log.jsonl : {initial_log_count}")

if __name__ == "__main__":
    main()
