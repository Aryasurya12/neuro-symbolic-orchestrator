"""Assertion verification script for Neurasym live study run_02."""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def verify_all(csv_path_str: str, manifest_path_str: str, initial_log_count: int = 15):
    csv_path = Path(csv_path_str).resolve()
    manifest_path = Path(manifest_path_str).resolve()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_queries = {q["query_id"]: q for q in manifest_data.get("queries", [])}
    T = len(manifest_queries)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("=" * 80)
    print(" NEURASYM LIVE STUDY RUN ASSERTIONS ".center(80, "="))
    print("=" * 80)

    # a) Row count == T * 4
    expected_rows = T * 4
    actual_rows = len(rows)
    pass_a = (actual_rows == expected_rows)
    print(f"Assertion a) Row count == T * 4 ({expected_rows}):")
    print(f"  Actual: {actual_rows} | {'[PASS]' if pass_a else '[FAIL]'}")

    # b) Exactly one row per (query_id, mode, trial)
    seen_tuples: Dict[Tuple[str, int, int], int] = {}
    for r in rows:
        qid = r.get("query_id", "")
        mode_val = int(r.get("mode", "0") or "0")
        trial_val = int(r.get("trial", "1") or "1")
        key = (qid, mode_val, trial_val)
        seen_tuples[key] = seen_tuples.get(key, 0) + 1

    expected_tuples = set()
    for qid in manifest_queries.keys():
        for m in (1, 2, 3, 4):
            expected_tuples.add((qid, m, 1))

    missing = expected_tuples - set(seen_tuples.keys())
    duplicates = [k for k, count in seen_tuples.items() if count > 1]
    pass_b = (len(missing) == 0 and len(duplicates) == 0 and len(seen_tuples) == expected_rows)
    print(f"Assertion b) Exactly one row per (query_id, mode, trial):")
    print(f"  Missing combinations: {len(missing)} {list(missing)[:5] if missing else '[]'}")
    print(f"  Duplicate combinations: {len(duplicates)} {duplicates[:5] if duplicates else '[]'}")
    print(f"  Status: {'[PASS]' if pass_b else '[FAIL]'}")

    # c) is_mock is False on every row
    mock_rows = [r for r in rows if r.get("is_mock", "").lower() in ("true", "1")]
    pass_c = (len(mock_rows) == 0)
    print(f"Assertion c) is_mock is False on every row:")
    print(f"  Mock rows count: {len(mock_rows)} | {'[PASS]' if pass_c else '[FAIL]'}")

    # d) Every Mode 1, 2 and 4 row has an existing raw_response_path file
    missing_raw_files = []
    for r in rows:
        m = int(r.get("mode", "0") or "0")
        if m in (1, 2, 4):
            raw_p = r.get("raw_response_path", "")
            if not raw_p:
                missing_raw_files.append((r.get("query_id"), m, "No path recorded"))
            else:
                full_p = PROJECT_ROOT / raw_p
                if not full_p.exists() or full_p.stat().st_size == 0:
                    missing_raw_files.append((r.get("query_id"), m, f"File missing/empty: {raw_p}"))

    pass_d = (len(missing_raw_files) == 0)
    print(f"Assertion d) Every Mode 1, 2, 4 row has existing raw_response_path file:")
    print(f"  Checked {sum(1 for r in rows if int(r.get('mode', '0') or '0') in (1,2,4))} LLM rows. Missing files: {len(missing_raw_files)}")
    if missing_raw_files:
        for item in missing_raw_files[:5]:
            print(f"    - {item}")
    print(f"  Status: {'[PASS]' if pass_d else '[FAIL]'}")

    # e) No row has provider = MockProvider
    mock_prov_rows = [r for r in rows if "mock" in r.get("provider", "").lower()]
    pass_e = (len(mock_prov_rows) == 0)
    print(f"Assertion e) No row has provider = MockProvider:")
    print(f"  MockProvider rows count: {len(mock_prov_rows)} | {'[PASS]' if pass_e else '[FAIL]'}")

    # f) Groq request count rose by about 3*T (plus Stage 4.5 report calls if enabled)
    log_path = PROJECT_ROOT / "results" / "live_api_call_log.jsonl"
    final_log_count = 0
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as lf:
            final_log_count = sum(1 for line in lf if line.strip())
    calls_made = final_log_count - initial_log_count
    min_expected_calls = 3 * T  # 87
    max_expected_calls = 4 * T  # 116 (with Stage 4.5)
    pass_f = (calls_made >= min_expected_calls)
    print(f"Assertion f) Groq request count rose by >= 3*T ({min_expected_calls}):")
    print(f"  Initial log count : {initial_log_count}")
    print(f"  Final log count   : {final_log_count}")
    print(f"  Calls made        : {calls_made} (Expected: ~{min_expected_calls}-{max_expected_calls}) | {'[PASS]' if pass_f else '[FAIL]'}")

    # g) Every row has expected_outcome and expected_cost_usd (or an explicit non-cost outcome such as CLARIFY or UNSUPPORTED)
    missing_expectations = []
    for r in rows:
        exp_out = r.get("expected_outcome", "").strip()
        exp_cost = r.get("expected_optimal_cost_usd", "").strip()
        qid = r.get("query_id", "")
        if not exp_out:
            missing_expectations.append((qid, "missing expected_outcome"))
        if exp_out == "FEASIBLE" and not exp_cost:
            missing_expectations.append((qid, "FEASIBLE query missing expected_cost_usd"))

    pass_g = (len(missing_expectations) == 0)
    print(f"Assertion g) Every row has expected_outcome and expected_cost_usd (or non-cost outcome):")
    print(f"  Missing expectation rows: {len(missing_expectations)}")
    if missing_expectations:
        for item in missing_expectations[:5]:
            print(f"    - {item}")
    print(f"  Status: {'[PASS]' if pass_g else '[FAIL]'}")

    print("=" * 80)
    all_passed = pass_a and pass_b and pass_c and pass_d and pass_e and pass_f and pass_g
    print(f"OVERALL ASSERTION RESULT: {'ALL PASS [PASS]' if all_passed else 'SOME FAILED [FAIL]'}")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "results/study_run_02.csv"
    manifest_file = sys.argv[2] if len(sys.argv) > 2 else "data/final_query_manifest.json"
    init_count = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    verify_all(csv_file, manifest_file, init_count)
