"""Comprehensive automated audit script for Neurasym."""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def audit_study_data():
    csv_path = PROJECT_ROOT / "results" / "study_run_02.csv"
    manifest_path = PROJECT_ROOT / "data" / "final_query_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_queries = {q["query_id"]: q for q in manifest_data.get("queries", [])}
    
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"Total Rows: {len(rows)}")
    print(f"Total Manifest Queries: {len(manifest_queries)}")

    # 1. Inspect specific requested queries
    targeted_inspect = [
        ("Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING", 3),
        ("Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING", 4),
        ("Q10_VM_HINGLISH_GCP_AZURE_8VCPU_32GB", 4),
    ]

    print("\n=== TARGETED QUERY AUDIT ===")
    for qid, mode_val in targeted_inspect:
        matching = [r for r in rows if r["query_id"] == qid and int(r["mode"]) == mode_val]
        if matching:
            r = matching[0]
            mq = manifest_queries.get(qid, {})
            print(f"\n--- {qid} | Mode {mode_val} ---")
            print(f"  Expected Outcome        : {r.get('expected_outcome')} (Manifest: {mq.get('expected_outcome')})")
            print(f"  Mode Outcome            : {r.get('mode_outcome')}")
            print(f"  Normalization Status    : {r.get('normalization_status')}")
            print(f"  Task Success            : {r.get('task_success')}")
            print(f"  Explain Label           : {r.get('explain_label')}")
            print(f"  Explain Headline        : {r.get('explain_headline')}")
            print(f"  Plan Valid              : {r.get('plan_valid')}")
            print(f"  Claimed Cost USD        : {r.get('claimed_cost_usd')}")
            print(f"  Recomputed Cost USD     : {r.get('recomputed_cost_usd')}")
            print(f"  Interpretation Match    : {r.get('interpretation_match')}")
            print(f"  Interpretation Mismatch : {r.get('interpretation_mismatch_fields')}")
            print(f"  Violations              : {r.get('violations')}")
            print(f"  Proof Status            : {r.get('proof_status')}")
            print(f"  Allocated Summary       : {r.get('allocated_summary')}")

    # 2. Mode 1 CORRECT_REFUSAL cases
    print("\n=== MODE 1 CORRECT_REFUSAL CASES ===")
    m1_refusals = [r for r in rows if int(r["mode"]) == 1 and (r.get("explain_label") == "CORRECT_REFUSAL" or r.get("task_success") == "1")]
    for r in m1_refusals:
        print(f"  {r['query_id']}: label={r.get('explain_label')}, task_success={r.get('task_success')}, norm_status={r.get('normalization_status')}, mode_outcome={r.get('mode_outcome')}, violations={r.get('violations')}")

    # 3. Mode 2 TASK_INCOMPATIBLE cases
    print("\n=== MODE 2 TASK_INCOMPATIBLE / UNSUPPORTED CASES ===")
    m2_unsup = [r for r in rows if int(r["mode"]) == 2 and ("INCOMPATIBLE" in r.get("normalization_status", "") or r.get("mode_outcome") == "UNSUPPORTED" or r.get("expected_outcome") == "UNSUPPORTED")]
    for r in m2_unsup:
        print(f"  {r['query_id']}: label={r.get('explain_label')}, task_success={r.get('task_success')}, norm_status={r.get('normalization_status')}, mode_outcome={r.get('mode_outcome')}, violations={r.get('violations')}")

    # 4. Blank task_success values
    print("\n=== BLANK OR NONE TASK_SUCCESS VALUES ===")
    blank_ts = [r for r in rows if r.get("task_success") is None or r.get("task_success").strip() == ""]
    print(f"  Total blank task_success count: {len(blank_ts)}")
    for r in blank_ts:
        print(f"  {r['query_id']} | Mode {r['mode']}: expected={r.get('expected_outcome')}, norm_status={r.get('normalization_status')}, mode_outcome={r.get('mode_outcome')}, label={r.get('explain_label')}")

    # 5. Field Contradictions Audit across all 116 rows
    print("\n=== FIELD CONTRADICTIONS AUDIT ===")
    contradictions = []
    for r in rows:
        qid = r["query_id"]
        mode_val = int(r["mode"])
        ts = r.get("task_success", "").strip()
        pv = r.get("plan_valid", "").strip().lower()
        lbl = r.get("explain_label", "").strip()
        mo = r.get("mode_outcome", "").strip()
        ns = r.get("normalization_status", "").strip()
        eo = r.get("expected_outcome", "").strip()
        vio = r.get("violations", "").strip()
        im = r.get("interpretation_match", "").strip().lower()
        proof = r.get("proof_status", "").strip()

        # Check 1: FEASIBLE expected, but plan_valid is false while task_success is 1
        if eo == "FEASIBLE" and pv in ("false", "0") and ts in ("1", "true"):
            contradictions.append((qid, mode_val, "FEASIBLE expected & plan_valid=false but task_success=1", r))

        # Check 2: violations exist on FEASIBLE query but plan_valid is true
        if eo == "FEASIBLE" and vio and pv in ("true", "1"):
            contradictions.append((qid, mode_val, "Violations exist but plan_valid=true", r))

        # Check 3: explain_label is CORRECT / OPTIMAL_PLAN but task_success is 0
        if lbl in ("CORRECT", "OPTIMAL_PLAN", "PROVABLY_OPTIMAL") and ts in ("0", "false"):
            contradictions.append((qid, mode_val, f"explain_label={lbl} but task_success=0", r))

        # Check 4: explain_label is WRONG_PLAN / WRONG_OUTCOME but task_success is 1
        if lbl in ("WRONG_PLAN", "WRONG_OUTCOME", "FAILED_VERIFICATION") and ts in ("1", "true"):
            contradictions.append((qid, mode_val, f"explain_label={lbl} but task_success=1", r))

        # Check 5: PSO continuous scaling claimed as exact discrete minimum / provably optimal branch and bound
        if "scaling" in qid.lower() and "exact discrete" in proof.lower():
            contradictions.append((qid, mode_val, f"PSO continuous scaling has exact discrete proof_status: {proof}", r))

    print(f"  Total contradictions detected: {len(contradictions)}")
    for qid, m, reason, _ in contradictions:
        print(f"    - [{qid} | Mode {m}]: {reason}")

    # 6. Check Git commit hashes
    print("\n=== GIT COMMIT INVESTIGATION ===")
    commits_in_csv = set(r.get("code_commit") for r in rows)
    print(f"  Code commits in study_run_02.csv : {commits_in_csv}")
    
    import subprocess
    git_log = subprocess.run(["git", "log", "-n", "5", "--oneline"], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print("  Recent Git commits:")
    for line in git_log.stdout.strip().splitlines():
        print(f"    {line}")

    # 7. Check Visualizer & Published Figures
    print("\n=== PUBLISHED FIGURES & VISUALIZERS ===")
    fig_dir = PROJECT_ROOT / "results"
    fig_files = list(fig_dir.glob("*.pdf")) + list(fig_dir.glob("*.png")) + list(fig_dir.glob("*.jpg"))
    print(f"  Figure files in results/: {[f.name for f in fig_files]}")

if __name__ == "__main__":
    audit_study_data()
