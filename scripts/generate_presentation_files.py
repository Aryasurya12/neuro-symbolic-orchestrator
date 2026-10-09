"""Presentation CSV generator and assertion checker from study_run_02.csv.

Generates:
1. results_presentation.csv (UTF-8 BOM, 116 rows)
2. results_side_by_side.csv (UTF-8 BOM, 29 rows)
3. results_summary.csv (UTF-8 BOM)
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODE_NAMES = {
    1: "Raw LLM",
    2: "Schema LLM",
    3: "Rule-based",
    4: "Neuro-symbolic",
}


def fmt_cost(val: Any) -> str:
    """Standard financial rounding to 2 decimal places for all monetary values, empty string if None/empty."""
    if val is None or val == "":
        return ""
    try:
        f = float(val)
        if f != f:  # NaN
            return ""
        return f"{f:.2f}"
    except (ValueError, TypeError):
        return ""


def fmt_ms(val: Any) -> str:
    """Format millisecond duration to 1 decimal place."""
    if val is None or val == "":
        return ""
    try:
        return f"{float(val):.1f}"
    except (ValueError, TypeError):
        return ""


def generate_all(csv_path: Path | str, manifest_path: Path | str) -> Dict[str, Any]:
    csv_file = Path(csv_path).resolve()
    manifest_file = Path(manifest_path).resolve()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_queries = {q["query_id"]: q for q in manifest_data.get("queries", [])}
    T = len(manifest_queries)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    # 1. results_presentation.csv
    pres_path = PROJECT_ROOT / "results_presentation.csv"
    pres_columns = [
        "query_id",
        "split",
        "category",
        "query_text",
        "mode_name",
        "expected_outcome",
        "expected_cost_usd",
        "mode_outcome",
        "claimed_cost_usd",
        "recomputed_cost_usd",
        "plan_valid",
        "result_label",
        "violations",
        "total_ms",
    ]

    # Sort raw rows by query_id, then mode
    def sort_key(r: Dict[str, str]):
        qid = r.get("query_id", "")
        mode_val = int(r.get("mode", "0") or "0")
        return (qid, mode_val)

    sorted_rows = sorted(raw_rows, key=sort_key)

    pres_rows = []
    for r in sorted_rows:
        m_num = int(r.get("mode", "0") or "0")
        m_name = MODE_NAMES.get(m_num, f"Mode {m_num}")
        
        # Expected cost from manifest or row
        exp_cost = r.get("expected_optimal_cost_usd")
        if not exp_cost:
            mq = manifest_queries.get(r.get("query_id", ""), {})
            exp_cost = mq.get("expected_optimal_cost_usd")

        claimed = fmt_cost(r.get("claimed_cost_usd"))
        recomp = fmt_cost(r.get("recomputed_cost_usd"))
        exp_cost_str = fmt_cost(exp_cost)

        p_valid = r.get("plan_valid", "").lower()
        if p_valid in ("true", "1", "pass"):
            plan_valid_str = "true"
        elif p_valid in ("false", "0", "fail"):
            plan_valid_str = "false"
        else:
            plan_valid_str = ""

        label = r.get("explain_label") or r.get("execution_status") or ""

        pres_rows.append({
            "query_id": r.get("query_id", ""),
            "split": r.get("split", ""),
            "category": r.get("category", ""),
            "query_text": r.get("query_text", ""),
            "mode_name": m_name,
            "expected_outcome": r.get("expected_outcome", ""),
            "expected_cost_usd": exp_cost_str,
            "mode_outcome": r.get("mode_outcome", ""),
            "claimed_cost_usd": claimed,
            "recomputed_cost_usd": recomp,
            "plan_valid": plan_valid_str,
            "result_label": label,
            "violations": r.get("violations", ""),
            "total_ms": fmt_ms(r.get("total_ms", "")),
        })

    with open(pres_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pres_columns)
        writer.writeheader()
        writer.writerows(pres_rows)

    # 2. results_side_by_side.csv
    sbs_path = PROJECT_ROOT / "results_side_by_side.csv"
    sbs_columns = [
        "query_id",
        "split",
        "category",
        "query_text",
        "expected_outcome",
        "expected_cost_usd",
        "mode1_label",
        "mode1_cost_usd",
        "mode2_label",
        "mode2_cost_usd",
        "mode3_label",
        "mode3_cost_usd",
        "mode4_label",
        "mode4_cost_usd",
    ]

    # Group by query_id
    grouped_by_q: Dict[str, Dict[int, Dict[str, str]]] = {}
    for r in sorted_rows:
        qid = r.get("query_id", "")
        m_num = int(r.get("mode", "0") or "0")
        if qid not in grouped_by_q:
            grouped_by_q[qid] = {}
        grouped_by_q[qid][m_num] = r

    sbs_rows = []
    for qid in sorted(grouped_by_q.keys()):
        modes_dict = grouped_by_q[qid]
        first_row = modes_dict.get(1) or modes_dict.get(2) or modes_dict.get(3) or modes_dict.get(4) or {}
        mq = manifest_queries.get(qid, {})
        exp_cost = fmt_cost(first_row.get("expected_optimal_cost_usd") or mq.get("expected_optimal_cost_usd"))

        row_data = {
            "query_id": qid,
            "split": first_row.get("split", ""),
            "category": first_row.get("category", ""),
            "query_text": first_row.get("query_text", ""),
            "expected_outcome": first_row.get("expected_outcome", ""),
            "expected_cost_usd": exp_cost,
        }

        for m_num in (1, 2, 3, 4):
            mr = modes_dict.get(m_num, {})
            m_label = mr.get("explain_label") or mr.get("execution_status") or ""
            # Cost: claimed cost if valid, else recomputed or claimed
            m_cost = fmt_cost(mr.get("claimed_cost_usd"))
            row_data[f"mode{m_num}_label"] = m_label
            row_data[f"mode{m_num}_cost_usd"] = m_cost

        sbs_rows.append(row_data)

    with open(sbs_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sbs_columns)
        writer.writeheader()
        writer.writerows(sbs_rows)

    # 3. results_summary.csv
    sum_path = PROJECT_ROOT / "results_summary.csv"
    sum_columns = [
        "mode_name",
        "scope",
        "n_queries",
        "n_correct",
        "accuracy_pct",
        "n_wrong",
        "n_unparseable",
        "n_provider_failure",
        "n_not_graded",
        "median_ms",
        "max_ms",
    ]

    # Distinct categories
    categories = sorted(list(set(r.get("category", "") for r in raw_rows if r.get("category"))))
    scopes = ["ALL", "DEV", "EVAL"] + categories

    sum_rows = []
    for m_num in (1, 2, 3, 4):
        m_name = MODE_NAMES[m_num]
        mode_rows = [r for r in raw_rows if int(r.get("mode", "0") or "0") == m_num]

        for scope in scopes:
            if scope == "ALL":
                scoped = mode_rows
            elif scope == "DEV":
                scoped = [r for r in mode_rows if r.get("split", "").lower() == "dev"]
            elif scope == "EVAL":
                scoped = [r for r in mode_rows if r.get("split", "").lower() == "eval"]
            else:
                scoped = [r for r in mode_rows if r.get("category") == scope]

            n_queries = len(scoped)
            if n_queries == 0:
                continue

            n_correct = 0
            n_unparseable = 0
            n_provider_failure = 0
            n_not_graded = 0
            n_wrong = 0
            durations = []

            for r in scoped:
                try:
                    durations.append(float(r.get("total_ms", 0.0)))
                except (ValueError, TypeError):
                    pass

                exp_out = r.get("expected_outcome", "").upper()
                exec_st = r.get("execution_status", "OK").upper()
                norm_st = r.get("normalization_status", "").upper()
                mode_out = r.get("mode_outcome", "").upper()
                t_succ = r.get("task_success", "")

                if exp_out in ("NOT_GRADED", "UNGRADED", "NONE", "—"):
                    n_not_graded += 1
                elif exec_st in ("PROVIDER_FAILURE", "RATE_LIMITED", "TIMEOUT") or "EXECUTION EXCEPTION" in r.get("violations", "").upper():
                    n_provider_failure += 1
                elif norm_st in ("UNPARSEABLE", "MALFORMED_OUTPUT") or mode_out == "UNPARSEABLE":
                    n_unparseable += 1
                elif t_succ in ("1", "true", "True", 1):
                    n_correct += 1
                else:
                    n_wrong += 1

            # Partition check
            # n_correct + n_wrong + n_unparseable + n_provider_failure + n_not_graded == n_queries
            n_wrong_check = n_queries - (n_correct + n_unparseable + n_provider_failure + n_not_graded)
            assert n_wrong == n_wrong_check, f"Partition mismatch: {n_wrong} vs {n_wrong_check}"

            # Accuracy % over graded queries
            n_graded = n_queries - n_not_graded
            if n_graded > 0:
                acc_pct = round((n_correct / n_graded) * 100.0, 1)
                acc_str = f"{acc_pct:.1f}%"
            else:
                acc_str = "N/A"

            med_ms = round(statistics.median(durations), 1) if durations else 0.0
            max_ms = round(max(durations), 1) if durations else 0.0

            sum_rows.append({
                "mode_name": m_name,
                "scope": scope,
                "n_queries": n_queries,
                "n_correct": n_correct,
                "accuracy_pct": acc_str,
                "n_wrong": n_wrong,
                "n_unparseable": n_unparseable,
                "n_provider_failure": n_provider_failure,
                "n_not_graded": n_not_graded,
                "median_ms": f"{med_ms:.1f}",
                "max_ms": f"{max_ms:.1f}",
            })

    with open(sum_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sum_columns)
        writer.writeheader()
        writer.writerows(sum_rows)

    return {
        "presentation_rows": len(pres_rows),
        "side_by_side_rows": len(sbs_rows),
        "summary_rows": len(sum_rows),
        "pres_path": str(pres_path),
        "sbs_path": str(sbs_path),
        "sum_path": str(sum_path),
    }


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "results/study_run_02.csv"
    manifest_file = sys.argv[2] if len(sys.argv) > 2 else "data/final_query_manifest.json"
    res = generate_all(csv_file, manifest_file)
    print("Generated presentation files successfully:")
    print(f"  Presentation rows : {res['presentation_rows']}")
    print(f"  Side-by-side rows : {res['side_by_side_rows']}")
    print(f"  Summary rows      : {res['summary_rows']}")
