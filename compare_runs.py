"""Benchmark Run Comparator for Neurasym Comparative Evaluation.

Compares two benchmark runs (e.g. Run A vs Run B) from their archived run folders
or CSV files, producing:
1. compare_<runA>_vs_<runB>.csv (one row per query_id x mode: labels, claimed costs, and changed=yes/no)
2. Side-by-side per-mode accuracy summary table printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


MODE_NAMES = {
    1: "Raw LLM",
    2: "Schema LLM",
    3: "Rule-based",
    4: "Neuro-symbolic",
}


def load_run_data(target_path: Path | str) -> Tuple[str, Dict[Tuple[str, int], Dict[str, Any]]]:
    """Loads run data from a run folder (containing presentation/study CSV) or directly from a CSV."""
    p = Path(target_path).resolve()
    csv_file: Optional[Path] = None
    run_name = p.stem

    if p.is_dir():
        run_name = p.name
        # Look for presentation CSV first, then study_run CSV
        cand_files = [
            p / "results_presentation.csv",
            p / "study_run_02.csv",
            p / "study_run_01.csv",
        ]
        for c in cand_files:
            if c.exists() and c.stat().st_size > 0:
                csv_file = c
                break
        if not csv_file:
            csv_matches = list(p.glob("*.csv"))
            if csv_matches:
                csv_file = csv_matches[0]
    elif p.is_file():
        csv_file = p
        run_name = p.stem

    if not csv_file or not csv_file.exists():
        raise FileNotFoundError(f"Could not find valid benchmark CSV in '{target_path}'.")

    records: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("query_id", "").strip()
            mode_raw = row.get("mode") or row.get("mode_name") or "0"
            if str(mode_raw).isdigit():
                m_num = int(mode_raw)
            else:
                # Map name to int
                m_num = 0
                for k, v in MODE_NAMES.items():
                    if v.lower() in str(mode_raw).lower():
                        m_num = k
                        break
            if not qid or m_num == 0:
                continue

            label = row.get("result_label") or row.get("explain_label") or row.get("execution_status") or ""
            claimed_cost = row.get("claimed_cost_usd", "").strip()
            recomp_cost = row.get("recomputed_cost_usd", "").strip()
            plan_valid = row.get("plan_valid", "").strip().lower() in ("true", "1", "pass")
            task_succ = row.get("task_success", "").strip()
            exp_out = row.get("expected_outcome", "").strip()

            # Determine correctness if not explicitly given
            is_correct = False
            if task_succ in ("1", "true", "True"):
                is_correct = True
            elif label in ("OPTIMAL_PLAN", "PROVABLY_OPTIMAL", "CORRECT_REFUSAL", "FEASIBLE_CERTIFIED"):
                is_correct = True
            elif label == "SUBOPTIMAL_PLAN" and plan_valid and exp_out == "FEASIBLE":
                is_correct = True

            records[(qid, m_num)] = {
                "query_id": qid,
                "mode": m_num,
                "mode_name": MODE_NAMES.get(m_num, f"Mode {m_num}"),
                "split": row.get("split", ""),
                "category": row.get("category", ""),
                "query_text": row.get("query_text", ""),
                "expected_outcome": exp_out,
                "expected_cost_usd": row.get("expected_cost_usd") or row.get("expected_optimal_cost_usd", ""),
                "result_label": label,
                "claimed_cost_usd": claimed_cost,
                "recomputed_cost_usd": recomp_cost,
                "plan_valid": plan_valid,
                "is_correct": is_correct,
            }

    return run_name, records


def compare_runs(run_a_path: Path | str, run_b_path: Path | str, out_csv: Optional[Path | str] = None) -> Path:
    name_a, data_a = load_run_data(run_a_path)
    name_b, data_b = load_run_data(run_b_path)

    all_keys = sorted(list(set(data_a.keys()) | set(data_b.keys())))

    diff_rows = []
    total_changed = 0

    for key in all_keys:
        qid, m_num = key
        rec_a = data_a.get(key, {})
        rec_b = data_b.get(key, {})

        label_a = rec_a.get("result_label", "MISSING")
        cost_a = rec_a.get("claimed_cost_usd", "")
        label_b = rec_b.get("result_label", "MISSING")
        cost_b = rec_b.get("claimed_cost_usd", "")

        # Compare label and cost
        label_diff = (label_a != label_b)
        cost_diff = False
        try:
            fa = float(cost_a) if cost_a else None
            fb = float(cost_b) if cost_b else None
            if fa is not None and fb is not None:
                cost_diff = abs(fa - fb) > 0.001
            elif fa != fb:
                cost_diff = True
        except ValueError:
            cost_diff = (cost_a != cost_b)

        is_changed = label_diff or cost_diff
        if is_changed:
            total_changed += 1

        first_rec = rec_a or rec_b
        diff_rows.append({
            "query_id": qid,
            "mode": m_num,
            "mode_name": MODE_NAMES.get(m_num, f"Mode {m_num}"),
            "split": first_rec.get("split", ""),
            "category": first_rec.get("category", ""),
            f"label_{name_a}": label_a,
            f"cost_{name_a}": cost_a,
            f"label_{name_b}": label_b,
            f"cost_{name_b}": cost_b,
            "changed": "yes" if is_changed else "no",
        })

    # Determine output file path
    if out_csv:
        out_file = Path(out_csv).resolve()
    else:
        out_file = PROJECT_ROOT / f"compare_{name_a}_vs_{name_b}.csv"

    fieldnames = [
        "query_id",
        "mode",
        "mode_name",
        "split",
        "category",
        f"label_{name_a}",
        f"cost_{name_a}",
        f"label_{name_b}",
        f"cost_{name_b}",
        "changed",
    ]

    with open(out_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diff_rows)

    # Compute per-mode accuracy for both runs
    print("=" * 88)
    print(f" BENCHMARK RUN COMPARISON: {name_a} vs {name_b} ".center(88, "="))
    print("=" * 88)
    print(f"  Total Comparison Rows : {len(diff_rows)}")
    print(f"  Rows Changed          : {total_changed} / {len(diff_rows)}")
    print(f"  Diff CSV Output Path  : {out_file}")
    print("-" * 88)
    print(f" {'Mode':<22} | {'Run A (' + name_a[:16] + ')':<25} | {'Run B (' + name_b[:16] + ')':<25} | {'Delta':<10}")
    print("-" * 88)

    for m_num in (1, 2, 3, 4):
        m_name = MODE_NAMES[m_num]
        q_a = [r for k, r in data_a.items() if k[1] == m_num]
        q_b = [r for k, r in data_b.items() if k[1] == m_num]

        corr_a = sum(1 for r in q_a if r.get("is_correct"))
        tot_a = len(q_a)
        acc_a = (corr_a / tot_a * 100.0) if tot_a > 0 else 0.0

        corr_b = sum(1 for r in q_b if r.get("is_correct"))
        tot_b = len(q_b)
        acc_b = (corr_b / tot_b * 100.0) if tot_b > 0 else 0.0

        delta = acc_b - acc_a
        delta_str = f"{delta:+.1f}%" if delta != 0 else "0.0%"

        str_a = f"{acc_a:.1f}% ({corr_a}/{tot_a})"
        str_b = f"{acc_b:.1f}% ({corr_b}/{tot_b})"
        print(f" {m_name:<22} | {str_a:<25} | {str_b:<25} | {delta_str:<10}")

    print("=" * 88 + "\n")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Compare two Neurasym benchmark runs")
    parser.add_argument("run_a", type=str, help="Path to Run A folder or CSV")
    parser.add_argument("run_b", type=str, help="Path to Run B folder or CSV")
    parser.add_argument("--out", type=str, default=None, help="Output comparison CSV path")
    args = parser.parse_args()

    compare_runs(args.run_a, args.run_b, args.out)


if __name__ == "__main__":
    main()
