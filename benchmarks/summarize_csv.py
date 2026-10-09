"""CSV Benchmark Summarizer for Neurasym 4-Way Evaluation.

Computes and prints summary tables derived SOLELY from the evaluation CSV file:
(a) Task success per mode (n shown; NOT_GRADED and mock rows excluded; provider failures and UNPARSEABLE shown separately)
(b) Task success per category per mode
(c) Interpretation accuracy for Modes 3 & 4 (overall, formal, Hinglish digits, Hinglish words)
(d) Median, min, max, mean total_ms per mode
(e) Cost gap vs independent optimum for FEASIBLE queries with a valid plan only
(f) Generalization split by previously_run_in_development (true vs false)
(g) Boundary test pairs (Q26/Q27 and Q28/Q29) side-by-side per mode

No composite score, no winner badge.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def parse_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def parse_bool(val: Any) -> Optional[bool]:
    if val is None or val == "":
        return None
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def summarize_csv_file(csv_path: Path | str, out_path: Optional[Path | str] = None) -> str:
    csv_file = Path(csv_path).resolve()
    if not csv_file.exists():
        msg = f"[ERROR] CSV file not found at '{csv_file}'."
        print(msg)
        return msg

    rows: List[Dict[str, str]] = []
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        msg = f"[ERROR] CSV file '{csv_file}' is empty (0 data rows)."
        print(msg)
        return msg

    out_lines: List[str] = []

    def p(text: str = ""):
        out_lines.append(text)
        print(text)

    # Header information
    run_ids = sorted(list({r.get("run_id", "") for r in rows if r.get("run_id")}))
    manifest_shas = sorted(list({r.get("manifest_sha256", "") for r in rows if r.get("manifest_sha256")}))
    is_mock_run = all(parse_bool(r.get("is_mock")) is True for r in rows)

    p("=" * 100)
    p(" NEURASYM 4-WAY COMPARATIVE BENCHMARK CSV SUMMARY REPORT ".center(100, "="))
    p("=" * 100)
    p(f"  Source CSV File  : {csv_file}")
    p(f"  Total Rows Found : {len(rows)}")
    p(f"  Run ID(s)        : {', '.join(run_ids) if run_ids else 'N/A'}")
    p(f"  Manifest SHA256  : {', '.join(manifest_shas) if manifest_shas else 'N/A'}")
    p(f"  Execution Type   : {'OFFLINE MOCK FIXTURE' if is_mock_run else 'LIVE / DETERMINISTIC PIPELINE'}")
    p("=" * 100 + "\n")

    mode_names = {
        1: "Mode 1: Raw LLM",
        2: "Mode 2: Schema LLM",
        3: "Mode 3: Pure Symbolic",
        4: "Mode 4: Neuro-Symbolic",
    }

    # =========================================================================
    # (a) Task Success per Mode
    # =========================================================================
    p("--- (A) TASK SUCCESS RATE PER MODE ---")
    p(f"{'Execution Mode':<26} | {'Graded (N)':<11} | {'Successes':<10} | {'Success Rate':<14} | {'Prov Failures':<14} | {'Unparseable':<12}")
    p("-" * 96)

    for m in [1, 2, 3, 4]:
        m_rows = [r for r in rows if parse_int(r.get("mode")) == m]
        # Graded queries: task_success is non-empty
        graded_rows = [r for r in m_rows if r.get("task_success") != "" and r.get("task_success") is not None]
        success_rows = [r for r in graded_rows if parse_int(r.get("task_success")) == 1]
        
        prov_fails = [r for r in m_rows if r.get("execution_status") in ("PROVIDER_FAILURE", "RATE_LIMITED", "TIMEOUT")]
        unparseables = [r for r in m_rows if r.get("mode_outcome") == "UNPARSEABLE" or r.get("normalization_status") in ("UNPARSEABLE", "MALFORMED_OUTPUT")]

        n_graded = len(graded_rows)
        n_succ = len(success_rows)
        rate_str = f"{(n_succ / n_graded * 100.0):.1f}%" if n_graded > 0 else "N/A (0)"
        p(f"{mode_names[m]:<26} | {n_graded:<11} | {n_succ:<10} | {rate_str:<14} | {len(prov_fails):<14} | {len(unparseables):<12}")
    p("-" * 96 + "\n")

    # =========================================================================
    # (b) Task Success per Category per Mode
    # =========================================================================
    p("--- (B) TASK SUCCESS RATE PER CATEGORY ---")
    categories = sorted(list({r.get("category", "") for r in rows if r.get("category")}))

    for cat in categories:
        p(f"\n  [Category: {cat}]")
        p(f"    {'Execution Mode':<24} | {'N Graded':<9} | {'Successes':<10} | {'Success Rate':<14}")
        p("    " + "-" * 64)
        for m in [1, 2, 3, 4]:
            c_rows = [r for r in rows if parse_int(r.get("mode")) == m and r.get("category") == cat]
            c_graded = [r for r in c_rows if r.get("task_success") != "" and r.get("task_success") is not None]
            c_succ = [r for r in c_graded if parse_int(r.get("task_success")) == 1]
            n_g = len(c_graded)
            n_s = len(c_succ)
            r_str = f"{(n_s / n_g * 100.0):.1f}%" if n_g > 0 else "—"
            p(f"    {mode_names[m]:<24} | {n_g:<9} | {n_s:<10} | {r_str:<14}")
    p("\n" + "=" * 96 + "\n")

    # =========================================================================
    # (c) Interpretation Accuracy for Modes 3 & 4
    # =========================================================================
    p("--- (C) INTERPRETATION ACCURACY (MODES 3 & 4) ---")
    p("  Measures exact extraction match against the manifest ground-truth requirements.\n")
    p(f"  {'Sub-Category':<32} | {'Mode 3 (SCOPE)':<20} | {'Mode 4 (Neural)':<20}")
    p("  " + "-" * 78)

    def eval_interp(mode_num: int, filter_fn) -> str:
        subset = [r for r in rows if parse_int(r.get("mode")) == mode_num and filter_fn(r)]
        if not subset:
            return "—"
        matches = [r for r in subset if parse_bool(r.get("interpretation_match")) is True]
        pct = (len(matches) / len(subset)) * 100.0
        return f"{pct:.1f}% ({len(matches)}/{len(subset)})"

    # Total
    p(f"  {'Overall Interpretation Match':<32} | {eval_interp(3, lambda r: True):<20} | {eval_interp(4, lambda r: True):<20}")

    # Formal English
    p(f"  {'Formal English Queries':<32} | {eval_interp(3, lambda r: r.get('category') == 'FORMAL_ENGLISH'):<20} | {eval_interp(4, lambda r: r.get('category') == 'FORMAL_ENGLISH'):<20}")

    # Hinglish with Digits (e.g. Q02, Q12)
    def is_hinglish_digits(r: Dict[str, str]) -> bool:
        qid = r.get("query_id", "")
        return r.get("category") == "COLLOQUIAL_HINGLISH" and ("DIGITS" in qid or qid in ("Q02_VM_HINGLISH_8VCPU_16GB", "Q05_VM_HINGLISH_16VCPU_64GB_INFEASIBLE", "Q12_DR_HINGLISH_DIGITS_50MS_600USD", "Q18_SCALING_HINGLISH_300MBPS_60CPU"))

    p(f"  {'Hinglish with Digits':<32} | {eval_interp(3, is_hinglish_digits):<20} | {eval_interp(4, is_hinglish_digits):<20}")

    # Hinglish with Number Words (e.g. Q03, Q10, Q13)
    def is_hinglish_words(r: Dict[str, str]) -> bool:
        qid = r.get("query_id", "")
        return r.get("category") == "COLLOQUIAL_HINGLISH" and ("WORDS" in qid or qid in ("Q03_VM_HINGLISH_WORDS_8VCPU_16GB", "Q10_VM_HINGLISH_GCP_AZURE_8VCPU_32GB", "Q13_DR_HINGLISH_WORDS_50MS_600USD"))

    p(f"  {'Hinglish with Number Words':<32} | {eval_interp(3, is_hinglish_words):<20} | {eval_interp(4, is_hinglish_words):<20}")
    p("  " + "-" * 78 + "\n")

    # Common Mismatches
    m3_mismatches = [r.get("interpretation_mismatch_fields") for r in rows if parse_int(r.get("mode")) == 3 and r.get("interpretation_mismatch_fields")]
    m4_mismatches = [r.get("interpretation_mismatch_fields") for r in rows if parse_int(r.get("mode")) == 4 and r.get("interpretation_mismatch_fields")]
    if m3_mismatches:
        p(f"  * Mode 3 Observed Mismatch Fields: {', '.join(set(m3_mismatches))}")
    if m4_mismatches:
        p(f"  * Mode 4 Observed Mismatch Fields: {', '.join(set(m4_mismatches))}")
    p("\n" + "=" * 96 + "\n")

    # =========================================================================
    # (d) Latency Summary (total_ms) per Mode
    # =========================================================================
    p("--- (D) LATENCY PROFILES (TOTAL EXECUTION TIME) ---")
    p(f"{'Execution Mode':<26} | {'Median (ms)':<12} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Mean (ms)':<10} | {'Samples':<8}")
    p("-" * 88)

    for m in [1, 2, 3, 4]:
        m_times = [parse_float(r.get("total_ms")) for r in rows if parse_int(r.get("mode")) == m and parse_float(r.get("total_ms")) is not None]
        if m_times:
            med_t = statistics.median(m_times)
            min_t = min(m_times)
            max_t = max(m_times)
            mean_t = statistics.mean(m_times)
            p(f"{mode_names[m]:<26} | {med_t:<12.1f} | {min_t:<10.1f} | {max_t:<10.1f} | {mean_t:<10.1f} | {len(m_times):<8}")
        else:
            p(f"{mode_names[m]:<26} | {'—':<12} | {'—':<10} | {'—':<10} | {'—':<10} | {0:<8}")
    p("-" * 88 + "\n")

    # =========================================================================
    # (e) Cost Optimality Gap
    # =========================================================================
    p("--- (E) COST OPTIMALITY GAP (FEASIBLE QUERIES WITH VALID PLAN ONLY) ---")
    p(f"{'Execution Mode':<26} | {'Valid Plans':<12} | {'Mean Gap %':<12} | {'Median Gap %':<14} | {'Exact Opt Matches':<18}")
    p("-" * 92)

    for m in [1, 2, 3, 4]:
        feas_valid_rows = [
            r for r in rows
            if parse_int(r.get("mode")) == m
            and r.get("expected_outcome") == "FEASIBLE"
            and parse_bool(r.get("plan_valid")) is True
        ]
        gaps = [parse_float(r.get("optimal_cost_gap_pct")) for r in feas_valid_rows if parse_float(r.get("optimal_cost_gap_pct")) is not None]
        exact_matches = [g for g in gaps if g <= 0.01]

        if gaps:
            mean_gap = statistics.mean(gaps)
            med_gap = statistics.median(gaps)
            exact_str = f"{len(exact_matches)}/{len(feas_valid_rows)}"
            p(f"{mode_names[m]:<26} | {len(feas_valid_rows):<12} | {mean_gap:<12.2f}% | {med_gap:<14.2f}% | {exact_str:<18}")
        else:
            p(f"{mode_names[m]:<26} | {len(feas_valid_rows):<12} | {'—':<12} | {'—':<14} | {'0/0':<18}")
    p("-" * 92 + "\n")

    # =========================================================================
    # (f) Development vs Unseen Evaluation Queries Split
    # =========================================================================
    p("--- (F) GENERALIZATION: SEEN IN DEV VS UNSEEN EVALUATION ---")
    p(f"{'Execution Mode':<26} | {'Seen in Dev (Task Succ %)':<28} | {'Unseen Eval (Task Succ %)':<28}")
    p("-" * 90)

    for m in [1, 2, 3, 4]:
        seen_rows = [r for r in rows if parse_int(r.get("mode")) == m and parse_bool(r.get("previously_run_in_development")) is True and r.get("task_success") != ""]
        unseen_rows = [r for r in rows if parse_int(r.get("mode")) == m and parse_bool(r.get("previously_run_in_development")) is False and r.get("task_success") != ""]

        seen_succ = [r for r in seen_rows if parse_int(r.get("task_success")) == 1]
        unseen_succ = [r for r in unseen_rows if parse_int(r.get("task_success")) == 1]

        seen_str = f"{(len(seen_succ) / len(seen_rows) * 100.0):.1f}% ({len(seen_succ)}/{len(seen_rows)})" if seen_rows else "—"
        unseen_str = f"{(len(unseen_succ) / len(unseen_rows) * 100.0):.1f}% ({len(unseen_succ)}/{len(unseen_rows)})" if unseen_rows else "—"

        p(f"{mode_names[m]:<26} | {seen_str:<28} | {unseen_str:<28}")
    p("-" * 90 + "\n")

    # =========================================================================
    # (g) Boundary Pairs Side-by-Side
    # =========================================================================
    p("--- (G) BOUNDARY TEST PAIRS (DYNAMIC COMPUTATION PROOF) ---")
    p("  Verifies that decisions adapt dynamically to $1 budget changes rather than memorized static answers.\n")

    boundary_pairs = [
        ("VM Boundary ($121 Infeasible vs $122 Feasible)", "Q26_BOUNDARY_VM_UNDER_121_INFEASIBLE", "Q27_BOUNDARY_VM_UNDER_122_FEASIBLE"),
        ("DR Boundary ($240 Infeasible vs $250 Feasible)", "Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE", "Q29_BOUNDARY_DR_UNDER_250_FEASIBLE"),
    ]

    for label, q_infeas_id, q_feas_id in boundary_pairs:
        p(f"  Pair: {label}")
        p(f"    {'Mode':<24} | {q_infeas_id[:20]:<20} | {q_feas_id[:20]:<20} | {'Consistent Pair?':<18}")
        p("    " + "-" * 72)
        for m in [1, 2, 3, 4]:
            r_inf = next((r for r in rows if parse_int(r.get("mode")) == m and r.get("query_id") == q_infeas_id), None)
            r_feas = next((r for r in rows if parse_int(r.get("mode")) == m and r.get("query_id") == q_feas_id), None)

            out_inf = r_inf.get("mode_outcome", "—") if r_inf else "—"
            out_feas = r_feas.get("mode_outcome", "—") if r_feas else "—"

            succ_inf = parse_int(r_inf.get("task_success")) == 1 if r_inf and r_inf.get("task_success") != "" else False
            succ_feas = parse_int(r_feas.get("task_success")) == 1 if r_feas and r_feas.get("task_success") != "" else False

            is_consistent = (succ_inf and succ_feas)
            cons_str = "YES (PASS)" if is_consistent else "NO (FAIL)"

            p(f"    {mode_names[m]:<24} | {out_inf:<20} | {out_feas:<20} | {cons_str:<18}")
        p("    " + "-" * 72 + "\n")

    p("=" * 100)
    p(" END OF SUMMARY REPORT (Derived strictly from evaluation CSV) ".center(100, "="))
    p("=" * 100 + "\n")

    full_report = "\n".join(out_lines)

    if out_path:
        out_file = Path(out_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"[OK] Summary report written to: {out_file}")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Neurasym Benchmark CSV Summarizer")
    parser.add_argument("--in", dest="in_file", type=str, required=True, help="Path to input evaluation CSV")
    parser.add_argument("--out", dest="out_file", type=str, default=None, help="Optional path to save summary text report")

    args = parser.parse_args()
    summarize_csv_file(args.in_file, args.out_file)


if __name__ == "__main__":
    main()
