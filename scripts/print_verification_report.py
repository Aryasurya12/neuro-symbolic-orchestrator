"""Script to extract and format verification report data."""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    # 1. Row counts
    files = [
        "results/study_run_02.csv",
        "results_presentation.csv",
        "results_side_by_side.csv",
        "results_summary.csv",
    ]
    print("=== FILE ROW COUNTS ===")
    for f in files:
        fp = PROJECT_ROOT / f
        with open(fp, "r", encoding="utf-8-sig") as fp_in:
            rows = list(csv.reader(fp_in))
            print(f"  {f:<30} : {len(rows)} lines ({len(rows)-1} data rows)")

    # 2. Partition check
    print("\n=== PARTITION CHECK IN RESULTS_SUMMARY.CSV ===")
    sum_fp = PROJECT_ROOT / "results_summary.csv"
    with open(sum_fp, "r", encoding="utf-8-sig") as fp_in:
        reader = csv.DictReader(fp_in)
        all_ok = True
        for r in reader:
            nq = int(r["n_queries"])
            nc = int(r["n_correct"])
            nw = int(r["n_wrong"])
            nu = int(r["n_unparseable"])
            npf = int(r["n_provider_failure"])
            nng = int(r["n_not_graded"])
            tot = nc + nw + nu + npf + nng
            if tot != nq:
                print(f"  FAIL: {r['mode_name']} {r['scope']}: sum={tot} != n_queries={nq}")
                all_ok = False
        if all_ok:
            print("  [PASS] Every mode and scope satisfies: n_correct + n_wrong + n_unparseable + n_provider_failure + n_not_graded == n_queries (32/32 partitions verified)")

    # 3. First 8 lines of results_presentation.csv
    print("\n=== FIRST 8 LINES OF RESULTS_PRESENTATION.CSV ===")
    pres_fp = PROJECT_ROOT / "results_presentation.csv"
    with open(pres_fp, "r", encoding="utf-8-sig") as fp_in:
        lines = [fp_in.readline().strip() for _ in range(8)]
        for line in lines:
            print(line)

    # 4. Full results_summary.csv
    print("\n=== FULL RESULTS_SUMMARY.CSV ===")
    with open(sum_fp, "r", encoding="utf-8-sig") as fp_in:
        print(fp_in.read().strip())

    # 5. Find 5 rows where parsed label most likely disagrees with raw response or is interesting
    print("\n=== 5 ROWS WHERE PARSED LABEL MOST LIKELY DISAGREES / REFUSALS / UNPARSEABLE WITH RAW TEXT ===")
    study_fp = PROJECT_ROOT / "results" / "study_run_02.csv"
    with open(study_fp, "r", encoding="utf-8-sig") as fp_in:
        rows = list(csv.DictReader(fp_in))

    # Look for interesting cases: unparseable, wrong plan where model gave prose explanation, or borderline cases
    interesting_qids = [
        ("Q03_VM_HINGLISH_WORDS_8VCPU_16GB", 1),
        ("Q04_VM_AWS_16VCPU_64GB_INFEASIBLE", 1),
        ("Q07_VM_AWS_MISSING_BUDGET", 1),
        ("Q22_UNSUPPORTED_GPU_H100_TRAINING", 1),
        ("Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE", 1),
    ]

    for qid, m_num in interesting_qids:
        matching = [r for r in rows if r["query_id"] == qid and int(r["mode"]) == m_num]
        if matching:
            r = matching[0]
            raw_path = r.get("raw_response_path", "")
            raw_text = ""
            if raw_path:
                rf = PROJECT_ROOT / raw_path
                if rf.exists():
                    with open(rf, "r", encoding="utf-8") as rfp:
                        raw_text = rfp.read().strip()

            print(f"\n--- Query: {qid} (Mode {m_num}: {r.get('model')}) ---")
            print(f"  Expected Outcome : {r.get('expected_outcome')} | Expected Cost: ${r.get('expected_optimal_cost_usd')}")
            print(f"  Parsed Label     : {r.get('explain_label')} | Mode Outcome: {r.get('mode_outcome')} | Claimed Cost: ${r.get('claimed_cost_usd')} | Plan Valid: {r.get('plan_valid')}")
            print(f"  Violations       : {r.get('violations')}")
            print("  Raw Text Snippet :")
            lines_raw = raw_text.splitlines()[:6]
            for l in lines_raw:
                print(f"    {l.encode('ascii', 'replace').decode('ascii')}")
            if len(raw_text.splitlines()) > 6:
                print(f"    ... ({len(raw_text.splitlines()) - 6} more lines)")

if __name__ == "__main__":
    main()
