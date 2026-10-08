import csv
import json
import os
from collections import Counter
from pathlib import Path

CSV_HEADERS = [
    "timestamp",
    "query",
    "mode",
    "mode_name",
    "engine",
    "latency_ms",
    "reported_cost",
    "feasibility",
    "verdict",
    "proof_verdict",
    "explanation",
    "is_mock",
]


def export_jsonl_to_csv(jsonl_path: str, csv_paths: list[str]) -> int:
    """Synchronizes records from JSONL log to CSV files."""
    if not os.path.exists(jsonl_path):
        return 0

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = rec.get("timestamp", "")
            q = rec.get("query", "")
            results = rec.get("results", {})
            for m_key in ["mode1", "mode2", "mode3", "mode4"]:
                mdata = results.get(m_key, {})
                if isinstance(mdata, dict) and mdata:
                    rows.append({
                        "timestamp": ts,
                        "query": q,
                        "mode": m_key,
                        "mode_name": mdata.get("mode_name", m_key),
                        "engine": mdata.get("engine", "N/A"),
                        "latency_ms": f"{float(mdata.get('latency_ms', 0.0)):.2f}",
                        "reported_cost": str(mdata.get("reported_cost", "N/A")),
                        "feasibility": mdata.get("feasibility", "UNKNOWN"),
                        "verdict": mdata.get("verdict", "UNKNOWN"),
                        "proof_verdict": mdata.get("proof_verdict", "N/A"),
                        "explanation": mdata.get("explanation", "N/A"),
                        "is_mock": mdata.get("is_mock", False),
                    })

    for p in csv_paths:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


try:
    from results.export_to_xlsx import generate_results_xlsx
except ImportError:
    from export_to_xlsx import generate_results_xlsx


def main():
    log_file = Path("results/live_exploration_log.jsonl")
    if not log_file.exists():
        print("No live log file found at results/live_exploration_log.jsonl yet.")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Synchronize and populate results.csv and results/results.csv
    row_count = export_jsonl_to_csv("results/live_exploration_log.jsonl", ["results.csv", "results/results.csv"])

    # Synchronize and populate results.xlsx and results/results.xlsx
    try:
        generate_results_xlsx("results/live_exploration_log.jsonl", ["results.xlsx", "results/results.xlsx"])
    except Exception as exc:
        print(f"[Warning: XLSX sync error]: {exc}")

    print(f"Total live queries run: {len(records)} ({row_count} mode evaluations written to results.csv & results.xlsx)")
    for mode in ["mode1", "mode2", "mode3", "mode4"]:
        verdicts = Counter(r["results"].get(mode, {}).get("verdict", "UNKNOWN") for r in records)
        feas = Counter(r["results"].get(mode, {}).get("feasibility", "UNKNOWN") for r in records)
        print(f"{mode}: {dict(verdicts)} | Feasibility: {dict(feas)}")


if __name__ == "__main__":
    main()

