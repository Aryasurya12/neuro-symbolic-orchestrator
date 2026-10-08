"""Auditable Four-Mode Benchmark Runner & Evaluation CLI for Neurasym.

Usage:
    # 1. Run offline benchmark with development manifest and export all CSVs:
    python benchmarks/run_auditable_evaluation.py --mock-llm

    # 2. Resume an interrupted benchmark session:
    python benchmarks/run_auditable_evaluation.py --session-name run_01 --mock-llm

    # 3. Export CSVs and view analytics without running new queries:
    python benchmarks/run_auditable_evaluation.py --export-only --session-name run_01

    # 4. Run multi-trial repetition (e.g. 3 trials):
    python benchmarks/run_auditable_evaluation.py --trials 3 --mock-llm
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.benchmarks.benchmark_engine import BenchmarkEngine
from src.benchmarks.dataset_manifest import ManifestManager
from src.benchmarks.visualizer import BenchmarkVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Neurasym Auditable Four-Mode Evaluation and Benchmark CLI"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/development_query_manifest.json",
        help="Path to query manifest JSON or CSV file",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default="development_eval",
        help="Session identifier for durable ledger checkpointing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark_runs",
        help="Directory to store ledger JSONL and exported CSV files",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        default=True,
        help="Execute using offline deterministic mock fixtures (ZERO live network calls)",
    )
    parser.add_argument(
        "--live-llm",
        action="store_false",
        dest="mock_llm",
        help="Execute using live external LLM API endpoints",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per query for repeated-trial variance testing",
    )
    parser.add_argument(
        "--force-resume",
        action="store_true",
        help="Force resume even if environment fingerprint has changed",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Do not execute new queries; generate CSVs and analytics from existing ledger",
    )
    parser.add_argument(
        "--final-study",
        action="store_true",
        help="Enforces that all queries in manifest are frozen and human-approved before execution",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose streaming output for all stages and audit checks",
    )

    args = parser.parse_args()

    print("=" * 88)
    print(" NEURASYM AUDITABLE FOUR-MODE EVALUATION CONTROLLER ".center(88, "="))
    print("=" * 88)
    print(f"  Session Name   : {args.session_name}")
    print(f"  Output Dir     : {args.output_dir}")
    print(f"  Execution Mode : {'[OFFLINE DETERMINISTIC FIXTURES - ZERO LIVE CALLS]' if args.mock_llm else '[LIVE API ENDPOINTS]'}")
    print(f"  Trials/Query   : {args.trials}")
    print(f"  Manifest Path  : {args.manifest}")
    print(f"  Study Type     : {'[FINAL FROZEN STUDY]' if args.final_study else '[DEVELOPMENT EVALUATION]'}")
    print("=" * 88 + "\n")

    # Load query manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[!] Manifest not found at '{args.manifest}'. Generating default 36-query development manifest...")
        ManifestManager.export_development_manifest_json(str(manifest_path))

    with open(manifest_path, "r", encoding="utf-8") as f:
        query_manifest = json.load(f)

    print(f"Loaded {len(query_manifest)} queries from manifest (Approval: {query_manifest[0].get('approval_status', 'UNAPPROVED')})\n")

    # Initialize Engine
    engine = BenchmarkEngine(
        output_dir=args.output_dir,
        session_name=args.session_name,
        mock_llm=args.mock_llm,
        trials_per_query=args.trials,
        force_resume=args.force_resume,
    )

    print(f"Environment Fingerprint: {engine.fingerprint.version_id}")
    print(f"Existing Completed Runs : {len(engine.completed_runs)} (query, mode, trial) keys in ledger\n")

    if not args.export_only:
        def on_completed(rec):
            status_tag = f"[{rec.feasibility.value}]"
            cost_str = f"${rec.recomputed_cost_usd:.2f}" if rec.recomputed_cost_usd is not None else "—"
            q_id = getattr(rec, "query_id", "")
            t_num = getattr(rec, "trial", 1)
            print(f"  ✓ Saved: {q_id:<22} | Mode {rec.mode} (Trial {t_num}) | Feasibility: {status_tag:<8} | Cost: {cost_str:<10} | Latency: {rec.total_duration_ms:.1f}ms")

        print("Executing benchmark runs across manifest...")
        records, tt_summary = engine.run_benchmark_on_manifest(
            query_manifest=query_manifest,
            on_record_completed=on_completed,
            verbose=args.verbose,
            final_study=args.final_study,
        )
    else:
        records = engine.saved_records
        # Reconstruct matched runs
        q_lookup = {q.get("query_id", ""): q for q in query_manifest if q.get("query_id")}
        from collections import defaultdict
        from src.benchmarks.truth_table import TruthTableEngine
        q_trials = defaultdict(dict)
        for r in records:
            q_id = getattr(r, "query_id", None) or r.requirements.get("query_id", "")
            trial_num = getattr(r, "trial", 1)
            q_trials[(q_id, trial_num)][r.mode] = r
        matched_runs = []
        for (q_id, trial_num), mode_dict in q_trials.items():
            exp = q_lookup.get(q_id, {}).get("expected_outcome", "FEASIBLE")
            pat = TruthTableEngine.compute_query_pattern(mode_dict, expected_outcome=exp)
            pat["query_id"] = q_id
            pat["trial"] = trial_num
            matched_runs.append(pat)
        engine.export_all_csvs(query_manifest, matched_runs)

    # Compute & Print Analytics
    analytics = BenchmarkVisualizer.compute_analytics(engine.saved_records, query_manifest)
    cli_report = BenchmarkVisualizer.render_cli_summary(analytics)
    print("\n" + cli_report)

    # List Exported Files
    csv_files = engine.export_all_csvs(query_manifest, matched_runs=getattr(engine, "matched_runs", None))
    print("\nExported Canonical CSV Artifacts:")
    for name, p in csv_files.items():
        print(f"  • {name:<20}: {p}")
    print(f"  • durable_ledger      : {engine.ledger_path}\n")


if __name__ == "__main__":
    main()
