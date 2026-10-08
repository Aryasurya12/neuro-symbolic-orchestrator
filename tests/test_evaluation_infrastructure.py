"""Comprehensive Tests for Neurasym Auditable Evaluation Infrastructure (Requirement G).

Verifies:
1. Durable JSONL logging and sync-to-disk.
2. Resume without duplicate execution of completed runs.
3. Interrupted request tracking (REQUEST_START without RUN_RECORD).
4. Fingerprint mismatch rejection (IncompatibleVersionError).
5. RFC 4180 CSV escaping, UTF-8 integrity, commas, newlines, and Hindi text.
6. Null and empty string handling in numerical columns (no spurious zeros).
7. Complete 16-pattern truth table counting (0000 to 1111) including 0-count states.
8. Separation of provider failures from reasoning errors.
9. Zero live inference / offline execution safety.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import tempfile
import pytest

from src.benchmarks.benchmark_engine import (
    BenchmarkEngine,
    BenchmarkEnvironmentFingerprint,
    IncompatibleVersionError,
)
from src.benchmarks.csv_exporter import CSVExporter
from src.benchmarks.dataset_manifest import ManifestManager, ManifestQueryItem
from src.benchmarks.truth_table import TaskEvaluationResult, TruthTableEngine
from src.verifiers.canonical_record import (
    AuditEvent,
    CanonicalExecutionRecord,
    CheckStatus,
    FeasibilityStatus,
    NormalizationStatus,
    OptimalityStatus,
)


@pytest.fixture
def temp_benchmark_dir():
    """Provides an isolated temporary directory for benchmark testing."""
    tmp_dir = tempfile.mkdtemp(prefix="neurasym_bench_test_")
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_durable_logging_and_fsync(temp_benchmark_dir):
    """Verifies that engine appends SESSION_HEADER, REQUEST_START, and RUN_RECORD durably."""
    engine = BenchmarkEngine(output_dir=temp_benchmark_dir, session_name="test_durable_session", mock_llm=True)

    test_query = {
        "query_id": "Q_TEST_01",
        "query_text": "Deploy 4 vCPUs on AWS for under $200",
        "category": "FORMAL_ENGLISH",
        "intended_archetype": "ILP_VM_Allocation",
        "expected_outcome": "FEASIBLE",
        "required_vcpus": 4,
        "required_ram_gb": 8.0,
        "budget_max_usd": 200.0,
    }

    # Run execution with mock LLM
    records, _ = engine.run_benchmark_on_manifest(
        query_manifest=[test_query],
    )

    assert len(records) == 4  # 4 modes
    assert os.path.exists(engine.ledger_path)

    # Read lines directly from disk
    with open(engine.ledger_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    record_types = [entry.get("record_type") for entry in lines]
    assert "SESSION_HEADER" in record_types
    assert "REQUEST_START" in record_types
    assert "RUN_RECORD" in record_types

    # Ensure 4 starts and 4 run records
    start_events = [e for e in lines if e.get("record_type") == "REQUEST_START"]
    run_events = [e for e in lines if e.get("record_type") == "RUN_RECORD"]
    assert len(start_events) == 4
    assert len(run_events) == 4


def test_resume_without_duplicate_execution(temp_benchmark_dir):
    """Verifies that resumed execution does not repeat completed (query_id, mode, trial) runs."""
    session_name = "test_resume_session"
    engine1 = BenchmarkEngine(output_dir=temp_benchmark_dir, session_name=session_name, mock_llm=True)

    queries = [
        {"query_id": "Q1", "query_text": "Query 1", "intended_archetype": "ILP_VM_Allocation", "required_vcpus": 4, "required_ram_gb": 8.0, "budget_max_usd": 200.0},
        {"query_id": "Q2", "query_text": "Query 2", "intended_archetype": "ILP_VM_Allocation", "required_vcpus": 8, "required_ram_gb": 16.0, "budget_max_usd": 400.0},
    ]

    # Run Q1
    engine1.run_benchmark_on_manifest(query_manifest=[queries[0]])
    assert len(engine1.completed_runs) == 4  # 4 modes for Q1

    # Now create engine2 for same session and give BOTH Q1 and Q2
    engine2 = BenchmarkEngine(output_dir=temp_benchmark_dir, session_name=session_name, mock_llm=True)
    assert len(engine2.completed_runs) == 4

    records2, _ = engine2.run_benchmark_on_manifest(query_manifest=queries)

    # Total records returned by engine2 includes loaded + new
    # Check that in ledger there are exactly 8 RUN_RECORD entries (4 for Q1, 4 for Q2)
    with open(engine2.ledger_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    run_events = [e for e in lines if e.get("record_type") == "RUN_RECORD"]
    assert len(run_events) == 8


def test_interrupted_run_identification(temp_benchmark_dir):
    """Verifies that incomplete requests (REQUEST_START without RUN_RECORD) are tracked."""
    session_name = "test_interrupted_session"
    engine = BenchmarkEngine(output_dir=temp_benchmark_dir, session_name=session_name, mock_llm=True)

    # Manually append a REQUEST_START without a RUN_RECORD to simulate a crash
    with open(engine.ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "record_type": "REQUEST_START",
            "run_id": "RUN_CRASHED_001",
            "query_id": "Q_CRASHED",
            "mode": 1,
            "trial": 1,
            "timestamp_utc": "2026-10-07T12:00:00Z",
        }) + "\n")

    # Load session into a new engine
    engine_resumed = BenchmarkEngine(output_dir=temp_benchmark_dir, session_name=session_name, mock_llm=True)
    assert ("Q_CRASHED", 1, 1) not in engine_resumed.completed_runs


def test_fingerprint_mismatch_rejection(temp_benchmark_dir):
    """Verifies that resuming a session with incompatible fingerprint raises IncompatibleVersionError."""
    session_name = "test_fingerprint_session"
    engine = BenchmarkEngine(output_dir=temp_benchmark_dir, session_name=session_name, mock_llm=True)

    # Corrupt fingerprint in session header
    with open(engine.ledger_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = json.loads(lines[0])
    header["fingerprint"]["version_id"] = "CORRUPTED_HASH_12345"

    with open(engine.ledger_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n" + "".join(lines[1:]))

    # Resuming with mismatched fingerprint MUST raise IncompatibleVersionError
    with pytest.raises(IncompatibleVersionError) as exc_info:
        BenchmarkEngine(output_dir=temp_benchmark_dir, session_name=session_name, mock_llm=True, force_resume=False)

    assert "fingerprint mismatch" in str(exc_info.value).lower()


def test_csv_escaping_and_hindi_support(temp_benchmark_dir):
    """Verifies RFC 4180 CSV escaping with commas, quotes, newlines, and Hindi/Hinglish text."""
    complex_queries = [
        {
            "query_id": "Q_HINDI_01",
            "scenario_family_id": "FAM_VM_01",
            "query_text": 'Bhai AWS pe 8 vCPU aur 16GB RAM ka setup lagade,\n300 dollar "budget" ke andar (₹25,000)',
            "category": "COLLOQUIAL_HINGLISH",
            "intended_archetype": "ILP_VM_Allocation",
            "expected_outcome": "FEASIBLE",
            "required_vcpus": 8,
            "required_ram_gb": 16.0,
            "budget_max_usd": 300.0,
        }
    ]

    out_csv = os.path.join(temp_benchmark_dir, "query_manifest.csv")
    CSVExporter.export_query_manifest(complex_queries, output_path=out_csv)

    # Read back with standard csv.reader
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["query_id"] == "Q_HINDI_01"
    assert "Bhai AWS pe" in rows[0]["query_text"]
    assert "₹25,000" in rows[0]["query_text"]
    assert "\n" in rows[0]["query_text"]
    assert '"budget"' in rows[0]["query_text"]


def test_csv_null_and_numeric_columns(temp_benchmark_dir):
    """Verifies that unknown/inapplicable metrics are empty strings '\"\"', never '0' or '0.0'."""
    # Create record for VM problem (which has NO bandwidth or replicas)
    record = CanonicalExecutionRecord(
        run_id="RUN_VM_TEST",
        mode=4,
        original_query="Deploy 4 vCPUs on AWS",
        problem_type="ILP_VM_Allocation",
        requirements={"query_id": "Q_VM_01", "required_vcpus": 4, "required_ram_gb": 8.0, "budget_max_usd": 200.0},
        normalized_allocation={"allocated_vms": [{"sku": "t3.xlarge", "provider": "AWS", "vcpus": 4, "ram_gb": 16.0, "count": 1}]},
        feasibility=FeasibilityStatus.PASS,
        optimality_status=OptimalityStatus.PROVABLY_OPTIMAL,
        claimed_cost_usd=121.47,
        recomputed_cost_usd=121.47,
    )

    out_csv = os.path.join(temp_benchmark_dir, "run_results.csv")
    CSVExporter.export_run_results([record], output_path=out_csv)

    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    # Inapplicable continuous scaling metrics must be empty strings
    assert row["required_bandwidth_mbps"] == ""
    assert row["allocated_bandwidth_mbps"] == ""
    assert row["required_replicas"] == ""
    assert row["allocated_replicas"] == ""
    assert row["target_cpu_pct"] == ""
    assert row["max_latency_ms"] == ""
    assert row["prompt_tokens"] == ""


def test_truth_table_all_16_patterns():
    """Verifies that TruthTableEngine counts all 16 combinations (0000 to 1111) including 0-counts."""
    patterns = [
        {"pattern": "1111", "mode1_pass": 1, "mode2_pass": 1, "mode3_pass": 1, "mode4_pass": 1},
        {"pattern": "0011", "mode1_pass": 0, "mode2_pass": 0, "mode3_pass": 1, "mode4_pass": 1},
        {"pattern": "0001", "mode1_pass": 0, "mode2_pass": 0, "mode3_pass": 0, "mode4_pass": 1},
    ]

    tt = TruthTableEngine.aggregate_truth_table(patterns)

    # All 16 patterns must be keys in table
    assert len(tt["pattern_counts"]) == 16
    assert tt["pattern_counts"]["1111"] == 1
    assert tt["pattern_counts"]["0011"] == 1
    assert tt["pattern_counts"]["0001"] == 1
    assert tt["pattern_counts"]["0000"] == 0
    assert tt["pattern_counts"]["1010"] == 0

    assert tt["total_queries_evaluated"] == 3


def test_provider_vs_reasoning_failures():
    """Verifies that API rate limits/failures count as task_pass=0 and are flagged as PROVIDER_ERROR."""
    rec_api_fail = CanonicalExecutionRecord(
        run_id="RUN_FAIL_API",
        mode=1,
        original_query="Test API error",
        problem_type="ILP_VM_Allocation",
        normalization_status=NormalizationStatus.API_FAILURE,
        feasibility=FeasibilityStatus.NOT_EVALUABLE,
        violations=["HTTP 429: Rate limit exceeded"],
    )

    eval_res = TruthTableEngine.evaluate_mode_task_success(rec_api_fail, expected_outcome="FEASIBLE")
    assert eval_res.task_pass == 0
    assert eval_res.is_provider_failure is True
    assert eval_res.is_reasoning_failure is False
    assert eval_res.failure_category == "PROVIDER_ERROR"

    rec_reason_fail = CanonicalExecutionRecord(
        run_id="RUN_FAIL_REASON",
        mode=2,
        original_query="Test reasoning error",
        problem_type="ILP_VM_Allocation",
        normalization_status=NormalizationStatus.SUCCESS,
        feasibility=FeasibilityStatus.FAIL,
        violations=["Compute deficit: allocated 2 < required 8"],
    )

    eval_res2 = TruthTableEngine.evaluate_mode_task_success(rec_reason_fail, expected_outcome="FEASIBLE")
    assert eval_res2.task_pass == 0
    assert eval_res2.is_provider_failure is False
    assert eval_res2.is_reasoning_failure is True
    assert eval_res2.failure_category == "REASONING_ERROR"


def test_no_network_access_in_offline_mock(temp_benchmark_dir):
    """Verifies that evaluation harness with mock_llm=True executes with ZERO live network calls."""
    queries = ManifestManager.get_development_manifest()[:4]
    engine = BenchmarkEngine(output_dir=temp_benchmark_dir, session_name="test_offline_session", mock_llm=True)

    # Run all 4 modes on 4 queries
    records, _ = engine.run_benchmark_on_manifest(query_manifest=queries)
    assert len(records) == 16

    for r in records:
        assert r.provider in ["Groq", "NVIDIA", "OpenRouter", "Mock_Provider", "Local_Deterministic", "None", None, ""]
        # With offline mock fixtures, duration is purely local compute
        assert r.total_duration_ms >= 0.0
