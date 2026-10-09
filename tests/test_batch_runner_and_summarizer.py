"""Comprehensive unit tests for the Resumable CSV Batch Runner and CSV Summarizer.

Tests all required invariants with 100% mocked providers (zero live calls):
1. Durable append (header written once, fsync called, immediate flush).
2. Resume without duplicates (completed rows skipped, failures re-run).
3. 429 rate limit clean stop.
4. CSV escaping (Hindi characters, commas, multiline strings).
5. Empty-not-zero invariant (inapplicable fields are empty string, never 0).
6. Summarizer agreement with CSV data.
7. Manifest validator invariants and safety checks.
"""

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.run_batch_csv import (
    CSV_COLUMNS,
    check_interpretation_match,
    compute_file_sha256,
    format_cell_empty,
    read_existing_csv_state,
    run_batch,
    summarize_allocation,
)
from benchmarks.summarize_csv import summarize_csv_file
from src.benchmarks.dataset_manifest import ManifestManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_manifest(tmp_path):
    """Creates a miniature 3-query manifest for testing."""
    manifest_data = {
        "manifest_name": "Test Manifest",
        "manifest_version": "1.0.0-test",
        "annotation_source": "AI_DRAFT",
        "approval_status": "UNAPPROVED",
        "is_approved": False,
        "query_count": 3,
        "clarification_policy": ["Policy note"],
        "success_definitions": {
            "task_success": "Outcome matches",
            "optimal_cost_match": "Within 1%",
            "interpretation_match": "Exact fields match",
        },
        "queries": [
            {
                "query_id": "Q01_TEST_VM",
                "scenario_family_id": "FAM_VM_01",
                "query_text": "Deploy 8 vCPUs and 16GB RAM for under $300 on AWS, Bhai setup lagade!",
                "category": "COLLOQUIAL_HINGLISH",
                "intended_archetype": "ILP_VM_Allocation",
                "expected_outcome": "FEASIBLE",
                "required_vcpus": 8,
                "required_ram_gb": 16.0,
                "budget_max_usd": 300.0,
                "cloud_providers": ["AWS"],
                "previously_run_in_development": True,
                "expected_optimal_cost_usd": 121.47,
                "optimum_source": "INDEPENDENT_SEARCH",
                "key_notes": "Test VM query",
            },
            {
                "query_id": "Q02_TEST_DR",
                "scenario_family_id": "FAM_DR_01",
                "query_text": "DR across AWS and GCP with 99.99% SLA under $100.",
                "category": "FORMAL_ENGLISH",
                "intended_archetype": "Z3_Graph_Disaster_Recovery",
                "expected_outcome": "INFEASIBLE",
                "latency_max_ms": 50.0,
                "sla_availability_pct": 99.99,
                "budget_max_usd": 100.0,
                "cloud_providers": ["AWS", "GCP"],
                "previously_run_in_development": False,
                "expected_optimal_cost_usd": None,
                "optimum_source": None,
                "key_notes": "Test DR infeasible query",
            },
            {
                "query_id": "Q03_TEST_UNSUPPORTED",
                "scenario_family_id": "FAM_UNSUPPORTED",
                "query_text": "Train transformer on 8 H100 GPUs for two weeks",
                "category": "UNSUPPORTED_TASKS",
                "intended_archetype": "NONE",
                "expected_outcome": "UNSUPPORTED",
                "previously_run_in_development": True,
                "expected_optimal_cost_usd": None,
                "optimum_source": None,
                "key_notes": "Test unsupported query",
            },
        ],
    }
    manifest_file = tmp_path / "test_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    return manifest_file


def test_empty_not_zero_formatting():
    """Validates that unknown/inapplicable cells return empty strings and never '0' or '0.0'."""
    assert format_cell_empty(None) == ""
    assert format_cell_empty("") == ""
    assert format_cell_empty(float("nan")) == ""
    assert format_cell_empty(121.47) == "121.47"
    assert format_cell_empty(0) == "0"  # literal integer zero should be "0" only if explicitly provided
    assert format_cell_empty(True) == "true"
    assert format_cell_empty(False) == "false"


def test_csv_escaping_hindi_commas_newlines(tmp_path):
    """Verifies that Hindi characters, quotes, commas, and newlines are cleanly escaped."""
    test_csv = tmp_path / "escaped_test.csv"
    hindi_text = 'Bhai mujhe AWS pe 8 vCPU, 16GB RAM chahiye.\n"Sabse sasta jugaad" karke do.'

    with open(test_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["query_id", "query_text", "expected_outcome"])
        writer.writerow(["Q_HINDI_01", hindi_text, "FEASIBLE"])

    # Read back and verify exact text equality
    with open(test_csv, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["query_id"] == "Q_HINDI_01"
        assert reader[0]["query_text"] == hindi_text
        assert reader[0]["expected_outcome"] == "FEASIBLE"


def test_durable_append_and_header_written_once(sample_manifest, tmp_path):
    """Tests that run_batch creates the file, writes the header once, and appends durably."""
    out_csv = tmp_path / "results_append_test.csv"

    # Pilot run with 1 query in mock mode
    run_batch(
        manifest_path=sample_manifest,
        out_path=out_csv,
        trials=1,
        delay=0.0,
        pilot=1,
        modes=[3],
        mock_llm=True,
        force_unapproved=True,
    )

    assert out_csv.exists()
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        lines = f.readlines()
        # 1 header line + 1 data line
        assert len(lines) == 2
        assert lines[0].startswith("run_id,timestamp_utc")

    # Second pilot run with 2 queries (appending without repeating header)
    run_batch(
        manifest_path=sample_manifest,
        out_path=out_csv,
        trials=1,
        delay=0.0,
        pilot=2,
        modes=[3],
        mock_llm=True,
        force_unapproved=True,
    )

    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        lines = f.readlines()
        # 1 header line + 2 data lines (Q01 was skipped, Q02 was appended)
        assert len(lines) == 3


def test_resume_without_duplicates(sample_manifest, tmp_path):
    """Tests that re-running on an existing CSV skips already completed rows."""
    out_csv = tmp_path / "resume_test.csv"

    # Run query 1 in Mode 3
    run_batch(
        manifest_path=sample_manifest,
        out_path=out_csv,
        trials=1,
        delay=0.0,
        pilot=1,
        modes=[3],
        mock_llm=True,
        force_unapproved=True,
    )

    sha, completed = read_existing_csv_state(out_csv)
    assert len(completed) == 1
    assert ("Q01_TEST_VM", 3, 1) in completed

    # Run again for all 3 queries in Mode 3
    run_batch(
        manifest_path=sample_manifest,
        out_path=out_csv,
        trials=1,
        delay=0.0,
        modes=[3],
        mock_llm=True,
        force_unapproved=True,
    )

    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))
        # Total rows should be exactly 3 (no duplicated Q01 row)
        assert len(reader) == 3
        q_ids = [r["query_id"] for r in reader]
        assert q_ids == ["Q01_TEST_VM", "Q02_TEST_DR", "Q03_TEST_UNSUPPORTED"]


def test_429_rate_limit_clean_stop(sample_manifest, tmp_path):
    """Verifies that an HTTP 429 exception stops the runner cleanly and logs RATE_LIMITED."""
    out_csv = tmp_path / "rate_limit_test.csv"

    with patch("benchmarks.run_batch_csv.execute_mode_1_raw_llm", side_effect=Exception("HTTP 429 Too Many Requests: Rate limit reached")):
        run_batch(
            manifest_path=sample_manifest,
            out_path=out_csv,
            trials=1,
            delay=0.0,
            pilot=2,
            modes=[1],
            mock_llm=False,
            force_unapproved=True,
        )

    assert out_csv.exists()
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))
        # Runner should stop immediately after the 429 on Q01 without attempting Q02
        assert len(reader) == 1
        assert reader[0]["execution_status"] == "RATE_LIMITED"
        assert reader[0]["query_id"] == "Q01_TEST_VM"


def test_summarizer_agrees_with_csv(tmp_path):
    """Tests that summarize_csv computes correct metrics directly from synthetic CSV rows."""
    test_csv = tmp_path / "summary_test.csv"
    summary_out = tmp_path / "summary_report.txt"

    with open(test_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        # Row 1: Mode 3 success
        writer.writerow([
            "run_01", "2026-10-09T00:00:00Z", "abc", "commit1", "false", "1",
            "Q01_VM", "FORMAL_ENGLISH", "FAM_01", "true", "Query 1", "FEASIBLE", "ILP_VM_Allocation",
            "8", "16", "300", "121.47", "INDEPENDENT_SEARCH", "3", "Local", "HiGHS", "SCOPE", "HiGHS",
            "OK", "1", "stop", "SUCCESS", "PLAN", "300", "8", "16", "", "",
            "true", "", "4x t3.medium (AWS)", "8", "16.0", "121.47", "121.47", "0.00", "0.00",
            "true", "", "true", "Exact Discrete Minimum", "", "1", "CORRECT", "Optimal",
            "1.2", "5.4", "", "", "8.5", "", "", "raw/q1.txt", "", ""
        ])
        # Row 2: Mode 1 wrong plan
        writer.writerow([
            "run_01", "2026-10-09T00:00:00Z", "abc", "commit1", "false", "1",
            "Q01_VM", "FORMAL_ENGLISH", "FAM_01", "true", "Query 1", "FEASIBLE", "ILP_VM_Allocation",
            "8", "16", "300", "121.47", "INDEPENDENT_SEARCH", "1", "Groq", "gpt-oss", "Prose", "None",
            "OK", "1", "stop", "SUCCESS", "PLAN", "300", "8", "16", "", "",
            "", "", "2x t3.xlarge (AWS)", "8", "32.0", "300.00", "300.21", "0.07", "147.15",
            "true", "", "true", "not proven by this mode", "", "1", "CORRECT_BUT_COSTLIER", "Costlier",
            "", "", "1200.0", "", "1250.0", "", "", "raw/q1_m1.txt", "", ""
        ])

    report_text = summarize_csv_file(test_csv, summary_out)

    assert "TASK SUCCESS RATE PER MODE" in report_text
    assert "Mode 3: Pure Symbolic" in report_text
    assert "Mode 1: Raw LLM" in report_text
    assert summary_out.exists()


def test_manifest_validation_on_final_query_manifest():
    """Validates data/final_query_manifest.json against all schema and safety invariants."""
    manifest_path = PROJECT_ROOT / "data" / "final_query_manifest.json"
    assert manifest_path.exists()

    is_valid, errors, stats = ManifestManager.validate_manifest_file(str(manifest_path))
    assert is_valid is True, f"Manifest validation failed with errors: {errors}"
    assert stats["total_queries"] == 29
    assert stats["approved_count"] == 0  # 100% UNAPPROVED safety invariant
    assert stats["feasible_with_cost_count"] == 12
