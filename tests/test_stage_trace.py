"""Unit and Integration Tests for Step 4 Terminal Stage Trace."""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.proof.stage_trace import (
    run_pipeline_trace,
    load_queries_from_file,
)


class TestStep4StageTrace(unittest.TestCase):
    """Test suite verifying genuine Step 4 pipeline terminal stage-by-stage tracing."""

    def test_trace_mode_4_vm_success(self):
        """Verify complete 6-stage trace for Feasible VM query."""
        query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
        res = run_pipeline_trace(query, mode=4, silent=True)

        self.assertTrue(res["passed"])
        self.assertEqual(res["mode"], 4)
        self.assertIsNone(res["failed_stage"])
        self.assertEqual(res["verdict"], "Feasible against checked constraints")
        self.assertGreater(res["cost"], 0.0)

    def test_trace_mode_3_scaling_violation(self):
        """Verify Mode 3 execution stops Stage 6 and catches scaling budget overflow / constraint violation."""
        query = "Continuous dynamic scaling with target CPU 70% under $20 budget cap"
        res = run_pipeline_trace(query, mode=3, silent=True)

        self.assertFalse(res["passed"])
        self.assertEqual(res["mode"], 3)
        self.assertEqual(res["failed_stage"], 5)
        self.assertTrue(
            "Constraint violation" in res["verdict"] or "Solver reported infeasible" in res["verdict"]
        )

    def test_trace_dr_z3_success(self):
        """Verify Z3 Disaster Recovery multi-region routing and independent verification."""
        query = "We need disaster recovery setup across AWS and GCP with 99.99% SLA and max 50ms latency."
        res = run_pipeline_trace(query, mode=4, silent=True)

        self.assertTrue(res["passed"])
        self.assertEqual(res["verdict"], "Feasible against checked constraints")
        self.assertGreater(res["cost"], 0.0)

    def test_trace_parser_failure_stops_at_stage_1(self):
        """Verify non-cloud input genuinely stops at Stage 1 without fabricating later stages."""
        query = "Quantum circuit simulation tensor network optimization"
        res = run_pipeline_trace(query, mode=4, silent=True)

        self.assertFalse(res["passed"])
        self.assertEqual(res["failed_stage"], 1)
        self.assertIn("PARSER_FAILED", res["verdict"])

    def test_trace_infeasible_budget_detection(self):
        """Verify impossible budget ceiling reports infeasible at Stage 5."""
        query = "Deploy a heavy database requiring 64 vCPUs and 256GB RAM on Azure with budget $15"
        res = run_pipeline_trace(query, mode=4, silent=True)

        self.assertFalse(res["passed"])
        self.assertEqual(res["failed_stage"], 5)
        self.assertEqual(res["verdict"], "Solver reported infeasible")

    def test_load_queries_file(self):
        """Verify loading query batches from JSON."""
        queries = load_queries_from_file("data/diagnostic_queries.json")
        self.assertEqual(len(queries), 6)
        self.assertIn("query", queries[0])
        self.assertIn("id", queries[0])


if __name__ == "__main__":
    unittest.main()
