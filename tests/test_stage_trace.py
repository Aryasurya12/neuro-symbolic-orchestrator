"""Unit and Integration Tests for Stage-by-Stage Terminal Trace."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.proof.stage_trace import (
    run_stage_by_stage_trace,
    save_stage_trace_json,
    trace_ilp_solver_internals,
    trace_pso_solver_internals,
    trace_z3_solver_internals,
)
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.scope_parser import SCOPEParser
from src.semantic.carm_matcher import CARMMatcher


class TestStageTrace(unittest.TestCase):
    """Test suite verifying real internal stage-by-stage tracing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trace_ilp_query(self):
        """Verify complete 6-stage trace for ILP VM Allocation query."""
        query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
        trace = run_stage_by_stage_trace(query, query_id="test_ilp", verbose_terminal=False)

        # Stage 1 Checks
        self.assertIn("stage_1_scope", as_dict(trace))
        s1 = trace.stage_1_scope
        self.assertIn("Budget_Limit_Max", s1["extracted_constraint_tokens"])
        self.assertIn("Resource_Min_vCPU", s1["extracted_constraint_tokens"])
        vcpu_trace = next(p for p in s1["parameter_extractions"] if p["field"] == "required_vcpus")
        self.assertTrue(vcpu_trace["matched"])
        self.assertEqual(vcpu_trace["converted_value"], 8)
        self.assertEqual(vcpu_trace["provenance"], "[EXPLICIT]")

        # Stage 2 Checks
        s2 = trace.stage_2_carm
        self.assertEqual(s2["winner"], "ILP_VM_Allocation")
        self.assertEqual(len(s2["archetype_scores"]), 3)
        self.assertGreater(s2["winner_score"], 0.0)

        # Stage 3 Checks
        s3 = trace.stage_3_contract
        self.assertEqual(s3["contract_fields"]["problem_type"], "ILP_VM_Allocation")
        self.assertEqual(s3["contract_fields"]["budget_max_usd"], 300.0)
        self.assertEqual(s3["contract_fields"]["required_vcpus"], 8)
        self.assertEqual(s3["contract_fields"]["required_ram_gb"], 16.0)

        # Stage 4 Checks
        s4 = trace.stage_4_dispatch
        self.assertEqual(s4["problem_type"], "ILP_VM_Allocation")
        self.assertEqual(s4["dispatched_function_name"], "solve_ilp_vm_knapsack")
        self.assertEqual(s4["dispatched_module"], "templates.ILP_VM_Knapsack_Allocation")

        # Stage 5 Checks
        s5 = trace.stage_5_solver
        self.assertEqual(s5["solver_type"], "ILP")
        self.assertTrue(s5["is_feasible"])
        self.assertEqual(len(s5["constraint_matrix_shape"]), 2)
        self.assertGreater(len(s5["sku_names"]), 0)

        # Stage 6 Checks
        s6 = trace.stage_6_explainer
        self.assertIn("status", s6["consumed_solver_fields"])
        self.assertIn("total_monthly_cost_usd", s6["consumed_solver_fields"])
        self.assertIn("NEURASYM FINOPS EXECUTIVE DEPLOYMENT REPORT", s6["generated_finops_report"])

    def test_trace_pso_query(self):
        """Verify complete 6-stage trace for Continuous PSO query."""
        query = "Continuous dynamic scaling with target CPU 70% under $1500"
        trace = run_stage_by_stage_trace(query, query_id="test_pso", verbose_terminal=False)

        # Stage 2 Route
        self.assertEqual(trace.stage_2_carm["winner"], "PSO_Continuous_Scaling")

        # Stage 4 Dispatch
        self.assertEqual(trace.stage_4_dispatch["dispatched_function_name"], "solve_pso_continuous_scaling")

        # Stage 5 Iterations
        s5 = trace.stage_5_solver
        self.assertEqual(s5["solver_type"], "PSO")
        self.assertGreaterEqual(len(s5["iteration_trace"]), 5)
        self.assertEqual(s5["iteration_trace"][0]["iteration"], 1)
        self.assertEqual(s5["iteration_trace"][-1]["iteration"], 50)

    def test_trace_z3_query(self):
        """Verify complete 6-stage trace for Z3 Disaster Recovery query."""
        query = "We need disaster recovery setup across AWS and GCP with 99.99% SLA and max 50ms latency."
        trace = run_stage_by_stage_trace(query, query_id="test_z3", verbose_terminal=False)

        # Stage 2 Route
        self.assertEqual(trace.stage_2_carm["winner"], "Z3_Graph_Disaster_Recovery")

        # Stage 4 Dispatch
        self.assertEqual(trace.stage_4_dispatch["dispatched_function_name"], "solve_z3_graph_disaster_recovery")

        # Stage 5 Z3 Assertions
        s5 = trace.stage_5_solver
        self.assertEqual(s5["solver_type"], "Z3_SMT")
        self.assertGreater(len(s5["constraint_clauses"]), 3)
        self.assertEqual(s5["solver_check_result"], "sat")
        self.assertTrue(s5["is_feasible"])

    def test_save_json_trace(self):
        """Verify JSON serialization and file writing."""
        query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
        trace = run_stage_by_stage_trace(query, query_id="test_save", verbose_terminal=False)
        out_file = save_stage_trace_json(trace, output_dir=self.temp_dir)

        self.assertTrue(Path(out_file).exists())
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["query_id"], "test_save")
        self.assertIn("stages", data)
        self.assertIn("stage_1_scope_parsing", data["stages"])
        self.assertIn("stage_2_carm_matching", data["stages"])
        self.assertIn("stage_3_contract_validation", data["stages"])
        self.assertIn("stage_4_solver_dispatch", data["stages"])
        self.assertIn("stage_5_solver_internals", data["stages"])
        self.assertIn("stage_6_explainer", data["stages"])


def as_dict(trace_obj):
    return {
        "stage_1_scope": trace_obj.stage_1_scope,
        "stage_2_carm": trace_obj.stage_2_carm,
        "stage_3_contract": trace_obj.stage_3_contract,
        "stage_4_dispatch": trace_obj.stage_4_dispatch,
        "stage_5_solver": trace_obj.stage_5_solver,
        "stage_6_explainer": trace_obj.stage_6_explainer,
    }


if __name__ == "__main__":
    unittest.main()
