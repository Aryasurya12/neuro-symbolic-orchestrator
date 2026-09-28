"""Part A Unit Test Suite: Semantic Front-End (SEM-2, SEM-3, SEM-4, SEM-5)

Compatible with both `pytest` and `python -m unittest`.

Covers:
    - CloudOptimizationContract validation boundaries (SEM-4).
    - CARMMatcher Jaccard score computation and template matching (SEM-2).
    - SCOPEParser natural language constraint extraction, USD and INR budget
      parsing, and full query-to-contract pipeline (SEM-3).
    - FinOpsExplainer dual currency (USD / INR) executive report generation (SEM-5).
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch, MagicMock
import pytest
from pydantic import ValidationError


from config.settings import settings
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.carm_matcher import CARMMatcher
from src.semantic.scope_parser import (
    SCOPEParser,
    parse_fallback_nemotron,
    parse_query_hybrid,
    parse_query_local,
)
from src.semantic.explainer import FinOpsExplainer



# ---------------------------------------------------------------------------
# SEM-4: CloudOptimizationContract
# ---------------------------------------------------------------------------

class TestCloudOptimizationContract(unittest.TestCase):
    def _valid_kwargs(self, **overrides) -> dict:
        base = dict(
            problem_type="ILP_VM_Allocation",
            cloud_providers=["AWS"],
            budget_max_usd=300.0,
            service_count=5,
            required_vcpus=2,
            required_ram_gb=4.0,
            latency_max_ms=50.0,
            sla_availability_pct=99.9,
        )
        base.update(overrides)
        return base

    def test_valid_contract_constructs_successfully(self):
        contract = CloudOptimizationContract(**self._valid_kwargs())
        self.assertEqual(contract.problem_type, "ILP_VM_Allocation")
        self.assertEqual(contract.budget_max_usd, 300.0)

    def test_budget_below_minimum_raises_value_error(self):
        with self.assertRaises(ValidationError) as ctx:
            CloudOptimizationContract(**self._valid_kwargs(budget_max_usd=5.0))
        self.assertIn("budget_max_usd", str(ctx.exception))

    def test_budget_exactly_at_zero_raises_value_error(self):
        with self.assertRaises(ValidationError):
            CloudOptimizationContract(**self._valid_kwargs(budget_max_usd=0.0))

    def test_budget_just_below_ten_raises_value_error(self):
        with self.assertRaises(ValidationError):
            CloudOptimizationContract(**self._valid_kwargs(budget_max_usd=9.99))

    def test_budget_exactly_ten_is_valid(self):
        contract = CloudOptimizationContract(**self._valid_kwargs(budget_max_usd=10.00))
        self.assertEqual(contract.budget_max_usd, 10.00)

    def test_invalid_problem_type_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            CloudOptimizationContract(**self._valid_kwargs(problem_type="Nonexistent_Solver"))

    def test_invalid_cloud_provider_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            CloudOptimizationContract(**self._valid_kwargs(cloud_providers=["IBM"]))

    def test_latency_above_1000ms_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            CloudOptimizationContract(**self._valid_kwargs(latency_max_ms=1500.0))

    def test_sla_below_90_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            CloudOptimizationContract(**self._valid_kwargs(sla_availability_pct=50.0))

    def test_sla_above_99_999_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            CloudOptimizationContract(**self._valid_kwargs(sla_availability_pct=99.9999))

    def test_default_sla_availability_pct(self):
        kwargs = self._valid_kwargs()
        kwargs.pop("sla_availability_pct")
        contract = CloudOptimizationContract(**kwargs)
        self.assertEqual(contract.sla_availability_pct, 99.9)

    def test_default_cloud_providers(self):
        kwargs = self._valid_kwargs()
        kwargs.pop("cloud_providers")
        contract = CloudOptimizationContract(**kwargs)
        self.assertEqual(contract.cloud_providers, ["AWS"])

    def test_non_positive_service_count_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            CloudOptimizationContract(**self._valid_kwargs(service_count=0))

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            CloudOptimizationContract(**self._valid_kwargs(unexpected_field="oops"))


# ---------------------------------------------------------------------------
# SEM-2: CARMMatcher
# ---------------------------------------------------------------------------

class TestCARMMatcher(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = CARMMatcher()

    def test_jaccard_score_identical_sets_is_one(self):
        s = {"Budget_Limit_Max", "Latency_Bound_Max"}
        self.assertEqual(self.matcher.compute_jaccard_score(s, s), 1.0)

    def test_jaccard_score_disjoint_sets_is_zero(self):
        a = {"Budget_Limit_Max"}
        b = {"Multi_Region_Disjoint"}
        self.assertEqual(self.matcher.compute_jaccard_score(a, b), 0.0)

    def test_jaccard_score_partial_overlap(self):
        a = {"Bounded_Integer_Variables", "Budget_Limit_Max"}
        b = {"Budget_Limit_Max", "Latency_Bound_Max"}
        self.assertAlmostEqual(self.matcher.compute_jaccard_score(a, b), 1.0 / 3.0, places=4)

    def test_jaccard_score_both_empty_is_zero(self):
        self.assertEqual(self.matcher.compute_jaccard_score(set(), set()), 0.0)

    def test_match_template_exact_ilp_match(self):
        query = {
            "Bounded_Integer_Variables",
            "Budget_Limit_Max",
            "Latency_Bound_Max",
            "Resource_Min_vCPU",
        }
        archetype, filename, score = self.matcher.match_template(query)
        self.assertEqual(archetype, "ILP_VM_Allocation")
        self.assertEqual(filename, "ilp_vm_allocation_template.py")
        self.assertEqual(score, 1.0)

    def test_match_template_exact_z3_match(self):
        query = {
            "Multi_Region_Disjoint",
            "SLA_Availability_Min",
            "Inter_Node_Latency_Max",
        }
        archetype, filename, score = self.matcher.match_template(query)
        self.assertEqual(archetype, "Z3_Graph_Disaster_Recovery")
        self.assertEqual(score, 1.0)

    def test_match_template_partial_match_still_selects_best_archetype(self):
        query = {"Continuous_Bandwidth_Range", "CPU_Threshold_Max"}
        archetype, _, score = self.matcher.match_template(query)
        self.assertEqual(archetype, "PSO_Continuous_Scaling")
        self.assertTrue(0.0 < score < 1.0)

    def test_match_template_raises_on_empty_constraints(self):
        with self.assertRaises(ValueError):
            self.matcher.match_template(set())

    def test_template_index_has_three_archetypes(self):
        self.assertEqual(len(CARMMatcher.TEMPLATE_INDEX), 3)
        self.assertEqual(
            set(CARMMatcher.TEMPLATE_INDEX.keys()),
            {
                "ILP_VM_Allocation",
                "PSO_Continuous_Scaling",
                "Z3_Graph_Disaster_Recovery",
            },
        )


# ---------------------------------------------------------------------------
# SEM-3: SCOPEParser
# ---------------------------------------------------------------------------

class TestSCOPEParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = SCOPEParser()

    def test_extract_constraints_detects_ilp_signals(self):
        text = "Deploy 5 microservices on AWS, max budget $300, 2 vCPUs, latency under 50ms."
        constraints = self.parser.extract_constraints_from_text(text)
        self.assertIn("Bounded_Integer_Variables", constraints)
        self.assertIn("Budget_Limit_Max", constraints)
        self.assertIn("Resource_Min_vCPU", constraints)
        self.assertIn("Latency_Bound_Max", constraints)

    def test_extract_constraints_detects_z3_signals(self):
        text = "Distribute database across AWS and GCP for disaster recovery, 99.99% SLA, max 20ms latency."
        constraints = self.parser.extract_constraints_from_text(text)
        self.assertIn("SLA_Availability_Min", constraints)
        self.assertIn("Inter_Node_Latency_Max", constraints)

    def test_extract_constraints_detects_rupee_budget_signals(self):
        text_rupee = "Deploy discrete VMs for microservices under ₹25000 budget with 4 vCPUs"
        constraints = self.parser.extract_constraints_from_text(text_rupee)
        self.assertIn("Budget_Limit_Max", constraints)

        text_inr = "Continuous dynamic scaling with target CPU 70% under 30000 INR"
        constraints_inr = self.parser.extract_constraints_from_text(text_inr)
        self.assertIn("Budget_Limit_Max", constraints_inr)

    def test_extract_constraints_empty_text_returns_empty_set(self):
        self.assertEqual(self.parser.extract_constraints_from_text(""), set())

    def test_extract_constraints_irrelevant_text_returns_empty_set(self):
        constraints = self.parser.extract_constraints_from_text("The weather today is sunny and pleasant.")
        self.assertEqual(constraints, set())

    def test_parse_query_to_contract_ilp_case(self):
        query = (
            "Deploy 5 microservices on AWS for e-commerce, max budget $300, "
            "2 vCPUs and 4GB RAM per service, latency under 50ms."
        )
        contract, filename, score = self.parser.parse_query_to_contract(query)

        self.assertIsInstance(contract, CloudOptimizationContract)
        self.assertEqual(contract.problem_type, "ILP_VM_Allocation")
        self.assertEqual(contract.cloud_providers, ["AWS"])
        self.assertEqual(contract.budget_max_usd, 300.0)
        self.assertEqual(contract.service_count, 5)
        self.assertEqual(contract.required_vcpus, 2)
        self.assertEqual(contract.required_ram_gb, 4.0)
        self.assertEqual(contract.latency_max_ms, 50.0)
        self.assertEqual(filename, "ilp_vm_allocation_template.py")
        self.assertTrue(0.0 < score <= 1.0)

    def test_parse_query_to_contract_z3_case(self):
        query = (
            "Distribute database across AWS and GCP for disaster recovery, "
            "99.99% SLA, max 20ms latency."
        )
        contract, filename, score = self.parser.parse_query_to_contract(query)

        self.assertEqual(contract.problem_type, "Z3_Graph_Disaster_Recovery")
        self.assertIn("AWS", contract.cloud_providers)
        self.assertIn("GCP", contract.cloud_providers)
        self.assertEqual(contract.latency_max_ms, 20.0)
        self.assertEqual(contract.sla_availability_pct, 99.99)
        self.assertEqual(filename, "z3_graph_disaster_recovery_template.py")
        self.assertTrue(0.0 < score <= 1.0)

    def test_parse_query_to_contract_inr_budget_conversion(self):
        # ₹25,500 INR / 85.0 = $300.00 USD
        query = "Deploy 5 microservices on AWS, budget of ₹25,500 INR, 2 vCPUs, latency under 50ms."
        contract, filename, score = self.parser.parse_query_to_contract(query)

        self.assertIsInstance(contract, CloudOptimizationContract)
        self.assertAlmostEqual(contract.budget_max_usd, 300.0, delta=1.0)
        self.assertEqual(contract.problem_type, "ILP_VM_Allocation")

    def test_parse_query_to_contract_returns_validated_contract_instance(self):
        query = "Deploy 3 microservices on Azure, budget $150, 4 vCPUs, latency under 100ms."
        contract, _, _ = self.parser.parse_query_to_contract(query)
        self.assertTrue(bool(contract.model_dump_json()))

    def test_parse_query_to_contract_falls_back_gracefully_on_sparse_query(self):
        query = "Optimize my cloud budget."
        contract, filename, score = self.parser.parse_query_to_contract(query)
        self.assertIsInstance(contract, CloudOptimizationContract)
        self.assertIsInstance(filename, str)
        self.assertTrue(0.0 <= score <= 1.0)


# ---------------------------------------------------------------------------
# SEM-5: FinOpsExplainer Dual Currency (USD / INR)
# ---------------------------------------------------------------------------

class TestFinOpsExplainer(unittest.TestCase):
    def test_usd_to_inr_conversion(self):
        usd_val = 100.0
        inr_val = FinOpsExplainer.usd_to_inr(usd_val, rate=85.0)
        self.assertEqual(inr_val, 8500.0)

    def test_report_includes_both_currencies(self):
        contract = CloudOptimizationContract(
            problem_type="ILP_VM_Allocation",
            budget_max_usd=300.0,
            required_vcpus=4,
            required_ram_gb=16.0,
        )
        solver_result = {
            "status": "OPTIMAL",
            "solver": "Exact_Branch_and_Bound_ILP",
            "total_monthly_cost_usd": 121.47,
            "cost_savings_usd": 178.53,
            "budget_utilized_pct": 40.49,
            "allocated_vms": [{
                "instance_type": "t3.xlarge",
                "provider": "AWS",
                "count": 1,
                "vcpus_per_vm": 4,
                "ram_gb_per_vm": 16.0,
                "monthly_cost": 121.47,
            }],
        }
        report = FinOpsExplainer.generate_report(contract, solver_result, exchange_rate=85.0)

        self.assertIn("USD", report)
        self.assertIn("INR", report)
        self.assertIn("₹", report)
        self.assertIn("$300.00 USD (₹25,500.00 INR)", report)
        self.assertIn("Exchange Rate        : 1 USD = 85.00 INR", report)

    def test_clean_llm_recommendations_strips_meta_prompts_and_placeholders(self):
        raw_llm_output = (
            "Here are the actionable recommendations:\n"
            "1. **Analyze the Input:** The workload requires 8 vCPUs and 16GB RAM on AWS.\n"
            "2. **Analyze the Input:** <recommendation> Enroll in AWS 1-Year Compute Savings Plans to save up to 34% on steady-state t3 compute.\n"
            "3. <recommendation> Track sustained CPU credit usage for 14 days and right-size instance types if average utilization remains below 40%.\n"
            "4. **Governance & Tagging:** Apply mandatory Cost Allocation Tags (`Environment`, `CostCenter`) across all resources for 100% cost attribution.\n"
        )
        recs = FinOpsExplainer._clean_llm_recommendations(raw_llm_output)

        self.assertEqual(len(recs), 3)
        # Verify no meta tags, headers, or placeholders remain
        for r in recs:
            self.assertNotIn("Analyze the Input", r)
            self.assertNotIn("<recommendation>", r)
            self.assertNotIn("</recommendation>", r)
            self.assertNotIn("Here are the actionable", r)
            self.assertFalse(r.startswith("1."))
            self.assertFalse(r.startswith("2."))
            self.assertFalse(r.startswith("3."))
            self.assertFalse(r.startswith("4."))

        self.assertTrue(recs[0].startswith("Enroll in AWS 1-Year Compute Savings Plans"))
        self.assertTrue(recs[1].startswith("Track sustained CPU credit usage"))
        self.assertTrue(recs[2].startswith("Governance & Tagging: Apply mandatory Cost Allocation Tags"))

    def test_clean_llm_recommendations_strips_bullet_points_and_step_headers(self):
        raw_llm_output = (
            "- Step 1: Commit to Azure 1-Year Reserved Instances with AHB to reduce base compute expenses by 45%.\n"
            "* Step 2: Establish Horizontal Pod Autoscaler cooldown periods targeting 70% CPU to prevent flapping.\n"
            "3. [Recommendation] Enable VPC endpoints and zstd compression to minimize cross-region data transfer fees.\n"
        )
        recs = FinOpsExplainer._clean_llm_recommendations(raw_llm_output)

        self.assertEqual(len(recs), 3)
        self.assertIn("Commit to Azure 1-Year Reserved Instances", recs[0])
        self.assertIn("Establish Horizontal Pod Autoscaler", recs[1])
        self.assertIn("Enable VPC endpoints and zstd compression", recs[2])
        for r in recs:
            self.assertNotIn("Step 1", r)
            self.assertNotIn("Step 2", r)
            self.assertNotIn("[Recommendation]", r)

    def test_clean_llm_recommendations_rejects_pure_meta_prompt_leakage(self):
        """Secondary Item 4 Verification: Reject prompt-echo and meta-text leakage."""
        raw_leaked_prompt = (
            "Role: Principal Cloud FinOps Architect\n"
            "1. Analyze the User's Request: The user wants disaster recovery.\n"
            "Problem Type: Z3_Graph_Disaster_Recovery\n"
            "Target Cloud: AWS and GCP\n"
            "Required Resources: 1 service, 4 vCPUs, 8GB RAM\n"
            "Monthly Budget Cap: $1000\n"
            "Placed Resources: AWS us-east-1, GCP us-central1\n"
            "Action item: analyze\n"
            "Done.\n"
        )
        recs = FinOpsExplainer._clean_llm_recommendations(raw_leaked_prompt)
        # All lines are meta-text or too short or lack valid FinOps recommendations
        self.assertEqual(len(recs), 0)

    def test_orchestrator_solver_hoisting_and_reuse(self):
        """Secondary Item 3 Verification: Hoisted solvers are initialized once and reused across queries."""
        from src.orchestrator.service import NeuroSymbolicOrchestrator
        orch = NeuroSymbolicOrchestrator()
        
        # Verify solvers exist as instance attributes
        self.assertIsNotNone(orch.ga_solver)
        self.assertIsNotNone(orch.pso_solver)
        self.assertIsNotNone(orch.z3_solver)
        
        z3_solver_id = id(orch.z3_solver)
        ga_solver_id = id(orch.ga_solver)
        pso_solver_id = id(orch.pso_solver)
        
        # Optimize 3 sequential contracts
        contract_z3 = CloudOptimizationContract(
            problem_type="Z3_Graph_Disaster_Recovery",
            cloud_providers=["AWS", "GCP"],
            budget_max_usd=1000.0,
            latency_max_ms=50.0,
            sla_availability_pct=99.99
        )
        contract_pso = CloudOptimizationContract(
            problem_type="PSO_Continuous_Scaling",
            cloud_providers=["AWS"],
            budget_max_usd=1500.0
        )
        contract_ilp = CloudOptimizationContract(
            problem_type="ILP_VM_Allocation",
            cloud_providers=["AWS"],
            budget_max_usd=300.0,
            required_vcpus=4,
            required_ram_gb=8.0
        )
        
        res1 = orch.optimize_contract(contract_z3)
        res2 = orch.optimize_contract(contract_pso)
        res3 = orch.optimize_contract(contract_ilp)
        
        # Solvers must maintain the exact same identity (no per-query re-instantiation)
        self.assertEqual(id(orch.z3_solver), z3_solver_id)
        self.assertEqual(id(orch.ga_solver), ga_solver_id)
        self.assertEqual(id(orch.pso_solver), pso_solver_id)
        
        self.assertEqual(res1["status"], "Feasible")
        self.assertEqual(res2["status"], "Feasible")
        self.assertEqual(res3["status"], "Feasible")



# ---------------------------------------------------------------------------
# SEM-3: Hybrid Architecture & Nemotron Fallback Tests
# ---------------------------------------------------------------------------

class TestHybridSCOPEParser(unittest.TestCase):
    def test_parse_query_local_success(self):
        query = "I need 4 vCPUs and 16GB RAM on AWS with budget $300"
        contract = parse_query_local(query)
        self.assertEqual(contract.problem_type, "ILP_VM_Allocation")
        self.assertEqual(contract.required_vcpus, 4)
        self.assertEqual(contract.required_ram_gb, 16.0)

    def test_parse_query_hybrid_local_path(self):
        query = "Continuous stream scaling with bandwidth 100 to 500 mbps autoscale cpu 70%"
        contract = parse_query_hybrid(query)
        self.assertEqual(contract.problem_type, "PSO_Continuous_Scaling")

    @patch("src.semantic.scope_parser.parse_fallback_nemotron")
    @patch("src.semantic.scope_parser.parse_query_local", side_effect=Exception("Local parse failed"))
    def test_parse_query_hybrid_fallback_on_local_error(self, mock_local, mock_fallback):
        mock_fallback.return_value = CloudOptimizationContract(
            problem_type="Z3_Graph_Disaster_Recovery",
            cloud_providers=["AWS", "Azure"],
            budget_max_usd=600.0,
            sla_availability_pct=99.99,
        )
        contract = parse_query_hybrid("complex ambiguous multi-region query")
        mock_local.assert_called_once()
        mock_fallback.assert_called_once_with("complex ambiguous multi-region query")
        self.assertEqual(contract.problem_type, "Z3_Graph_Disaster_Recovery")
        self.assertEqual(contract.budget_max_usd, 600.0)

    def test_has_cloud_intent_detects_cloud_queries(self):
        self.assertTrue(SCOPEParser.has_cloud_intent("I need 4 vCPUs on AWS"))
        self.assertTrue(SCOPEParser.has_cloud_intent("Deploy containers under $300 budget"))
        self.assertTrue(SCOPEParser.has_cloud_intent("Multi-region disaster recovery with 99.99% SLA"))
        self.assertFalse(SCOPEParser.has_cloud_intent("What is the capital of France?"))
        self.assertFalse(SCOPEParser.has_cloud_intent("Good morning!"))
        self.assertFalse(SCOPEParser.has_cloud_intent(""))

    def test_parse_query_local_raises_on_non_cloud_query(self):
        with self.assertRaises(ValueError) as ctx:
            parse_query_local("Tell me a funny joke about cats")
        self.assertIn("Smart Intent Check failed", str(ctx.exception))

    def test_numeric_validation_detects_mismatch(self):
        from src.semantic.scope_parser import validate_parsed_numbers
        # Contract with budget 500 but query mentions $300
        contract = CloudOptimizationContract(
            problem_type="ILP_VM_Allocation",
            budget_max_usd=500.0,
            required_vcpus=1,
            required_ram_gb=1.0,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_parsed_numbers("Deploy on AWS with budget $300", contract)
        self.assertIn("Local parser missed custom constraints", str(ctx.exception))

        # Contract with vCPUs 1 but query mentions 8 cores
        with self.assertRaises(ValueError) as ctx:
            validate_parsed_numbers("Deploy on AWS with 8 cores", contract)
        self.assertIn("Local parser missed custom constraints", str(ctx.exception))

    @patch("src.semantic.scope_parser.parse_fallback_nemotron")
    def test_parse_query_hybrid_fallback_on_missed_constraints(self, mock_fallback):
        mock_fallback.return_value = CloudOptimizationContract(
            problem_type="ILP_VM_Allocation",
            cloud_providers=["AWS"],
            budget_max_usd=300.0,
            required_vcpus=4,
            required_ram_gb=16.0,
        )
        with patch("src.semantic.scope_parser.parse_query_local") as mock_local:
            mock_local.return_value = CloudOptimizationContract(
                problem_type="ILP_VM_Allocation",
                budget_max_usd=500.0,
                required_vcpus=1,
                required_ram_gb=1.0,
            )
            contract = parse_query_hybrid("Need 4 vCPUs on AWS with budget $300")
            mock_fallback.assert_called_once_with("Need 4 vCPUs on AWS with budget $300")
            self.assertEqual(contract.budget_max_usd, 300.0)


# ---------------------------------------------------------------------------
# Benchmark Harness: Reasoning-Preamble Stripping & Extraction Tests
# ---------------------------------------------------------------------------

class TestBenchmarkReasoningHarness(unittest.TestCase):
    def test_mode1_cost_extraction_regression_budget_echo(self):
        """Fix 3 Step 4 Regression Test: Never extract input budget echo over calculated cost."""
        from benchmarks.run_4way_benchmark import extract_mode1_cost, strip_reasoning_preamble
        raw_text = "...Budget cap: $850...Estimated Total Monthly Cost: $612.40"
        cost = extract_mode1_cost(raw_text)
        self.assertEqual(cost, 612.40)
        self.assertNotEqual(cost, 850.00)

    def test_strip_reasoning_preamble_with_final_answer_delimiter(self):
        from benchmarks.run_4way_benchmark import strip_reasoning_preamble, extract_mode1_cost
        raw_content = (
            "Here's a thinking process:\n"
            "1. **Analyze User Input:**\n"
            "   - Target budget: $850\n"
            "   - Latency cap: 50ms\n"
            "2. **Evaluate SKUs:**\n"
            "   - c5.xlarge costs $124/mo\n"
            "Final Answer:\n"
            "Recommended Plan: 2x AWS c5.xlarge. Estimated Total Monthly Cost: $248.00."
        )
        cleaned = strip_reasoning_preamble(raw_content)
        self.assertNotIn("thinking process", cleaned.lower())
        self.assertTrue(cleaned.startswith("Recommended Plan: 2x AWS c5.xlarge"))
        cost = extract_mode1_cost(raw_content)
        self.assertEqual(cost, 248.00)

    def test_strip_reasoning_preamble_json_extraction(self):
        from benchmarks.run_4way_benchmark import strip_reasoning_preamble
        raw_mode2_content = (
            "Here's a thinking process:\n"
            "1. **Analyze User Input:**\n"
            "   - Need JSON format for DR placement\n"
            "{\n"
            '  "provider": "AWS",\n'
            '  "allocated_vms": [\n'
            '    {\n'
            '      "instance_type": "t3.medium",\n'
            '      "count": 2,\n'
            '      "vcpus_per_vm": 2,\n'
            '      "ram_gb_per_vm": 4.0,\n'
            '      "estimated_monthly_cost_usd": 60.50\n'
            '    }\n'
            '  ],\n'
            '  "total_vcpus": 4,\n'
            '  "total_ram_gb": 8.0,\n'
            '  "total_monthly_cost_usd": 60.50\n'
            "}"
        )
    def test_strip_reasoning_preamble_xml_tags(self):
        from benchmarks.run_4way_benchmark import strip_reasoning_preamble, extract_mode1_cost
        raw_xml_content = (
            "<thinking>\n"
            "The user needs 2 nodes with $850 budget cap.\n"
            "Let's select t3.large instances.\n"
            "</thinking>\n"
            "Recommendation: 2x AWS t3.large. Estimated Total Monthly Cost: $121.00."
        )
        cleaned = strip_reasoning_preamble(raw_xml_content)
        self.assertNotIn("thinking", cleaned.lower())
        self.assertIn("121.00", cleaned)
        cost = extract_mode1_cost(raw_xml_content)
        self.assertEqual(cost, 121.00)

    def test_extract_mode1_cost_truncation_safety(self):
        from benchmarks.run_4way_benchmark import extract_mode1_cost
        truncated_content = (
            "Here's a thinking process:\n"
            "1. **Analyze User Request:**\n"
            "   - Target budget cap: $850.00\n"
            "   - Minimum RAM: 16GB\n"
            "2. **Calculate"
        )
        # When finish_reason == 'length', cost MUST be None (no false fallback to $850)
        cost = extract_mode1_cost(truncated_content, finish_reason="length")
        self.assertIsNone(cost)

        # Even without finish_reason, if response only contains thinking preamble, cost MUST be None
        cost_no_finish = extract_mode1_cost(truncated_content)
        self.assertIsNone(cost_no_finish)

    def test_settings_constants_and_module_exports(self):
        """Verify USD_TO_INR_RATE, OPENROUTER_API_KEY, and OPENROUTER_MODEL are present and exported."""
        from config.settings import USD_TO_INR_RATE, OPENROUTER_API_KEY, OPENROUTER_MODEL, settings
        self.assertEqual(USD_TO_INR_RATE, 85.0)
        self.assertEqual(settings.USD_TO_INR_RATE, 85.0)
        self.assertIsInstance(OPENROUTER_API_KEY, str)
        self.assertIsInstance(settings.OPENROUTER_API_KEY, str)
        self.assertEqual(OPENROUTER_MODEL, "nvidia/nemotron-3.5-lightning:free")
        self.assertEqual(settings.OPENROUTER_MODEL, "nvidia/nemotron-3.5-lightning:free")

    def test_solver_templates_accept_standard_keyword_arguments(self):
        """Verify ILP, PSO, and Z3 templates accept budget_max_usd, target_providers, required_vcpus, required_ram_gb."""
        from templates.ILP_VM_Knapsack_Allocation import solve_ilp_vm_knapsack
        from templates.Continuous_PSO_Dynamic_Scaling import solve_pso_continuous_scaling
        from templates.Graph_SMT_Z3_MultiRegion_Placement import solve_z3_graph_disaster_recovery

        # 1. ILP
        res_ilp = solve_ilp_vm_knapsack(
            budget_max_usd=600.0,
            target_providers=["AWS"],
            required_vcpus=4,
            required_ram_gb=16.0,
        )
        self.assertIn("status", res_ilp)
        self.assertIn("total_monthly_cost_usd", res_ilp)

        # 2. PSO
        res_pso = solve_pso_continuous_scaling(
            budget_max_usd=600.0,
            target_providers=["AWS"],
            required_vcpus=4,
            required_ram_gb=16.0,
        )
        self.assertIn("status", res_pso)
        self.assertIn("estimated_monthly_cost_usd", res_pso)

        # 3. Z3
        res_z3 = solve_z3_graph_disaster_recovery(
            budget_max_usd=600.0,
            target_providers=["AWS", "GCP"],
            required_vcpus=4,
            required_ram_gb=16.0,
        )
        self.assertIn("status", res_z3)
        self.assertIn("total_monthly_cost_usd", res_z3)

    def test_scope_parser_instance_methods_and_matcher_alignment(self):
        """Verify SCOPEParser provides _extract_parameters, extract_parameters, and .matcher attribute."""
        parser = SCOPEParser()
        self.assertIsNotNone(parser.matcher)
        self.assertTrue(hasattr(parser.matcher, "match_template"))

        query = "Deploy 4 microservices on AWS, max budget $400, 4 vCPUs and 8GB RAM"
        params_private = parser._extract_parameters(query)
        params_public = parser.extract_parameters(query)
        self.assertEqual(params_private, params_public)
        self.assertEqual(params_public.get("budget_max_usd"), 400.0)
        self.assertEqual(params_public.get("required_vcpus"), 4)
        self.assertEqual(params_public.get("required_ram_gb"), 8.0)
        self.assertEqual(params_public.get("cloud_providers"), ["AWS"])


if __name__ == "__main__":
    unittest.main()