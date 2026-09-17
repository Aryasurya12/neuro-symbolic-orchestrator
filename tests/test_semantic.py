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

import unittest
import pytest
from pydantic import ValidationError

from config.settings import settings
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.carm_matcher import CARMMatcher
from src.semantic.scope_parser import SCOPEParser
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


if __name__ == "__main__":
    unittest.main()