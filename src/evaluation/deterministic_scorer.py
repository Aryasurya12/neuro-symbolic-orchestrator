"""Deterministic Evaluation Scorer for Neurasym Four-Mode Benchmarks.

Implements the unified 7-dimension scoring contract defined in
research_eval/EVALUATION_PROTOCOL.md.

All four execution modes (Raw LLM, Schema LLM, Pure Symbolic, Neuro-Symbolic)
are graded using identical verification invariants, mathematical oracles,
and ground-truth rules.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class EvaluationVerdict:
    """Unified evaluation record representing the complete 7-dimension evaluation."""
    query_id: str
    mode: int
    mode_name: str
    
    # 7 Evaluation Dimensions
    interpretation_correct: Optional[bool] = None
    outcome_correct: bool = False
    constraints_satisfied: Optional[bool] = None
    cost_correct: Optional[bool] = None
    optimality_status: str = "NOT_APPLICABLE"  # "PROVED_OPTIMAL", "HEURISTIC_FEASIBLE", "SUBOPTIMAL", "NOT_APPLICABLE"
    refusal_correct: Optional[bool] = None
    strict_success: int = 0  # 1 = Success, 0 = Failure
    
    # Explanations & Telemetry
    failure_reason: str = "NONE"
    explain_label: str = "CORRECT"
    adjudication_status: str = "AUTOMATED_VERIFIED"  # "AUTOMATED_VERIFIED", "UNRESOLVED_HUMAN_REVIEW"
    
    # Independent Numerical Verifications
    reported_cost_usd: Optional[float] = None
    recomputed_cost_usd: Optional[float] = None
    cost_gap_usd: Optional[float] = None
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    mismatch_fields: List[str] = field(default_factory=list)
    verification_errors: List[str] = field(default_factory=list)


class IndependentCatalogOracle:
    """Decoupled reference oracle querying the frozen FinOps catalog."""

    # Reference Ground-Truth VM SKUs
    GROUND_TRUTH_VM_SKUS: Dict[str, Dict[str, Any]] = {
        "t3.medium": {"provider": "AWS", "vcpus": 2, "ram_gb": 4.0, "hourly_cost_usd": 0.0416, "monthly_cost": 30.37},
        "t3.large": {"provider": "AWS", "vcpus": 2, "ram_gb": 8.0, "hourly_cost_usd": 0.0832, "monthly_cost": 60.74},
        "t3.xlarge": {"provider": "AWS", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1664, "monthly_cost": 121.47},
        "c5.large": {"provider": "AWS", "vcpus": 2, "ram_gb": 4.0, "hourly_cost_usd": 0.0850, "monthly_cost": 62.05},
        "c5.xlarge": {"provider": "AWS", "vcpus": 4, "ram_gb": 8.0, "hourly_cost_usd": 0.1700, "monthly_cost": 124.10},
        "m5.large": {"provider": "AWS", "vcpus": 2, "ram_gb": 8.0, "hourly_cost_usd": 0.0960, "monthly_cost": 70.08},
        "m5.xlarge": {"provider": "AWS", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1920, "monthly_cost": 140.16},
        "m5.2xlarge": {"provider": "AWS", "vcpus": 8, "ram_gb": 32.0, "hourly_cost_usd": 0.3840, "monthly_cost": 280.32},
        "Standard_D4s_v5": {"provider": "Azure", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1920, "monthly_cost": 140.16},
        "e2-standard-4": {"provider": "GCP", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1340, "monthly_cost": 97.82},
    }

    # Reference Ground-Truth Regional Infrastructure Graph
    GROUND_TRUTH_REGIONS: Dict[str, Dict[str, Any]] = {
        "us-east-1": {
            "provider": "AWS",
            "base_cost_usd": 120.0,
            "sla_pct": 99.95,
            "peer_latencies_ms": {"us-west-2": 65.0, "eu-west-1": 85.0, "eastus": 12.0, "us-central1": 32.0},
            "transit_costs": {"us-west-2": 15.0, "eu-west-1": 25.0, "eastus": 5.0, "us-central1": 8.0},
        },
        "us-west-2": {
            "provider": "AWS",
            "base_cost_usd": 130.0,
            "sla_pct": 99.95,
            "peer_latencies_ms": {"us-east-1": 65.0, "eu-west-1": 135.0, "eastus": 72.0, "us-central1": 45.0},
            "transit_costs": {"us-east-1": 15.0, "eu-west-1": 35.0, "eastus": 18.0, "us-central1": 10.5},
        },
        "eu-west-1": {
            "provider": "AWS",
            "base_cost_usd": 140.0,
            "sla_pct": 99.95,
            "peer_latencies_ms": {"us-east-1": 85.0, "us-west-2": 135.0, "eastus": 90.0, "us-central1": 110.0},
            "transit_costs": {"us-east-1": 25.0, "us-west-2": 35.0, "eastus": 22.0, "us-central1": 28.0},
        },
        "eastus": {
            "provider": "Azure",
            "base_cost_usd": 110.0,
            "sla_pct": 99.95,
            "peer_latencies_ms": {"us-east-1": 12.0, "us-west-2": 72.0, "eu-west-1": 90.0, "us-central1": 38.0},
            "transit_costs": {"us-east-1": 5.0, "us-west-2": 18.0, "eu-west-1": 22.0, "us-central1": 9.0},
        },
        "us-central1": {
            "provider": "GCP",
            "base_cost_usd": 115.0,
            "sla_pct": 99.95,
            "peer_latencies_ms": {"us-east-1": 32.0, "us-west-2": 45.0, "eu-west-1": 110.0, "eastus": 38.0},
            "transit_costs": {"us-east-1": 8.0, "us-west-2": 10.5, "eu-west-1": 28.0, "eastus": 9.0},
        },
    }

    def __init__(self) -> None:
        self.skus = self.GROUND_TRUTH_VM_SKUS
        self.regions = self.GROUND_TRUTH_REGIONS

    def _clean_region_name(self, region_str: Optional[str]) -> Optional[str]:
        """Extracts standard region identifier from formatted strings like 'AWS:us-east-1 (US_East)'."""
        if not region_str:
            return None
        m = re.search(r"(us-east-1|us-west-2|eu-west-1|eastus|us-central1)", str(region_str), re.IGNORECASE)
        if m:
            return m.group(1).lower()
        return str(region_str).strip()

    def recompute_vm_cost_and_capacity(
        self, allocated_vms: List[Dict[str, Any]]
    ) -> Tuple[float, int, float, List[str]]:
        """Recomputes monthly cost, total vCPUs, and total RAM from concrete SKU tuples."""
        total_cost = 0.0
        total_vcpu = 0
        total_ram = 0.0
        errors = []

        for item in allocated_vms:
            sku = item.get("sku") or item.get("instance_type")
            qty = int(item.get("quantity") or item.get("count") or 1)
            if qty <= 0:
                continue
            if not sku or sku not in self.skus:
                errors.append(f"Unknown SKU '{sku}' in allocation plan.")
                continue
            sku_info = self.skus[sku]
            total_vcpu += sku_info["vcpus"] * qty
            total_ram += sku_info["ram_gb"] * qty
            total_cost += sku_info["monthly_cost"] * qty

        return round(total_cost, 2), total_vcpu, total_ram, errors

    def recompute_dr_cost_and_metrics(
        self, primary_region: str, secondary_region: str
    ) -> Tuple[float, float, float, List[str]]:
        """Recomputes DR monthly cost, composite availability, and latency."""
        errors = []
        r1_clean = self._clean_region_name(primary_region)
        r2_clean = self._clean_region_name(secondary_region)

        if not r1_clean or r1_clean not in self.regions:
            errors.append(f"Unknown primary region '{primary_region}'.")
        if not r2_clean or r2_clean not in self.regions:
            errors.append(f"Unknown secondary region '{secondary_region}'.")
        if errors:
            return 0.0, 0.0, 0.0, errors

        r1 = self.regions[r1_clean]
        r2 = self.regions[r2_clean]

        # Topology lookup
        if r2_clean not in r1["peer_latencies_ms"]:
            errors.append(f"No direct topology link between '{primary_region}' and '{secondary_region}'.")
            return 0.0, 0.0, 0.0, errors

        latency_ms = r1["peer_latencies_ms"][r2_clean]
        transit_cost = r1["transit_costs"].get(r2_clean, 8.0)

        # Composite availability: A_comp = 1 - (1 - A1) * (1 - A2)
        a1 = r1["sla_pct"] / 100.0
        a2 = r2["sla_pct"] / 100.0
        composite_sla_pct = round((1.0 - (1.0 - a1) * (1.0 - a2)) * 100.0, 6)

        total_cost = round(r1["base_cost_usd"] + r2["base_cost_usd"] + transit_cost, 2)
        return total_cost, composite_sla_pct, latency_ms, errors


class DeterministicScorer:
    """Universal Deterministic Evaluation Engine for Neurasym."""

    def __init__(self, oracle: Optional[IndependentCatalogOracle] = None) -> None:
        self.oracle = oracle or IndependentCatalogOracle()

    def evaluate_record(
        self,
        query_id: str,
        mode: int,
        manifest_entry: Dict[str, Any],
        raw_output: Any,
        extracted_contract: Optional[Dict[str, Any]] = None,
        telemetry_record: Optional[Dict[str, Any]] = None,
    ) -> EvaluationVerdict:
        """Grades a single benchmark execution record against the frozen manifest ground truth."""
        mode_names = {1: "Raw LLM", 2: "Schema LLM", 3: "Pure Symbolic", 4: "Neuro-Symbolic"}
        verdict = EvaluationVerdict(
            query_id=query_id,
            mode=mode,
            mode_name=mode_names.get(mode, f"Mode {mode}"),
        )

        expected_outcome = manifest_entry.get("expected_outcome", "FEASIBLE").strip().upper()

        # Check for provider/timeout failure first
        if telemetry_record:
            ps = str(telemetry_record.get("proof_status", "")).lower()
            el = str(telemetry_record.get("explain_label", "")).upper()
            if "timeout" in ps or el == "PROVIDER_FAILURE" or "provider" in ps:
                verdict.outcome_correct = False
                verdict.constraints_satisfied = False
                verdict.strict_success = 0
                verdict.explain_label = "PROVIDER_FAILURE"
                verdict.failure_reason = "Provider timeout or API failure."
                return verdict
        
        # 1. Semantic Interpretation Fidelity Evaluation
        if extracted_contract is not None:
            verdict.extracted_fields = extracted_contract
            interp_ok, mismatches = self._check_interpretation(manifest_entry, extracted_contract)
            verdict.interpretation_correct = interp_ok
            verdict.mismatch_fields = mismatches
        elif mode in [1, 2]:
            parsed_json = self._extract_json_payload(raw_output)
            if parsed_json:
                verdict.extracted_fields = parsed_json
                interp_ok, mismatches = self._check_generative_interpretation(manifest_entry, parsed_json)
                verdict.interpretation_correct = interp_ok
                verdict.mismatch_fields = mismatches
            else:
                verdict.interpretation_correct = None

        # 2. Outcome Correctness & Dimension Grading
        if expected_outcome == "FEASIBLE":
            self._grade_feasible_workload(verdict, manifest_entry, raw_output, mode, telemetry_record)
        elif expected_outcome in ["INFEASIBLE", "CONFLICTING_REQUIREMENTS", "CLARIFICATION_REQUIRED", "UNSUPPORTED"]:
            self._grade_refusal_workload(verdict, manifest_entry, raw_output, expected_outcome, mode, telemetry_record)
        else:
            verdict.explain_label = "NOT_GRADED"
            verdict.adjudication_status = "NOT_GRADED"

        return verdict

    def _check_interpretation(
        self, manifest: Dict[str, Any], contract: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Verifies symbolic contract against manifest requirements."""
        mismatches = []

        # Check vCPUs
        req_vcpu = manifest.get("required_vcpus")
        if req_vcpu is not None:
            ext_vcpu = contract.get("required_vcpus")
            if ext_vcpu is None or abs(float(ext_vcpu) - float(req_vcpu)) > 0.01:
                mismatches.append(f"required_vcpus: expected {req_vcpu}, extracted {ext_vcpu}")

        # Check RAM
        req_ram = manifest.get("required_ram_gb")
        if req_ram is not None:
            ext_ram = contract.get("required_ram_gb")
            if ext_ram is None or abs(float(ext_ram) - float(req_ram)) > 0.01:
                mismatches.append(f"required_ram_gb: expected {req_ram}, extracted {ext_ram}")

        # Check Budget
        req_budget = manifest.get("budget_max_usd")
        if req_budget is not None:
            ext_budget = contract.get("budget_max_usd")
            if ext_budget is None or abs(float(ext_budget) - float(req_budget)) > 0.01:
                mismatches.append(f"budget_max_usd: expected {req_budget}, extracted {ext_budget}")

        # Check Bandwidth / CPU scaling
        req_bw = manifest.get("target_bandwidth_mbps")
        if req_bw is not None:
            ext_bw = contract.get("target_bandwidth_mbps")
            if ext_bw is None or abs(float(ext_bw) - float(req_bw)) > 0.1:
                mismatches.append(f"target_bandwidth_mbps: expected {req_bw}, extracted {ext_bw}")

        req_cpu_target = manifest.get("target_cpu_pct")
        if req_cpu_target is not None:
            ext_cpu_target = contract.get("target_cpu_pct")
            if ext_cpu_target is None or abs(float(ext_cpu_target) - float(req_cpu_target)) > 0.1:
                mismatches.append(f"target_cpu_pct: expected {req_cpu_target}, extracted {ext_cpu_target}")

        req_cpu_ceil = manifest.get("max_cpu_pct")
        if req_cpu_ceil is not None:
            ext_cpu_ceil = contract.get("max_cpu_pct")
            if ext_cpu_ceil is None or abs(float(ext_cpu_ceil) - float(req_cpu_ceil)) > 0.1:
                mismatches.append(f"max_cpu_pct: expected {req_cpu_ceil}, extracted {ext_cpu_ceil}")

        return len(mismatches) == 0, mismatches

    def _check_generative_interpretation(
        self, manifest: Dict[str, Any], payload: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Verifies structured JSON payload against manifest requirements."""
        mismatches = []
        expected_type = manifest.get("intended_archetype", manifest.get("problem_type"))
        task_type = payload.get("task_type")
        if expected_type and task_type:
            type_mapping = {
                "ILP_VM_Allocation": ["ILP_VM_Allocation", "VM_Allocation", "VM_Placement"],
                "Z3_Graph_Disaster_Recovery": ["Z3_Graph_Disaster_Recovery", "Disaster_Recovery", "DR"],
                "PSO_Continuous_Scaling": ["PSO_Continuous_Scaling", "Continuous_Scaling", "Scaling"],
            }
            valid_types = type_mapping.get(expected_type, [expected_type])
            if task_type not in valid_types and task_type not in ["infeasible", "unsupported", "needs_clarification"]:
                mismatches.append(f"task_type: expected {expected_type}, got {task_type}")

        return len(mismatches) == 0, mismatches

    def _grade_feasible_workload(
        self,
        verdict: EvaluationVerdict,
        manifest: Dict[str, Any],
        raw_output: Any,
        mode: int,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Evaluates execution on a FEASIBLE benchmark query."""
        plan = self._extract_plan_data(raw_output, mode, telemetry)
        
        if plan is None or plan.get("is_refusal", False):
            verdict.outcome_correct = False
            verdict.constraints_satisfied = False
            verdict.strict_success = 0
            verdict.explain_label = "WRONG_OUTCOME"
            verdict.failure_reason = "System refused or asked for clarification when a valid feasible plan exists."
            return

        verdict.outcome_correct = True
        archetype = manifest.get("intended_archetype", manifest.get("problem_type", "ILP_VM_Allocation"))

        # Check constraints & cost by archetype
        if "VM" in archetype or archetype == "ILP_VM_Allocation":
            self._verify_vm_plan(verdict, manifest, plan)
        elif "DR" in archetype or archetype == "Z3_Graph_Disaster_Recovery":
            self._verify_dr_plan(verdict, manifest, plan)
        elif "Scaling" in archetype or archetype == "PSO_Continuous_Scaling":
            self._verify_scaling_plan(verdict, manifest, plan)
        else:
            verdict.constraints_satisfied = False
            verdict.failure_reason = f"Unsupported archetype '{archetype}'."
            verdict.explain_label = "WRONG_PLAN"

        # Check strict end-to-end success
        if (
            verdict.outcome_correct
            and verdict.constraints_satisfied
            and verdict.cost_correct
        ):
            if verdict.interpretation_correct is False:
                verdict.strict_success = 0
                verdict.explain_label = "RIGHT_ANSWER_WRONG_READING"
            else:
                verdict.strict_success = 1
                if verdict.cost_gap_usd and verdict.cost_gap_usd > 0.01:
                    verdict.explain_label = "CORRECT_BUT_COSTLIER"
                    verdict.optimality_status = "SUBOPTIMAL"
                else:
                    verdict.explain_label = "CORRECT"
        else:
            verdict.strict_success = 0
            if not verdict.constraints_satisfied:
                verdict.explain_label = "WRONG_PLAN"

    def _grade_refusal_workload(
        self,
        verdict: EvaluationVerdict,
        manifest: Dict[str, Any],
        raw_output: Any,
        expected_outcome: str,
        mode: int,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Evaluates execution on an INFEASIBLE, CONFLICTING, CLARIFICATION, or UNSUPPORTED query."""
        is_refusal, refusal_reason, refusal_type = self._detect_refusal(raw_output, mode, telemetry)

        if not is_refusal:
            verdict.outcome_correct = False
            verdict.refusal_correct = False
            verdict.strict_success = 0
            verdict.explain_label = "WRONG_OUTCOME"
            verdict.failure_reason = f"Expected {expected_outcome}, but system generated an unconstrained allocation plan."
            return

        # Refusal type matching
        outcome_matched = False
        if expected_outcome == "INFEASIBLE" and refusal_type in ["infeasible", "solver_infeasible", "prose_refusal", "proven_infeasible"]:
            outcome_matched = True
        elif expected_outcome == "CONFLICTING_REQUIREMENTS" and refusal_type in ["conflicting_requirements", "conflicting", "infeasible", "prose_refusal"]:
            outcome_matched = True
        elif expected_outcome == "CLARIFICATION_REQUIRED" and refusal_type in ["needs_clarification", "clarification_required", "clarification", "prose_refusal"]:
            outcome_matched = True
        elif expected_outcome == "UNSUPPORTED" and refusal_type in ["unsupported", "task_incompatible", "prose_refusal"]:
            outcome_matched = True

        if not outcome_matched:
            verdict.outcome_correct = False
            verdict.refusal_correct = False
            verdict.strict_success = 0
            verdict.explain_label = "WRONG_OUTCOME"
            verdict.failure_reason = f"Refusal category mismatch: expected {expected_outcome}, received {refusal_type} ({refusal_reason})."
            return

        verdict.outcome_correct = True
        verdict.refusal_correct = True
        verdict.strict_success = 1
        verdict.explain_label = "CORRECT_REFUSAL"
        verdict.failure_reason = "NONE"

    def _verify_vm_plan(
        self, verdict: EvaluationVerdict, manifest: Dict[str, Any], plan: Dict[str, Any]
    ) -> None:
        """Verifies VM allocation plan against constraints and recomputes cost."""
        allocated_vms = plan.get("allocated_vms", [])
        if not allocated_vms:
            verdict.constraints_satisfied = False
            verdict.failure_reason = "Allocation plan contains no VM instances."
            return

        recomp_cost, total_vcpu, total_ram, errors = self.oracle.recompute_vm_cost_and_capacity(allocated_vms)
        verdict.recomputed_cost_usd = recomp_cost
        verdict.reported_cost_usd = plan.get("total_monthly_cost_usd")
        verdict.verification_errors.extend(errors)

        # Hardware checks
        req_vcpu = manifest.get("required_vcpus", 0)
        req_ram = manifest.get("required_ram_gb", 0.0)
        req_budget = manifest.get("budget_max_usd")
        allowed_providers = set(manifest.get("cloud_providers", ["AWS", "Azure", "GCP"]))

        for item in allocated_vms:
            prov = item.get("provider", "").upper()
            if prov and prov not in allowed_providers:
                errors.append(f"VM provider '{prov}' not in allowed providers {allowed_providers}.")

        if total_vcpu < req_vcpu:
            errors.append(f"Insufficient vCPUs: provided {total_vcpu} < required {req_vcpu}.")
        if total_ram < req_ram:
            errors.append(f"Insufficient RAM: provided {total_ram:.1f} GB < required {req_ram:.1f} GB.")
        if req_budget is not None and recomp_cost > req_budget + 0.01:
            errors.append(f"Cost exceeds budget: ${recomp_cost:,.2f} > ${req_budget:,.2f}.")

        verdict.constraints_satisfied = (len(errors) == 0)
        if errors:
            verdict.failure_reason = "; ".join(errors)

        if verdict.reported_cost_usd is not None:
            diff = abs(verdict.reported_cost_usd - recomp_cost)
            verdict.cost_correct = (diff <= 0.02)
        else:
            verdict.cost_correct = True

        opt_cost = manifest.get("optimal_cost_usd")
        if opt_cost is not None and verdict.constraints_satisfied:
            verdict.cost_gap_usd = round(recomp_cost - float(opt_cost), 2)
            if verdict.cost_gap_usd <= 0.01:
                verdict.optimality_status = "PROVED_OPTIMAL"
            else:
                verdict.optimality_status = "SUBOPTIMAL"

    def _verify_dr_plan(
        self, verdict: EvaluationVerdict, manifest: Dict[str, Any], plan: Dict[str, Any]
    ) -> None:
        """Verifies Disaster Recovery plan against latency, SLA, and budget."""
        p_region = plan.get("primary_region")
        s_region = plan.get("secondary_region")

        if not p_region or not s_region:
            verdict.constraints_satisfied = False
            verdict.failure_reason = "Missing primary or secondary region in DR plan."
            return

        recomp_cost, comp_sla, latency_ms, errors = self.oracle.recompute_dr_cost_and_metrics(p_region, s_region)
        verdict.recomputed_cost_usd = recomp_cost
        verdict.reported_cost_usd = plan.get("total_monthly_cost_usd")
        verdict.verification_errors.extend(errors)

        req_sla = manifest.get("sla_availability_pct", 99.9)
        req_latency = manifest.get("latency_max_ms", 100.0)
        req_budget = manifest.get("budget_max_usd")

        if comp_sla < req_sla - 1e-5:
            errors.append(f"SLA violation: composite {comp_sla:.4f}% < required {req_sla:.4f}%.")
        if latency_ms > req_latency + 1e-5:
            errors.append(f"Latency violation: inter-region {latency_ms:.1f}ms > max {req_latency:.1f}ms.")
        if req_budget is not None and recomp_cost > req_budget + 0.01:
            errors.append(f"DR Cost exceeds budget: ${recomp_cost:,.2f} > ${req_budget:,.2f}.")

        verdict.constraints_satisfied = (len(errors) == 0)
        if errors:
            verdict.failure_reason = "; ".join(errors)

        if verdict.reported_cost_usd is not None:
            diff = abs(verdict.reported_cost_usd - recomp_cost)
            verdict.cost_correct = (diff <= 0.02)
        else:
            verdict.cost_correct = True

        opt_cost = manifest.get("optimal_cost_usd")
        if opt_cost is not None and verdict.constraints_satisfied:
            verdict.cost_gap_usd = round(recomp_cost - float(opt_cost), 2)
            if verdict.cost_gap_usd <= 0.01:
                verdict.optimality_status = "PROVED_OPTIMAL"
            else:
                verdict.optimality_status = "SUBOPTIMAL"

    def _verify_scaling_plan(
        self, verdict: EvaluationVerdict, manifest: Dict[str, Any], plan: Dict[str, Any]
    ) -> None:
        """Verifies continuous scaling plan against traffic, replica bounds, and CPU ceiling."""
        replicas = plan.get("recommended_replicas") or plan.get("replicas")
        bw = plan.get("optimal_bandwidth_mbps") or plan.get("bandwidth_mbps") or manifest.get("target_bandwidth_mbps", 300.0)

        if replicas is None or int(replicas) <= 0:
            verdict.constraints_satisfied = False
            verdict.failure_reason = "Invalid or missing replica count in scaling plan."
            return

        replicas = int(replicas)
        traffic_capacity = replicas * 75.0
        achieved_cpu = (float(bw) / traffic_capacity) * 100.0

        errors = []
        max_cpu_ceil = manifest.get("max_cpu_pct")
        if max_cpu_ceil is not None and achieved_cpu > float(max_cpu_ceil) + 0.1:
            errors.append(f"CPU Ceiling violation: achieved {achieved_cpu:.2f}% > max {max_cpu_ceil:.2f}%.")

        req_budget = manifest.get("budget_max_usd")
        recomp_cost = round(80.0 + replicas * 37.0, 2)
        verdict.recomputed_cost_usd = recomp_cost
        verdict.reported_cost_usd = plan.get("total_monthly_cost_usd")

        if req_budget is not None and recomp_cost > req_budget + 0.01:
            errors.append(f"Scaling cost exceeds budget: ${recomp_cost:,.2f} > ${req_budget:,.2f}.")

        verdict.constraints_satisfied = (len(errors) == 0)
        if errors:
            verdict.failure_reason = "; ".join(errors)

        if verdict.reported_cost_usd is not None:
            diff = abs(verdict.reported_cost_usd - recomp_cost)
            verdict.cost_correct = (diff <= 10.0)
        else:
            verdict.cost_correct = True

        if verdict.constraints_satisfied:
            verdict.optimality_status = "HEURISTIC_FEASIBLE"

    def _extract_plan_data(
        self, raw_output: Any, mode: int, telemetry: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Extracts allocation parameters from raw model response or telemetry trace."""
        if mode == 2 or isinstance(raw_output, dict):
            if isinstance(raw_output, str):
                try:
                    payload = json.loads(raw_output)
                except Exception:
                    return None
            else:
                payload = raw_output
            if payload.get("outcome") in ["infeasible", "unsupported", "needs_clarification"]:
                return {"is_refusal": True, "reason": payload.get("reason", "")}
            return payload

        if mode in [3, 4]:
            if telemetry:
                if telemetry.get("proof_status") in ["Infeasible", "Conflicting Requirements", "Unsupported", "Clarification Required"]:
                    return {"is_refusal": True, "reason": telemetry.get("proof_status")}
                if telemetry.get("plan_valid") is False and not telemetry.get("allocated_vms"):
                    return {"is_refusal": True, "reason": "No valid plan generated."}
                return telemetry

        if mode == 1 and isinstance(raw_output, str):
            lower = raw_output.lower()
            if any(w in lower for w in ["cannot be built", "cannot be satisfied", "impossible", "infeasible", "no combination", "i can't help with that", "i’m sorry, but i can’t help"]):
                return {"is_refusal": True, "reason": raw_output}
            
            allocated_vms = []
            for sku in self.oracle.skus.keys():
                pattern = rf"(\d+)\s*[x×]\s*{re.escape(sku)}"
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    allocated_vms.append({"sku": sku, "quantity": int(match.group(1))})
            if allocated_vms:
                return {"allocated_vms": allocated_vms}

        return None

    def _detect_refusal(
        self, raw_output: Any, mode: int, telemetry: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, str]:
        """Detects whether a model response constitutes an explicit task refusal or clarification."""
        if mode == 2:
            payload = self._extract_json_payload(raw_output)
            if payload:
                outcome = str(payload.get("outcome", "")).lower()
                reason = payload.get("reason", "")
                if outcome in ["infeasible", "unsupported", "needs_clarification"]:
                    return True, reason, outcome

        if mode in [3, 4]:
            # Inspect raw_output if dict
            if isinstance(raw_output, dict):
                io = str(raw_output.get("interpretation_outcome", raw_output.get("outcome", ""))).lower()
                if io == "conflicting_requirements" or raw_output.get("conflicting_reasons"):
                    return True, "Conflicting requirements detected", "conflicting_requirements"
                if io == "needs_clarification" or raw_output.get("clarification_questions"):
                    return True, "Clarification questions generated", "clarification_required"
                if io == "unsupported" or raw_output.get("unsupported_reasons"):
                    return True, "Unsupported workload", "unsupported"
                if io == "infeasible" or raw_output.get("is_feasible") is False:
                    return True, "Infeasible request", "infeasible"

            if telemetry:
                ps = str(telemetry.get("proof_status", "")).lower()
                norm_stat = str(telemetry.get("normalization_status", "")).lower()

                if "conflicting" in ps or "conflicting" in norm_stat:
                    return True, ps, "conflicting_requirements"
                if "clarification" in ps or "clarification" in norm_stat or "missing" in ps:
                    return True, ps, "clarification_required"
                if "unsupported" in ps or "incompatible" in norm_stat:
                    return True, ps, "unsupported"
                if "infeasible" in ps or "infeasible" in norm_stat:
                    return True, ps, "infeasible"

        if mode == 1 and isinstance(raw_output, str):
            lower = raw_output.lower()
            if any(w in lower for w in [
                "cannot be built", "cannot be satisfied", "impossible", "infeasible",
                "no combination", "exceeds the budget", "exceeds budget",
                "i can't help with that", "i’m sorry, but i can’t help",
                "outside the scope", "requires clarification"
            ]):
                return True, raw_output[:200], "prose_refusal"

        return False, "", ""

    def _extract_json_payload(self, raw: Any) -> Optional[Dict[str, Any]]:
        """Safely parses JSON payload from string or dict."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw.strip())
            except Exception:
                match = re.search(r"(\{.*\})", raw, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except Exception:
                        return None
        return None
