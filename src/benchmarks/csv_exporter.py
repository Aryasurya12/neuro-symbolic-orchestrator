"""Canonical CSV Exporters for Neurasym Evaluation & Auditability.

Produces 4 standardized CSV exports directly from canonical execution records:
1. query_manifest.csv: Query metadata, archetype intent, expected outcomes, explicit requirements, and human approval status.
2. run_results.csv: One row per query × mode × trial with complete provenance, parameters, metrics, and timings.
3. verification_checks.csv: One row per verified parameter with structured audit event fields.
4. outcome_patterns.csv: Four-mode truth table patterns (e.g. '0011') and per-mode success reasons.

Strict Rules:
- Null, unknown, or inapplicable numeric metrics MUST be formatted as empty strings '""', never '0' or '0.0'.
- UTF-8 encoding with standard RFC 4180 CSV quoting for text with commas, newlines, and Hindi/Hinglish characters.
- Reads directly from saved records without re-solving or live inference.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from src.benchmarks.truth_table import TruthTableEngine
from src.verifiers.canonical_record import (
    AuditEvent,
    CanonicalExecutionRecord,
    CheckStatus,
    FeasibilityStatus,
    NormalizationStatus,
    OptimalityStatus,
)


def _fmt_val(val: Any) -> str:
    """Formats numeric or optional values as empty string if None or N/A."""
    if val is None:
        return ""
    if isinstance(val, float):
        if val != val:  # NaN
            return ""
        return f"{val:.4f}".rstrip("0").rstrip(".") if "." in f"{val:.4f}" else str(val)
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


class CSVExporter:
    """Generates all 4 standard benchmark and evaluation CSV tables."""

    @classmethod
    def export_query_manifest(
        cls,
        queries: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> str:
        """Exports query_manifest.csv containing test suite definitions."""
        fieldnames = [
            "query_id",
            "scenario_family_id",
            "query_text",
            "category",
            "intended_archetype",
            "expected_outcome",
            "required_vcpus",
            "required_ram_gb",
            "target_bandwidth_mbps",
            "target_cpu_pct",
            "max_cpu_pct",
            "latency_max_ms",
            "sla_availability_pct",
            "budget_max_usd",
            "cloud_providers",
            "annotation_source",
            "approval_status",
        ]

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writeheader()

        for q in queries:
            row = {
                "query_id": _fmt_val(q.get("query_id")),
                "scenario_family_id": _fmt_val(q.get("scenario_family_id")),
                "query_text": q.get("query_text", ""),
                "category": _fmt_val(q.get("category")),
                "intended_archetype": _fmt_val(q.get("intended_archetype", q.get("problem_type"))),
                "expected_outcome": _fmt_val(q.get("expected_outcome", "FEASIBLE")),
                "required_vcpus": _fmt_val(q.get("required_vcpus")),
                "required_ram_gb": _fmt_val(q.get("required_ram_gb")),
                "target_bandwidth_mbps": _fmt_val(q.get("target_bandwidth_mbps")),
                "target_cpu_pct": _fmt_val(q.get("target_cpu_pct")),
                "max_cpu_pct": _fmt_val(q.get("max_cpu_pct")),
                "latency_max_ms": _fmt_val(q.get("latency_max_ms")),
                "sla_availability_pct": _fmt_val(q.get("sla_availability_pct")),
                "budget_max_usd": _fmt_val(q.get("budget_max_usd")),
                "cloud_providers": json.dumps(q.get("cloud_providers", ["AWS"])) if q.get("cloud_providers") else "",
                "annotation_source": _fmt_val(q.get("annotation_source", "AI_DRAFT/DEVELOPMENT")),
                "approval_status": _fmt_val(q.get("approval_status", "UNAPPROVED")),
            }
            writer.writerow(row)

        csv_text = buffer.getvalue()
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)

        return csv_text

    @classmethod
    def export_run_results(
        cls,
        records: List[CanonicalExecutionRecord],
        output_path: Optional[str] = None,
        query_metadata_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Exports run_results.csv containing one row per query × mode × trial."""
        fieldnames = [
            "run_id",
            "query_id",
            "query_text",
            "category",
            "scenario_family_id",
            "mode",
            "trial",
            "is_mock",
            "timestamp",
            "provider",
            "model",
            "parser",
            "solver",
            "solver_seed",
            "execution_status",
            "normalization_status",
            "interpretation_status",
            "task_pass",
            "feasibility_status",
            "failure_stage",
            "failure_reason",
            "required_vcpus",
            "allocated_vcpus",
            "required_ram_gb",
            "allocated_ram_gb",
            "instance_quantities",
            "required_bandwidth_mbps",
            "allocated_bandwidth_mbps",
            "required_replicas",
            "allocated_replicas",
            "target_cpu_pct",
            "max_cpu_pct",
            "achieved_cpu_pct",
            "max_latency_ms",
            "achieved_latency_ms",
            "required_sla_pct",
            "achieved_sla_pct",
            "provider_choices",
            "region_choices",
            "budget_usd",
            "claimed_monthly_cost_usd",
            "recomputed_monthly_cost_usd",
            "cost_error_pct",
            "budget_headroom_usd",
            "parsing_duration_ms",
            "solving_duration_ms",
            "verification_duration_ms",
            "explanation_duration_ms",
            "total_duration_ms",
            "prompt_tokens",
            "completion_tokens",
            "api_cost_usd",
            "optimality_status",
            "allocated_decision_json",
        ]

        lookup = query_metadata_lookup or {}
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writeheader()

        for r in records:
            # Query metadata lookup
            q_id = getattr(r, "query_id", None) or r.requirements.get("query_id", "")
            q_meta = lookup.get(q_id, {})
            expected_outcome = q_meta.get("expected_outcome", "FEASIBLE")

            # Evaluate task pass
            eval_res = TruthTableEngine.evaluate_mode_task_success(r, expected_outcome=expected_outcome)

            reqs = r.requirements or {}
            norm_alloc = r.normalized_allocation or {}

            # Parameter unpacking according to problem archetype
            prob = r.problem_type
            req_vcpu = reqs.get("required_vcpus")
            req_ram = reqs.get("required_ram_gb")
            req_bw = reqs.get("target_bandwidth_mbps") or reqs.get("min_bandwidth_mbps")
            req_rep = reqs.get("target_replicas") or reqs.get("min_replicas")
            tgt_cpu = reqs.get("target_cpu_pct")
            max_cpu = reqs.get("max_cpu_pct")
            max_lat = reqs.get("latency_max_ms")
            req_sla = reqs.get("sla_availability_pct")
            budget = reqs.get("budget_max_usd")

            # Allocated unpacking
            alloc_vcpu = None
            alloc_ram = None
            inst_qtys = None
            alloc_bw = None
            alloc_rep = None
            achieved_cpu = None
            achieved_lat = None
            achieved_sla = None
            prov_choices = None
            region_choices = None

            if prob == "ILP_VM_Allocation":
                vms = norm_alloc.get("allocated_vms", [])
                if isinstance(vms, list) and vms:
                    alloc_vcpu = sum(item.get("vcpus", 0) * item.get("count", 1) for item in vms)
                    alloc_ram = sum(item.get("ram_gb", 0) * item.get("count", 1) for item in vms)
                    inst_qtys = ", ".join(f"{item.get('sku')}x{item.get('count', 1)}" for item in vms)
                    prov_choices = list({item.get("provider") for item in vms if item.get("provider")})

            elif prob == "PSO_Continuous_Scaling":
                alloc_bw = norm_alloc.get("optimal_bandwidth_mbps", norm_alloc.get("bandwidth_mbps"))
                alloc_rep = norm_alloc.get("recommended_replicas", norm_alloc.get("replicas"))
                if alloc_bw is not None and alloc_rep is not None and alloc_rep > 0:
                    achieved_cpu = round((float(alloc_bw) / (float(alloc_rep) * 75.0)) * 100.0, 2)

            elif prob == "Z3_Graph_Disaster_Recovery":
                reg_a = norm_alloc.get("primary_region")
                reg_b = norm_alloc.get("secondary_region")
                if reg_a and reg_b:
                    region_choices = [str(reg_a), str(reg_b)]

            # Check audit event for achieved values if available
            for ev in r.audit_events:
                c_name = ev.check_name.lower()
                if "latency" in c_name and ev.recomputed_result is not None:
                    achieved_lat = ev.recomputed_result
                elif "sla" in c_name and ev.recomputed_result is not None:
                    achieved_sla = ev.recomputed_result
                elif "cpu utilization" in c_name and ev.recomputed_result is not None:
                    achieved_cpu = ev.recomputed_result

            # Failure stage & reason
            fail_stage = r.failed_stage
            fail_reason = ""
            if r.feasibility != FeasibilityStatus.PASS or eval_res.task_pass == 0:
                fail_reason = "; ".join(r.violations) if r.violations else eval_res.evaluation_reason

            row = {
                "run_id": r.run_id,
                "query_id": q_id,
                "query_text": r.original_query,
                "category": q_meta.get("category", ""),
                "scenario_family_id": q_meta.get("scenario_family_id", ""),
                "mode": r.mode,
                "trial": getattr(r, "trial", 1),
                "is_mock": getattr(r, "is_mock", False),
                "timestamp": r.timestamp_utc,
                "provider": r.provider or "",
                "model": r.model or "",
                "parser": "SCOPE" if r.mode == 3 else ("NVIDIA_Neural" if r.mode == 4 else "LLM_Direct"),
                "solver": r.solver_name or "",
                "solver_seed": _fmt_val(r.solver_seed),
                "execution_status": "COMPLETED" if r.normalization_status != NormalizationStatus.API_FAILURE else "FAILED",
                "normalization_status": r.normalization_status.value if isinstance(r.normalization_status, NormalizationStatus) else str(r.normalization_status),
                "interpretation_status": "READY" if r.normalization_status == NormalizationStatus.SUCCESS else r.normalization_status.value,
                "task_pass": _fmt_val(eval_res.task_pass),
                "feasibility_status": r.feasibility.value if isinstance(r.feasibility, FeasibilityStatus) else str(r.feasibility),
                "failure_stage": _fmt_val(fail_stage),
                "failure_reason": fail_reason,
                "required_vcpus": _fmt_val(req_vcpu),
                "allocated_vcpus": _fmt_val(alloc_vcpu),
                "required_ram_gb": _fmt_val(req_ram),
                "allocated_ram_gb": _fmt_val(alloc_ram),
                "instance_quantities": inst_qtys or "",
                "required_bandwidth_mbps": _fmt_val(req_bw),
                "allocated_bandwidth_mbps": _fmt_val(alloc_bw),
                "required_replicas": _fmt_val(req_rep),
                "allocated_replicas": _fmt_val(alloc_rep),
                "target_cpu_pct": _fmt_val(tgt_cpu),
                "max_cpu_pct": _fmt_val(max_cpu),
                "achieved_cpu_pct": _fmt_val(achieved_cpu),
                "max_latency_ms": _fmt_val(max_lat),
                "achieved_latency_ms": _fmt_val(achieved_latency := achieved_lat),
                "required_sla_pct": _fmt_val(req_sla),
                "achieved_sla_pct": _fmt_val(achieved_sla),
                "provider_choices": json.dumps(prov_choices) if prov_choices else "",
                "region_choices": json.dumps(region_choices) if region_choices else "",
                "budget_usd": _fmt_val(budget),
                "claimed_monthly_cost_usd": _fmt_val(r.claimed_cost_usd),
                "recomputed_monthly_cost_usd": _fmt_val(r.recomputed_cost_usd),
                "cost_error_pct": _fmt_val(r.cost_error_pct),
                "budget_headroom_usd": _fmt_val(r.budget_headroom_usd),
                "parsing_duration_ms": _fmt_val(round(r.parsing_ms, 2)),
                "solving_duration_ms": _fmt_val(round(r.solving_ms, 2)),
                "verification_duration_ms": _fmt_val(round(r.verification_ms, 2)),
                "explanation_duration_ms": _fmt_val(round(r.explanation_ms, 2)),
                "total_duration_ms": _fmt_val(round(r.total_duration_ms, 2)),
                "prompt_tokens": "",
                "completion_tokens": "",
                "api_cost_usd": "",
                "optimality_status": r.optimality_status.value if isinstance(r.optimality_status, OptimalityStatus) else str(r.optimality_status),
                "allocated_decision_json": json.dumps(norm_alloc, ensure_ascii=False) if norm_alloc else "",
            }
            writer.writerow(row)

        csv_text = buffer.getvalue()
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)

        return csv_text

    @classmethod
    def export_verification_checks(
        cls,
        records: List[CanonicalExecutionRecord],
        output_path: Optional[str] = None,
    ) -> str:
        """Exports verification_checks.csv with one row per audit check event."""
        fieldnames = [
            "run_id",
            "query_id",
            "mode",
            "trial",
            "check_id",
            "check_name",
            "verifier_module",
            "input_values",
            "provenance",
            "observed_value",
            "required_value",
            "operator",
            "source",
            "formula",
            "recomputed_result",
            "tolerance",
            "signed_margin",
            "status",
            "reason",
        ]

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writeheader()

        for r in records:
            q_id = getattr(r, "query_id", None) or r.requirements.get("query_id", "")
            trial_num = getattr(r, "trial", 1)

            for ev in r.audit_events:
                row = {
                    "run_id": r.run_id,
                    "query_id": q_id,
                    "mode": r.mode,
                    "trial": trial_num,
                    "check_id": getattr(ev, "check_id", "CHK_GENERAL"),
                    "check_name": ev.check_name,
                    "verifier_module": getattr(ev, "verifier_module", "src.verifiers.independent_checker"),
                    "input_values": json.dumps(ev.input_values, ensure_ascii=False) if getattr(ev, "input_values", None) else "",
                    "provenance": getattr(ev, "provenance", "") or ev.source or "",
                    "observed_value": str(ev.observed_value),
                    "required_value": str(ev.required_value),
                    "operator": ev.operator,
                    "source": ev.source,
                    "formula": ev.formula,
                    "recomputed_result": _fmt_val(ev.recomputed_result),
                    "tolerance": getattr(ev, "tolerance", "") or "",
                    "signed_margin": ev.signed_margin or "",
                    "status": ev.status.value if isinstance(ev.status, CheckStatus) else str(ev.status),
                    "reason": ev.reason,
                }
                writer.writerow(row)

        csv_text = buffer.getvalue()
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)

        return csv_text

    @classmethod
    def export_outcome_patterns(
        cls,
        matched_runs: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> str:
        """Exports outcome_patterns.csv containing 4-mode truth table patterns."""
        fieldnames = [
            "query_id",
            "trial",
            "mode1_task_pass",
            "mode2_task_pass",
            "mode3_task_pass",
            "mode4_task_pass",
            "observed_pattern",
            "mode1_reason",
            "mode2_reason",
            "mode3_reason",
            "mode4_reason",
        ]

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writeheader()

        for item in matched_runs:
            row = {
                "query_id": item.get("query_id", ""),
                "trial": item.get("trial", 1),
                "mode1_task_pass": _fmt_val(item.get("mode1_pass")),
                "mode2_task_pass": _fmt_val(item.get("mode2_pass")),
                "mode3_task_pass": _fmt_val(item.get("mode3_pass")),
                "mode4_task_pass": _fmt_val(item.get("mode4_pass")),
                "observed_pattern": item.get("observed_pattern") or item.get("pattern", "????"),
                "mode1_reason": item.get("mode1_reason", ""),
                "mode2_reason": item.get("mode2_reason", ""),
                "mode3_reason": item.get("mode3_reason", ""),
                "mode4_reason": item.get("mode4_reason", ""),
            }
            writer.writerow(row)

        csv_text = buffer.getvalue()
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)

        return csv_text
