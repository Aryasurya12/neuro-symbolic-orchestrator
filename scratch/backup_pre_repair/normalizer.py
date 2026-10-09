"""Output Normalization Engine for Neurasym.

Parses, normalizes, and validates candidate allocations from unstructured prose (Mode 1),
structured JSON (Mode 2), and symbolic solver results (Modes 3 & 4).

Key Principles:
1. Preserve units and exact text spans for all extracted values.
2. Distinguish monthly rates ($/mo) from hourly rates ($/hr); never scale hourly to monthly implicitly or round $0.096/hr into $0.10/mo.
3. Distinguish deployment quantities (e.g., '2 x t3.medium') from passing mentions of alternatives.
4. Distinguish schema validity from task suitability (e.g. VM schema on dynamic scaling is TASK_INCOMPATIBLE, not zero-bandwidth).
5. If extraction is uncertain, report 'NEEDS_REVIEW' rather than labeling it 'hallucinated'.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.verifiers.canonical_record import (
    ExtractedEvidence,
    NormalizationStatus,
)


class OutputNormalizer:
    """Robust normalizer for LLM prose, JSON schema, and solver outputs."""

    # Ground-truth VM SKUs
    KNOWN_SKUS: Dict[str, Dict[str, Any]] = {
        "t3.medium": {"provider": "AWS", "vcpus": 2, "ram_gb": 4.0, "hourly_cost_usd": 0.0416},
        "t3.large": {"provider": "AWS", "vcpus": 2, "ram_gb": 8.0, "hourly_cost_usd": 0.0832},
        "t3.xlarge": {"provider": "AWS", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1664},
        "c5.large": {"provider": "AWS", "vcpus": 2, "ram_gb": 4.0, "hourly_cost_usd": 0.0850},
        "c5.xlarge": {"provider": "AWS", "vcpus": 4, "ram_gb": 8.0, "hourly_cost_usd": 0.1700},
        "m5.large": {"provider": "AWS", "vcpus": 2, "ram_gb": 8.0, "hourly_cost_usd": 0.0960},
        "m5.xlarge": {"provider": "AWS", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1920},
        "m5.2xlarge": {"provider": "AWS", "vcpus": 8, "ram_gb": 32.0, "hourly_cost_usd": 0.3840},
        "Standard_D4s_v5": {"provider": "Azure", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1920},
        "e2-standard-4": {"provider": "GCP", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.1340},
    }

    HOURS_PER_MONTH: float = 730.0

    @classmethod
    def normalize_mode1_prose(
        cls,
        raw_text: str,
        problem_type: str = "ILP_VM_Allocation",
        contract_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NormalizationStatus, Optional[float], Any, List[str], List[ExtractedEvidence]]:
        """Extracts monthly cost and structured decision vector from Mode 1 prose."""
        if not raw_text or not raw_text.strip():
            return (
                NormalizationStatus.MALFORMED_OUTPUT,
                None,
                None,
                ["Raw prose response is empty."],
                [],
            )

        errors: List[str] = []
        evidence_list: List[ExtractedEvidence] = []

        # 1. Extract Stated Monthly Cost vs Hourly Cost
        # Priority 1: Explicit monthly patterns (e.g. "$300.21/month", "$300.21 per month", "total cost: $300.21")
        monthly_patterns = [
            r"(?:total\s+(?:monthly\s+)?cost|total\s+spend|monthly\s+total|cost)\s*(?:is|:|=|of)?\s*\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:/\s*month|/\s*mo|/month|\bper\s+month|\bmonthly)?",
            r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:/\s*month|/\s*mo|\bper\s+month|\bmonthly)",
            r"(?:total|estimated\s+cost)\s*:\s*\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        hourly_patterns = [
            r"\$\s*(\d+(?:\.\d+)?)\s*(?:/\s*hour|/\s*hr|\bper\s+hour|\bhourly)",
        ]

        extracted_monthly_cost: Optional[float] = None
        extracted_hourly_cost: Optional[float] = None

        for pat in monthly_patterns:
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                val_str = m.group(1).replace(",", "")
                try:
                    val = float(val_str)
                    extracted_monthly_cost = val
                    evidence_list.append(
                        ExtractedEvidence(
                            field_name="total_monthly_cost_usd",
                            extracted_value=val,
                            unit="USD/month",
                            text_span=m.group(0),
                            confidence="HIGH",
                        )
                    )
                    break
                except ValueError:
                    pass

        for pat in hourly_patterns:
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                val_str = m.group(1).replace(",", "")
                try:
                    val = float(val_str)
                    extracted_hourly_cost = val
                    evidence_list.append(
                        ExtractedEvidence(
                            field_name="hourly_rate_usd",
                            extracted_value=val,
                            unit="USD/hour",
                            text_span=m.group(0),
                            confidence="HIGH",
                        )
                    )
                    break
                except ValueError:
                    pass

        # Fallback: General dollar amount if no monthly/hourly qualifier was matched
        if extracted_monthly_cost is None:
            gen_match = re.search(r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)", raw_text)
            if gen_match:
                val_str = gen_match.group(1).replace(",", "")
                try:
                    val = float(val_str)
                    # If this dollar amount is very small (e.g. < $1.00) and an hourly pattern was near it, DO NOT treat it as monthly!
                    if extracted_hourly_cost is not None and abs(val - extracted_hourly_cost) < 0.001:
                        errors.append(f"Only hourly rate (${val:.4f}/hr) found in prose. Monthly total was not explicitly stated.")
                    else:
                        extracted_monthly_cost = val
                        evidence_list.append(
                            ExtractedEvidence(
                                field_name="total_monthly_cost_usd",
                                extracted_value=val,
                                unit="USD (unqualified)",
                                text_span=gen_match.group(0),
                                confidence="NEEDS_REVIEW",
                            )
                        )
                except ValueError:
                    pass

        if extracted_monthly_cost is None:
            errors.append("No unambiguous monthly dollar cost ($XX.XX/month) found in prose.")

        # 2. Extract Decision Allocation based on problem type
        if problem_type == "ILP_VM_Allocation":
            # Extract VM allocations with explicit quantities
            # Look for patterns like "2 x t3.medium", "2 instances of t3.medium", "t3.medium (quantity: 2)"
            allocated_vms = []
            for sku_name, sdata in cls.KNOWN_SKUS.items():
                # Check for explicit quantity with SKU
                qty_patterns = [
                    r"(\d+)\s*(?:x|\*|instances?|nodes?|vms?)\s*(?:of\s*)?" + re.escape(sku_name),
                    re.escape(sku_name) + r"\s*(?:x|\*|:\s*|\(quantity:\s*|\(count:\s*)(\d+)",
                    r"deploy\s+(\d+)\s+" + re.escape(sku_name),
                ]
                sku_found = False
                for q_pat in qty_patterns:
                    qm = re.search(q_pat, raw_text, re.IGNORECASE)
                    if qm:
                        qty = int(qm.group(1) if qm.group(1).isdigit() else qm.group(2))
                        if qty > 0:
                            allocated_vms.append({
                                "sku": sku_name,
                                "instance_type": sku_name,
                                "count": qty,
                                "provider": sdata["provider"],
                                "vcpus": sdata["vcpus"],
                                "ram_gb": sdata["ram_gb"],
                                "monthly_cost": round(sdata["hourly_cost_usd"] * cls.HOURS_PER_MONTH * qty, 2),
                            })
                            evidence_list.append(
                                ExtractedEvidence(
                                    field_name=f"allocated_vm_{sku_name}",
                                    extracted_value=qty,
                                    unit="instances",
                                    text_span=qm.group(0),
                                    confidence="HIGH",
                                )
                            )
                            sku_found = True
                            break

                # If SKU is mentioned but without explicit quantity, check if it's an active recommendation vs passing mention
                if not sku_found:
                    mention_pat = r"\b" + re.escape(sku_name) + r"\b"
                    if re.search(mention_pat, raw_text, re.IGNORECASE):
                        # Check context: if it says "consider", "alternative", "such as", mark as passing mention (don't add)
                        context_match = re.search(
                            r"(?:consider|alternative|such as|option|instead of|or)\s+[^.\n]*" + re.escape(sku_name),
                            raw_text,
                            re.IGNORECASE,
                        )
                        if not context_match:
                            # Assume 1 instance but flag as NEEDS_REVIEW
                            allocated_vms.append({
                                "sku": sku_name,
                                "instance_type": sku_name,
                                "count": 1,
                                "provider": sdata["provider"],
                                "vcpus": sdata["vcpus"],
                                "ram_gb": sdata["ram_gb"],
                                "monthly_cost": round(sdata["hourly_cost_usd"] * cls.HOURS_PER_MONTH, 2),
                            })
                            evidence_list.append(
                                ExtractedEvidence(
                                    field_name=f"allocated_vm_{sku_name}",
                                    extracted_value=1,
                                    unit="instances (inferred)",
                                    text_span=sku_name,
                                    confidence="NEEDS_REVIEW",
                                )
                            )

            has_any_sku_mention = any(re.search(r"\b" + re.escape(sku_name) + r"\b", raw_text, re.IGNORECASE) for sku_name in cls.KNOWN_SKUS)
            if not allocated_vms:
                if has_any_sku_mention:
                    errors.append("Prose mentions cloud instance SKUs without actionable deployment quantities.")
                    status = NormalizationStatus.NEEDS_REVIEW
                else:
                    errors.append("No cloud VM instances could be extracted from prose.")
                    status = NormalizationStatus.NORMALIZATION_FAILURE
            elif any(e.confidence == "NEEDS_REVIEW" for e in evidence_list):
                status = NormalizationStatus.NEEDS_REVIEW
            else:
                status = NormalizationStatus.SUCCESS

            normalized_decision = {"allocated_vms": allocated_vms} if allocated_vms else None

        elif problem_type == "PSO_Continuous_Scaling":
            # Extract bandwidth and replicas
            bw_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mbps|gbps|bandwidth)", raw_text, re.IGNORECASE)
            rep_m = re.search(r"(\d+)\s*(?:replicas?|pods?|instances?|nodes?|workers?)", raw_text, re.IGNORECASE)

            bw_val = float(bw_m.group(1)) if bw_m else None
            rep_val = int(rep_m.group(1)) if rep_m else None

            if bw_m:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="bandwidth_mbps",
                        extracted_value=bw_val,
                        unit="Mbps",
                        text_span=bw_m.group(0),
                        confidence="HIGH",
                    )
                )
            if rep_m:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="replicas",
                        extracted_value=rep_val,
                        unit="replicas",
                        text_span=rep_m.group(0),
                        confidence="HIGH",
                    )
                )

            if bw_val is None or rep_val is None:
                errors.append(f"Incomplete scaling parameters: bandwidth={bw_val}, replicas={rep_val}")
                status = NormalizationStatus.NORMALIZATION_FAILURE
            else:
                status = NormalizationStatus.SUCCESS

            normalized_decision = {
                "optimal_bandwidth_mbps": bw_val,
                "recommended_replicas": rep_val,
            }

        elif problem_type == "Z3_Graph_Disaster_Recovery":
            # Extract region names
            reg_matches = re.findall(r"\b(us-east-1|us-west-2|eu-west-1|eastus|us-central1)\b", raw_text, re.IGNORECASE)
            if len(reg_matches) >= 2:
                primary = reg_matches[0].lower()
                secondary = reg_matches[1].lower()
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="primary_region",
                        extracted_value=primary,
                        unit="region_id",
                        text_span=primary,
                        confidence="HIGH",
                    )
                )
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="secondary_region",
                        extracted_value=secondary,
                        unit="region_id",
                        text_span=secondary,
                        confidence="HIGH",
                    )
                )
                status = NormalizationStatus.SUCCESS
                normalized_decision = {
                    "primary_region": primary,
                    "secondary_region": secondary,
                }
            else:
                errors.append(f"Could not extract two distinct regions (found: {reg_matches})")
                status = NormalizationStatus.NORMALIZATION_FAILURE
                normalized_decision = {
                    "primary_region": reg_matches[0] if reg_matches else None,
                    "secondary_region": None,
                }
        else:
            status = NormalizationStatus.TASK_INCOMPATIBLE
            errors.append(f"Unsupported problem type: {problem_type}")
            normalized_decision = None

        return status, extracted_monthly_cost, normalized_decision, errors, evidence_list

    @classmethod
    def normalize_mode2_json(
        cls,
        raw_json_or_text: Any,
        requested_problem_type: str = "ILP_VM_Allocation",
        contract_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NormalizationStatus, Optional[float], Any, List[str], List[ExtractedEvidence]]:
        """Parses and validates Mode 2 structured JSON against the requested problem task."""
        errors: List[str] = []
        evidence_list: List[ExtractedEvidence] = []

        if isinstance(raw_json_or_text, str):
            first_b = raw_json_or_text.find("{")
            last_b = raw_json_or_text.rfind("}")
            if first_b == -1 or last_b == -1:
                return (
                    NormalizationStatus.MALFORMED_OUTPUT,
                    None,
                    None,
                    ["No JSON object found in response."],
                    [],
                )
            try:
                parsed_json = json.loads(raw_json_or_text[first_b : last_b + 1])
            except Exception as exc:
                return (
                    NormalizationStatus.MALFORMED_OUTPUT,
                    None,
                    None,
                    [f"JSON parse error: {exc}"],
                    [],
                )
        elif isinstance(raw_json_or_text, dict):
            parsed_json = raw_json_or_text
        else:
            return (
                NormalizationStatus.MALFORMED_OUTPUT,
                None,
                None,
                ["Invalid JSON input type."],
                [],
            )

        # Extract claimed cost
        raw_cost = (
            parsed_json.get("total_monthly_cost")
            if parsed_json.get("total_monthly_cost") is not None
            else (
                parsed_json.get("total_monthly_cost_usd")
                if parsed_json.get("total_monthly_cost_usd") is not None
                else parsed_json.get("cost_usd")
            )
        )
        claimed_cost: Optional[float] = None
        if raw_cost is not None:
            try:
                claimed_cost = float(raw_cost)
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="total_monthly_cost_usd",
                        extracted_value=claimed_cost,
                        unit="USD/month",
                        text_span=str(raw_cost),
                        confidence="HIGH",
                    )
                )
            except (ValueError, TypeError):
                errors.append(f"Non-numeric total_monthly_cost: {raw_cost}")

        # Check Task Suitability & Discriminated Outcome
        task_type_claimed = parsed_json.get("task_type") or parsed_json.get("problem_type")
        outcome_claimed = str(
            parsed_json.get("outcome")
            or parsed_json.get("task_outcome")
            or parsed_json.get("status")
            or "ready"
        ).lower()

        if outcome_claimed in ["needs_clarification", "clarification_required"]:
            errors.append(parsed_json.get("reason", parsed_json.get("reasoning", "Model indicated query needs clarification.")))
            return (
                NormalizationStatus.CLARIFICATION_REQUIRED,
                claimed_cost,
                None,
                errors,
                evidence_list,
            )
        if outcome_claimed in ["unsupported", "task_incompatible"]:
            errors.append(parsed_json.get("reason", parsed_json.get("reasoning", "Model indicated query is unsupported.")))
            return (
                NormalizationStatus.TASK_INCOMPATIBLE,
                claimed_cost,
                None,
                errors,
                evidence_list,
            )
        if outcome_claimed in ["infeasible", "solver_infeasible"]:
            errors.append(parsed_json.get("reason", parsed_json.get("reasoning", "Model indicated requirements are infeasible.")))
            return (
                NormalizationStatus.SOLVER_INFEASIBLE,
                claimed_cost,
                None,
                errors,
                evidence_list,
            )

        has_vm_fields = ("instances" in parsed_json or "allocated_vms" in parsed_json) and bool(parsed_json.get("instances") or parsed_json.get("allocated_vms"))
        has_scaling_fields = ("bandwidth_mbps" in parsed_json or "optimal_bandwidth_mbps" in parsed_json or "bandwidth" in parsed_json) and ("recommended_replicas" in parsed_json or "replicas" in parsed_json or "worker_replicas" in parsed_json)
        has_dr_fields = "primary_region" in parsed_json and "secondary_region" in parsed_json and bool(parsed_json.get("primary_region")) and bool(parsed_json.get("secondary_region"))

        if requested_problem_type == "ILP_VM_Allocation" and not has_vm_fields and (has_scaling_fields or has_dr_fields or task_type_claimed in ["PSO_Continuous_Scaling", "Z3_Graph_Disaster_Recovery"]):
            errors.append(
                f"Task Incompatibility: Prompt requested VM Knapsack Allocation, but JSON emitted "
                f"'{task_type_claimed or ('Dynamic Scaling' if has_scaling_fields else 'Disaster Recovery')}' task schema."
            )
            return (
                NormalizationStatus.TASK_INCOMPATIBLE,
                claimed_cost,
                None,
                errors,
                evidence_list,
            )

        if requested_problem_type == "PSO_Continuous_Scaling" and not has_scaling_fields and (has_vm_fields or has_dr_fields or task_type_claimed in ["ILP_VM_Allocation", "Z3_Graph_Disaster_Recovery"]):
            errors.append(
                f"Task Incompatibility: Prompt requested Continuous Scaling task, but JSON emitted "
                f"'{task_type_claimed or ('VM Allocation' if has_vm_fields else 'Disaster Recovery')}' schema. "
                "Emitted structure cannot satisfy dynamic bandwidth/replica requirements."
            )
            return (
                NormalizationStatus.TASK_INCOMPATIBLE,
                claimed_cost,
                None,
                errors,
                evidence_list,
            )

        if requested_problem_type == "Z3_Graph_Disaster_Recovery" and not has_dr_fields and (has_vm_fields or has_scaling_fields or task_type_claimed in ["ILP_VM_Allocation", "PSO_Continuous_Scaling"]):
            errors.append(
                f"Task Incompatibility: Prompt requested Disaster Recovery task, but JSON emitted "
                f"'{task_type_claimed or ('VM Allocation' if has_vm_fields else 'Dynamic Scaling')}' schema. "
                "Emitted structure does not specify multi-region topological placement."
            )
            return (
                NormalizationStatus.TASK_INCOMPATIBLE,
                claimed_cost,
                None,
                errors,
                evidence_list,
            )

        # Normalization according to target problem
        if requested_problem_type == "ILP_VM_Allocation":
            raw_vms = parsed_json.get("allocated_vms") or parsed_json.get("instances") or []
            normalized_vms = []
            for item in raw_vms:
                if isinstance(item, dict):
                    s_name = item.get("sku") or item.get("instance_type") or item.get("name")
                    qty = item.get("quantity") if item.get("quantity") is not None else item.get("count", 1)
                    try:
                        qty_int = int(qty)
                    except (ValueError, TypeError):
                        qty_int = 1
                    sdata = cls.KNOWN_SKUS.get(s_name, {})
                    normalized_vms.append({
                        "sku": s_name,
                        "instance_type": s_name,
                        "count": qty_int,
                        "provider": item.get("cloud_provider", item.get("provider", sdata.get("provider", "AWS"))),
                        "monthly_cost": float(item.get("monthly_cost", 0.0)),
                    })
            if not normalized_vms:
                errors.append("No valid VM instances found in JSON.")
                return (
                    NormalizationStatus.NORMALIZATION_FAILURE,
                    claimed_cost,
                    {"allocated_vms": []},
                    errors,
                    evidence_list,
                )
            return (
                NormalizationStatus.SUCCESS,
                claimed_cost,
                {"allocated_vms": normalized_vms},
                errors,
                evidence_list,
            )

        elif requested_problem_type == "PSO_Continuous_Scaling":
            raw_bw = parsed_json.get("optimal_bandwidth_mbps", parsed_json.get("bandwidth_mbps", parsed_json.get("bandwidth")))
            raw_reps = parsed_json.get("recommended_replicas", parsed_json.get("replicas", parsed_json.get("worker_replicas")))
            try:
                bw_val = float(raw_bw) if raw_bw is not None else None
                reps_val = int(raw_reps) if raw_reps is not None else None
            except (ValueError, TypeError):
                bw_val = None
                reps_val = None

            if bw_val is None or reps_val is None:
                errors.append(f"Missing or invalid bandwidth ({raw_bw}) or replicas ({raw_reps}) in scaling JSON.")
                return (
                    NormalizationStatus.NORMALIZATION_FAILURE,
                    claimed_cost,
                    None,
                    errors,
                    evidence_list,
                )
            return (
                NormalizationStatus.SUCCESS,
                claimed_cost,
                {"optimal_bandwidth_mbps": bw_val, "recommended_replicas": reps_val},
                errors,
                evidence_list,
            )

        elif requested_problem_type == "Z3_Graph_Disaster_Recovery":
            prim = parsed_json.get("primary_region")
            sec = parsed_json.get("secondary_region")
            if not prim or not sec:
                errors.append(f"Missing primary ({prim}) or secondary ({sec}) region in DR JSON.")
                return (
                    NormalizationStatus.NORMALIZATION_FAILURE,
                    claimed_cost,
                    None,
                    errors,
                    evidence_list,
                )
            return (
                NormalizationStatus.SUCCESS,
                claimed_cost,
                {"primary_region": str(prim), "secondary_region": str(sec)},
                errors,
                evidence_list,
            )

        return (
            NormalizationStatus.TASK_INCOMPATIBLE,
            claimed_cost,
            None,
            [f"Unsupported problem type: {requested_problem_type}"],
            evidence_list,
        )
