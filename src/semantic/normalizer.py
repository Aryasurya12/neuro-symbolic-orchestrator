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
    def _filter_monthly_dollar_candidates(
        cls,
        raw_text: str,
        contract_data: Optional[Dict[str, Any]] = None,
        extracted_hourly_cost: Optional[float] = None,
    ) -> List[Tuple[float, str]]:
        """Filters dollar amounts in prose, strictly excluding unit rates, budget caps, arithmetic factors, and refusal figures."""
        matches = list(re.finditer(r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)", raw_text))
        valid_candidates = []
        budget_cap = float(contract_data.get("budget_max_usd", -999.0)) if contract_data and contract_data.get("budget_max_usd") is not None else -999.0

        for m in matches:
            val_str = m.group(1).replace(",", "")
            try:
                val = float(val_str)
            except ValueError:
                continue

            start_ctx = max(0, m.start() - 35)
            end_ctx = min(len(raw_text), m.end() + 35)
            imm_prefix = raw_text[start_ctx:m.start()].lower()
            imm_suffix = raw_text[m.end():end_ctx].lower()

            # 1. Exclude if equal to hourly cost or explicitly marked hourly
            if extracted_hourly_cost is not None and abs(val - extracted_hourly_cost) < 0.001:
                continue
            if re.search(r"^\s*(?:/\s*(?:hr|hour|h\b)|\bper\s+(?:hr|hour|h\b)|\bhourly\b)", imm_suffix):
                continue
            if re.search(r"(?:hourly\s+rate|hourly\s+cost|per\s+hour|per\s+hr|at)\s*[:=]?\s*$", imm_prefix):
                continue

            # 2. Exclude unit rates (per mbps, per replica, per instance, per vcpu, per gb, /mo per, base replica fee)
            if re.search(r"^\s*(?:/\s*(?:mbps|replica|rep|instance|core|gb|unit|node)|\bper\s+(?:mbps|replica|rep|instance|core|gb|unit|node))", imm_suffix):
                continue
            if re.search(r"(?:base\s+replica\s+fee|unit\s+price|price\s+per\s+mbps)\s*[:=]?\s*$", imm_prefix):
                continue

            # 3. Exclude budget caps: "budget of $X", "budget $X", "under $X", "limit of $X", "cap of $X", "for under $X"
            if re.search(r"(?:budget|budget\s+cap|budget\s+of|under|limit\s+of|limit\s+is|spend\s+of|within\s+(?:the|your)?|max\s+budget)\s*(?:of|is|:|=|under|capped\s+at)?\s*$", imm_prefix):
                continue
            if abs(val - budget_cap) < 0.01 and re.search(r"(?:budget|under|limit|cap|ceiling)", imm_prefix + " " + imm_suffix):
                continue

            # 4. Exclude arithmetic operators / formulas: e.g. "4 * $121.47" or "(105.0 * $0.08)"
            if re.search(r"[\*\+]\s*$", imm_prefix) or re.search(r"^\s*[\*\+]", imm_suffix):
                continue

            valid_candidates.append((val, m.group(0)))

        return valid_candidates

    @classmethod
    def normalize_mode1_prose(
        cls,
        raw_text: str,
        problem_type: str = "ILP_VM_Allocation",
        contract_data: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
    ) -> Tuple[NormalizationStatus, Optional[float], Any, List[str], List[ExtractedEvidence]]:
        """Extracts monthly cost and structured decision vector from Mode 1 prose with scope awareness."""
        if finish_reason == "length" and (not raw_text or not raw_text.strip()):
            return (
                NormalizationStatus.TRUNCATION_FAILURE,
                None,
                None,
                ["Provider/config truncation failure: model exceeded completion token budget (finish_reason='length')."],
                [],
            )

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
        lower_text = raw_text.lower()

        # 0. Check for explicit refusal or infeasibility statement from the LLM
        unsupported_phrases = [
            "no gpu skus",
            "no gpu instances",
            "gpu instances are not supported",
            "does not offer gpu",
            "cannot compute a deployment that satisfies",
            "do not have access to gpu",
            "hardware not supported",
            "unsupported workload",
        ]
        if any(p in lower_text for p in unsupported_phrases):
            return (
                NormalizationStatus.TASK_INCOMPATIBLE,
                None,
                None,
                ["Model explicitly reported workload is unsupported/refused in this environment."],
                evidence_list,
            )

        infeasible_phrases = [
            "cannot be satisfied",
            "is infeasible",
            "are infeasible",
            "mutually exclusive",
            "cannot provide a feasible deployment",
            "exceeds the budget",
            "exceeds your budget",
            "exceed your budget",
            "exceed the budget",
            "budget is too low",
            "no combination of available",
            "impossible under",
            "impossible to satisfy",
        ]
        if any(p in lower_text for p in infeasible_phrases):
            return (
                NormalizationStatus.SOLVER_INFEASIBLE,
                None,
                None,
                ["Model explicitly stated the requested workload is infeasible within constraints."],
                evidence_list,
            )

        # 1. Extract Hourly Rate if explicitly stated
        hourly_match = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*(?:/\s*hour|/\s*hr|\bper\s+hour|\bhourly)", raw_text, re.IGNORECASE)
        extracted_hourly_cost = float(hourly_match.group(1)) if hourly_match else None
        if hourly_match and extracted_hourly_cost is not None:
            evidence_list.append(
                ExtractedEvidence(
                    field_name="hourly_rate_usd",
                    extracted_value=extracted_hourly_cost,
                    unit="USD/hour",
                    text_span=hourly_match.group(0),
                    confidence="HIGH",
                )
            )

        # 2. Scope-Aware Cost Breakdown Extraction
        # Look for Grand / Overall total
        grand_total_patterns = [
            r"(?:grand\s+total|overall\s+total|overall\s+deployment\s+total|total\s+deployment\s+total|deployment\s+total|across\s+all\s+components|combined\s+monthly\s+cost|total\s+spend|total\s+monthly\s+spend)[^\S\r\n]*(?:is|:|=)?[^\S\r\n]*(?:[^\n$]*?)\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?:total\s+(?:estimated\s+)?(?:monthly\s+)?cost|estimated\s+total\s+(?:monthly\s+)?cost|monthly\s+cost|total\s+monthly\s+cost)[^\S\r\n]*(?:of|is|:|=|of\s+approximately)?[^\S\r\n]*\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:/\s*month|/\s*mo|\bper\s+month|\bmonthly)?",
            r"\|\s*\*\*Total(?:\s+monthly\s+cost)?\*\*\s*\|[^\n|]*?\*\*(?:Grand\s+total\s*=\s*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)[^\n|]*?\*\*",
            r"\|\s*\*\*Total\*\*\s*\|\s*\*\*\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\*\*",
            r"\btotal\s+(?:estimated\s+)?cost\s*:\s*\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        grand_total: Optional[float] = None
        grand_total_span: Optional[str] = None
        for pat in grand_total_patterns:
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    grand_total = val
                    grand_total_span = m.group(0)
                    break
                except ValueError:
                    pass

        # Look for Scaling subtotal (Bandwidth + Replicas)
        scaling_subtotal_patterns = [
            r"(?:scaling\s+subtotal|scaling\s+total|scaling\s+cost)[^\S\r\n]*(?:is|:|=|→|->)?[^\S\r\n]*(?:[^\n$|]*?)\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"\|\s*(?:Dynamic[\s-]*scaling|Scaling\s+cost|Dynamic[\s-]*scaling\s*\(bandwidth\s*\+\s*replicas\))\s*\|[^\S\r\n]*(?:[^\n$|]*?)\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"Replicas\s*\$\d+.*?->\s*\*\*\$(\d+(?:,\d+)*(?:\.\d+)?)\*\*",
            r"Dynamic[\s-]*scaling\s+total\s*=\s*\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?:subtotal\s+for\s+scaling|scaling\s+workload\s+subtotal)[^\S\r\n]*:[^\S\r\n]*\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        scaling_subtotal: Optional[float] = None
        scaling_subtotal_span: Optional[str] = None
        for pat in scaling_subtotal_patterns:
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    scaling_subtotal = val
                    scaling_subtotal_span = m.group(0)
                    break
                except ValueError:
                    pass

        # Look for VM subtotal / VM fleet
        vm_subtotal_patterns = [
            r"(?:vm\s+cost|vm\s+fleet|vm\s+subtotal|vm\s+total|subtotal\s+for\s+compute(?:\s+nodes)?|compute\s+nodes\s+subtotal)[^\S\r\n]*(?:is|:|=|→|->)?[^\S\r\n]*(?:[^\n$|]*?)\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"\|\s*(?:VM\s+fleet|VM\s+cost|VM\s+SKU)[^|]*\|[^\S\r\n]*(?:[^\n$|]*?)\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"VM\s+cost\s*\$\d+(?:\.\d+)?\s*\+\s*\$\d+(?:\.\d+)?\s*=\s*\$(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        vm_subtotal: Optional[float] = None
        vm_subtotal_span: Optional[str] = None
        for pat in vm_subtotal_patterns:
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    vm_subtotal = val
                    vm_subtotal_span = m.group(0)
                    break
                except ValueError:
                    pass

        # Look for DR subtotal / Region pair
        dr_subtotal_patterns = [
            r"(?:dr\s+cost|dr\s+subtotal|dr\s+total|region\s+total|region\s+pair|disaster\s+recovery(?:\s+cost|\s+subtotal)?|secondary\s+region)[^\S\r\n]*(?:is|:|=|→|->)?[^\S\r\n]*(?:[^\n$|]*?)\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"\|\s*(?:Region\s+pair|DR\s+cost|Disaster\s+Recovery)[^|]*\|[^\S\r\n]*(?:[^\n$|]*?)\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"Base\s+[^=\n]*=\s*\*\*\$(\d+(?:,\d+)*(?:\.\d+)?)\*\*",
            r"Region\s+total\s*=\s*\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        dr_subtotal: Optional[float] = None
        dr_subtotal_span: Optional[str] = None
        for pat in dr_subtotal_patterns:
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    dr_subtotal = val
                    dr_subtotal_span = m.group(0)
                    break
                except ValueError:
                    pass

        extracted_monthly_cost: Optional[float] = None
        cost_ambiguous: bool = False

        # Select claimed cost scoped to requested problem type
        if problem_type == "PSO_Continuous_Scaling":
            if scaling_subtotal is not None:
                extracted_monthly_cost = scaling_subtotal
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="scaling_subtotal_cost_usd",
                        extracted_value=scaling_subtotal,
                        unit="USD/month",
                        text_span=scaling_subtotal_span or str(scaling_subtotal),
                        confidence="HIGH",
                    )
                )
            elif grand_total is not None and vm_subtotal is None and dr_subtotal is None:
                extracted_monthly_cost = grand_total
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="total_monthly_cost_usd",
                        extracted_value=grand_total,
                        unit="USD/month",
                        text_span=grand_total_span or str(grand_total),
                        confidence="HIGH",
                    )
                )
            elif grand_total is not None:
                extracted_monthly_cost = grand_total
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="grand_total_monthly_cost_usd",
                        extracted_value=grand_total,
                        unit="USD/month",
                        text_span=grand_total_span or str(grand_total),
                        confidence="NEEDS_REVIEW",
                    )
                )
            else:
                valid_cands = cls._filter_monthly_dollar_candidates(raw_text, contract_data, extracted_hourly_cost)
                if len(valid_cands) == 1:
                    extracted_monthly_cost = valid_cands[0][0]
                    evidence_list.append(
                        ExtractedEvidence(
                            field_name="total_monthly_cost_usd",
                            extracted_value=valid_cands[0][0],
                            unit="USD/month",
                            text_span=valid_cands[0][1],
                            confidence="HIGH",
                        )
                    )
                elif len(valid_cands) > 1:
                    cost_ambiguous = True
                    errors.append("Multiple ambiguous dollar figures in prose without explicit scaling subtotal.")

            # Record breakdown evidence and unsolicited components
            if grand_total is not None and scaling_subtotal is not None:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="grand_total_monthly_cost_usd",
                        extracted_value=grand_total,
                        unit="USD/month",
                        text_span=grand_total_span or str(grand_total),
                        confidence="HIGH",
                    )
                )
            if vm_subtotal is not None:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="unsolicited_vm_cost_usd",
                        extracted_value=vm_subtotal,
                        unit="USD/month",
                        text_span=vm_subtotal_span or str(vm_subtotal),
                        confidence="HIGH",
                    )
                )
            if dr_subtotal is not None:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="unsolicited_dr_cost_usd",
                        extracted_value=dr_subtotal,
                        unit="USD/month",
                        text_span=dr_subtotal_span or str(dr_subtotal),
                        confidence="HIGH",
                    )
                )
            if vm_subtotal is not None or dr_subtotal is not None:
                errors.append("Unsolicited VM/DR components detected; scope expansion preserved in evidence.")

        elif problem_type == "ILP_VM_Allocation":
            if vm_subtotal is not None:
                extracted_monthly_cost = vm_subtotal
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="vm_subtotal_cost_usd",
                        extracted_value=vm_subtotal,
                        unit="USD/month",
                        text_span=vm_subtotal_span or str(vm_subtotal),
                        confidence="HIGH",
                    )
                )
            elif grand_total is not None:
                extracted_monthly_cost = grand_total
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="total_monthly_cost_usd",
                        extracted_value=grand_total,
                        unit="USD/month",
                        text_span=grand_total_span or str(grand_total),
                        confidence="HIGH",
                    )
                )
            else:
                valid_cands = cls._filter_monthly_dollar_candidates(raw_text, contract_data, extracted_hourly_cost)
                if len(valid_cands) == 1:
                    extracted_monthly_cost = valid_cands[0][0]
                    evidence_list.append(
                        ExtractedEvidence(
                            field_name="total_monthly_cost_usd",
                            extracted_value=valid_cands[0][0],
                            unit="USD/month",
                            text_span=valid_cands[0][1],
                            confidence="HIGH",
                        )
                    )
                elif len(valid_cands) > 1:
                    cost_ambiguous = True
                    errors.append("Multiple ambiguous dollar figures in prose without explicit VM subtotal.")

            if grand_total is not None and vm_subtotal is not None and grand_total != vm_subtotal:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="grand_total_monthly_cost_usd",
                        extracted_value=grand_total,
                        unit="USD/month",
                        text_span=grand_total_span or str(grand_total),
                        confidence="HIGH",
                    )
                )
            if scaling_subtotal is not None:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="unsolicited_scaling_cost_usd",
                        extracted_value=scaling_subtotal,
                        unit="USD/month",
                        text_span=scaling_subtotal_span or str(scaling_subtotal),
                        confidence="HIGH",
                    )
                )
            if dr_subtotal is not None:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="unsolicited_dr_cost_usd",
                        extracted_value=dr_subtotal,
                        unit="USD/month",
                        text_span=dr_subtotal_span or str(dr_subtotal),
                        confidence="HIGH",
                    )
                )
            if scaling_subtotal is not None or dr_subtotal is not None:
                errors.append("Unsolicited Scaling/DR components detected; scope expansion preserved in evidence.")

        elif problem_type == "Z3_Graph_Disaster_Recovery":
            if dr_subtotal is not None:
                extracted_monthly_cost = dr_subtotal
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="dr_subtotal_cost_usd",
                        extracted_value=dr_subtotal,
                        unit="USD/month",
                        text_span=dr_subtotal_span or str(dr_subtotal),
                        confidence="HIGH",
                    )
                )
            elif grand_total is not None:
                extracted_monthly_cost = grand_total
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="total_monthly_cost_usd",
                        extracted_value=grand_total,
                        unit="USD/month",
                        text_span=grand_total_span or str(grand_total),
                        confidence="HIGH",
                    )
                )
            else:
                valid_cands = cls._filter_monthly_dollar_candidates(raw_text, contract_data, extracted_hourly_cost)
                if len(valid_cands) == 1:
                    extracted_monthly_cost = valid_cands[0][0]
                    evidence_list.append(
                        ExtractedEvidence(
                            field_name="total_monthly_cost_usd",
                            extracted_value=valid_cands[0][0],
                            unit="USD/month",
                            text_span=valid_cands[0][1],
                            confidence="HIGH",
                        )
                    )
                elif len(valid_cands) > 1:
                    cost_ambiguous = True
                    errors.append("Multiple ambiguous dollar figures in prose without explicit DR subtotal.")

            if grand_total is not None and dr_subtotal is not None and grand_total != dr_subtotal:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="grand_total_monthly_cost_usd",
                        extracted_value=grand_total,
                        unit="USD/month",
                        text_span=grand_total_span or str(grand_total),
                        confidence="HIGH",
                    )
                )
            if vm_subtotal is not None:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="unsolicited_vm_cost_usd",
                        extracted_value=vm_subtotal,
                        unit="USD/month",
                        text_span=vm_subtotal_span or str(vm_subtotal),
                        confidence="HIGH",
                    )
                )
            if scaling_subtotal is not None:
                evidence_list.append(
                    ExtractedEvidence(
                        field_name="unsolicited_scaling_cost_usd",
                        extracted_value=scaling_subtotal,
                        unit="USD/month",
                        text_span=scaling_subtotal_span or str(scaling_subtotal),
                        confidence="HIGH",
                    )
                )
            if vm_subtotal is not None or scaling_subtotal is not None:
                errors.append("Unsolicited VM/Scaling components detected; scope expansion preserved in evidence.")

        # 3. Extract Decision Allocation based on problem type
        if problem_type == "ILP_VM_Allocation":
            allocated_vms = []
            for sku_name, sdata in cls.KNOWN_SKUS.items():
                qty_patterns = [
                    r"(\d+)\s*(?:x|\*|instances?|nodes?|vms?)\s*(?:of\s*)?" + re.escape(sku_name),
                    re.escape(sku_name) + r"\s*(?:x|\*|:\s*|\(quantity:\s*|\(count:\s*)(\d+)",
                    r"deploy\s+(\d+)\s+" + re.escape(sku_name),
                    r"provision\s+(\d+)\s+" + re.escape(sku_name),
                ]
                sku_found = False
                for q_pat in qty_patterns:
                    qm = re.search(q_pat, raw_text, re.IGNORECASE)
                    if qm:
                        match_start = max(0, qm.start() - 60)
                        match_ctx = raw_text[match_start:qm.end()].lower()
                        if "would need" in match_ctx or "exceed" in match_ctx or "instead of" in match_ctx or "alternative" in match_ctx or "option" in match_ctx:
                            continue
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

            has_any_sku_mention = any(re.search(r"\b" + re.escape(sku_name) + r"\b", raw_text, re.IGNORECASE) for sku_name in cls.KNOWN_SKUS)
            if cost_ambiguous:
                status = NormalizationStatus.AMBIGUOUS
            elif not allocated_vms:
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
            bw_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mbps|gbps)", raw_text, re.IGNORECASE)
            if not bw_m:
                bw_m = re.search(r"(?:bandwidth|traffic|load)[^\d\n]{0,30}(\d+(?:\.\d+)?)", raw_text, re.IGNORECASE)

            rep_m = re.search(r"(\d+)\s*(?:x\s*)?(?:worker\s+replicas?|replicas?|pods?|workers?|nodes?)", raw_text, re.IGNORECASE)
            if not rep_m:
                rep_m = re.search(r"(?:replicas?|pods?|workers?|worker\s+replicas?|nodes?)[:=\s\*\u202f]+(\d+)", raw_text, re.IGNORECASE)

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

            if cost_ambiguous:
                status = NormalizationStatus.AMBIGUOUS
            elif bw_val is None or rep_val is None:
                errors.append(f"Incomplete scaling parameters: bandwidth={bw_val}, replicas={rep_val}")
                status = NormalizationStatus.NORMALIZATION_FAILURE
            elif any(e.confidence == "NEEDS_REVIEW" for e in evidence_list):
                status = NormalizationStatus.NEEDS_REVIEW
            else:
                status = NormalizationStatus.SUCCESS

            normalized_decision = {
                "optimal_bandwidth_mbps": bw_val,
                "recommended_replicas": rep_val,
            }

        elif problem_type == "Z3_Graph_Disaster_Recovery":
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
                if cost_ambiguous:
                    status = NormalizationStatus.AMBIGUOUS
                else:
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
        finish_reason: Optional[str] = None,
    ) -> Tuple[NormalizationStatus, Optional[float], Any, List[str], List[ExtractedEvidence]]:
        """Parses and validates Mode 2 structured JSON against the requested problem task."""
        errors: List[str] = []
        evidence_list: List[ExtractedEvidence] = []

        if finish_reason == "length" and (not raw_json_or_text or (isinstance(raw_json_or_text, str) and not raw_json_or_text.strip())):
            return (
                NormalizationStatus.TRUNCATION_FAILURE,
                None,
                None,
                ["Provider/config truncation failure: model exceeded completion token budget (finish_reason='length')."],
                [],
            )

        if isinstance(raw_json_or_text, str):
            first_b = raw_json_or_text.find("{")
            last_b = raw_json_or_text.rfind("}")
            if first_b == -1 or last_b == -1:
                if finish_reason == "length":
                    return (
                        NormalizationStatus.TRUNCATION_FAILURE,
                        None,
                        None,
                        ["Provider/config truncation failure: JSON output truncated mid-stream (finish_reason='length')."],
                        [],
                    )
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
