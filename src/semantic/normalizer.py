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
    def _normalize_unicode(cls, text: str) -> str:
        """Normalizes Unicode hyphens (U+2010 to U+2015, U+2212) and whitespace to standard ASCII."""
        if not text:
            return ""
        # Unicode hyphens/dashes: U+2010 (HYPHEN), U+2011 (NON-BREAKING HYPHEN), U+2012 (FIGURE DASH),
        # U+2013 (EN DASH), U+2014 (EM DASH), U+2015 (HORIZONTAL BAR), U+2212 (MINUS SIGN),
        # plus compatibility dashes U+FE58, U+FE63, U+FF0D, U+2043, U+2500
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d\u2043\u2500]", "-", text)
        # Unicode spaces: U+00A0 (NO-BREAK SPACE), U+2000-U+200A, U+202F (NARROW NO-BREAK SPACE),
        # U+205F (MATH SPACE), U+3000 (IDEOGRAPHIC SPACE), U+FEFF (ZERO WIDTH NO-BREAK SPACE)
        text = re.sub(r"[\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\ufeff]", " ", text)
        return text

    @classmethod
    def _strip_markdown(cls, text: str) -> str:
        """Strips markdown asterisks (* and **) from text."""
        if not text:
            return ""
        return re.sub(r"\*\*|\*", "", text)

    @classmethod
    def _filter_monthly_dollar_candidates(
        cls,
        raw_text: str,
        contract_data: Optional[Dict[str, Any]] = None,
        extracted_hourly_cost: Optional[float] = None,
    ) -> List[Tuple[float, str]]:
        """Filters dollar amounts in prose, strictly excluding unit rates, budget caps, arithmetic factors, component prices, and refusal figures."""
        raw_text = cls._normalize_unicode(raw_text)
        matches = list(re.finditer(r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)", raw_text))
        valid_candidates = []
        budget_cap = float(contract_data.get("budget_max_usd", -999.0)) if contract_data and contract_data.get("budget_max_usd") is not None else -999.0

        for m in matches:
            val_str = m.group(1).replace(",", "")
            try:
                val = float(val_str)
            except ValueError:
                continue

            start_ctx = max(0, m.start() - 55)
            end_ctx = min(len(raw_text), m.end() + 55)
            imm_prefix = raw_text[start_ctx:m.start()].lower()
            imm_suffix = raw_text[m.end():end_ctx].lower()

            imm_prefix_clean = cls._strip_markdown(imm_prefix)
            imm_suffix_clean = cls._strip_markdown(imm_suffix)

            # 1. Exclude if equal to hourly cost or explicitly marked hourly
            if extracted_hourly_cost is not None and abs(val - extracted_hourly_cost) < 0.001:
                continue
            if re.search(r"^\s*(?:/\s*(?:hr|hour|h\b)|\bper\s+(?:hr|hour|h\b)|\bhourly\b)", imm_suffix_clean):
                continue
            if re.search(r"(?:hourly\s+rate|hourly\s+cost|per\s+hour|per\s+hr|at)\s*[:=]?\s*$", imm_prefix_clean):
                continue

            # 2. Exclude unit rates (preceded by ×, \u00d7, *, @, x or followed by /ms, /hr, /Mbps, per, /replica, /instance, /vcpu, /gb, /unit, /node)
            if re.search(r"^\s*(?:/\s*(?:ms|hr|hour|h\b|mbps|gbps|replica|rep|instance|core|vcpu|gb|unit|node|vm)\b|\bper\b)", imm_suffix_clean):
                continue
            if re.search(r"(?:base\s+replica\s+fee|unit\s+price|price\s+per\s+mbps|rate)\s*[:=]?\s*$", imm_prefix_clean):
                continue
            if re.search(r"(?:[×\u00d7\*\@x]|times|multiplied\s+by|\bat\b)\s*$", imm_prefix.rstrip()):
                continue
            if re.search(r"^\s*[×\u00d7\*\+x]", imm_suffix.lstrip()):
                continue

            # 3. Exclude budget caps: "budget of $X", "budget $X", "under $X", "limit of $X", "cap of $X", "for under $X", "$X budget", "$X limit"
            if re.search(r"(?:budget|budget\s+cap|budget\s+of|under|limit\s+of|limit\s+is|spend\s+of|within\s+(?:the|your)?|max\s+budget|budget\s+limit|ceiling\s+of|cap\s+of)\s*(?:of|is|:|=|under|capped\s+at)?\s*$", imm_prefix_clean):
                continue
            if re.search(r"^\s*(?:budget|cap|ceiling|limit|monthly\s+budget|max\s+budget|monthly\s+spend|monthly\s+limit)", imm_suffix_clean):
                continue
            if abs(val - budget_cap) < 0.01 and re.search(r"(?:budget|under|limit|cap|ceiling|spend)", imm_prefix_clean + " " + imm_suffix_clean):
                continue

            # 4. Exclude arithmetic operators / formulas: e.g. "4 * $121.47" or "(105.0 * $0.08)"
            if re.search(r"[\*\+\=]\s*$", imm_prefix_clean) and not re.search(r"(?:total|cost|spend)\s*=\s*$", imm_prefix_clean):
                continue
            if re.search(r"^\s*[\*\+]", imm_suffix_clean):
                continue

            # 5. Exclude single component line items: e.g. "base DR cost $120", "base compute $120", "replication $8", "latency surcharge $8"
            if re.search(r"(?:base\s+dr\s+cost|base\s+dr|base\s+compute|compute\s+only|standby\s+instance|standby|surcharge|latency\s+surcharge|replication|storage\s+only|each\s+vm|per\s+vm|each\s+instance|per\s+instance|primary\s+region(?:\s+base)?|secondary\s+region(?:\s+base)?|base\s+cost)\s*(?:of|is|:|=|at)?\s*$", imm_prefix_clean):
                continue

            # 6. Exclude alternative / rejected option figures: e.g. "Alternative 1 would cost $485", "Option A ($485) exceeds budget", "would cost $224"
            if re.search(r"(?:alternative|option\s+[a-z0-9]|instead\s+of|would\s+cost|would\s+need|would\s+require|requires\s+at\s+least|starting\s+at|minimal\s+cost\s+of|minimum\s+cost\s+of|cheapest\s+valid|exceeds?\s+budget|exceeds?\s+your\s+budget|exceeds?\s+the\s+budget|exceeds?\s+the\s+limit|rejected)\b", imm_prefix_clean):
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

        # 0. Normalise Unicode hyphens/dashes and whitespace to ASCII
        raw_text = cls._normalize_unicode(raw_text)

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
            "unsupported task",
            "outside the supported domain",
            "not supported by the catalog",
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
            "cannot satisfy",
            "is infeasible",
            "are infeasible",
            "infeasible",
            "unfeasible",
            "mutually exclusive",
            "cannot provide a feasible deployment",
            "cannot provide a feasible allocation",
            "cannot provide a feasible plan",
            "cannot provide a feasible",
            "cannot find a feasible",
            "no feasible deployment",
            "no feasible allocation",
            "no feasible configuration",
            "no feasible plan",
            "no feasible",
            "cannot fulfill",
            "cannot meet",
            "unable to fulfill",
            "unable to provide",
            "not possible within",
            "not possible under",
            "not feasible within",
            "not feasible under",
            "impossible under",
            "impossible to satisfy",
            "impossible to provide",
            "exceeds the budget",
            "exceeds your budget",
            "exceed your budget",
            "exceed the budget",
            "exceeds the specified budget",
            "exceeds the allocated budget",
            "exceeds the monthly budget",
            "exceeds the budget cap",
            "exceeds the limit",
            "exceeds budget",
            "budget is too low",
            "budget is too small",
            "budget is too tight",
            "budget is insufficient",
            "insufficient budget",
            "budget is not enough",
            "not enough budget",
            "no combination of available",
            "no combination of",
            "cannot deploy within",
            "cannot be deployed within",
            "cannot fit within",
        ]

        # Check if the entire response is a refusal/infeasibility statement without a chosen/recommended plan
        has_chosen_plan_header = bool(re.search(r"(?:recommendation|chosen\s+plan|recommended\s+plan|recommended\s+deployment|selected\s+plan|recommended\s+configuration|option\s+[0-9]+\s*\(recommended\)|optimal\s+deployment)\b", lower_text))
        
        # If response states infeasibility and has no chosen plan section, return SOLVER_INFEASIBLE with cost=None
        if any(p in lower_text for p in infeasible_phrases) and not has_chosen_plan_header:
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
        clean_text = cls._strip_markdown(raw_text)

        # Look for Grand / Overall total of the CHOSEN plan
        # Prefer the figure on a line labelled "Total" or "Total monthly cost", preferring the last labelled total.
        grand_total: Optional[float] = None
        grand_total_span: Optional[str] = None

        # Pass 1: Line-by-line search on stripped text for explicit total labels
        total_line_candidates: List[Tuple[float, str, int]] = []
        for line in clean_text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            line_lower = line_str.lower()
            if any(rej in line_lower for rej in ["alternative", "rejected", "option 1 (exceed", "option a (exceed", "option 1 (reject", "option a (reject", "instead of", "would cost", "would need", "would require"]):
                continue
            if any(b in line_lower for b in ["budget of", "limit of", "budget cap", "max budget", "under $"]):
                continue
            # Exclude subtotal lines (e.g. "scaling subtotal", "vm subtotal", "dr subtotal", "compute nodes subtotal")
            if re.search(r"\b(?:subtotal|base\s+dr|base\s+compute|replication|surcharge|bandwidth|replicas)\b", line_lower):
                continue

            # Match lines with "total monthly cost", "total monthly spend", "overall deployment total", "grand total", "overall total", "total cost", "total spend", "total"
            m_tot = re.search(r"(?:overall\s+(?:deployment\s+)?total|grand\s+total|total\s+monthly\s+(?:cost|spend|deployment\s+cost)|total\s+(?:cost|spend|deployment\s+cost)|total)\s*[:=]\s*\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)", line_str, re.IGNORECASE)
            if m_tot:
                m_val_str = m_tot.group(1).replace(",", "")
                try:
                    val = float(m_val_str)
                    imm_suffix = line_str[m_tot.end():].lower()
                    if not re.search(r"^\s*(?:/\s*(?:ms|hr|hour|h\b|mbps|gbps|replica|rep|instance|core|vcpu|gb|unit|node|vm)\b|\bper\b)", imm_suffix):
                        priority = 1
                        if "overall" in line_lower or "grand total" in line_lower:
                            priority = 3
                        elif "total monthly" in line_lower or "monthly cost" in line_lower:
                            priority = 2
                        total_line_candidates.append((val, line_str, priority))
                except ValueError:
                    pass

        if total_line_candidates:
            max_prio = max(c[2] for c in total_line_candidates)
            matching = [c for c in total_line_candidates if c[2] == max_prio]
            grand_total = matching[-1][0]
            grand_total_span = matching[-1][1]

        # Pass 2: Grand total regex patterns as fallback
        if grand_total is None:
            grand_total_patterns = [
                # "Exact Estimated Total Monthly Cost: $243.00", "Exact Total Monthly Cost: $243.00", "Exact Estimated Total Monthly Cost ($/month): $243.00"
                r"(?:exact\s+)?(?:estimated\s+)?total\s+(?:monthly\s+)?(?:cost|spend|deployment\s+cost)[^\S\r\n]*(?:\(\$/month\)|\(\$/mo\))?[^\S\r\n]*(?:is|of|:|=|of\s+approximately)?[^\S\r\n]*\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:/\s*month|/\s*mo|\bper\s+month|\bmonthly)?",
                # "Grand Total: $243.00", "Overall Total: $243.00", "Total Deployment Total: $243.00", "Combined monthly cost: $243.00"
                r"(?:grand\s+total|overall\s+total|overall\s+deployment\s+total|total\s+deployment\s+total|deployment\s+total|across\s+all\s+components|combined\s+monthly\s+cost|total\s+spend|total\s+monthly\s+spend)[^\S\r\n]*(?:is|:|=)?[^\S\r\n]*(?:[^\n$]*?)\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)",
                # Markdown table rows: "| Total Monthly Cost | $243.00 |" or "| Total | $243.00 |"
                r"\|\s*Total(?:\s+Monthly\s+Cost|\s+Cost|\s+Estimated\s+Cost)?\s*\|[^\n|]*?(?:Grand\s+total\s*=\s*)?\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)[^\n|]*?",
                # "with a total cost of $243 per month", "with a total monthly cost of $243"
                r"\bwith\s+a\s+total\s+(?:monthly\s+)?cost\s+of\s+\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)",
                # "Total: $243.00", "Total = $243.00"
                r"\btotal\s*[:=]\s*\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)",
                # "Total of $243/month", "Total of $243.00"
                r"\btotal\s+of\s+\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:/\s*mo|/\s*month|\bper\s+month|\bmonthly)?",
                # "$243.00/month total", "$243 per month total"
                r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:/\s*month|/\s*mo|\bper\s+month|\bmonthly)\s*(?:total|in\s+total|all[\s-]inclusive)",
            ]
            for pat in grand_total_patterns:
                for m in re.finditer(pat, clean_text, re.IGNORECASE):
                    # Verify match is not inside an alternative/rejected option block or budget cap
                    m_start = m.start()
                    start_ctx = max(0, m_start - 60)
                    ctx = clean_text[start_ctx:m_start].lower()
                    if any(rej in ctx for rej in ["alternative", "rejected", "option 1 (exceed", "option a (exceed", "option 1 (reject", "option a (reject", "instead of", "would cost", "would need", "would require"]):
                        continue
                    if any(b in ctx for b in ["budget", "budget of", "under", "limit of", "budget cap", "max budget", "within"]):
                        continue
                    try:
                        val = float(m.group(1).replace(",", ""))
                        grand_total = val
                        grand_total_span = m.group(0)
                        break
                    except ValueError:
                        pass
                if grand_total is not None:
                    break

        # Look for Scaling subtotal (Bandwidth + Replicas)
        scaling_subtotal_patterns = [
            r"(?:scaling\s+subtotal|scaling\s+total|scaling\s+cost|total\s+scaling\s+cost)[^\S\r\n]*(?:is|:|=|→|->)?[^\S\r\n]*(?:[^\n$|]*?)(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"\|\s*(?:Dynamic[\s-]*scaling|Scaling\s+cost|Dynamic[\s-]*scaling\s*\(bandwidth\s*\+\s*replicas\))\s*\|[^\S\r\n]*(?:[^\n$|]*?)(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"Replicas\s*\$\d+.*?->\s*\*\*\$(\d+(?:,\d+)*(?:\.\d+)?)\*\*",
            r"Dynamic[\s-]*scaling\s+total\s*=\s*(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?:subtotal\s+for\s+scaling|scaling\s+workload\s+subtotal)[^\S\r\n]*:[^\S\r\n]*(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        scaling_subtotal: Optional[float] = None
        scaling_subtotal_span: Optional[str] = None
        for pat in scaling_subtotal_patterns:
            for m in re.finditer(pat, raw_text, re.IGNORECASE):
                m_start = m.start()
                start_ctx = max(0, m_start - 60)
                ctx = raw_text[start_ctx:m_start].lower()
                if any(rej in ctx for rej in ["alternative", "rejected", "option 1", "option a", "instead of"]):
                    continue
                try:
                    val = float(m.group(1).replace(",", ""))
                    scaling_subtotal = val
                    scaling_subtotal_span = m.group(0)
                    break
                except ValueError:
                    pass
            if scaling_subtotal is not None:
                break

        # Look for VM subtotal / VM fleet
        vm_subtotal_patterns = [
            r"(?:vm\s+cost|vm\s+fleet|vm\s+subtotal|vm\s+total|subtotal\s+for\s+compute(?:\s+nodes)?|compute\s+nodes\s+subtotal|total\s+vm\s+cost)[^\S\r\n]*(?:is|:|=|→|->)?[^\S\r\n]*(?:[^\n$|]*?)(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"\|\s*(?:VM\s+fleet|VM\s+cost|VM\s+SKU|Compute\s+Nodes)[^|]*\|[^\S\r\n]*(?:[^\n$|]*?)(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"VM\s+cost\s*\$\d+(?:\.\d+)?\s*\+\s*\$\d+(?:\.\d+)?\s*=\s*(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        vm_subtotal: Optional[float] = None
        vm_subtotal_span: Optional[str] = None
        for pat in vm_subtotal_patterns:
            for m in re.finditer(pat, raw_text, re.IGNORECASE):
                m_start = m.start()
                start_ctx = max(0, m_start - 60)
                ctx = raw_text[start_ctx:m_start].lower()
                if any(rej in ctx for rej in ["alternative", "rejected", "option 1", "option a", "instead of"]):
                    continue
                try:
                    val = float(m.group(1).replace(",", ""))
                    vm_subtotal = val
                    vm_subtotal_span = m.group(0)
                    break
                except ValueError:
                    pass
            if vm_subtotal is not None:
                break

        # Look for DR subtotal / Region pair total
        dr_subtotal_patterns = [
            r"(?:total\s+dr\s+cost|dr\s+total|dr\s+subtotal|region\s+pair\s+total|total\s+disaster\s+recovery\s+cost|disaster\s+recovery\s+total|combined\s+dr\s+cost|disaster\s+recovery\s+cost)[^\S\r\n]*(?:is|:|=|→|->)?[^\S\r\n]*(?:[^\n$|]*?)(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"\|\s*(?:Disaster[\s-]*recovery\s+pair(?:\s*\(region\s+base\))?|Region\s+pair\s+total|DR\s+Total|Disaster\s+Recovery\s+Total|Combined\s+DR)[^|]*\|[^\n|]*?=\s*(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"\|\s*(?:Disaster[\s-]*recovery\s+pair(?:\s*\(region\s+base\))?|Region\s+pair\s+total|DR\s+Total|Disaster\s+Recovery\s+Total|Combined\s+DR)[^|]*\|[^\S\r\n]*(?:[^\n$|]*?)(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?:secondary\s+region\s+in\s+[a-z0-9-]+|multi-region\s+disaster\s+recovery)[^\S\r\n]*:[^\S\r\n]*(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"DR\s+Total\s*=\s*(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)(?:\*\*)?",
            r"Region\s+total\s*=\s*(?:\*\*)?\$\s*(\d+(?:,\d+)*(?:\.\d+)?)(?:\*\*)?",
        ]
        dr_subtotal: Optional[float] = None
        dr_subtotal_span: Optional[str] = None
        for pat in dr_subtotal_patterns:
            for m in re.finditer(pat, raw_text, re.IGNORECASE):
                m_start = m.start()
                start_ctx = max(0, m_start - 60)
                ctx = raw_text[start_ctx:m_start].lower()
                if any(rej in ctx for rej in ["alternative", "rejected", "option 1", "option a", "instead of"]):
                    continue
                try:
                    val = float(m.group(1).replace(",", ""))
                    dr_subtotal = val
                    dr_subtotal_span = m.group(0)
                    break
                except ValueError:
                    pass
            if dr_subtotal is not None:
                break

        extracted_monthly_cost: Optional[float] = None
        cost_ambiguous: bool = False

        # Determine effective problem type if defaulted to ILP_VM_Allocation with no VM content
        effective_problem_type = problem_type
        if effective_problem_type == "ILP_VM_Allocation":
            has_vm_sku_mention = any(re.search(r"\b" + re.escape(sku_name) + r"\b", raw_text, re.IGNORECASE) for sku_name in cls.KNOWN_SKUS)
            has_dr_regions_count = len(re.findall(r"\b(us-east-1|us-west-2|eu-west-1|eastus|us-central-?1|us-central\s+1)\b", raw_text, re.IGNORECASE))
            has_scaling_signals = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mbps|gbps)\b", raw_text, re.IGNORECASE) and re.search(r"\b\d+\s*(?:worker\s+replicas?|replicas?|pods?|workers?)\b", raw_text, re.IGNORECASE))

            if has_dr_regions_count >= 2 and not has_vm_sku_mention:
                effective_problem_type = "Z3_Graph_Disaster_Recovery"
            elif has_scaling_signals and not has_vm_sku_mention:
                effective_problem_type = "PSO_Continuous_Scaling"

        # Select claimed cost scoped to requested/effective problem type
        if effective_problem_type == "PSO_Continuous_Scaling":
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
                unique_cands = {round(v[0], 2) for v in valid_cands}
                if len(unique_cands) == 1:
                    extracted_monthly_cost = list(unique_cands)[0]
                    first_span = next(s for v, s in valid_cands if round(v, 2) == extracted_monthly_cost)
                    evidence_list.append(
                        ExtractedEvidence(
                            field_name="total_monthly_cost_usd",
                            extracted_value=extracted_monthly_cost,
                            unit="USD/month",
                            text_span=first_span,
                            confidence="HIGH",
                        )
                    )
                elif len(unique_cands) > 1:
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

        elif effective_problem_type == "ILP_VM_Allocation":
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
                unique_cands = {round(v[0], 2) for v in valid_cands}
                if len(unique_cands) == 1:
                    extracted_monthly_cost = list(unique_cands)[0]
                    first_span = next(s for v, s in valid_cands if round(v, 2) == extracted_monthly_cost)
                    evidence_list.append(
                        ExtractedEvidence(
                            field_name="total_monthly_cost_usd",
                            extracted_value=extracted_monthly_cost,
                            unit="USD/month",
                            text_span=first_span,
                            confidence="HIGH",
                        )
                    )
                elif len(unique_cands) > 1:
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

        elif effective_problem_type == "Z3_Graph_Disaster_Recovery":
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
                # Filter out candidates that are explicitly labeled as single component items
                filtered_cands = []
                for cand_val, cand_span in valid_cands:
                    cand_m = re.search(re.escape(cand_span), raw_text)
                    if cand_m:
                        start_ctx = max(0, cand_m.start() - 40)
                        ctx = raw_text[start_ctx:cand_m.start()].lower()
                        component_labels = [
                            "base compute", "compute + storage", "compute only", "standby instance",
                            "standby:", "storage only", "bandwidth:", "replication:", "data replication",
                            "egress:", "ingress:", "component:", "each vm:", "per vm:", "base dr cost",
                            "base dr", "surcharge", "latency surcharge", "primary region", "secondary region",
                            "base cost", "latency:", "rate:"
                        ]
                        if any(cl in ctx for cl in component_labels) and not any(tl in ctx for tl in ["total", "grand total", "overall", "monthly cost", "sum", "exact estimated total"]):
                            continue
                    filtered_cands.append((cand_val, cand_span))

                unique_cand_vals = {round(v[0], 2) for v in filtered_cands}
                if len(unique_cand_vals) == 1:
                    extracted_monthly_cost = list(unique_cand_vals)[0]
                    first_span = next(s for v, s in filtered_cands if round(v, 2) == extracted_monthly_cost)
                    evidence_list.append(
                        ExtractedEvidence(
                            field_name="total_monthly_cost_usd",
                            extracted_value=extracted_monthly_cost,
                            unit="USD/month",
                            text_span=first_span,
                            confidence="HIGH",
                        )
                    )
                elif len(unique_cand_vals) > 1:
                    cost_ambiguous = True
                    errors.append("Multiple ambiguous dollar figures in prose without explicit DR total.")

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

        # 3. Extract Decision Allocation based on effective problem type
        if effective_problem_type == "ILP_VM_Allocation":
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
                    for qm in re.finditer(q_pat, raw_text, re.IGNORECASE):
                        match_start = max(0, qm.start() - 60)
                        match_ctx = raw_text[match_start:qm.end()].lower()
                        if "would need" in match_ctx or "exceed" in match_ctx or "instead of" in match_ctx or "alternative" in match_ctx or "option" in match_ctx or "rejected" in match_ctx:
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
                    if sku_found:
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

        elif effective_problem_type == "PSO_Continuous_Scaling":
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

        elif effective_problem_type == "Z3_Graph_Disaster_Recovery":
            # Normalize region name matches including central variants (us-central1, us-central-1, us-central 1)
            # and collect unique regions in order of appearance
            region_pattern = r"\b(us-east-1|us-west-2|eu-west-1|eastus|us-central-?1|us-central\s+1)\b"
            unique_regions = []
            for rm in re.finditer(region_pattern, raw_text, re.IGNORECASE):
                matched_reg = rm.group(1).lower().replace(" ", "").replace("us-central-1", "us-central1")
                if matched_reg == "us-central1" or matched_reg in ["us-east-1", "us-west-2", "eu-west-1", "eastus"]:
                    if matched_reg not in unique_regions:
                        unique_regions.append(matched_reg)

            if len(unique_regions) >= 2:
                primary = unique_regions[0]
                secondary = unique_regions[1]
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
                errors.append(f"Could not extract two distinct regions (found: {unique_regions})")
                status = NormalizationStatus.NORMALIZATION_FAILURE
                normalized_decision = {
                    "primary_region": unique_regions[0] if unique_regions else None,
                    "secondary_region": None,
                }
        else:
            status = NormalizationStatus.TASK_INCOMPATIBLE
            errors.append(f"Unsupported problem type: {effective_problem_type}")
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
            raw_json_or_text = cls._normalize_unicode(raw_json_or_text)
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

        # Normalize any unicode strings within parsed_json
        if isinstance(parsed_json, dict):
            for k, v in list(parsed_json.items()):
                if isinstance(v, str):
                    parsed_json[k] = cls._normalize_unicode(v)

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
                None,
                None,
                errors,
                evidence_list,
            )
        if outcome_claimed in ["unsupported", "task_incompatible"]:
            errors.append(parsed_json.get("reason", parsed_json.get("reasoning", "Model indicated query is unsupported.")))
            return (
                NormalizationStatus.TASK_INCOMPATIBLE,
                None,
                None,
                errors,
                evidence_list,
            )
        if outcome_claimed in ["infeasible", "solver_infeasible"]:
            errors.append(parsed_json.get("reason", parsed_json.get("reasoning", "Model indicated requirements are infeasible.")))
            return (
                NormalizationStatus.SOLVER_INFEASIBLE,
                None,
                None,
                errors,
                evidence_list,
            )

        has_vm_fields = ("instances" in parsed_json or "allocated_vms" in parsed_json) and bool(parsed_json.get("instances") or parsed_json.get("allocated_vms"))
        has_scaling_fields = ("bandwidth_mbps" in parsed_json or "optimal_bandwidth_mbps" in parsed_json or "bandwidth" in parsed_json) and ("recommended_replicas" in parsed_json or "replicas" in parsed_json or "worker_replicas" in parsed_json)
        has_dr_fields = "primary_region" in parsed_json and "secondary_region" in parsed_json and bool(parsed_json.get("primary_region")) and bool(parsed_json.get("secondary_region"))

        # Determine effective problem type with priority on model's explicit task_type or structural fields
        # This prevents default VM bias from falsely rejecting valid DR and Scaling JSON schemas
        if task_type_claimed in ["ILP_VM_Allocation", "PSO_Continuous_Scaling", "Z3_Graph_Disaster_Recovery"]:
            effective_problem_type = task_type_claimed
        elif has_dr_fields and not has_vm_fields and not has_scaling_fields:
            effective_problem_type = "Z3_Graph_Disaster_Recovery"
        elif has_scaling_fields and not has_vm_fields and not has_dr_fields:
            effective_problem_type = "PSO_Continuous_Scaling"
        elif has_vm_fields and not has_dr_fields and not has_scaling_fields:
            effective_problem_type = "ILP_VM_Allocation"
        else:
            effective_problem_type = requested_problem_type or "ILP_VM_Allocation"

        # Normalization according to effective problem type
        if effective_problem_type == "ILP_VM_Allocation":
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

        elif effective_problem_type == "PSO_Continuous_Scaling":
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

        elif effective_problem_type == "Z3_Graph_Disaster_Recovery":
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
            [f"Unsupported problem type: {effective_problem_type}"],
            evidence_list,
        )
