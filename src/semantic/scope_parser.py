"""SEM-3: Disentangled SCOPE Parser for Natural Language Cloud Queries.

Implements a Hybrid Parsing Architecture combining a sub-5ms local regex/CARM parser
with an OpenRouter NVIDIA Nemotron LLM fallback for high-accuracy intent extraction.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, cast
from dotenv import load_dotenv
from openai import OpenAI

# Ensure UTF-8 stdout encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config.settings import settings
from src.semantic.carm_matcher import CARMMatcher
from src.semantic.schemas import CloudOptimizationContract

# 1. Load OPENROUTER_API_KEY from .env using python-dotenv
load_dotenv()



class SCOPEParser:
    """Disentangles user natural language intent into structured constraints

    and produces validated Pydantic contracts via CARM template matching,
    supporting dual USD ($) and INR (₹) budget inputs.
    """

    @staticmethod
    def has_cloud_intent(text: str) -> bool:
        """Smart Cloud Intent Check: Identifies if the user query contains domain-relevant

        cloud computing, infrastructure sizing, or FinOps keywords/patterns.
        """
        if not text or not text.strip():
            return False

        lower = text.lower()
        cloud_keywords = [
            "aws", "azure", "gcp", "google cloud", "amazon", "cloud",
            "vm", "vms", "instance", "instances", "vcpu", "vcpus", "core", "cores", "v-cpu",
            "ram", "memory", "gb", "gigabytes", "compute", "storage",
            "budget", "cost", "price", "spend", "spending", "savings", "discount",
            "usd", "$", "dollar", "dollars", "₹", "inr", "rs", "rs.", "rupee", "rupees",
            "latency", "ms", "response time", "ping", "delay",
            "sla", "availability", "uptime", "fault tolerant", "redundant",
            "autoscale", "autoscaling", "scaling", "bandwidth", "mbps", "gbps", "throughput",
            "stream", "traffic", "pso",
            "microservice", "microservices", "service", "services", "container", "containers",
            "disaster recovery", "failover", "multi-region", "multi region", "multiregion",
            "disjoint", "cross-region", "knapsack", "z3", "milp", "ilp", "placement", "topology",
            "allocate", "allocation", "provision", "deploy", "workload", "cluster", "node", "nodes"
        ]

        return any(
            re.search(r"\b" + re.escape(kw) + r"\b", lower) if kw not in ["$", "₹"] else kw in lower
            for kw in cloud_keywords
        )

    def __init__(self, matcher: CARMMatcher | None = None) -> None:
        self.matcher = matcher or CARMMatcher()

    def extract_constraints_from_text(self, text: str) -> Set[str]:


        """Analyzes natural language text and extracts symbolic constraint tokens.

        Returns an empty set if no relevant domain signals are present.
        """
        if not text or not text.strip():
            return set()

        lower = text.lower()
        constraints: Set[str] = set()

        # Check for disaster recovery / topological context
        is_dr_context = any(
            k in lower
            for k in [
                "disaster recovery",
                "disaster-recovery",
                "multi-region",
                "multi region",
                "multiregion",
                "disjoint",
                "failover",
                "dr",
                "geo-redundant",
                "cross-region",
                "distribute",
                "placement",
                "topology",
                "graph",
                "inter-node",
                "inter-region",
            ]
        )

        # Check for ILP / Discrete Knapsack Allocation constraints
        if any(
            k in lower
            for k in [
                "microservice",
                "microservices",
                "vm",
                "vms",
                "instance",
                "instances",
                "knapsack",
                "discrete",
                "integer",
                "allocate",
                "allocation",
                "container",
                "containers",
            ]
        ):
            constraints.add("Bounded_Integer_Variables")

        # Check for budget signals (USD, INR, and general budget keywords)
        if any(
            k in lower
            for k in [
                "budget",
                "max budget",
                "cost",
                "usd",
                "$",
                "dollar",
                "dollars",
                "spend",
                "spending",
                "price limit",
                "₹",
                "rs",
                "rs.",
                "inr",
                "rupee",
                "rupees",
            ]
        ):
            constraints.add("Budget_Limit_Max")

        if any(
            k in lower
            for k in [
                "vcpu",
                "vcpus",
                "core",
                "cores",
                "v-cpu",
                "ram",
                "memory",
                "gb",
                "compute",
                "high compute",
                "heavy workload",
                "high memory",
                "memory intensive",
                "large ram",
                "heavy database",
                "database tier",
                "enterprise cluster",
                "large cluster",
                "intensive batch",
            ]
        ):
            constraints.add("Resource_Min_vCPU")

        # Check for latency signals and disambiguate context
        has_latency_signal = any(
            k in lower
            for k in [
                "latency",
                "ms",
                "response time",
                "roundtrip",
                "delay",
                "ping",
                "inter-node",
                "inter-region",
                "backbone",
                "edge latency",
                "sync latency",
            ]
        )

        if has_latency_signal:
            if is_dr_context:
                constraints.add("Inter_Node_Latency_Max")
            else:
                constraints.add("Latency_Bound_Max")

        # Check for Continuous PSO Scaling constraints
        if any(
            k in lower
            for k in [
                "bandwidth",
                "mbps",
                "gbps",
                "throughput",
                "continuous",
                "stream",
                "traffic",
                "dynamic range",
            ]
        ):
            constraints.add("Continuous_Bandwidth_Range")

        if any(
            k in lower
            for k in [
                "threshold",
                "utilization",
                "target cpu",
                "cpu percent",
                "autoscale",
                "autoscaling",
                "scaling",
                "pso",
            ]
        ):
            constraints.add("CPU_Threshold_Max")

        if any(
            k in lower
            for k in [
                "cost minimization",
                "minimize cost",
                "cost objective",
                "efficient cost",
            ]
        ):
            constraints.add("Cost_Minimization_Objective")

        # Check for Z3 Graph Disaster Recovery constraints
        if is_dr_context:
            constraints.add("Multi_Region_Disjoint")

        if any(
            k in lower
            for k in [
                "sla",
                "availability",
                "uptime",
                "99.",
                "fault tolerant",
                "high availability",
            ]
        ):
            constraints.add("SLA_Availability_Min")

        return constraints

    def _extract_parameters(self, text: str) -> dict:
        """Extracts numerical and categorical parameters from natural language query,

        converting INR budgets automatically to USD using settings.USD_TO_INR_RATE.
        """
        params: dict = {}

        # 1. Check for explicit INR Budget: e.g. "₹25000", "25000 INR", "Rs 25000", "25000 rupees"
        inr_patterns = [
            r"(?:₹|rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:inr|rupees?|rs\b)",
            r"budget(?:\s+of|\s+max|\s+limit)?\s*[:=]?\s*(?:₹|rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"budget(?:\s+of|\s+max|\s+limit)?\s*[:=]?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:inr|rupees?|rs\b)",
        ]
        inr_budget = None
        for pat in inr_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    raw_val = m.group(1).replace(",", "")
                    inr_budget = float(raw_val)
                    break
                except ValueError:
                    pass

        if inr_budget is not None:
            # Convert INR to USD for contract consistency
            converted_usd = round(inr_budget / settings.USD_TO_INR_RATE, 2)
            params["budget_max_usd"] = converted_usd
        else:
            # 2. Check for USD or general Budget: e.g. "$300", "300 USD", "budget of $350", "budget 300", "under $300"
            usd_patterns = [
                r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
                r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:usd|dollars?)",
                r"(?:budget|cost|price|spend|limit|under|cap)\s*(?:of|max|limit|under|is|to)?\s*[:=]?\s*\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            ]
            for pat in usd_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        raw_val = m.group(1).replace(",", "")
                        val = float(raw_val)
                        if val > 0:
                            params["budget_max_usd"] = val
                            break
                    except ValueError:
                        pass


        # Service / Microservice count: e.g. "5 microservices", "3 services"
        service_match = re.search(
            r"(\d+)\s*(?:microservices?|services?|components?|nodes?)",
            text,
            re.IGNORECASE,
        )
        if service_match:
            try:
                params["service_count"] = int(service_match.group(1))
            except ValueError:
                pass

        # Explicit vCPU extraction: e.g. "2 vCPUs", "16 vCPUs", "8 cores"
        vcpu_match = re.search(
            r"(\d+)\s*(?:vcpus?|cores?|v-cpu)", text, re.IGNORECASE
        )
        if vcpu_match:
            try:
                params["required_vcpus"] = int(vcpu_match.group(1))
            except ValueError:
                pass

        # Explicit RAM extraction: e.g. "4GB RAM", "64 GB RAM", "32GB memory"
        ram_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)\s*(?:ram|memory)?",
            text,
            re.IGNORECASE,
        )
        if ram_match:
            try:
                params["required_ram_gb"] = float(ram_match.group(1))
            except ValueError:
                pass

        # Qualitative Intent Keyword Detection (when explicit numeric values may be missing)
        lower = text.lower()
        qualitative_intent = None
        qual_vcpus = None
        qual_ram = None

        # 1. SAGE-GNN Cloud Application Benchmark Topologies
        if "secure_web_container" in lower:
            qualitative_intent = "SAGE-GNN Secure Web Container Topology"
            qual_vcpus = 10
            qual_ram = 28.0
        elif "wordpress_multitier" in lower or "wordpress" in lower:
            qualitative_intent = "SAGE-GNN WordPress MultiTier Topology"
            qual_vcpus = 18
            qual_ram = 60.0
        elif "oryx2_lambda_pipeline" in lower or "oryx2" in lower or "lambda_pipeline" in lower:
            qualitative_intent = "SAGE-GNN Oryx2 Lambda Pipeline Topology"
            qual_vcpus = 18
            qual_ram = 68.0
        elif any(k in lower for k in ["sage-gnn", "sage_gnn", "anti-affinity", "anti affinity"]):
            qualitative_intent = "SAGE-GNN Neural Graph Topology"
            qual_vcpus = 8
            qual_ram = 32.0
        # 2. Heavy Database / Enterprise Cluster
        elif any(
            p in lower
            for p in [
                "heavy database",
                "database tier",
                "enterprise cluster",
                "large cluster",
            ]
        ):
            qualitative_intent = "Heavy Database / Enterprise Cluster"
            qual_vcpus = 8
            qual_ram = 32.0
        # 3. High Memory
        elif any(
            p in lower
            for p in [
                "high memory",
                "memory intensive",
                "memory-intensive",
                "large ram",
                "high-memory",
            ]
        ):
            qualitative_intent = "High Memory"
            qual_vcpus = 4
            qual_ram = 32.0
        # 4. High Compute / Scaling
        elif any(
            p in lower
            for p in [
                "high compute",
                "heavy workload",
                "high compute scaling",
                "intensive batch",
                "compute-intensive",
                "compute intensive",
            ]
        ):
            qualitative_intent = "High Compute Scaling"
            qual_vcpus = 4
            qual_ram = 16.0

        metadata = {}
        if qualitative_intent:
            metadata["qualitative_intent"] = qualitative_intent

        # Precedence: Explicit numeric constraints strictly take precedence over qualitative defaults.
        # If no explicit numbers AND no qualitative intensity keywords are matched, maintain fallback (1 vCPU, 1.0 GB RAM).
        if "required_vcpus" not in params:
            if qual_vcpus is not None:
                params["required_vcpus"] = qual_vcpus
            else:
                params["required_vcpus"] = 1

        if "required_ram_gb" not in params:
            if qual_ram is not None:
                params["required_ram_gb"] = qual_ram
            else:
                params["required_ram_gb"] = 1.0

        params["metadata"] = metadata

        # Latency extraction: e.g. "50ms", "20ms latency", "max 80 ms"
        latency_match = re.search(
            r"(?:max\s+|under\s+|latency\s+(?:of\s+|under\s+)?|\b)(\d+(?:\.\d+)?)\s*ms",
            text,
            re.IGNORECASE,
        )
        if latency_match:
            try:
                params["latency_max_ms"] = float(latency_match.group(1))
            except ValueError:
                pass

        # SLA availability extraction: e.g. "99.99%", "99.9% SLA"
        sla_match = re.search(
            r"(9\d(?:\.\d+)?)\s*%", text, re.IGNORECASE
        )
        if sla_match:
            try:
                params["sla_availability_pct"] = float(sla_match.group(1))
            except ValueError:
                pass

        # Cloud Provider extraction preserving appearance order
        detected_providers = []
        for match in re.finditer(r"\b(AWS|Azure|GCP)\b", text, re.IGNORECASE):
            prov = match.group(1).upper()
            if prov == "AZURE":
                prov = "Azure"
            if prov not in detected_providers:
                detected_providers.append(prov)
        if detected_providers:
            params["cloud_providers"] = detected_providers

        # Build Field Provenance Mapping: Explicit vs Inferred vs Default
        provenance = {}

        # 1. Budget Provenance
        if inr_budget is not None or "budget_max_usd" in params:
            provenance["budget_max_usd"] = "[EXPLICIT]"
        else:
            provenance["budget_max_usd"] = "[DEFAULT: Baseline Fallback]"

        # 2. Service count Provenance
        if "service_count" in params:
            provenance["service_count"] = "[EXPLICIT]"
        else:
            provenance["service_count"] = "[DEFAULT: Baseline Fallback]"

        # 3. vCPU Provenance
        if vcpu_match:
            provenance["required_vcpus"] = "[EXPLICIT]"
        elif qual_vcpus is not None:
            provenance["required_vcpus"] = f"[INFERRED: {qualitative_intent}]"
        else:
            provenance["required_vcpus"] = "[DEFAULT: Baseline Fallback]"

        # 4. RAM Provenance
        if ram_match:
            provenance["required_ram_gb"] = "[EXPLICIT]"
        elif qual_ram is not None:
            provenance["required_ram_gb"] = f"[INFERRED: {qualitative_intent}]"
        else:
            provenance["required_ram_gb"] = "[DEFAULT: Baseline Fallback]"

        # 5. Latency Provenance
        if latency_match:
            provenance["latency_max_ms"] = "[EXPLICIT]"
        else:
            provenance["latency_max_ms"] = "[DEFAULT: Baseline Fallback]"

        # 6. SLA Provenance
        if sla_match:
            provenance["sla_availability_pct"] = "[EXPLICIT]"
        else:
            provenance["sla_availability_pct"] = "[DEFAULT: Baseline Fallback]"

        # 7. Cloud Providers Provenance
        if detected_providers:
            provenance["cloud_providers"] = "[EXPLICIT]"
        else:
            provenance["cloud_providers"] = "[DEFAULT: Baseline Fallback]"

        metadata["field_provenance"] = provenance
        params["metadata"] = metadata

        return params

    def extract_parameters(self, text: str) -> dict:
        """Public interface for parameter extraction from user query text."""
        return self._extract_parameters(text)

    def extract_constraints_trace(self, text: str) -> Tuple[Set[str], List[dict]]:
        """Extracts constraints while capturing detailed pattern-matching trace per token."""
        if not text or not text.strip():
            return set(), []

        lower = text.lower()
        extracted: Set[str] = set()
        trace_list: List[dict] = []

        dr_keywords = [
            "disaster recovery", "disaster-recovery", "multi-region", "multi region",
            "multiregion", "disjoint", "failover", "dr", "geo-redundant", "cross-region",
            "distribute", "placement", "topology", "graph", "inter-node", "inter-region"
        ]
        matched_dr = [k for k in dr_keywords if k in lower]
        is_dr_context = bool(matched_dr)

        # 1. Bounded_Integer_Variables
        ilp_kws = ["microservice", "microservices", "vm", "vms", "instance", "instances", "knapsack", "discrete", "integer", "allocate", "allocation", "container", "containers"]
        m_ilp = [k for k in ilp_kws if k in lower]
        if m_ilp:
            extracted.add("Bounded_Integer_Variables")
        trace_list.append({
            "constraint_token": "Bounded_Integer_Variables",
            "keywords_checked": ilp_kws,
            "matched": bool(m_ilp),
            "matched_keywords": m_ilp,
            "status": "MATCHED" if m_ilp else "NO MATCH",
        })

        # 2. Budget_Limit_Max
        bud_kws = ["budget", "max budget", "cost", "usd", "$", "dollar", "dollars", "spend", "spending", "price limit", "₹", "rs", "rs.", "inr", "rupee", "rupees"]
        m_bud = [k for k in bud_kws if (k in ["$", "₹"] and k in lower) or (k not in ["$", "₹"] and k in lower)]
        if m_bud:
            extracted.add("Budget_Limit_Max")
        trace_list.append({
            "constraint_token": "Budget_Limit_Max",
            "keywords_checked": bud_kws,
            "matched": bool(m_bud),
            "matched_keywords": m_bud,
            "status": "MATCHED" if m_bud else "NO MATCH",
        })

        # 3. Resource_Min_vCPU
        cpu_kws = ["vcpu", "vcpus", "core", "cores", "v-cpu", "ram", "memory", "gb", "compute", "high compute", "heavy workload", "high memory", "memory intensive", "large ram", "heavy database", "database tier", "enterprise cluster", "large cluster", "intensive batch"]
        m_cpu = [k for k in cpu_kws if k in lower]
        if m_cpu:
            extracted.add("Resource_Min_vCPU")
        trace_list.append({
            "constraint_token": "Resource_Min_vCPU",
            "keywords_checked": cpu_kws,
            "matched": bool(m_cpu),
            "matched_keywords": m_cpu,
            "status": "MATCHED" if m_cpu else "NO MATCH",
        })

        # 4. Latency
        lat_kws = ["latency", "ms", "response time", "roundtrip", "delay", "ping", "inter-node", "inter-region", "backbone", "edge latency", "sync latency"]
        m_lat = [k for k in lat_kws if k in lower]
        if m_lat:
            token = "Inter_Node_Latency_Max" if is_dr_context else "Latency_Bound_Max"
            extracted.add(token)
            trace_list.append({
                "constraint_token": token,
                "keywords_checked": lat_kws,
                "matched": True,
                "matched_keywords": m_lat,
                "status": f"MATCHED (Context: {'DR/Inter-Node' if is_dr_context else 'Standard Bound'})",
            })
        else:
            trace_list.append({
                "constraint_token": "Latency_Bound_Max / Inter_Node_Latency_Max",
                "keywords_checked": lat_kws,
                "matched": False,
                "matched_keywords": [],
                "status": "NO MATCH",
            })

        # 5. Continuous_Bandwidth_Range
        bw_kws = ["bandwidth", "mbps", "gbps", "throughput", "continuous", "stream", "traffic", "dynamic range"]
        m_bw = [k for k in bw_kws if k in lower]
        if m_bw:
            extracted.add("Continuous_Bandwidth_Range")
        trace_list.append({
            "constraint_token": "Continuous_Bandwidth_Range",
            "keywords_checked": bw_kws,
            "matched": bool(m_bw),
            "matched_keywords": m_bw,
            "status": "MATCHED" if m_bw else "NO MATCH",
        })

        # 6. CPU_Threshold_Max
        thresh_kws = ["threshold", "utilization", "target cpu", "cpu percent", "autoscale", "autoscaling", "scaling", "pso"]
        m_thresh = [k for k in thresh_kws if k in lower]
        if m_thresh:
            extracted.add("CPU_Threshold_Max")
        trace_list.append({
            "constraint_token": "CPU_Threshold_Max",
            "keywords_checked": thresh_kws,
            "matched": bool(m_thresh),
            "matched_keywords": m_thresh,
            "status": "MATCHED" if m_thresh else "NO MATCH",
        })

        # 7. Cost_Minimization_Objective
        cost_kws = ["cost minimization", "minimize cost", "cost objective", "efficient cost"]
        m_cost = [k for k in cost_kws if k in lower]
        if m_cost:
            extracted.add("Cost_Minimization_Objective")
        trace_list.append({
            "constraint_token": "Cost_Minimization_Objective",
            "keywords_checked": cost_kws,
            "matched": bool(m_cost),
            "matched_keywords": m_cost,
            "status": "MATCHED" if m_cost else "NO MATCH",
        })

        # 8. Multi_Region_Disjoint
        if is_dr_context:
            extracted.add("Multi_Region_Disjoint")
        trace_list.append({
            "constraint_token": "Multi_Region_Disjoint",
            "keywords_checked": dr_keywords,
            "matched": is_dr_context,
            "matched_keywords": matched_dr,
            "status": "MATCHED" if is_dr_context else "NO MATCH",
        })

        # 9. SLA_Availability_Min
        sla_kws = ["sla", "availability", "uptime", "99.", "fault tolerant", "high availability"]
        m_sla = [k for k in sla_kws if k in lower]
        if m_sla:
            extracted.add("SLA_Availability_Min")
        trace_list.append({
            "constraint_token": "SLA_Availability_Min",
            "keywords_checked": sla_kws,
            "matched": bool(m_sla),
            "matched_keywords": m_sla,
            "status": "MATCHED" if m_sla else "NO MATCH",
        })

        return extracted, trace_list

    def extract_parameters_trace(self, text: str) -> Tuple[dict, List[dict]]:
        """Extracts parameters while recording the exact regex pattern match status,
        matched substrings, and raw vs converted values for each field.
        """
        params: dict = {}
        trace_list: List[dict] = []

        # 1. INR Budget
        inr_patterns = [
            r"(?:₹|rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:inr|rupees?|rs\b)",
            r"budget(?:\s+of|\s+max|\s+limit)?\s*[:=]?\s*(?:₹|rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"budget(?:\s+of|\s+max|\s+limit)?\s*[:=]?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:inr|rupees?|rs\b)",
        ]
        inr_budget = None
        inr_match_obj = None
        inr_matched_pat = None
        for pat in inr_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    raw_val = m.group(1).replace(",", "")
                    inr_budget = float(raw_val)
                    inr_match_obj = m
                    inr_matched_pat = pat
                    break
                except ValueError:
                    pass

        # 2. USD Budget
        usd_patterns = [
            r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:usd|dollars?)",
            r"(?:budget|cost|price|spend|limit|under|cap)\s*(?:of|max|limit|under|is|to)?\s*[:=]?\s*\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        usd_budget = None
        usd_match_obj = None
        usd_matched_pat = None
        if inr_budget is None:
            for pat in usd_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        raw_val = m.group(1).replace(",", "")
                        val = float(raw_val)
                        if val > 0:
                            usd_budget = val
                            usd_match_obj = m
                            usd_matched_pat = pat
                            break
                    except ValueError:
                        pass

        # Record Budget Trace
        if inr_budget is not None:
            converted_usd = round(inr_budget / settings.USD_TO_INR_RATE, 2)
            params["budget_max_usd"] = converted_usd
            trace_list.append({
                "field": "budget_max_usd",
                "patterns_evaluated": inr_patterns + usd_patterns,
                "matched": True,
                "matched_pattern": inr_matched_pat,
                "matched_substring": inr_match_obj.group(0) if inr_match_obj else None,
                "raw_extracted": inr_match_obj.group(1) if inr_match_obj else None,
                "converted_value": converted_usd,
                "field_type": "float (USD converted from INR)",
                "status": "MATCHED (INR)",
                "provenance": "[EXPLICIT: INR converted to USD]",
            })
        elif usd_budget is not None:
            params["budget_max_usd"] = usd_budget
            trace_list.append({
                "field": "budget_max_usd",
                "patterns_evaluated": inr_patterns + usd_patterns,
                "matched": True,
                "matched_pattern": usd_matched_pat,
                "matched_substring": usd_match_obj.group(0) if usd_match_obj else None,
                "raw_extracted": usd_match_obj.group(1) if usd_match_obj else None,
                "converted_value": usd_budget,
                "field_type": "float (USD)",
                "status": "MATCHED (USD)",
                "provenance": "[EXPLICIT]",
            })
        else:
            params["budget_max_usd"] = settings.DEFAULT_BUDGET_USD
            trace_list.append({
                "field": "budget_max_usd",
                "patterns_evaluated": inr_patterns + usd_patterns,
                "matched": False,
                "matched_pattern": None,
                "matched_substring": None,
                "raw_extracted": None,
                "converted_value": settings.DEFAULT_BUDGET_USD,
                "field_type": "float (USD)",
                "status": "NO MATCH (Default Applied)",
                "provenance": "[DEFAULT: Baseline Fallback]",
            })

        # 3. Service / Microservice count
        svc_pat = r"(\d+)\s*(?:microservices?|services?|components?|nodes?)"
        svc_match = re.search(svc_pat, text, re.IGNORECASE)
        if svc_match:
            try:
                s_count = int(svc_match.group(1))
                params["service_count"] = s_count
                trace_list.append({
                    "field": "service_count",
                    "patterns_evaluated": [svc_pat],
                    "matched": True,
                    "matched_pattern": svc_pat,
                    "matched_substring": svc_match.group(0),
                    "raw_extracted": svc_match.group(1),
                    "converted_value": s_count,
                    "field_type": "int",
                    "status": "MATCHED",
                    "provenance": "[EXPLICIT]",
                })
            except ValueError:
                params["service_count"] = 1
                trace_list.append({
                    "field": "service_count",
                    "patterns_evaluated": [svc_pat],
                    "matched": False,
                    "matched_pattern": None,
                    "matched_substring": None,
                    "raw_extracted": None,
                    "converted_value": 1,
                    "field_type": "int",
                    "status": "NO MATCH (Parse Error, Default Applied)",
                    "provenance": "[DEFAULT: Baseline Fallback]",
                })
        else:
            params["service_count"] = 1
            trace_list.append({
                "field": "service_count",
                "patterns_evaluated": [svc_pat],
                "matched": False,
                "matched_pattern": None,
                "matched_substring": None,
                "raw_extracted": None,
                "converted_value": 1,
                "field_type": "int",
                "status": "NO MATCH (Default Applied)",
                "provenance": "[DEFAULT: Baseline Fallback]",
            })

        # 4. Explicit vCPU extraction
        vcpu_pat = r"(\d+)\s*(?:vcpus?|cores?|v-cpu)"
        vcpu_match = re.search(vcpu_pat, text, re.IGNORECASE)

        # 5. Explicit RAM extraction
        ram_pat = r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)\s*(?:ram|memory)?"
        ram_match = re.search(ram_pat, text, re.IGNORECASE)

        # 6. Qualitative Intent
        lower = text.lower()
        qualitative_intent = None
        qual_vcpus = None
        qual_ram = None

        if "secure_web_container" in lower:
            qualitative_intent = "SAGE-GNN Secure Web Container Topology"
            qual_vcpus, qual_ram = 10, 28.0
        elif "wordpress_multitier" in lower or "wordpress" in lower:
            qualitative_intent = "SAGE-GNN WordPress MultiTier Topology"
            qual_vcpus, qual_ram = 18, 60.0
        elif "oryx2_lambda_pipeline" in lower or "oryx2" in lower or "lambda_pipeline" in lower:
            qualitative_intent = "SAGE-GNN Oryx2 Lambda Pipeline Topology"
            qual_vcpus, qual_ram = 18, 68.0
        elif any(k in lower for k in ["sage-gnn", "sage_gnn", "anti-affinity", "anti affinity"]):
            qualitative_intent = "SAGE-GNN Neural Graph Topology"
            qual_vcpus, qual_ram = 8, 32.0
        elif any(p in lower for p in ["heavy database", "database tier", "enterprise cluster", "large cluster"]):
            qualitative_intent = "Heavy Database / Enterprise Cluster"
            qual_vcpus, qual_ram = 8, 32.0
        elif any(p in lower for p in ["high memory", "memory intensive", "memory-intensive", "large ram", "high-memory"]):
            qualitative_intent = "High Memory"
            qual_vcpus, qual_ram = 4, 32.0
        elif any(p in lower for p in ["high compute", "heavy workload", "high compute scaling", "intensive batch", "compute-intensive", "compute intensive"]):
            qualitative_intent = "High Compute Scaling"
            qual_vcpus, qual_ram = 4, 16.0

        trace_list.append({
            "field": "qualitative_intent",
            "patterns_evaluated": ["SAGE-GNN topologies", "heavy database", "high memory", "high compute"],
            "matched": bool(qualitative_intent),
            "matched_pattern": qualitative_intent,
            "matched_substring": qualitative_intent,
            "raw_extracted": qualitative_intent,
            "converted_value": qualitative_intent,
            "field_type": "str",
            "status": f"DETECTED: {qualitative_intent}" if qualitative_intent else "NO MATCH",
            "provenance": f"[INFERRED: {qualitative_intent}]" if qualitative_intent else "[NONE]",
        })

        # Process vCPU
        if vcpu_match:
            try:
                v_val = int(vcpu_match.group(1))
                params["required_vcpus"] = v_val
                trace_list.append({
                    "field": "required_vcpus",
                    "patterns_evaluated": [vcpu_pat],
                    "matched": True,
                    "matched_pattern": vcpu_pat,
                    "matched_substring": vcpu_match.group(0),
                    "raw_extracted": vcpu_match.group(1),
                    "converted_value": v_val,
                    "field_type": "int",
                    "status": "MATCHED",
                    "provenance": "[EXPLICIT]",
                })
            except ValueError:
                params["required_vcpus"] = qual_vcpus if qual_vcpus is not None else 1
                trace_list.append({
                    "field": "required_vcpus",
                    "patterns_evaluated": [vcpu_pat],
                    "matched": False,
                    "matched_pattern": None,
                    "matched_substring": None,
                    "raw_extracted": None,
                    "converted_value": params["required_vcpus"],
                    "field_type": "int",
                    "status": "NO MATCH (Parse Error, Fallback Applied)",
                    "provenance": f"[INFERRED: {qualitative_intent}]" if qual_vcpus else "[DEFAULT: Baseline Fallback]",
                })
        elif qual_vcpus is not None:
            params["required_vcpus"] = qual_vcpus
            trace_list.append({
                "field": "required_vcpus",
                "patterns_evaluated": [vcpu_pat],
                "matched": False,
                "matched_pattern": None,
                "matched_substring": None,
                "raw_extracted": None,
                "converted_value": qual_vcpus,
                "field_type": "int",
                "status": "NO MATCH (Qualitative Intent Applied)",
                "provenance": f"[INFERRED: {qualitative_intent}]",
            })
        else:
            params["required_vcpus"] = 1
            trace_list.append({
                "field": "required_vcpus",
                "patterns_evaluated": [vcpu_pat],
                "matched": False,
                "matched_pattern": None,
                "matched_substring": None,
                "raw_extracted": None,
                "converted_value": 1,
                "field_type": "int",
                "status": "NO MATCH (Default Applied)",
                "provenance": "[DEFAULT: Baseline Fallback]",
            })

        # Process RAM
        if ram_match:
            try:
                r_val = float(ram_match.group(1))
                params["required_ram_gb"] = r_val
                trace_list.append({
                    "field": "required_ram_gb",
                    "patterns_evaluated": [ram_pat],
                    "matched": True,
                    "matched_pattern": ram_pat,
                    "matched_substring": ram_match.group(0),
                    "raw_extracted": ram_match.group(1),
                    "converted_value": r_val,
                    "field_type": "float",
                    "status": "MATCHED",
                    "provenance": "[EXPLICIT]",
                })
            except ValueError:
                params["required_ram_gb"] = qual_ram if qual_ram is not None else 1.0
                trace_list.append({
                    "field": "required_ram_gb",
                    "patterns_evaluated": [ram_pat],
                    "matched": False,
                    "matched_pattern": None,
                    "matched_substring": None,
                    "raw_extracted": None,
                    "converted_value": params["required_ram_gb"],
                    "field_type": "float",
                    "status": "NO MATCH (Parse Error, Fallback Applied)",
                    "provenance": f"[INFERRED: {qualitative_intent}]" if qual_ram else "[DEFAULT: Baseline Fallback]",
                })
        elif qual_ram is not None:
            params["required_ram_gb"] = qual_ram
            trace_list.append({
                "field": "required_ram_gb",
                "patterns_evaluated": [ram_pat],
                "matched": False,
                "matched_pattern": None,
                "matched_substring": None,
                "raw_extracted": None,
                "converted_value": qual_ram,
                "field_type": "float",
                "status": "NO MATCH (Qualitative Intent Applied)",
                "provenance": f"[INFERRED: {qualitative_intent}]",
            })
        else:
            params["required_ram_gb"] = 1.0
            trace_list.append({
                "field": "required_ram_gb",
                "patterns_evaluated": [ram_pat],
                "matched": False,
                "matched_pattern": None,
                "matched_substring": None,
                "raw_extracted": None,
                "converted_value": 1.0,
                "field_type": "float",
                "status": "NO MATCH (Default Applied)",
                "provenance": "[DEFAULT: Baseline Fallback]",
            })

        # 7. Latency extraction
        lat_pat = r"(?:max\s+|under\s+|latency\s+(?:of\s+|under\s+)?|\b)(\d+(?:\.\d+)?)\s*ms"
        lat_match = re.search(lat_pat, text, re.IGNORECASE)
        if lat_match:
            try:
                lat_val = float(lat_match.group(1))
                params["latency_max_ms"] = lat_val
                trace_list.append({
                    "field": "latency_max_ms",
                    "patterns_evaluated": [lat_pat],
                    "matched": True,
                    "matched_pattern": lat_pat,
                    "matched_substring": lat_match.group(0),
                    "raw_extracted": lat_match.group(1),
                    "converted_value": lat_val,
                    "field_type": "float",
                    "status": "MATCHED",
                    "provenance": "[EXPLICIT]",
                })
            except ValueError:
                params["latency_max_ms"] = 100.0
                trace_list.append({
                    "field": "latency_max_ms",
                    "patterns_evaluated": [lat_pat],
                    "matched": False,
                    "matched_pattern": None,
                    "matched_substring": None,
                    "raw_extracted": None,
                    "converted_value": 100.0,
                    "field_type": "float",
                    "status": "NO MATCH (Parse Error, Default Applied)",
                    "provenance": "[DEFAULT: Baseline Fallback]",
                })
        else:
            params["latency_max_ms"] = 100.0
            trace_list.append({
                "field": "latency_max_ms",
                "patterns_evaluated": [lat_pat],
                "matched": False,
                "matched_pattern": None,
                "matched_substring": None,
                "raw_extracted": None,
                "converted_value": 100.0,
                "field_type": "float",
                "status": "NO MATCH (Default Applied)",
                "provenance": "[DEFAULT: Baseline Fallback]",
            })

        # 8. SLA availability extraction
        sla_pat = r"(9\d(?:\.\d+)?)\s*%"
        sla_match = re.search(sla_pat, text, re.IGNORECASE)
        if sla_match:
            try:
                sla_val = float(sla_match.group(1))
                params["sla_availability_pct"] = sla_val
                trace_list.append({
                    "field": "sla_availability_pct",
                    "patterns_evaluated": [sla_pat],
                    "matched": True,
                    "matched_pattern": sla_pat,
                    "matched_substring": sla_match.group(0),
                    "raw_extracted": sla_match.group(1),
                    "converted_value": sla_val,
                    "field_type": "float",
                    "status": "MATCHED",
                    "provenance": "[EXPLICIT]",
                })
            except ValueError:
                params["sla_availability_pct"] = 99.9
                trace_list.append({
                    "field": "sla_availability_pct",
                    "patterns_evaluated": [sla_pat],
                    "matched": False,
                    "matched_pattern": None,
                    "matched_substring": None,
                    "raw_extracted": None,
                    "converted_value": 99.9,
                    "field_type": "float",
                    "status": "NO MATCH (Parse Error, Default Applied)",
                    "provenance": "[DEFAULT: Baseline Fallback]",
                })
        else:
            params["sla_availability_pct"] = 99.9
            trace_list.append({
                "field": "sla_availability_pct",
                "patterns_evaluated": [sla_pat],
                "matched": False,
                "matched_pattern": None,
                "matched_substring": None,
                "raw_extracted": None,
                "converted_value": 99.9,
                "field_type": "float",
                "status": "NO MATCH (Default Applied)",
                "provenance": "[DEFAULT: Baseline Fallback]",
            })

        # 9. Cloud Provider extraction
        prov_pat = r"\b(AWS|Azure|GCP)\b"
        detected_providers = []
        raw_prov_matches = []
        for match in re.finditer(prov_pat, text, re.IGNORECASE):
            prov = match.group(1).upper()
            if prov == "AZURE":
                prov = "Azure"
            if prov not in detected_providers:
                detected_providers.append(prov)
                raw_prov_matches.append(match.group(0))

        if detected_providers:
            params["cloud_providers"] = detected_providers
            trace_list.append({
                "field": "cloud_providers",
                "patterns_evaluated": [prov_pat],
                "matched": True,
                "matched_pattern": prov_pat,
                "matched_substring": ", ".join(raw_prov_matches),
                "raw_extracted": detected_providers,
                "converted_value": detected_providers,
                "field_type": "List[str]",
                "status": "MATCHED",
                "provenance": "[EXPLICIT]",
            })
        else:
            params["cloud_providers"] = ["AWS"]
            trace_list.append({
                "field": "cloud_providers",
                "patterns_evaluated": [prov_pat],
                "matched": False,
                "matched_pattern": None,
                "matched_substring": None,
                "raw_extracted": None,
                "converted_value": ["AWS"],
                "field_type": "List[str]",
                "status": "NO MATCH (Default Applied)",
                "provenance": "[DEFAULT: Baseline Fallback]",
            })

        metadata: dict = {}
        if qualitative_intent:
            metadata["qualitative_intent"] = qualitative_intent

        provenance = {t["field"]: t["provenance"] for t in trace_list if "field" in t}
        metadata["field_provenance"] = provenance
        params["metadata"] = metadata

        return params, trace_list

    def trace_parse(self, user_query: str) -> Dict[str, Any]:
        """Deep stage-1 and stage-2 trace through real parsing execution."""
        extracted_constraints, constraint_traces = self.extract_constraints_trace(user_query)
        params, param_traces = self.extract_parameters_trace(user_query)

        carm_details = self.matcher.match_template_detailed(extracted_constraints) if extracted_constraints else {
            "scores": self.matcher.score_all_archetypes(set()),
            "winner": "ILP_VM_Allocation",
            "winner_template": "ilp_vm_allocation_template.py",
            "winner_score": 0.0,
            "runner_up": None,
            "runner_up_score": 0.0,
            "margin": 0.0,
            "is_near_tie": False,
        }

        template_name = carm_details["winner"]
        template_filename = carm_details["winner_template"]
        score = carm_details["winner_score"]

        typed_problem = cast(
            Literal[
                "ILP_VM_Allocation",
                "PSO_Continuous_Scaling",
                "Z3_Graph_Disaster_Recovery",
            ],
            template_name,
        )
        providers_list = params.get("cloud_providers", ["AWS"])
        typed_providers = cast(
            List[Literal["AWS", "Azure", "GCP"]], providers_list
        )

        validation_errors: List[dict] = []
        try:
            contract = CloudOptimizationContract(
                problem_type=typed_problem,
                cloud_providers=typed_providers,
                budget_max_usd=params.get("budget_max_usd", settings.DEFAULT_BUDGET_USD),
                service_count=params.get("service_count", 1),
                required_vcpus=params.get("required_vcpus", 1),
                required_ram_gb=params.get("required_ram_gb", 1.0),
                latency_max_ms=params.get("latency_max_ms", 100.0),
                sla_availability_pct=params.get("sla_availability_pct", 99.9),
                metadata=params.get("metadata", {}),
            )
        except Exception as exc:
            validation_errors.append({
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            # Fallback contract
            contract = CloudOptimizationContract(
                problem_type=typed_problem,
                cloud_providers=typed_providers,
                budget_max_usd=max(settings.MIN_VIABLE_BUDGET_USD, params.get("budget_max_usd", settings.DEFAULT_BUDGET_USD)),
                service_count=max(1, params.get("service_count", 1)),
                required_vcpus=max(1, params.get("required_vcpus", 1)),
                required_ram_gb=max(1.0, params.get("required_ram_gb", 1.0)),
                latency_max_ms=min(1000.0, max(1.0, params.get("latency_max_ms", 100.0))),
                sla_availability_pct=min(99.999, max(90.0, params.get("sla_availability_pct", 99.9))),
                metadata=params.get("metadata", {}),
            )

        return {
            "contract": contract,
            "template_filename": template_filename,
            "score": score,
            "carm_details": carm_details,
            "parameter_traces": param_traces,
            "constraint_traces": constraint_traces,
            "extracted_constraints": sorted(extracted_constraints),
            "validation_errors": validation_errors,
        }

    def parse_query_to_contract(
        self, user_query: str
    ) -> Tuple[CloudOptimizationContract, str, float]:
        """Parses a user query into a validated CloudOptimizationContract and matched template.

        Returns:
            Tuple of (CloudOptimizationContract, template_filename, jaccard_score)
        """
        extracted_constraints = self.extract_constraints_from_text(user_query)

        if extracted_constraints:
            template_name, template_filename, score = self.matcher.match_template(
                extracted_constraints
            )
        else:
            template_name = "ILP_VM_Allocation"
            template_filename = "ilp_vm_allocation_template.py"
            score = 0.0

        params = self._extract_parameters(user_query)

        typed_problem = cast(
            Literal[
                "ILP_VM_Allocation",
                "PSO_Continuous_Scaling",
                "Z3_Graph_Disaster_Recovery",
            ],
            template_name,
        )
        providers_list = params.get("cloud_providers", ["AWS"])
        typed_providers = cast(
            List[Literal["AWS", "Azure", "GCP"]], providers_list
        )

        contract = CloudOptimizationContract(
            problem_type=typed_problem,
            cloud_providers=typed_providers,
            budget_max_usd=params.get(
                "budget_max_usd", settings.DEFAULT_BUDGET_USD
            ),
            service_count=params.get("service_count", 1),
            required_vcpus=params.get("required_vcpus", 1),
            required_ram_gb=params.get("required_ram_gb", 1.0),
            latency_max_ms=params.get("latency_max_ms", 100.0),
            sla_availability_pct=params.get("sla_availability_pct", 99.9),
            metadata=params.get("metadata", {}),
        )

        return contract, template_filename, score

    def parse_query_local(self, user_query: str) -> CloudOptimizationContract:
        """Parses query using local regex and heuristic SCOPE parser."""
        contract, _, _ = self.parse_query_to_contract(user_query)
        return contract

    def parse_fallback_nemotron(self, user_query: str) -> CloudOptimizationContract:
        """Fallback LLM parser delegating to OpenRouter NVIDIA Nemotron."""
        return parse_fallback_nemotron(user_query)

    def parse_query_hybrid(self, user_query: str) -> CloudOptimizationContract:
        """Hybrid parsing entrypoint method on SCOPEParser."""
        return parse_query_hybrid(user_query)



# ===========================================================================
# 2 & 3. OpenRouter NVIDIA Nemotron Fallback Implementation
# ===========================================================================

def parse_fallback_nemotron(user_query: str) -> CloudOptimizationContract:
    """Fallback function using OpenRouter NVIDIA Nemotron LLM to extract a validated
    CloudOptimizationContract from complex, ambiguous natural language queries.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or getattr(settings, "OPENROUTER_API_KEY", "")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    primary_model = os.getenv("OPENROUTER_MODEL") or getattr(
        settings, "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"
    )
    candidate_models = [
        primary_model,
        "nvidia/nemotron-3.5-lightning:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ]
    # Deduplicate while preserving order
    models_to_try = list(dict.fromkeys(candidate_models))

    schema_json = json.dumps(CloudOptimizationContract.model_json_schema(), indent=2)
    system_prompt = (
        "You are an expert FinOps cloud resource optimizer and structured parser. "
        "Extract cloud optimization requirements from the user's natural language query "
        "and return a JSON object that strictly adheres to the following JSON schema:\n\n"
        f"{schema_json}\n\n"
        "Carefully extract any explicit numbers and constraints mentioned in the query "
        "(such as budget, node/service count, vCPUs, RAM, latency, SLA percentage). "
        "If a budget is explicitly mentioned (e.g. $300, ₹25000), set budget_max_usd to that value (converting INR to USD at 85 if needed). "
        "Return ONLY the valid raw JSON object matching the schema. Do not output markdown fences or explanatory text."
    )

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.0,
            )

            raw_content = (response.choices[0].message.content or "{}").strip()
            # Clean markdown fences if present
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
                raw_content = re.sub(r"\s*```$", "", raw_content)
            raw_content = raw_content.strip()

            print("\n" + "=" * 60)
            print("📄 RAW NEMOTRON LLM OUTPUT:")
            print("=" * 60)
            print(raw_content)
            print("=" * 60)

            contract = CloudOptimizationContract.model_validate_json(raw_content)

            print("\n" + "=" * 60)
            print("🔒 PYDANTIC VALIDATED JSON CONTRACT:")
            print("=" * 60)
            print(contract.model_dump_json(indent=2))
            print("=" * 60 + "\n")

            return contract
        except Exception as err:
            last_error = err
            continue

    raise RuntimeError(f"All candidate OpenRouter models failed. Last error: {last_error}")


def has_cloud_intent(user_query: str) -> bool:
    """Convenience module function to check if a query has cloud/FinOps intent."""
    return SCOPEParser.has_cloud_intent(user_query)


def validate_parsed_numbers(user_query: str, contract: CloudOptimizationContract) -> None:
    """Checks if numeric quantities mentioned in user_query (budget, nodes, vCPUs, RAM)
    were accurately captured. If the local parser missed or replaced with default values,
    raises ValueError("Local parser missed custom constraints") to trigger parse_fallback_nemotron.
    """
    # 1. Check budget mentions ($300, 300 USD, 300 dollars, ₹25000, 25000 INR, budget 300, etc.)
    budget_patterns = [
        r"(?:\$|₹|rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)",
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:usd|dollars?|inr|rupees?)",
        r"(?:budget|spend|cost)\s*(?:of|max|limit|under|is|to)?\s*[:=]?\s*\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)",
    ]
    for pat in budget_patterns:
        m = re.search(pat, user_query, re.IGNORECASE)
        if m:
            raw_val = m.group(1) or (m.group(2) if len(m.groups()) >= 2 else None)
            if raw_val:
                num_val = float(raw_val.replace(",", ""))
                inr_converted = round(num_val / getattr(settings, "USD_TO_INR_RATE", 85.0), 2)
                # If contract budget does not match either direct USD or INR-converted value
                if abs(contract.budget_max_usd - num_val) > 0.01 and abs(contract.budget_max_usd - inr_converted) > 1.0:
                    raise ValueError(
                        f"Local parser missed custom constraints: explicit budget '{num_val}' in query (contract has {contract.budget_max_usd})"
                    )
            break

    # 2. Check nodes / services / instances count (e.g. 4 nodes, 5 microservices, 3 instances)
    node_match = re.search(
        r"(\d+)\s*(?:nodes?|instances?|microservices?|services?|components?)",
        user_query,
        re.IGNORECASE,
    )
    if node_match:
        expected_nodes = int(node_match.group(1))
        if contract.service_count != expected_nodes:
            raise ValueError(
                f"Local parser missed custom constraints: explicit node/service count '{expected_nodes}' in query (contract has {contract.service_count})"
            )

    # 3. Check vCPU mentions (4 vCPUs, 8 cores, etc.)
    vcpu_match = re.search(r"(\d+)\s*(?:vcpus?|cores?|v-cpu)", user_query, re.IGNORECASE)
    if vcpu_match:
        expected_vcpus = int(vcpu_match.group(1))
        if contract.required_vcpus != expected_vcpus:
            raise ValueError(
                f"Local parser missed custom constraints: explicit vCPU count '{expected_vcpus}' in query (contract has {contract.required_vcpus})"
            )

    # 4. Check RAM mentions (16GB RAM, 32 GB memory, etc.)
    ram_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)\s*(?:ram|memory)?", user_query, re.IGNORECASE)
    if ram_match:
        expected_ram = float(ram_match.group(1))
        if abs(contract.required_ram_gb - expected_ram) > 0.01:
            raise ValueError(
                f"Local parser missed custom constraints: explicit RAM '{expected_ram}' in query (contract has {contract.required_ram_gb})"
            )


def parse_query_local(user_query: str) -> CloudOptimizationContract:
    """Parses a query using the fast local SCOPE regex & CARM matcher.

    Raises ValueError if query lacks confident cloud intent or is empty.
    """
    if not user_query or not user_query.strip():
        raise ValueError("Query is empty or whitespace.")

    if not has_cloud_intent(user_query):
        raise ValueError(f"Smart Intent Check failed: No cloud domain signals found in '{user_query}'.")

    parser = SCOPEParser()
    contract, _, _ = parser.parse_query_to_contract(user_query)
    return contract


def parse_query_hybrid(user_query: str) -> CloudOptimizationContract:
    """Hybrid entrypoint function:

    - Step 1: Attempts fast local SCOPE parsing (<5ms, $0 cost).
    - Step 2: Checks if the query contains explicit numeric constraints (e.g. '$300', '300 dollars', '4 nodes')
      that the local parser missed or replaced with default values (like budget_max_usd == 500).
      If custom constraints do not match, raises ValueError("Local parser missed custom constraints").
    - Step 3: On failure/mismatch, falls back to OpenRouter NVIDIA Nemotron LLM.
    - Step 4: Returns a safe default CloudOptimizationContract if all approaches fail.
    """
    # Step 1: Try executing the existing local parser
    try:
        contract = parse_query_local(user_query)
        # Check if query contains explicit numeric constraints missed or defaulted by local parser
        validate_parsed_numbers(user_query, contract)
        print("⚡ [Part A] Parsed via Local SCOPE Parser (<5ms, $0 Cost)")
        return contract
    except Exception as local_err:
        # Step 2: Fallback to OpenRouter NVIDIA Nemotron
        try:
            contract = parse_fallback_nemotron(user_query)
            print("🧠 [Part A] Parsed via OpenRouter NVIDIA Nemotron Fallback")
            return contract
        except Exception as api_err:
            # Step 3: Fail-safe default contract
            print(f"⚠️ [Part A] Parsing failed, falling back to default contract: {api_err}")
            return CloudOptimizationContract(
                problem_type="ILP_VM_Allocation",
                cloud_providers=["AWS"],
                budget_max_usd=getattr(settings, "DEFAULT_BUDGET_USD", 500.0),
                service_count=1,
                required_vcpus=1,
                required_ram_gb=1.0,
                latency_max_ms=100.0,
                sla_availability_pct=99.9,
            )

