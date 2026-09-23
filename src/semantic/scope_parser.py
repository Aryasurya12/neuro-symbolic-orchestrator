"""SEM-3: Disentangled SCOPE Parser for Natural Language Cloud Queries.

Implements a Hybrid Parsing Architecture combining a sub-5ms local regex/CARM parser
with an OpenRouter NVIDIA Nemotron LLM fallback for high-accuracy intent extraction.
"""

import json
import os
import re
import sys
from typing import List, Literal, Set, Tuple, cast
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
            # 2. Check for USD or general Budget: e.g. "$300", "300 USD", "budget of $350", "budget 300"
            usd_match = re.search(
                r"(?:budget(?:\s+of|\s+max|\s+limit)?\s*[:=]?\s*\$?|\$)\s*(\d+(?:,\d+)*(?:\.\d+)?)",
                text,
                re.IGNORECASE,
            )
            if usd_match:
                try:
                    raw_val = usd_match.group(1).replace(",", "")
                    params["budget_max_usd"] = float(raw_val)
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

        # vCPU extraction: e.g. "2 vCPUs", "16 vCPUs", "8 cores"
        vcpu_match = re.search(
            r"(\d+)\s*(?:vcpus?|cores?|v-cpu)", text, re.IGNORECASE
        )
        if vcpu_match:
            try:
                params["required_vcpus"] = int(vcpu_match.group(1))
            except ValueError:
                pass

        # RAM extraction: e.g. "4GB RAM", "64 GB RAM", "32GB memory"
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

        return params

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

    schema_json = json.dumps(CloudOptimizationContract.model_json_schema(), indent=2)
    system_prompt = (
        "You are an expert FinOps cloud resource optimizer and structured parser. "
        "Extract cloud optimization requirements from the user's natural language query "
        "and return a JSON object that strictly adheres to the following JSON schema:\n\n"
        f"{schema_json}\n\n"
        "Return ONLY the valid raw JSON object matching the schema. Do not output markdown fences or explanatory text."
    )

    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    raw_json = response.choices[0].message.content or "{}"
    return CloudOptimizationContract.model_validate_json(raw_json)


# ===========================================================================
# 4. Hybrid Entrypoint Function
# ===========================================================================

def parse_query_local(user_query: str) -> CloudOptimizationContract:
    """Parses a query using the fast local SCOPE regex & CARM matcher."""
    if not user_query or not user_query.strip():
        raise ValueError("Query is empty or whitespace.")
    parser = SCOPEParser()
    contract, _, _ = parser.parse_query_to_contract(user_query)
    return contract


def parse_query_hybrid(user_query: str) -> CloudOptimizationContract:
    """Hybrid entrypoint function:

    - Step 1: Attempts fast local SCOPE parsing (<5ms, $0 cost).
    - Step 2: On failure or exception, falls back to OpenRouter NVIDIA Nemotron LLM.
    - Step 3: Returns a safe default CloudOptimizationContract if all approaches fail.
    """
    # Step 1: Try executing the existing local parser
    try:
        contract = parse_query_local(user_query)
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
