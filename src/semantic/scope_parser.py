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

            return CloudOptimizationContract.model_validate_json(raw_content)
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
                inr_converted = round(num_val / getattr(settings, "USD_TO_INR_RATE", 83.0), 2)
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

