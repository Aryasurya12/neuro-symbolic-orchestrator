"""SEM-NVIDIA: Dedicated Genuine Neural Requirement Interpreter for Mode 4.

Invokes the NVIDIA API (nvidia/llama-3.1-nemotron-70b-instruct) directly on raw user queries
to interpret requirements and construct formal CloudOptimizationContracts without prior local
SCOPE/CARM archetype matching.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError

from config.settings import settings
from src.semantic.schemas import CloudOptimizationContract
from src.verifiers.canonical_record import ExtractedEvidence


@dataclass
class NVIDIAExtractionResult:
    """Carries the outcome, provenance, and structured payload of an NVIDIA requirement interpretation."""
    status: str  # "SUCCESS", "MISSING_CREDENTIALS", "TIMEOUT", "NETWORK_ERROR", "MALFORMED_JSON", "SCHEMA_ERROR", "NEEDS_CLARIFICATION", "UNSUPPORTED", "CONFLICTING_REQUIREMENTS"
    contract: Optional[CloudOptimizationContract] = None
    outcome: str = "ready"  # "ready", "needs_clarification", "unsupported", "conflicting_requirements"
    clarification_questions: List[str] = field(default_factory=list)
    unsupported_reasons: List[str] = field(default_factory=list)
    conflicting_reasons: List[str] = field(default_factory=list)
    raw_response: str = ""
    parsed_json: Optional[Dict[str, Any]] = None
    elapsed_ms: float = 0.0
    response_id: Optional[str] = None
    model: str = "llama-3.3-70b-versatile"
    provider: str = "Groq"
    extracted_evidence: List[ExtractedEvidence] = field(default_factory=list)
    error_message: Optional[str] = None
    unsupported_reason: Optional[str] = None
    conflicting_reason: Optional[str] = None
    raw_response_text: Optional[str] = None
    latency_ms_val: Optional[float] = None

    def __post_init__(self):
        if self.raw_response_text and not self.raw_response:
            self.raw_response = self.raw_response_text
        elif self.raw_response and not self.raw_response_text:
            self.raw_response_text = self.raw_response
            
        if self.latency_ms_val is not None and self.elapsed_ms == 0.0:
            self.elapsed_ms = self.latency_ms_val

        if self.unsupported_reason and not self.unsupported_reasons:
            self.unsupported_reasons = [self.unsupported_reason]
        elif self.unsupported_reasons and not self.unsupported_reason:
            self.unsupported_reason = self.unsupported_reasons[0]

        if self.conflicting_reason and not self.conflicting_reasons:
            self.conflicting_reasons = [self.conflicting_reason]
        elif self.conflicting_reasons and not self.conflicting_reason:
            self.conflicting_reason = self.conflicting_reasons[0]

    @property
    def latency_ms(self) -> float:
        return self.elapsed_ms

    @property
    def is_executable(self) -> bool:
        """Indicates whether a valid mathematical contract was extracted and is ready for solver dispatch."""
        return self.status == "SUCCESS" and self.contract is not None and self.outcome == "ready"


class NVIDIAExtractor:
    """Translates unstructured natural language cloud infrastructure queries into structured
    CloudOptimizationContracts using genuine NVIDIA LLM inference.
    """

    SYSTEM_PROMPT = (
        "You are a Principal Cloud Workload Requirements Specification Architect for Neurasym.\n"
        "Your task is to interpret a user's natural language cloud infrastructure request and extract structured requirements into a formal JSON contract for mathematical optimization.\n\n"
        "CRITICAL ARCHITECTURAL CONSTRAINTS:\n"
        "1. Do NOT solve the optimization problem. Do NOT select specific VM SKUs (e.g. t3.medium, c5.large), do NOT allocate replicas, and do NOT calculate total prices or costs.\n"
        "2. Extract ONLY the user's workload requirements, bounds, constraints, and operational targets.\n"
        "3. Classify the problem into EXACTLY ONE of the three supported problem archetypes:\n"
        "   - \"ILP_VM_Allocation\": Workloads requiring compute virtual machines with vCPUs, RAM in GB, monthly budget cap in USD, target cloud providers (AWS, Azure, GCP), and service/instance counts.\n"
        "   - \"PSO_Continuous_Scaling\": Autoscaling / dynamic traffic workloads with continuous bandwidth requirements (in Mbps), worker replica bounds, target CPU utilization (e.g., 70%), maximum CPU ceiling, and monthly budget.\n"
        "   - \"Z3_Graph_Disaster_Recovery\": High-availability multi-region disaster recovery setups with target SLA availability (e.g., 99.99%), maximum inter-region network latency (e.g., 50ms), budget, and target cloud providers/regions.\n\n"
        "STRUCTURED INTERPRETATION OUTCOMES:\n"
        "Set \"interpretation_outcome\" to one of:\n"
        "- \"ready\": The query provides sufficient information to formulate a valid mathematical contract.\n"
        "- \"needs_clarification\": The query is ambiguous, incomplete, or lacks essential workload parameters (e.g., 'deploy something fast'). Provide clear, specific questions in \"clarification_questions\".\n"
        "- \"unsupported\": The query requests non-cloud or out-of-scope workloads (e.g., quantum circuit optimization, blockchain mining, physical hardware). Provide reasons in \"unsupported_reasons\".\n"
        "- \"conflicting_requirements\": The query specifies mutually contradictory constraints (e.g., $0 budget, required negative latency). Provide reasons in \"conflicting_reasons\".\n\n"
        "EXTRACTION SCHEMA SPECIFICATION (JSON ONLY):\n"
        "{\n"
        '  "problem_type": "ILP_VM_Allocation" | "PSO_Continuous_Scaling" | "Z3_Graph_Disaster_Recovery",\n'
        '  "interpretation_outcome": "ready" | "needs_clarification" | "unsupported" | "conflicting_requirements",\n'
        '  "cloud_providers": ["AWS" | "Azure" | "GCP"],\n'
        '  "budget_max_usd": float,\n'
        '  "service_count": int (default 1),\n'
        '  "required_vcpus": int (total vCPUs required across allocation),\n'
        '  "required_ram_gb": float (total RAM in GB),\n'
        '  "target_bandwidth_mbps": float or null,\n'
        '  "min_bandwidth_mbps": float or null,\n'
        '  "max_bandwidth_mbps": float or null,\n'
        '  "min_replicas": int or null,\n'
        '  "max_replicas": int or null,\n'
        '  "target_replicas": int or null,\n'
        '  "target_cpu_pct": float or null (e.g. 70.0),\n'
        '  "max_cpu_pct": float or null (e.g. 70.0 or 100.0),\n'
        '  "latency_max_ms": float (default 100.0 for DR),\n'
        '  "sla_availability_pct": float (default 99.9 for DR),\n'
        '  "allowed_regions": list of str or null,\n'
        '  "primary_region": str or null,\n'
        '  "secondary_region": str or null,\n'
        '  "require_multi_region": bool or null,\n'
        '  "require_multi_cloud": bool or null,\n'
        '  "instance_count": int or null,\n'
        '  "min_instance_count": int or null,\n'
        '  "vcpus_per_instance": int or null,\n'
        '  "ram_gb_per_instance": float or null,\n'
        '  "clarification_questions": list of str,\n'
        '  "unsupported_reasons": list of str,\n'
        '  "conflicting_reasons": list of str,\n'
        '  "extracted_spans": {\n'
        '    "required_vcpus": "exact text from query if present",\n'
        '    "required_ram_gb": "exact text from query if present",\n'
        '    "budget_max_usd": "exact text from query if present",\n'
        '    "cloud_providers": "exact text from query if present",\n'
        '    "target_cpu_pct": "exact text from query if present",\n'
        '    "latency_max_ms": "exact text from query if present",\n'
        '    "sla_availability_pct": "exact text from query if present"\n'
        "  }\n"
        "}\n\n"
        "OUTPUT FORMAT: Return ONLY the raw JSON object. No markdown, no explanations, no wrapping."
    )

    @classmethod
    def _clean_and_parse_json(cls, raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Cleans and extracts JSON object from raw LLM output text.
        
        Handles thinking/reasoning tags, code fences, preambles, and conversational text.
        """
        if not raw_text or not raw_text.strip():
            return None, "Empty response from LLM"

        text = raw_text.strip()
        # Remove thinking/reasoning tags if present
        text = re.sub(r"<(?:thinking|thought|think|reasoning)>[\s\S]*?</(?:thinking|thought|think|reasoning)>", "", text, flags=re.IGNORECASE).strip()

        # 1. Look for ```json ... ``` code fence first
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            fence_content = fence_match.group(1).strip()
            try:
                data = json.loads(fence_content)
                if isinstance(data, dict):
                    return data, None
            except Exception:
                pass

        # 2. Try direct json.loads on stripped text
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data, None
        except Exception:
            pass

        # 3. Search for outermost balanced { ... } object in text
        start_idx = text.find("{")
        if start_idx != -1:
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if end_idx != -1:
                json_candidate = text[start_idx:end_idx].strip()
                try:
                    data = json.loads(json_candidate)
                    if isinstance(data, dict):
                        return data, None
                except Exception:
                    pass

        # 4. Fallback to greedy regex extract
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data, None
            except Exception as e:
                return None, f"JSON parse error: {str(e)}"

        return None, "Could not locate valid JSON object in output"

    @classmethod
    def _generate_offline_mock(cls, query: str) -> Dict[str, Any]:
        """Generates realistic, deterministic requirement extractions for offline testing and fixtures."""
        q_lower = query.lower()

        # 1. Non-cloud / Unsupported queries
        if any(w in q_lower for w in ["quantum", "tensor network", "blockchain", "crypto", "asic", "fpga hardware"]):
            return {
                "problem_type": "ILP_VM_Allocation",
                "interpretation_outcome": "unsupported",
                "cloud_providers": ["AWS"],
                "budget_max_usd": 500.0,
                "unsupported_reasons": [
                    "Query requests quantum circuit simulation / hardware optimization outside cloud resource placement archetypes."
                ],
                "clarification_questions": [],
                "conflicting_reasons": [],
                "extracted_spans": {},
            }

        # 2. Highly Ambiguous queries
        if len(query.split()) <= 4 and not any(char.isdigit() for char in query) and not any(w in q_lower for w in ["aws", "azure", "gcp"]):
            return {
                "problem_type": "ILP_VM_Allocation",
                "interpretation_outcome": "needs_clarification",
                "cloud_providers": ["AWS"],
                "budget_max_usd": 500.0,
                "clarification_questions": [
                    "What compute capacity (vCPUs / RAM) does the workload require?",
                    "Which cloud provider (AWS, Azure, or GCP) should host the deployment?",
                    "What is the maximum monthly budget ceiling?",
                ],
                "unsupported_reasons": [],
                "conflicting_reasons": [],
                "extracted_spans": {},
            }

        # 3. Conflicting queries
        if any(w in q_lower for w in ["$0", "zero budget", "negative latency", "-10ms"]):
            return {
                "problem_type": "ILP_VM_Allocation",
                "interpretation_outcome": "conflicting_requirements",
                "cloud_providers": ["AWS"],
                "budget_max_usd": 0.0,
                "conflicting_reasons": [
                    "Budget of $0 violates the minimum viable cloud allocation invariant of $10.00."
                ],
                "clarification_questions": [],
                "unsupported_reasons": [],
                "extracted_spans": {"budget_max_usd": "$0"},
            }

        # 4. Disaster Recovery queries
        if any(w in q_lower for w in ["disaster recovery", "multi-region", "replication", "cross-region", "failover"]) or (re.search(r"\bdr\b", q_lower) and "cpu" not in q_lower):
            providers = []
            if "aws" in q_lower:
                providers.append("AWS")
            if "gcp" in q_lower:
                providers.append("GCP")
            if "azure" in q_lower:
                providers.append("Azure")
            if not providers:
                providers = ["AWS", "GCP"]

            sla = 99.9
            sla_match = re.search(r"(\d{2}(?:\.\d+)?)\s*(?:%|\s*sla)", query, re.IGNORECASE)
            if sla_match and float(sla_match.group(1)) >= 90.0:
                sla = float(sla_match.group(1))

            lat = 100.0
            lat_match = re.search(r"(\d+(?:\.\d+)?)\s*ms", query, re.IGNORECASE)
            if lat_match:
                lat = float(lat_match.group(1))

            bud = 1000.0
            bud_match = re.search(r"\$(\d+(?:\.\d+)?)", query)
            if bud_match:
                bud = float(bud_match.group(1))

            return {
                "problem_type": "Z3_Graph_Disaster_Recovery",
                "interpretation_outcome": "ready",
                "cloud_providers": providers,
                "budget_max_usd": bud,
                "latency_max_ms": lat,
                "sla_availability_pct": sla,
                "require_multi_region": True,
                "require_multi_cloud": len(providers) >= 2,
                "clarification_questions": [],
                "unsupported_reasons": [],
                "conflicting_reasons": [],
                "extracted_spans": {
                    "cloud_providers": " & ".join(providers),
                    "sla_availability_pct": f"{sla}% SLA",
                    "latency_max_ms": f"{lat}ms latency",
                },
            }

        # 5. Continuous Scaling queries
        if any(w in q_lower for w in ["scale", "scaling", "continuous", "pso", "bandwidth", "replicas", "dynamic scaling", "rps", "traffic", "worker nodes"]):
            bud = 1500.0
            bud_match = re.search(r"\$(\d+(?:\.\d+)?)", query)
            if bud_match:
                bud = float(bud_match.group(1))

            cpu_tgt = 70.0
            cpu_match = re.search(r"(\d+(?:\.\d+)?)\s*%", query)
            if cpu_match:
                cpu_tgt = float(cpu_match.group(1))

            bw = 100.0
            bw_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mbps|gbps)", query, re.IGNORECASE)
            if bw_match:
                bw = float(bw_match.group(1))
                if "gbps" in bw_match.group(0).lower():
                    bw *= 1000.0

            reps = 4
            rep_match = re.search(r"(\d+)\s*(?:replicas?|workers?|nodes?|pods?)", query, re.IGNORECASE)
            if rep_match:
                reps = int(rep_match.group(1))

            return {
                "problem_type": "PSO_Continuous_Scaling",
                "interpretation_outcome": "ready",
                "cloud_providers": ["AWS"],
                "budget_max_usd": bud,
                "target_bandwidth_mbps": bw,
                "min_bandwidth_mbps": max(50.0, bw * 0.5),
                "max_bandwidth_mbps": max(1000.0, bw * 2.0),
                "min_replicas": 1,
                "max_replicas": max(16, reps * 2),
                "target_replicas": reps,
                "target_cpu_pct": cpu_tgt,
                "max_cpu_pct": cpu_tgt if "exceed" in q_lower or "max" in q_lower else 100.0,
                "clarification_questions": [],
                "unsupported_reasons": [],
                "conflicting_reasons": [],
                "extracted_spans": {
                    "target_cpu_pct": f"{cpu_tgt}%",
                    "budget_max_usd": f"${bud:.0f}",
                },
            }

        # 6. Default: VM Knapsack Allocation
        providers = []
        if "aws" in q_lower:
            providers.append("AWS")
        if "gcp" in q_lower:
            providers.append("GCP")
        if "azure" in q_lower:
            providers.append("Azure")
        if not providers:
            providers = ["AWS"]

        vcpus = 4
        vcpu_match = re.search(r"(\d+)\s*(?:vcpu|vcpus|cores?|v-cpus?)", query, re.IGNORECASE)
        if vcpu_match:
            vcpus = int(vcpu_match.group(1))
        elif re.search(r"\b(\d+)\s*(?:cpu|cpus)\b", query, re.IGNORECASE):
            vcpus = int(re.search(r"\b(\d+)\s*(?:cpu|cpus)\b", query, re.IGNORECASE).group(1))

        ram = 16.0
        ram_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?|gigs?|gib)", query, re.IGNORECASE)
        if ram_match:
            ram = float(ram_match.group(1))

        bud = 300.0
        bud_match = re.search(r"\$(\d+(?:\.\d+)?)", query)
        if bud_match:
            bud = float(bud_match.group(1))

        return {
            "problem_type": "ILP_VM_Allocation",
            "interpretation_outcome": "ready",
            "cloud_providers": providers,
            "budget_max_usd": bud,
            "required_vcpus": vcpus,
            "required_ram_gb": ram,
            "service_count": 1,
            "clarification_questions": [],
            "unsupported_reasons": [],
            "conflicting_reasons": [],
            "extracted_spans": {
                "required_vcpus": f"{vcpus} vCPUs",
                "required_ram_gb": f"{ram:.0f}GB RAM",
                "budget_max_usd": f"${bud:.0f}",
                "cloud_providers": ", ".join(providers),
            },
        }

    @classmethod
    def extract_contract_from_query(
        cls,
        query: str = "",
        query_text: str = "",
        timeout_seconds: Optional[float] = None,
        mock_response: Optional[Dict[str, Any]] = None,
        offline: bool = False,
        *args,
        **kwargs,
    ) -> NVIDIAExtractionResult:
        """Invokes neural requirement extraction directly on query to construct a Pydantic contract.
        
        Preserves the mandatory Mode 4 path: raw query -> neural extraction -> contract -> solver -> verifier.
        Zero local SCOPE/CARM pre-filtering is performed.
        """
        active_query = query or query_text or kwargs.get("query_text", "") or kwargs.get("query", "")
        load_dotenv()
        
        provider_name, api_key, base_url, target_model, default_timeout = settings.get_mode4_provider_config()
        effective_timeout = timeout_seconds if timeout_seconds is not None else default_timeout

        t0 = time.perf_counter()

        # Handle Mock / Offline execution
        if offline or mock_response is not None:
            raw_data = mock_response if mock_response is not None else cls._generate_offline_mock(active_query)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            raw_content = json.dumps(raw_data, indent=2)

            return cls._validate_and_build_result(
                raw_dict=raw_data,
                raw_response=raw_content,
                elapsed_ms=elapsed_ms,
                model=target_model,
                provider=provider_name,
                response_id=f"mock_{provider_name.lower()}_{int(time.time())}",
            )

        # Verify API credentials - classify missing credentials as infrastructure failure, NOT unsupported/infeasible
        if not api_key:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return NVIDIAExtractionResult(
                status="MISSING_CREDENTIALS",
                outcome="infrastructure_failure",
                error_message=f"{provider_name.upper()}_API_KEY is not set in environment or settings. Please provide credentials in .env.",
                model=target_model,
                provider=provider_name,
                elapsed_ms=elapsed_ms,
            )

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=effective_timeout,
                max_retries=0,
            )

            messages = [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract structured optimization contract from query:\n\"{active_query}\""},
            ]

            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if not response or not response.choices:
                return NVIDIAExtractionResult(
                    status="EMPTY_RESPONSE",
                    outcome="infrastructure_failure",
                    error_message=f"Empty response received from {provider_name} API",
                    model=target_model,
                    provider=provider_name,
                    elapsed_ms=elapsed_ms,
                )

            choice = response.choices[0]
            raw_content = (choice.message.content or "").strip()
            resp_id = getattr(response, "id", None)

            parsed_dict, parse_err = cls._clean_and_parse_json(raw_content)
            if parse_err or not parsed_dict:
                return NVIDIAExtractionResult(
                    status="MALFORMED_JSON",
                    outcome="infrastructure_failure",
                    raw_response=raw_content,
                    error_message=f"Failed to parse {provider_name} response as JSON: {parse_err}",
                    model=target_model,
                    provider=provider_name,
                    elapsed_ms=elapsed_ms,
                    response_id=resp_id,
                )

            return cls._validate_and_build_result(
                raw_dict=parsed_dict,
                raw_response=raw_content,
                elapsed_ms=elapsed_ms,
                model=target_model,
                provider=provider_name,
                response_id=resp_id,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err_str = str(e)
            status_tag = "TIMEOUT" if "timeout" in err_str.lower() else "NETWORK_ERROR"
            return NVIDIAExtractionResult(
                status=status_tag,
                outcome="infrastructure_failure",
                error_message=f"{provider_name} API Error ({status_tag}): {err_str}",
                model=target_model,
                provider=provider_name,
                elapsed_ms=elapsed_ms,
            )

    @classmethod
    def _validate_and_build_result(
        cls,
        raw_dict: Dict[str, Any],
        raw_response: str,
        elapsed_ms: float,
        model: str,
        provider: str = "Groq",
        response_id: Optional[str] = None,
    ) -> NVIDIAExtractionResult:
        """Validates the extracted dictionary against Pydantic schema and returns a structured result."""
        outcome = str(raw_dict.get("interpretation_outcome", "ready")).lower()
        clarifications = raw_dict.get("clarification_questions", [])
        unsupported = raw_dict.get("unsupported_reasons", [])
        conflicting = raw_dict.get("conflicting_reasons", [])
        spans = raw_dict.get("extracted_spans", {})

        evidence_list: List[ExtractedEvidence] = []
        if isinstance(spans, dict):
            for field_name, span_text in spans.items():
                val = raw_dict.get(field_name)
                evidence_list.append(
                    ExtractedEvidence(
                        field_name=field_name,
                        extracted_value=val,
                        unit="N/A",
                        text_span=str(span_text),
                        confidence="HIGH",
                    )
                )

        if outcome in ["needs_clarification", "unsupported", "conflicting_requirements"]:
            status_map = {
                "needs_clarification": "NEEDS_CLARIFICATION",
                "unsupported": "UNSUPPORTED",
                "conflicting_requirements": "CONFLICTING_REQUIREMENTS",
            }
            return NVIDIAExtractionResult(
                status=status_map[outcome],
                outcome=outcome,
                clarification_questions=clarifications,
                unsupported_reasons=unsupported,
                conflicting_reasons=conflicting,
                raw_response=raw_response,
                parsed_json=raw_dict,
                elapsed_ms=elapsed_ms,
                model=model,
                provider=provider,
                response_id=response_id,
                extracted_evidence=evidence_list,
            )

        # Validate with CloudOptimizationContract
        try:
            contract = CloudOptimizationContract(**raw_dict)
            return NVIDIAExtractionResult(
                status="SUCCESS",
                contract=contract,
                outcome="ready",
                clarification_questions=[],
                unsupported_reasons=[],
                conflicting_reasons=[],
                raw_response=raw_response,
                parsed_json=raw_dict,
                elapsed_ms=elapsed_ms,
                model=model,
                provider=provider,
                response_id=response_id,
                extracted_evidence=evidence_list,
            )
        except ValidationError as val_err:
            return NVIDIAExtractionResult(
                status="SCHEMA_ERROR",
                outcome="infrastructure_failure",
                raw_response=raw_response,
                parsed_json=raw_dict,
                error_message=f"Schema validation failed: {str(val_err)}",
                model=model,
                provider=provider,
                elapsed_ms=elapsed_ms,
                response_id=response_id,
                extracted_evidence=evidence_list,
            )
        except (ValidationError, ValueError) as e:
            return NVIDIAExtractionResult(
                status="SCHEMA_ERROR",
                outcome="unsupported",
                error_message=f"Pydantic Contract Validation Failed: {str(e)}",
                raw_response=raw_response,
                parsed_json=raw_dict,
                elapsed_ms=elapsed_ms,
                model=model,
                response_id=response_id,
                extracted_evidence=evidence_list,
            )
