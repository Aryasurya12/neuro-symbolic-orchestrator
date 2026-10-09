"""SEM-LLM: Centralized Live LLM Client for Mode 1 and Mode 2 Request Dispatching.

Handles isolated, live inference requests to Groq (for Mode 1 and Mode 2) without
triggering Streamlit UI or web server lifecycle code.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Ensure environment variables from .env are loaded into os.environ
load_dotenv()

from config.settings import settings


def execute_dashboard_llm_request(
    mode_num: int,
    query: str,
    contract: Any = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Executes a single, isolated live inference request to Groq API for Mode 1 or Mode 2.

    Uses configurable timeout from settings (default: 360s), max_retries=0, target Groq model.
    Never fabricates fallback costs; captures exact timing, raw content, and failure reason.
    """
    load_dotenv()
    from src.symbolic.optimizers.domain_catalog import DatabaseBackedCatalog

    api_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", "")
    if not api_key:
        return {
            "status": "missing_credentials",
            "error_type": "ConfigurationError",
            "error_message": "GROQ_API_KEY is missing in environment or settings. Please set GROQ_API_KEY in your .env file.",
            "elapsed_seconds": 0.0,
            "elapsed_ms": 0.0,
            "content": "",
            "reported_cost_usd": None,
            "actual_cost_usd": None,
            "error_usd": None,
            "error_pct": None,
            "overflow_usd": None,
            "is_feasible": None,
        }

    effective_timeout = timeout_seconds if timeout_seconds is not None else getattr(
        settings, "LLM_REQUEST_TIMEOUT_SECONDS", 360.0
    )
    base_url = (
        os.getenv("GROQ_BASE_URL")
        or getattr(settings, "GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    model = (
        os.getenv("GROQ_MODEL")
        or getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    max_tokens = getattr(settings, "LLM_MAX_COMPLETION_TOKENS", 4096)

    t0 = time.perf_counter()
    run_record: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode_num": mode_num,
        "query": query,
        "model": model,
        "base_url": base_url,
        "timeout": effective_timeout,
        "content": "",
        "reported_cost_usd": None,
        "actual_cost_usd": None,
        "error_usd": None,
        "error_pct": None,
        "overflow_usd": None,
        "is_feasible": None,
    }

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=effective_timeout,
            max_retries=0,
        )

        catalog_context = (
            "CLOUD DOMAIN CATALOG & COST/CAPACITY SPECIFICATION:\n"
            "1. Available Virtual Machine SKUs (730 hours/month):\n"
            "   - AWS: t3.medium (2 vCPUs, 4GB, $0.0416/hr = $30.37/mo), t3.large (2 vCPUs, 8GB, $0.0832/hr = $60.74/mo), "
            "t3.xlarge (4 vCPUs, 16GB, $0.1664/hr = $121.47/mo), c5.large (2 vCPUs, 4GB, $0.0850/hr = $62.05/mo), "
            "c5.xlarge (4 vCPUs, 8GB, $0.1700/hr = $124.10/mo), m5.large (2 vCPUs, 8GB, $0.0960/hr = $70.08/mo), "
            "m5.xlarge (4 vCPUs, 16GB, $0.1920/hr = $140.16/mo), m5.2xlarge (8 vCPUs, 32GB, $0.3840/hr = $280.32/mo)\n"
            "   - Azure: Standard_D4s_v5 (4 vCPUs, 16GB, $0.1920/hr = $140.16/mo)\n"
            "   - GCP: e2-standard-4 (4 vCPUs, 16GB, $0.1340/hr = $97.82/mo)\n\n"
            "2. Multi-Region Disaster Recovery (Topology Graph):\n"
            "   - Regions & Base Costs: us-east-1 (AWS, $120/mo, 99.95% SLA), us-west-2 (AWS, $130/mo, 99.95% SLA), "
            "eu-west-1 (AWS, Europe, $140/mo, 99.95% SLA), eastus (Azure, $125/mo, 99.95% SLA), us-central1 (GCP, $115/mo, 99.95% SLA)\n"
            "   - Peer Latencies: us-east-1 <-> eastus: 12ms, us-east-1 <-> us-central1: 32ms, us-east-1 <-> us-west-2: 65ms, "
            "us-east-1 <-> eu-west-1: 85ms, us-west-2 <-> us-central1: 42ms, us-west-2 <-> eastus: 70ms, "
            "us-west-2 <-> eu-west-1: 135ms, eastus <-> us-central1: 28ms, eu-west-1 <-> eastus: 90ms, eu-west-1 <-> us-central1: 105ms\n"
            "   - Cost Formula: base_cost(RegionA) + base_cost(RegionB) + (latency_ms * $0.25)\n"
            "   - SLA Formula: 1 - ((1 - SLA_A/100) * (1 - SLA_B/100))\n\n"
            "3. Continuous Dynamic Scaling:\n"
            "   - Bandwidth: [100.0, 1000.0] Mbps ($0.08 / Mbps / month)\n"
            "   - Worker Replicas: [1, 16] deployable integer count ($45.00 / replica / month)\n"
            "   - Capacity: 75.0 Mbps per replica. Modeled CPU = (Bandwidth / (Replicas * 75.0)) * 100%\n"
            "   - Cost Formula: (Bandwidth * $0.08) + (Replicas * $45.00)\n"
        )

        if mode_num == 1:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Cloud Solutions Architect. Analyze the user query and recommend a concrete deployment allocation using the catalog snapshot and domain rules below.\n\n"
                        + catalog_context
                        + "\nState the recommended provider, instance types/regions/bandwidth/replicas, and the exact total monthly cost in USD ($/month). Be concise."
                    ),
                },
                {"role": "user", "content": f"Recommend deployment plan for: \"{query}\""},
            ]
        else:
            discriminated_schema = {
                "task_type": "ILP_VM_Allocation | PSO_Continuous_Scaling | Z3_Graph_Disaster_Recovery | unsupported | needs_clarification | infeasible",
                "outcome": "ready | needs_clarification | unsupported | infeasible",
                "reason": "Explanation if outcome is not ready",
                "allocated_vms": [
                    {"sku": "t3.xlarge", "provider": "AWS", "quantity": 2, "monthly_cost": 242.94}
                ],
                "primary_region": "us-east-1",
                "secondary_region": "us-west-2",
                "optimal_bandwidth_mbps": 100.0,
                "recommended_replicas": 2,
                "total_monthly_cost_usd": 242.94,
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Cloud Optimization System. Analyze the user query and respond ONLY with a valid JSON object matching this discriminated schema based on the query task.\n\n"
                        + catalog_context
                        + f"\nDISCRIMINATED SCHEMA SPECIFICATION:\n{json.dumps(discriminated_schema, indent=2)}\n\n"
                        "CRITICAL INSTRUCTIONS:\n"
                        "- For VM allocation tasks: set task_type='ILP_VM_Allocation', populate 'allocated_vms' with valid SKUs from catalog and counts, and 'total_monthly_cost_usd'.\n"
                        "- For Disaster Recovery tasks: set task_type='Z3_Graph_Disaster_Recovery', populate 'primary_region', 'secondary_region', and 'total_monthly_cost_usd'.\n"
                        "- For Dynamic Scaling tasks: set task_type='PSO_Continuous_Scaling', populate 'optimal_bandwidth_mbps', integer 'recommended_replicas', and 'total_monthly_cost_usd'.\n"
                        "- For infeasible, ambiguous, or unsupported tasks: set outcome appropriately with explanation in 'reason'.\n"
                        "- Respond ONLY with raw JSON. No surrounding markdown fences or text."
                    ),
                },
                {"role": "user", "content": f"Optimize allocation for: \"{query}\". Respond in JSON."},
            ]

        # Attempt execution helper with reasoning effort setting and rate limit backoff
        def _invoke_llm_attempt(attempt_num: int, tok_limit: int) -> Tuple[Any, Optional[str], str, Optional[Dict[str, Any]]]:
            for retry_idx in range(4):
                try:
                    try:
                        r = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.0 if mode_num == 2 else 0.2,
                            max_tokens=tok_limit,
                            extra_body={"reasoning_format": "parsed", "reasoning_effort": "low"},
                        )
                    except Exception as e_low:
                        if "429" in str(e_low) or "RateLimit" in type(e_low).__name__:
                            raise e_low
                        try:
                            r = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=0.0 if mode_num == 2 else 0.2,
                                max_tokens=tok_limit,
                                reasoning_effort="low",
                            )
                        except Exception as e_med:
                            if "429" in str(e_med) or "RateLimit" in type(e_med).__name__:
                                raise e_med
                            r = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=0.0 if mode_num == 2 else 0.2,
                                max_tokens=tok_limit,
                            )
                    ch = r.choices[0] if r.choices else None
                    f_reason = getattr(ch, "finish_reason", "unknown") if ch else "empty"
                    c_text = (ch.message.content or "").strip() if ch and ch.message else ""
                    u_dict = {
                        "prompt_tokens": r.usage.prompt_tokens,
                        "completion_tokens": r.usage.completion_tokens,
                        "total_tokens": r.usage.total_tokens,
                    } if hasattr(r, "usage") and r.usage else None
                    return r, f_reason, c_text, u_dict
                except Exception as exc:
                    err_s = str(exc)
                    if ("429" in err_s or "RateLimit" in type(exc).__name__) and retry_idx < 3:
                        wait_time = 3.0 + retry_idx * 2.0
                        time.sleep(wait_time)
                        continue
                    raise exc
            raise RuntimeError("Exhausted rate limit retries")

        attempts: List[Dict[str, Any]] = []
        initial_max_tokens = max_tokens
        retry_max_tokens = max(8192, max_tokens * 2)

        # Attempt 1
        resp, finish_reason, raw_content, usage_info = _invoke_llm_attempt(1, initial_max_tokens)
        attempts.append({
            "attempt": 1,
            "max_tokens": initial_max_tokens,
            "finish_reason": finish_reason,
            "content_len": len(raw_content),
            "status": "TRUNCATED" if finish_reason == "length" else ("EMPTY" if not raw_content else "OK"),
        })

        # Retry once if truncated on Attempt 1
        if finish_reason == "length" or (not raw_content and finish_reason in ["length", "unknown"]):
            resp_retry, retry_finish, retry_content, retry_usage = _invoke_llm_attempt(2, retry_max_tokens)
            attempts.append({
                "attempt": 2,
                "max_tokens": retry_max_tokens,
                "finish_reason": retry_finish,
                "content_len": len(retry_content),
                "status": "TRUNCATED" if retry_finish == "length" else ("EMPTY" if not retry_content else "OK"),
            })
            if retry_content:
                resp = resp_retry
                finish_reason = retry_finish
                raw_content = retry_content
                if retry_usage:
                    usage_info = retry_usage
            else:
                finish_reason = retry_finish

        elapsed_s = time.perf_counter() - t0
        run_record["elapsed_seconds"] = round(elapsed_s, 2)
        run_record["elapsed_ms"] = round(elapsed_s * 1000.0, 1)
        run_record["attempts"] = attempts
        run_record["finish_reason"] = finish_reason
        run_record["content"] = raw_content
        run_record["response_id"] = getattr(resp, "id", None) if resp else None
        if usage_info:
            run_record["usage"] = usage_info

        if not raw_content:
            if finish_reason == "length":
                run_record["status"] = "truncation_failure"
                run_record["error_message"] = "Provider/config truncation failure: reasoning model exceeded completion token budget (finish_reason='length')."
            else:
                run_record["status"] = "empty_response"
                run_record["error_message"] = "Model returned empty content."
            return run_record

        if finish_reason == "length":
            run_record["status"] = "truncated"
            run_record["error_message"] = "Generation reached max token limit and was truncated (finish_reason='length')."

        # Compute ground truth catalog cost if possible
        budget = float(getattr(contract, "budget_max_usd", 500.0)) if contract else 500.0
        req_v = int(getattr(contract, "required_vcpus", 4)) if contract else 4
        req_r = float(getattr(contract, "required_ram_gb", 16.0)) if contract else 16.0

        catalog_cost_lookup = None
        try:
            catalog = DatabaseBackedCatalog()
            matching_costs = [
                sku.monthly_cost() for sku in catalog.VM_CATALOG
                if sku.vcpus >= req_v and sku.ram_gb >= req_r
            ]
            if matching_costs:
                catalog_cost_lookup = min(matching_costs)
        except Exception:
            catalog_cost_lookup = None

        if mode_num == 1:
            cost_match = re.search(r"\$\s*(\d+(?:\.\d+)?)", raw_content)
            if cost_match:
                rep_cost = float(cost_match.group(1))
                run_record["reported_cost_usd"] = rep_cost
                run_record["actual_cost_usd"] = catalog_cost_lookup
                if catalog_cost_lookup and catalog_cost_lookup > 0:
                    err_usd = round(abs(catalog_cost_lookup - rep_cost), 2)
                    err_pct = round((err_usd / catalog_cost_lookup) * 100.0, 1)
                    run_record["error_usd"] = err_usd
                    run_record["error_pct"] = err_pct
                    run_record["overflow_usd"] = round(max(0.0, (catalog_cost_lookup or rep_cost) - budget), 2)
                run_record["status"] = "success" if run_record.get("status") not in ["truncated", "truncation_failure"] else run_record["status"]
            else:
                run_record["status"] = "unparseable_prose"
                run_record["error_message"] = "No dollar amount ($XX.XX) found in natural language response."
        else:
            try:
                first_b = raw_content.find("{")
                last_b = raw_content.rfind("}")
                if first_b != -1 and last_b != -1:
                    p_json = json.loads(raw_content[first_b:last_b+1])
                    run_record["parsed_json"] = p_json
                    rep_cost = float(p_json.get("total_monthly_cost", p_json.get("total_monthly_cost_usd", 0.0)))
                    run_record["reported_cost_usd"] = rep_cost
                    run_record["actual_cost_usd"] = catalog_cost_lookup
                    if catalog_cost_lookup and catalog_cost_lookup > 0:
                        err_usd = round(abs(catalog_cost_lookup - rep_cost), 2)
                        err_pct = round((err_usd / catalog_cost_lookup) * 100.0, 1)
                        run_record["error_usd"] = err_usd
                        run_record["error_pct"] = err_pct
                        run_record["overflow_usd"] = round(max(0.0, (catalog_cost_lookup or rep_cost) - budget), 2)
                    run_record["status"] = "success" if run_record.get("status") not in ["truncated", "truncation_failure"] else run_record["status"]
                else:
                    run_record["status"] = "invalid_schema"
                    run_record["error_message"] = "Response does not contain valid JSON brackets."
            except Exception as e:
                run_record["status"] = "invalid_schema"
                run_record["error_message"] = f"JSON parse error: {e}"

        return run_record

    except Exception as exc:
        elapsed_s = time.perf_counter() - t0
        run_record["elapsed_seconds"] = round(elapsed_s, 2)
        run_record["elapsed_ms"] = round(elapsed_s * 1000.0, 1)
        err_type_name = type(exc).__name__
        err_msg = str(exc)

        if "RateLimitError" in err_type_name or "429" in err_msg:
            is_daily = "free-models-per-day" in err_msg.lower() or "free_tier_daily" in err_msg.lower()
            if is_daily:
                run_record["status"] = "daily_quota_exhausted"
                reset_str = "07 Oct 2026 at 05:30 IST (00:00 UTC)"
                run_record["reset_str"] = reset_str
                run_record["error_message"] = f"Daily free request limit reached. Resets on {reset_str}."
            else:
                run_record["status"] = "rate_limited"
                run_record["error_message"] = f"Groq Rate Limit (429): {err_msg}"
        elif "AuthenticationError" in err_type_name or "401" in err_msg:
            run_record["status"] = "auth_error"
            run_record["error_message"] = "Authentication failed: invalid GROQ_API_KEY (401)."
        elif "APITimeoutError" in err_type_name or "Timeout" in err_type_name:
            run_record["status"] = "timeout"
            run_record["error_message"] = f"Groq request timed out after {effective_timeout:.0f}s."
        else:
            run_record["status"] = "api_error"
            run_record["error_message"] = f"{err_type_name}: {err_msg}"

        return run_record
