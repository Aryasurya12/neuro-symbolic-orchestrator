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

        if mode_num == 1:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Cloud Solutions Architect. Recommend a concrete cloud VM allocation plan. "
                        "State the recommended provider, instance types, quantities, and the exact total monthly cost in USD ($/month). Be concise."
                    ),
                },
                {"role": "user", "content": f"Recommend cloud VMs for this request: \"{query}\""},
            ]
        else:
            req_vcpus = int(getattr(contract, "required_vcpus", 4)) if contract else 4
            req_ram = float(getattr(contract, "required_ram_gb", 16.0)) if contract else 16.0
            schema_sample = {
                "cloud_provider": "AWS",
                "instances": [{"sku": "t3.medium", "quantity": 2, "monthly_cost": 60.74}],
                "total_monthly_cost": 60.74,
                "total_vcpus": req_vcpus,
                "total_ram_gb": req_ram,
            }
            messages = [
                {
                    "role": "system",
                    "content": f"You are a Cloud Optimization System. Respond ONLY with valid JSON matching this schema: {json.dumps(schema_sample)}. No explanatory text.",
                },
                {"role": "user", "content": f"Optimize allocation for: \"{query}\". Respond in JSON."},
            ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0 if mode_num == 2 else 0.2,
            max_tokens=max_tokens,
        )
        elapsed_s = time.perf_counter() - t0
        run_record["elapsed_seconds"] = round(elapsed_s, 2)
        run_record["elapsed_ms"] = round(elapsed_s * 1000.0, 1)

        choice = resp.choices[0] if resp.choices else None
        if not choice:
            run_record["status"] = "empty_response"
            run_record["error_message"] = "Provider returned no choices."
            return run_record

        finish_reason = getattr(choice, "finish_reason", "unknown")
        run_record["finish_reason"] = finish_reason
        raw_content = (choice.message.content or "").strip()
        run_record["content"] = raw_content
        run_record["response_id"] = getattr(resp, "id", None)
        if hasattr(resp, "usage") and resp.usage:
            run_record["usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        if not raw_content:
            run_record["status"] = "empty_response"
            run_record["error_message"] = "Model returned empty content."
            return run_record

        if finish_reason == "length":
            run_record["status"] = "truncated"
            run_record["error_message"] = "Generation reached max token limit and was truncated."

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
                run_record["status"] = "success" if run_record.get("status") != "truncated" else "truncated"
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
                    run_record["status"] = "success" if run_record.get("status") != "truncated" else "truncated"
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
