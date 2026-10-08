"""Standalone LLM Connection Verification Script for Neurasym.

Performs exactly one isolated inference request against the configured Groq model
using the project's centralized settings (config.settings).
Does NOT import or trigger the dashboard, solvers, orchestrator, explainer, or benchmarks.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# Ensure UTF-8 stdout on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config.settings import settings
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


def run_connection_test(
    model: str | None = None,
    timeout_seconds: float = 360.0,
    max_tokens: int = 4096,
) -> dict:
    """Executes a single, non-streaming test inference request to Groq API (or configured provider)."""
    api_key = (
        os.getenv("GROQ_API_KEY")
        or getattr(settings, "GROQ_API_KEY", "")
        or os.getenv("OPENROUTER_API_KEY")
        or getattr(settings, "OPENROUTER_API_KEY", "")
    )
    if not api_key:
        return {
            "auth_success": False,
            "request_success": False,
            "marker_success": False,
            "error_type": "ConfigurationError",
            "error_message": "GROQ_API_KEY is missing or empty in configuration / .env.",
        }

    base_url = (
        os.getenv("GROQ_BASE_URL")
        or getattr(settings, "GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    target_model = model or os.getenv("GROQ_MODEL") or getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
    prompt = "Reply with exactly: NEURASYM_CONNECTION_OK"
    expected_marker = "NEURASYM_CONNECTION_OK"

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=0,
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_model": target_model,
        "timeout_seconds": timeout_seconds,
        "max_tokens": max_tokens,
        "auth_success": False,
        "request_success": False,
        "marker_success": False,
    }

    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed_seconds = time.perf_counter() - start_time
        result["elapsed_seconds"] = round(elapsed_seconds, 2)
        result["auth_success"] = True
        result["request_success"] = True

        result["response_id"] = getattr(response, "id", None)
        result["response_created"] = getattr(response, "created", None)
        result["returned_model"] = getattr(response, "model", None)

        choice = response.choices[0] if response.choices else None
        if choice:
            result["finish_reason"] = getattr(choice, "finish_reason", None)
            raw_content = (choice.message.content or "").strip()
            result["content"] = raw_content
            result["content_length"] = len(raw_content)
            result["marker_success"] = expected_marker in raw_content
        else:
            result["finish_reason"] = "no_choices"
            result["content"] = ""
            result["content_length"] = 0
            result["marker_success"] = False

        if hasattr(response, "usage") and response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            # Check for reasoning / completion tokens breakdown if provided
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                details = response.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens") and details.reasoning_tokens is not None:
                    usage_dict["reasoning_tokens"] = details.reasoning_tokens
            result["usage"] = usage_dict
        else:
            result["usage"] = None

    except AuthenticationError as e:
        elapsed_seconds = time.perf_counter() - start_time
        result["elapsed_seconds"] = round(elapsed_seconds, 2)
        result["error_type"] = "AuthenticationError (401)"
        result["error_message"] = "Invalid API key or unauthorized access."
    except PermissionDeniedError as e:
        elapsed_seconds = time.perf_counter() - start_time
        result["elapsed_seconds"] = round(elapsed_seconds, 2)
        result["error_type"] = "PermissionDeniedError (403)"
        result["error_message"] = "Permission denied for this resource or model."
    except RateLimitError as e:
        elapsed_seconds = time.perf_counter() - start_time
        result["elapsed_seconds"] = round(elapsed_seconds, 2)
        result["auth_success"] = True  # Key was recognized, but rate limited
        result["error_type"] = "RateLimitError (429)"
        result["error_message"] = (
            f"Daily free request limit or rate limit reached on OpenRouter. Details: {e.message}"
        )
    except BadRequestError as e:
        elapsed_seconds = time.perf_counter() - start_time
        result["elapsed_seconds"] = round(elapsed_seconds, 2)
        result["auth_success"] = True
        result["error_type"] = f"BadRequestError ({e.status_code})"
        result["error_message"] = f"Bad request parameters: {e.message}"
    except APIConnectionError as e:
        elapsed_seconds = time.perf_counter() - start_time
        result["elapsed_seconds"] = round(elapsed_seconds, 2)
        result["error_type"] = "APIConnectionError"
        result["error_message"] = f"Failed to connect to OpenRouter endpoint: {e}"
    except TimeoutError as e:
        elapsed_seconds = time.perf_counter() - start_time
        result["elapsed_seconds"] = round(elapsed_seconds, 2)
        result["error_type"] = "TimeoutError"
        result["error_message"] = f"Request exceeded configured timeout of {timeout_seconds}s."
    except Exception as e:
        elapsed_seconds = time.perf_counter() - start_time
        result["elapsed_seconds"] = round(elapsed_seconds, 2)
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify single OpenRouter LLM connection for Neurasym."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model ID override (defaults to OPENROUTER_MODEL from config/settings.py).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=360.0,
        help="Request timeout in seconds (default: 360.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens limit (default: 4096).",
    )
    args = parser.parse_args()

    print("====================================================================")
    print("🔌 Neurasym: Single-Request Groq LLM Connection Verification")
    print("====================================================================")
    print(f"Target Model:    {args.model or getattr(settings, 'GROQ_MODEL', 'llama-3.3-70b-versatile')}")
    print(f"Request Timeout: {args.timeout}s | Max Retries: 0 | Max Tokens: {args.max_tokens}")
    print("Executing 1 non-streaming test request... (please wait)")
    print("--------------------------------------------------------------------")

    result = run_connection_test(
        model=args.model,
        timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
    )

    print("\n--- Test Outcome Summary ---")
    print(f"Timestamp (UTC):   {result.get('timestamp_utc')}")
    print(f"Elapsed Time:      {result.get('elapsed_seconds', 'N/A')} seconds")
    print(f"Authentication:    {'✅ OK' if result.get('auth_success') else '❌ Failed'}")
    print(f"Request Executed:  {'✅ OK' if result.get('request_success') else '❌ Failed'}")
    print(f"Marker Received:   {'✅ OK' if result.get('marker_success') else '❌ Failed'}")

    if result.get("request_success"):
        print(f"Response ID:       {result.get('response_id')}")
        print(f"Response Created:  {result.get('response_created')}")
        print(f"Returned Model:    {result.get('returned_model')}")
        print(f"Finish Reason:     {result.get('finish_reason')}")
        print(f"Token Usage:       {json.dumps(result.get('usage', {}))}")
        print("\n--- Raw Response Content ---")
        print(result.get("content", ""))
        print("----------------------------")
    else:
        print(f"Error Type:        {result.get('error_type')}")
        print(f"Error Message:     {result.get('error_message')}")

    print("====================================================================\n")

    # Return exit code based on complete marker success
    if result.get("marker_success"):
        sys.exit(0)
    elif result.get("request_success"):
        sys.exit(2)  # Request succeeded, but unexpected answer / empty content
    else:
        sys.exit(1)  # Request failed


if __name__ == "__main__":
    main()
