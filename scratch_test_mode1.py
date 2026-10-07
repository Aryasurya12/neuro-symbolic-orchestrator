import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.run_4way_benchmark import FourWayBenchmarker, extract_mode1_cost, strip_reasoning_preamble
from src.verifiers.proof_engine import verify_feasibility
from src.symbolic.optimizers.domain_catalog import VM_CATALOG

def run_test():
    benchmarker = FourWayBenchmarker()
    query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
    print(f"=== TESTING MODE 1 LIVE (3 RUNS) ===", flush=True)
    print(f"Query: {query}", flush=True)
    print(f"Model: {benchmarker.model}", flush=True)
    print(f"API Key present: {bool(benchmarker.api_key)}", flush=True)
    print(flush=True)

    for run_idx in range(1, 4):
        print(f"--- LIVE RUN {run_idx}/3 ---", flush=True)
        t0 = time.perf_counter()
        
        system_instruction = (
            "You are a Cloud Infrastructure FinOps architect. "
            "Your task is to recommend an optimal cloud virtual machine provisioning configuration "
            "that strictly satisfies user compute, memory, provider, and monthly budget constraints.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Output ONLY direct recommendations in plain text.\n"
            "- Do NOT include meta-instructions, step-by-step thinking, 'Analyze the Input' headers, or placeholders.\n"
            "- State the concrete instance types, provider, total vCPUs, total RAM, and estimated total monthly cost ($/month)."
        )
        user_prompt = (
            f"A customer sends this request:\n"
            f"\"{query}\"\n\n"
            f"Recommend a concrete cloud VM allocation plan. Provide:\n"
            f"1. Recommended Cloud Provider and Instance Types with quantities\n"
            f"2. Total vCPUs and Total RAM provided\n"
            f"3. Exact Estimated Total Monthly Cost ($/month)\n"
            f"Respond directly with your recommendation in plain text. Be concise."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]
        
        max_tok = 2048
        try:
            try:
                resp = benchmarker.client.chat.completions.create(
                    model=benchmarker.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tok,
                    extra_body={"reasoning": {"effort": "none", "max_tokens": 0}},
                )
            except Exception:
                resp = benchmarker.client.chat.completions.create(
                    model=benchmarker.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tok,
                )
            
            raw_choice = resp.choices[0]
            finish_reason = getattr(raw_choice, "finish_reason", "unknown")
            content = (raw_choice.message.content or "").strip()
            
            retry_fired = False
            if finish_reason == "length":
                retry_fired = True
                retry_resp = benchmarker.client.chat.completions.create(
                    model=benchmarker.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=4096,
                )
                retry_choice = retry_resp.choices[0]
                retry_finish = getattr(retry_choice, "finish_reason", "unknown")
                if retry_finish != "length":
                    finish_reason = retry_finish
                    content = (retry_choice.message.content or "").strip()

            elapsed = (time.perf_counter() - t0)
            
            cleaned = strip_reasoning_preamble(content)
            extracted_cost = extract_mode1_cost(content, finish_reason=finish_reason)
            
            print(f"Request max_tokens sent: {max_tok}", flush=True)
            print(f"Response finish_reason: {finish_reason}", flush=True)
            print(f"Retry fired: {retry_fired}", flush=True)
            print(f"Latency: {elapsed:.2f}s", flush=True)
            print(f"Raw content length: {len(content)} chars", flush=True)
            print(f"Cleaned content snippet:\n{cleaned[:400]}...", flush=True)
            print(f"Extracted Cost ($/mo): {extracted_cost}", flush=True)
            print(f"Input Budget Cap: $300.00", flush=True)
            print(flush=True)
            
        except Exception as e:
            print(f"Run {run_idx} failed with error: {e}", flush=True)
            print(flush=True)

if __name__ == "__main__":
    run_test()
