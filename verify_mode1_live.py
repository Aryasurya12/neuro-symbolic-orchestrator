import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI
from benchmarks.run_4way_benchmark import extract_mode1_cost, strip_reasoning_preamble
from config.settings import settings

def run_live_mode1_verification():
    api_key = os.getenv("OPENROUTER_API_KEY") or getattr(settings, "OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL") or getattr(settings, "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
    
    print(f"================================================================================", flush=True)
    print(f"MODE 1 LIVE TRUNCATION & VERIFICATION (3 LIVE RUNS)", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"API Key: {api_key[:10]}...{api_key[-6:] if api_key else 'NONE'}", flush=True)
    print(f"================================================================================", flush=True)
    
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=60.0)
    
    query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
    
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
    
    runs_data = []
    
    for i in range(1, 4):
        print(f"\n>>> RUN {i} of 3...", flush=True)
        t0 = time.perf_counter()
        sent_max_tokens = 2048
        
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=sent_max_tokens,
                    extra_body={"reasoning": {"effort": "none"}},
                )
            except Exception as e_extra:
                print(f"   [Notice: extra_body not supported, falling back to standard create: {e_extra}]", flush=True)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=sent_max_tokens,
                )
                
            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", "unknown")
            raw_content = (choice.message.content or "").strip()
            elapsed_sec = time.perf_counter() - t0
            
            retry_occurred = False
            if finish_reason == "length":
                retry_occurred = True
                print(f"   [Truncation detected on attempt 1 (finish_reason='length'). Retrying with max_tokens=4096...]", flush=True)
                sent_max_tokens = 4096
                retry_resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=sent_max_tokens,
                )
                retry_choice = retry_resp.choices[0]
                retry_finish = getattr(retry_choice, "finish_reason", "unknown")
                if retry_finish != "length":
                    finish_reason = retry_finish
                    raw_content = (retry_choice.message.content or "").strip()
                else:
                    finish_reason = "length"
            
            cleaned_content = strip_reasoning_preamble(raw_content)
            extracted_cost = extract_mode1_cost(raw_content, finish_reason=finish_reason)
            
            usage = getattr(resp, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            }
            
            run_info = {
                "run": i,
                "latency_sec": round(elapsed_sec, 2),
                "sent_max_tokens": sent_max_tokens,
                "finish_reason": finish_reason,
                "retry_occurred": retry_occurred,
                "usage": usage_dict,
                "raw_content_chars": len(raw_content),
                "extracted_cost": extracted_cost,
                "raw_content": raw_content,
                "cleaned_content": cleaned_content,
            }
            runs_data.append(run_info)
            
            print(f"   - Latency: {elapsed_sec:.2f}s", flush=True)
            print(f"   - max_tokens sent: {sent_max_tokens}", flush=True)
            print(f"   - finish_reason: {finish_reason}", flush=True)
            print(f"   - Usage tokens: {usage_dict}", flush=True)
            print(f"   - Extracted Cost: ${extracted_cost:.2f}/mo" if extracted_cost is not None else "   - Extracted Cost: None", flush=True)
            print(f"   - Cleaned Content:\n{cleaned_content}\n", flush=True)
            
        except Exception as err:
            print(f"   ERROR in run {i}: {err}", flush=True)
            runs_data.append({
                "run": i,
                "error": str(err)
            })

    with open("mode1_live_results.json", "w", encoding="utf-8") as f:
        json.dump(runs_data, f, indent=2)
    print("\nSaved run details to mode1_live_results.json", flush=True)

if __name__ == "__main__":
    run_live_mode1_verification()
