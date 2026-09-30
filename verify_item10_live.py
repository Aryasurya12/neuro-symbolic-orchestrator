"""Dedicated live verification script for Item 10: FinOpsExplainer.generate_llm_recommendations."""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.explainer import FinOpsExplainer
from config.settings import settings


def verify_item10_live():
    print("=" * 80)
    print("ITEM 10: generate_llm_recommendations live invocation & finish_reason check")
    print("=" * 80)

    api_key = os.getenv("OPENROUTER_API_KEY") or getattr(settings, "OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    model_name = os.getenv("OPENROUTER_MODEL") or getattr(
        settings, "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"
    )
    print(f"Target Model: {model_name}")

    contract = CloudOptimizationContract(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "GCP"],
        service_count=2,
        required_vcpus=4,
        required_ram_gb=16.0,
        budget_max_usd=850.0,
        latency_max_ms=50.0,
        sla_availability_pct=99.99,
    )

    solver_result = {
        "status": "Feasible",
        "solver": "GraphSteeredZ3",
        "total_monthly_cost_usd": 243.00,
        "estimated_monthly_cost_usd": 243.00,
        "primary_region": "us-east-1",
        "secondary_region": "us-central1",
        "inter_region_latency_ms": 32.0,
        "achieved_sla_pct": 99.99997,
        "budget_utilized_pct": 28.59,
        "allocated_vms": [
            {"provider": "AWS", "instance_type": "us-east-1", "count": 1},
            {"provider": "GCP", "instance_type": "us-central1", "count": 1},
        ],
    }

    # Direct raw call reproducing generate_llm_recommendations to capture finish_reason explicitly
    system_prompt = (
        "You are a Principal Cloud FinOps Architect. "
        "Analyze the given cloud resource deployment contract and solver result, and provide "
        "exactly 3 to 4 concise, highly-actionable, technical FinOps recommendations.\n\n"
        "Key Focus Areas:\n"
        "- Commitment pricing strategies (AWS Savings Plans/RIs, Azure AHB/Reservations, GCP CUDs/Spot)\n"
        "- Capacity rightsizing and autoscaling thresholds\n"
        "- Architecture optimization (data transfer, multi-region replication egress)\n"
        "- Cost governance, monitoring alarms, and allocation tagging\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Output ONLY direct recommendations, one per line.\n"
        "- Do NOT include meta-instructions, step-by-step thinking, 'Analyze the Input' headers, or placeholders.\n"
        "- Do NOT include numbering, bullet points, or introductory/concluding text."
    )

    user_prompt = (
        f"Cloud Deployment Details:\n"
        f"- Problem Type: {contract.problem_type}\n"
        f"- Target Cloud Provider(s): {', '.join(contract.cloud_providers)}\n"
        f"- Required Resources: {contract.service_count} service(s), {contract.required_vcpus} vCPUs, {contract.required_ram_gb}GB RAM\n"
        f"- Monthly Budget Cap: ${contract.budget_max_usd:.2f} USD\n"
        f"- Optimized Monthly Cost: $243.00 USD (28.6% budget utilized)\n"
        f"- Placed Resources: AWS us-east-1 x1, GCP us-central1 x1\n\n"
        f"Provide 3-4 actionable FinOps recommendations tailored specifically to this deployment."
    )

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=60.0)

    print("\n1. Making direct live API call with max_tokens=1500 to inspect raw response object...")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    choice = resp.choices[0]
    finish_reason = getattr(choice, "finish_reason", "unknown")
    raw_content = (choice.message.content or "").strip()

    print(f"finish_reason: {finish_reason}")
    print(f"Raw content length: {len(raw_content)} chars")

    print("\n2. Invoking FinOpsExplainer.generate_llm_recommendations(contract, solver_result)...")
    recs = FinOpsExplainer.generate_llm_recommendations(contract, solver_result, timeout_seconds=60.0)
    print(f"Returned recommendations count: {len(recs) if recs else 0}")
    if recs:
        for idx, r in enumerate(recs, 1):
            print(f"  {idx}. {r}")

    assert finish_reason == "stop", f"Expected finish_reason == 'stop', got '{finish_reason}'"
    assert recs is not None and len(recs) >= 2, "Expected at least 2 recommendations"
    print("\n✓ PASS: Item 10 verified live with finish_reason == 'stop'.")


if __name__ == "__main__":
    verify_item10_live()
