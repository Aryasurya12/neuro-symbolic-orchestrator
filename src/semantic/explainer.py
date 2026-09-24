"""SEM-5: Stage 6 FinOps Report Generator & Natural Language Explainer."""

import os
import re
from typing import Any, Dict, List, Optional
from config.settings import settings
from src.semantic.schemas import CloudOptimizationContract
from openai import OpenAI


class FinOpsExplainer:
    """Transforms raw symbolic optimization solver telemetry and Pydantic contracts
    into executive-level FinOps deployment reports with multi-currency support (USD / INR)
    and dynamic, contextualized FinOps recommendations.
    """

    @staticmethod
    def usd_to_inr(
        usd_amount: float, rate: float = settings.USD_TO_INR_RATE
    ) -> float:
        """Converts a USD amount to INR using the configured exchange rate."""
        return round(usd_amount * rate, 2)

    @classmethod
    def generate_dynamic_recommendations(
        cls,
        contract: CloudOptimizationContract,
        solver_result: Dict[str, Any],
        exchange_rate: float = settings.USD_TO_INR_RATE,
    ) -> List[str]:
        """Generates dynamic, non-hardcoded FinOps recommendations tailored specifically
        to the cloud provider, allocated instance families, budget headroom, and problem type.
        """
        recs: List[str] = []
        inr_sym = settings.CURRENCY_SYMBOL_INR
        usd_sym = settings.CURRENCY_SYMBOL_USD

        budget_max = contract.budget_max_usd
        total_cost = solver_result.get(
            "total_monthly_cost_usd",
            solver_result.get("estimated_monthly_cost_usd", 0.0),
        )
        savings = solver_result.get(
            "cost_savings_usd", max(0.0, budget_max - total_cost)
        )
        savings_inr = cls.usd_to_inr(savings, exchange_rate)
        utilization = solver_result.get(
            "budget_utilized_pct",
            round((total_cost / budget_max) * 100, 2) if budget_max > 0 else 0.0,
        )

        providers = [p.upper() for p in contract.cloud_providers]
        allocated_vms = solver_result.get("allocated_vms", [])
        instance_types = [vm.get("instance_type", "") for vm in allocated_vms if vm.get("instance_type")]
        instance_summary = ", ".join(set(instance_types)) if instance_types else "allocated compute"

        # 1. Cloud Provider & Commitment Strategy
        if "AWS" in providers:
            if any("t3" in it.lower() or "t4" in it.lower() for it in instance_types):
                recs.append(
                    f"Pricing Strategy: Workload utilizes burstable instances ({instance_summary}). "
                    f"Enroll in AWS 1-Year Compute Savings Plans (saves ~28-34%) or migrate to ARM64 Graviton (t4g) "
                    f"for up to 20% better price-to-performance."
                )
            else:
                recs.append(
                    f"Pricing Strategy: Commit to 1-Year AWS EC2 Instance Savings Plans or Standard Reserved Instances "
                    f"for steady-state compute ({instance_summary}), yielding 30-45% discount over on-demand rates."
                )
        elif "AZURE" in providers:
            recs.append(
                f"Pricing Strategy: Purchase Azure 1-Yr/3-Yr Reserved Virtual Machine Instances with Azure Hybrid Benefit (AHB) "
                f"to reduce base compute expenses by up to 40-65%."
            )
        elif "GCP" in providers:
            recs.append(
                f"Pricing Strategy: Activate Google Cloud 1-Year Committed Use Discounts (CUDs) and evaluate Spot VMs "
                f"for stateless batch components to achieve 60-90% cost reduction."
            )
        else:
            recs.append(
                f"Pricing Strategy: Leverage cross-cloud commitment tiers across {', '.join(contract.cloud_providers)} "
                f"and evaluate containerized spot instances for stateless service components."
            )

        # 2. Budget Headroom & Capacity Strategy
        if utilization < 30.0:
            recs.append(
                f"Budget Headroom ({utilization:.1f}% utilized): You have a monthly surplus of "
                f"{usd_sym}{savings:,.2f} USD ({inr_sym}{savings_inr:,.2f} INR). Reallocate surplus capital toward "
                f"multi-AZ automated failover and managed snapshot replication."
            )
        elif utilization > 75.0:
            recs.append(
                f"Budget Warning ({utilization:.1f}% utilized): Spending is near the {usd_sym}{budget_max:,.2f} USD cap "
                f"with only {usd_sym}{savings:,.2f} USD buffer. Configure automated billing alerts and scaling throttles "
                f"at 85% to prevent overage."
            )
        else:
            recs.append(
                f"Spend Governance ({utilization:.1f}% utilized): Optimal operating band with {usd_sym}{savings:,.2f} USD "
                f"({inr_sym}{savings_inr:,.2f} INR)/mo buffer. Establish automated CloudWatch/Prometheus anomaly alerts at 80%."
            )

        # 3. Problem & Architecture Specific Optimization
        if contract.problem_type == "ILP_VM_Allocation":
            if contract.service_count > 1:
                recs.append(
                    f"Workload Bin-Packing: Consolidate the {contract.service_count} requested microservices into containerized "
                    f"pods (ECS/EKS/Docker) with strict CPU/memory limits to maximize utilization density on {instance_summary}."
                )
            else:
                recs.append(
                    f"Compute Rightsizing: Track sustained CPU credit usage on {instance_summary} for 14 days; "
                    f"scale down instance size if average utilization remains below 40%."
                )
        elif contract.problem_type == "PSO_Continuous_Scaling":
            target_cpu = solver_result.get("target_cpu_utilization_pct", 70.0)
            recs.append(
                f"Autoscaling Governance: Set Horizontal Pod Autoscaler (HPA) cooldown periods (300s scale-down window) "
                f"targeting {target_cpu:.0f}% CPU to avoid resource flapping and transient billing ticks."
            )
        elif contract.problem_type == "Z3_Graph_Disaster_Recovery":
            p_region = solver_result.get("primary_region", "Primary")
            s_region = solver_result.get("secondary_region", "Secondary")
            lat = solver_result.get("inter_region_latency_ms", 0.0)
            recs.append(
                f"Data Transfer & Egress: Inter-region replication between {p_region} and {s_region} ({lat:.1f}ms latency) "
                f"incurs cross-region transfer fees; enable zstd/gzip compression to minimize data egress costs."
            )

        # 4. Mandatory Tagging & Attribution
        recs.append(
            f"Governance & Attribution: Apply mandatory Cost Allocation Tags (`Environment`, `CostCenter`, `Owner:FinOps`) "
            f"across all {contract.service_count} service resources for 100% cost attribution."
        )

        return recs

    @classmethod
    def generate_llm_recommendations(
        cls,
        contract: CloudOptimizationContract,
        solver_result: Dict[str, Any],
        timeout_seconds: float = 6.0,
    ) -> Optional[List[str]]:
        """Invokes OpenRouter NVIDIA Nemotron to generate dynamic, AI-reasoned FinOps advice.
        Returns a list of recommendation strings, or None on failure/timeout.
        """
        api_key = os.getenv("OPENROUTER_API_KEY") or getattr(settings, "OPENROUTER_API_KEY", "")
        if not api_key:
            return None

        primary_model = os.getenv("OPENROUTER_MODEL") or getattr(
            settings, "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"
        )
        candidate_models = [
            primary_model,
            "nvidia/nemotron-3.5-lightning:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ]
        models_to_try = list(dict.fromkeys(candidate_models))

        total_cost = solver_result.get(
            "total_monthly_cost_usd",
            solver_result.get("estimated_monthly_cost_usd", 0.0),
        )
        utilization = solver_result.get("budget_utilized_pct", 0.0)
        vms = [
            f"{v.get('provider', '')} {v.get('instance_type', '')} x{v.get('count', 1)}"
            for v in solver_result.get("allocated_vms", [])
        ]
        vms_str = ", ".join(vms) if vms else "Optimized compute instance"

        prompt = (
            f"You are a Principal FinOps Architect. Analyze this cloud deployment result:\n"
            f"- Problem Type: {contract.problem_type}\n"
            f"- Cloud Provider(s): {', '.join(contract.cloud_providers)}\n"
            f"- Target Services: {contract.service_count} service(s), {contract.required_vcpus} vCPUs, {contract.required_ram_gb}GB RAM\n"
            f"- Budget Cap: ${contract.budget_max_usd:.2f} USD\n"
            f"- Optimized Monthly Cost: ${total_cost:.2f} USD ({utilization:.1f}% budget utilized)\n"
            f"- Placed Resources: {vms_str}\n\n"
            f"Provide exactly 3 concise, highly-actionable, technical FinOps recommendations tailored to this specific result. "
            f"Include specific pricing models (Savings Plans/RIs/CUDs/Spot), monitoring thresholds, and architecture rightsizing tactics. "
            f"Output ONLY 3 numbered lines:\n"
            f"1. <recommendation>\n"
            f"2. <recommendation>\n"
            f"3. <recommendation>"
        )

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout_seconds,
        )

        for model_name in models_to_try:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=350,
                )
                if not resp or not resp.choices:
                    continue
                content = (resp.choices[0].message.content or "").strip()
                lines = [l.strip() for l in content.split("\n") if l.strip()]
                recs = []
                for line in lines:
                    if re.match(r"^\d+[\.\)]\s+", line):
                        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                        # Sanitize any non-standard unicode characters
                        cleaned = cleaned.encode("ascii", "replace").decode("ascii")
                        if cleaned:
                            recs.append(cleaned)
                if len(recs) >= 2:
                    return recs[:4]
            except Exception:
                continue

        return None

    @classmethod
    def generate_report(
        cls,
        contract: CloudOptimizationContract,
        solver_result: Dict[str, Any],
        exchange_rate: float = settings.USD_TO_INR_RATE,
        enable_llm_explainer: bool = True,
    ) -> str:
        """Generates a structured ASCII Executive FinOps Deployment Report
        displaying dual-currency metrics ($ and ₹) and dynamic, contextual recommendations.
        """
        problem = contract.problem_type
        status = solver_result.get("status", "UNKNOWN")
        solver_name = solver_result.get("solver", "Neuro-Symbolic Engine")
        budget_max = contract.budget_max_usd
        total_cost = solver_result.get(
            "total_monthly_cost_usd",
            solver_result.get("estimated_monthly_cost_usd", 0.0),
        )
        savings = solver_result.get(
            "cost_savings_usd", max(0.0, budget_max - total_cost)
        )
        utilization = solver_result.get(
            "budget_utilized_pct",
            round((total_cost / budget_max) * 100, 2) if budget_max > 0 else 0.0,
        )

        budget_inr = cls.usd_to_inr(budget_max, exchange_rate)
        cost_inr = cls.usd_to_inr(total_cost, exchange_rate)
        savings_inr = cls.usd_to_inr(savings, exchange_rate)
        inr_sym = settings.CURRENCY_SYMBOL_INR
        usd_sym = settings.CURRENCY_SYMBOL_USD

        lines = [
            "=" * 82,
            "               NEURASYM FINOPS EXECUTIVE DEPLOYMENT REPORT                ",
            "                  Stage 6: Symbolic Optimization Verdict                  ",
            "=" * 82,
            f" Problem Type         : {problem}",
            f" Optimization Engine  : {solver_name}",
            f" Feasibility Status   : {status}",
            f" Target Cloud(s)      : {', '.join(contract.cloud_providers)}",
            f" Exchange Rate        : 1 USD = {exchange_rate:.2f} INR",
            "-" * 82,
            " FINANCIAL & BUDGET SUMMARY (DUAL CURRENCY: USD / INR)",
            "-" * 82,
            f"  - Monthly Budget Cap       : {usd_sym}{budget_max:,.2f} USD ({inr_sym}{budget_inr:,.2f} INR) / month",
            f"  - Optimized Monthly Cost   : {usd_sym}{total_cost:,.2f} USD ({inr_sym}{cost_inr:,.2f} INR) / month",
            f"  - Monthly Net Savings      : {usd_sym}{savings:,.2f} USD ({inr_sym}{savings_inr:,.2f} INR) / month ({max(0.0, 100.0 - utilization):.1f}% under cap)",
            f"  - Budget Utilization Rate  : {utilization:>6.2f}%",
            "-" * 82,
        ]

        if problem == "ILP_VM_Allocation":
            allocated_vms = solver_result.get("allocated_vms", [])
            total_vcpus = solver_result.get("total_vcpus", contract.required_vcpus)
            total_ram = solver_result.get("total_ram_gb", contract.required_ram_gb)

            lines.extend([
                " COMPUTE & RESOURCE ALLOCATION BREAKDOWN",
                "-" * 82,
                f"  - Target Services Required : {contract.service_count} service(s)",
                f"  - Total Allocated Compute  : {total_vcpus} vCPUs (Required: {contract.required_vcpus})",
                f"  - Total Allocated Memory   : {total_ram:.1f} GB RAM (Required: {contract.required_ram_gb:.1f} GB)",
                "",
                "  Instance Placement Plan:",
            ])
            for idx, vm in enumerate(allocated_vms, 1):
                vm_cost_usd = vm.get("monthly_cost", 0.0)
                vm_cost_inr = cls.usd_to_inr(vm_cost_usd, exchange_rate)
                lines.append(
                    f"   [{idx}] {vm.get('provider')} {vm.get('instance_type')} x {vm.get('count')} instance(s) "
                    f"({vm.get('vcpus_per_vm')} vCPUs, {vm.get('ram_gb_per_vm')}GB) -> "
                    f"{usd_sym}{vm_cost_usd:,.2f} USD ({inr_sym}{vm_cost_inr:,.2f} INR)/mo"
                )

        elif problem == "PSO_Continuous_Scaling":
            bw = solver_result.get("optimal_bandwidth_mbps", 0.0)
            replicas = solver_result.get("recommended_replicas", 1)
            target_cpu = solver_result.get("target_cpu_utilization_pct", 70.0)
            hourly_cost_usd = solver_result.get("estimated_hourly_cost_usd", 0.0)
            hourly_cost_inr = round(hourly_cost_usd * exchange_rate, 4)

            lines.extend([
                " CONTINUOUS AUTOSCALING PARAMETERS",
                "-" * 82,
                f"  - Optimal Bandwidth Sizing : {bw:.2f} Mbps",
                f"  - Recommended Pod Replicas : {replicas} active replica(s)",
                f"  - Target CPU Utilization   : {target_cpu:.1f}%",
                f"  - Estimated Hourly Burn    : {usd_sym}{hourly_cost_usd:.4f} USD ({inr_sym}{hourly_cost_inr:.4f} INR)/hour",
            ])

        elif problem == "Z3_Graph_Disaster_Recovery":
            primary = solver_result.get("primary_region", "N/A")
            secondary = solver_result.get("secondary_region", "N/A")
            latency = solver_result.get("inter_region_latency_ms", 0.0)
            achieved_sla = solver_result.get("achieved_sla_pct", 99.99)
            topology = solver_result.get(
                "disaster_recovery_topology", "Active-Active Mesh"
            )

            lines.extend([
                " TOPOLOGICAL DISASTER RECOVERY PLACEMENT",
                "-" * 82,
                f"  - Primary Failure Domain   : {primary}",
                f"  - Secondary Failover Domain: {secondary}",
                f"  - DR Interconnect Topology : {topology}",
                f"  - Cross-Region Sync Latency: {latency:.2f} ms (Cap: {contract.latency_max_ms:.1f} ms)",
                f"  - Composite Availability   : {achieved_sla:.5f}% SLA (Target: {contract.sla_availability_pct:.3f}%)",
            ])

        # Obtain dynamic / LLM recommendations
        recommendations = None
        if enable_llm_explainer:
            try:
                recommendations = cls.generate_llm_recommendations(contract, solver_result)
            except Exception:
                recommendations = None

        if not recommendations:
            recommendations = cls.generate_dynamic_recommendations(
                contract, solver_result, exchange_rate
            )

        lines.extend([
            "-" * 82,
            " ACTIONABLE FINOPS RECOMMENDATIONS",
            "-" * 82,
        ])
        for idx, rec in enumerate(recommendations, 1):
            # Wrap lines cleanly if needed
            lines.append(f"  {idx}. {rec}")

        lines.append("=" * 82)
        return "\n".join(lines)
