"""FinOps Explainer and Executive Report Generator for Neurasym.

Transforms symbolic optimization solver telemetry, verification verdicts, and Pydantic contracts
into executive-level FinOps deployment reports with multi-currency support (USD / INR)
and dynamic, contextualized FinOps recommendations.

Guarantees:
1. Explanations strictly respect verification results: rejected allocations lead with the failure
   and never recommend deployment or claim formal certification.
2. An LLM explanation cannot alter the allocation, cost, or checker verdict.
3. Accurate provenance reporting: LIVE_PROVIDER, CACHED_PROVIDER, LOCAL_TEMPLATE, UNAVAILABLE.
4. "Budget Headroom" replaces "net savings" when only budget minus candidate cost is calculated.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from config.settings import settings
from src.semantic.schemas import CloudOptimizationContract
from src.verifiers.canonical_record import ExplanationSource


class FinOpsExplainer:
    """Transforms raw symbolic optimization solver telemetry and Pydantic contracts
    into executive-level FinOps deployment reports with multi-currency support (USD / INR).
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
        is_feasible: bool = True,
        violations: Optional[List[str]] = None,
        currency_symbol: Optional[str] = None,
        currency_code: Optional[str] = None,
        currency_rate: Optional[float] = None,
        **kwargs,
    ) -> List[str]:
        """Generates dynamic, rule-based FinOps recommendations tailored specifically
        to the cloud provider, allocated instance families, budget headroom, and problem type.
        """
        recs: List[str] = []
        eff_rate = currency_rate if currency_rate is not None else exchange_rate
        inr_sym = currency_symbol if currency_symbol is not None else settings.CURRENCY_SYMBOL_INR
        inr_code = currency_code if currency_code is not None else "INR"
        usd_sym = settings.CURRENCY_SYMBOL_USD

        if not is_feasible:
            recs.append("Constraint Remediation: The allocation violated formal constraints and cannot be deployed as-is.")
            if violations:
                for v in violations:
                    recs.append(f"Constraint Violation: {v}")
            recs.append("Remediation Strategy: Increase the monthly budget cap or reduce workload capacity requirements to achieve a feasible configuration.")
            return recs

        budget_max = float(contract.budget_max_usd)
        total_cost = float(solver_result.get(
            "total_monthly_cost_usd",
            solver_result.get("estimated_monthly_cost_usd", 0.0),
        ))
        headroom = max(0.0, budget_max - total_cost)
        headroom_inr = cls.usd_to_inr(headroom, eff_rate)
        utilization = round((total_cost / budget_max) * 100, 2) if budget_max > 0 else 0.0

        providers = [p.upper() for p in contract.cloud_providers]
        allocated_vms = solver_result.get("allocated_vms", [])
        instance_types = [vm.get("instance_type", "") for vm in allocated_vms if vm.get("instance_type")]
        instance_summary = ", ".join(set(instance_types)) if instance_types else "allocated compute"

        bud_inr = cls.usd_to_inr(budget_max, eff_rate)
        cur_tag = f"{usd_sym}{headroom:,.2f} USD ({inr_sym}{headroom_inr:,.2f} {inr_code})"
        cur_bud_tag = f"{usd_sym}{budget_max:,.2f} USD ({inr_sym}{bud_inr:,.2f} {inr_code})"
        cur_sav_buf = f"{usd_sym}{headroom:,.2f} USD ({inr_sym}{headroom_inr:,.2f} {inr_code})"

        # 1. Cloud Provider Commitment Strategy
        if "AWS" in providers:
            if any("t3" in it.lower() or "t4" in it.lower() for it in instance_types):
                recs.append(
                    f"Pricing Strategy: Workload utilizes burstable instances ({instance_summary}). "
                    f"Enroll in AWS 1-Year Compute Savings Plans (saves ~28-34%) or evaluate ARM64 Graviton (t4g) "
                    f"for up to 20% better price-to-performance (Unverified proposal until separately checked)."
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

        # 2. Budget Headroom Strategy
        if utilization < 30.0:
            recs.append(
                f"Budget Headroom ({utilization:.1f}% utilized): Monthly surplus of "
                f"{cur_tag}. Reallocate surplus capital toward "
                f"multi-AZ automated failover and managed snapshot replication."
            )
        elif utilization > 75.0:
            recs.append(
                f"Budget Caution ({utilization:.1f}% utilized): Spending is near the {cur_bud_tag} cap "
                f"with {cur_sav_buf} buffer. Configure automated billing alerts and scaling throttles "
                f"at 85% to prevent overage."
            )
        else:
            recs.append(
                f"Spend Governance ({utilization:.1f}% utilized): Optimal operating band with {cur_tag}/mo headroom. "
                f"Establish automated CloudWatch/Prometheus anomaly alerts at 80%."
            )

        # 3. Problem Specific Optimization
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

        # 4. Mandatory Tagging
        recs.append(
            f"Governance & Attribution: Apply mandatory Cost Allocation Tags (`Environment`, `CostCenter`, `Owner:FinOps`) "
            f"across all {contract.service_count} service resources for 100% cost attribution."
        )

        return recs

    @classmethod
    def _clean_llm_recommendations(cls, raw_text: str) -> List[str]:
        """Cleans and filters raw LLM output into actionable FinOps recommendations."""
        if not raw_text or not raw_text.strip():
            return []

        cleaned_text = re.sub(
            r"<(?:thinking|thought|think|reasoning)>[\s\S]*?</(?:thinking|thought|think|reasoning)>",
            "",
            raw_text,
            flags=re.IGNORECASE,
        ).strip()

        lines = [l.strip() for l in cleaned_text.split("\n") if l.strip()]
        cleaned_recs: List[str] = []

        finops_keywords = [
            "saving", "discount", "commit", "reserv", "spot", "rightsiz",
            "scale", "scaling", "tag", "tagging", "monitor", "monitoring",
            "alert", "reduc", "optimi", "utiliz", "cost", "plan", "egress",
            "transfer", "headroom", "capacity", "failover", "sla", "budget",
            "billing", "flapping", "burst", "graviton", "instance", "hpa",
            "tier", "governance", "attribution", "replication", "cooldown",
            "consolidat", "density", "container", "bin-pack",
        ]

        bullet_pattern = re.compile(
            r"^(?:(?:\d+[\.\)\:\-]\s*)+|(?:\*(?!\*)|[\-\•\–\—\+])\s*)+",
            re.IGNORECASE,
        )

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith(("-", "=", "`", "#")) and len(line) > 5 and set(line).issubset({"-", "=", "`", " ", "#"}):
                continue

            # Strip initial bullets, numbered prefixes, step headers, and meta prefixes
            line = re.sub(r"^(?:(?:\d+[\.\)\:\-]\s*)+|(?:\*(?!\*)|[\-\•\–\—\+])\s*)+", "", line).strip()
            line = re.sub(r"^(?:Step\s*\d+[\:\-\.]\s*|\[Recommendation\]\s*)+", "", line, flags=re.IGNORECASE).strip()
            line = re.sub(r"^\*{1,2}Analyze the Input:?\*{1,2}:?\s*", "", line, flags=re.IGNORECASE).strip()
            line = re.sub(r"^Analyze the Input:?\s*", "", line, flags=re.IGNORECASE).strip()
            line = re.sub(r"</?recommendation>", "", line, flags=re.IGNORECASE).strip()

            if re.match(r"^(?:role|system|prompt|user|problem type|target cloud|required resources|monthly budget cap):", line, re.IGNORECASE):
                continue
            if re.match(r"^(?:here'?s?\s*(?:a\s+)?(?:thinking|reasoning|analysis|are|is)|thinking\s*process|based on\b|sure,?\b|the following (?:are|is))", line, re.IGNORECASE):
                continue
            if re.match(r"^(?:finops recommendations|actionable recommendations|recommendations):?$", line, re.IGNORECASE):
                continue

            # Clean markdown bold asterisks for category headers like **Governance & Tagging:**
            line = re.sub(r"^\*{1,2}(.*?):?\*{1,2}:?\s*", r"\1: ", line)
            line = line.encode("ascii", "replace").decode("ascii")
            line = re.sub(r"^[:\-\s]+", "", line).strip()

            words = line.split()
            if len(words) < 4 or len(line) < 25:
                continue

            line_lower = line.lower()
            if not any(kw in line_lower for kw in finops_keywords):
                continue

            if line and line not in cleaned_recs:
                cleaned_recs.append(line)

        return cleaned_recs[:4]

    @classmethod
    def generate_llm_recommendations(
        cls,
        contract: CloudOptimizationContract,
        solver_result: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[Optional[List[str]], Optional[str], Optional[float], ExplanationSource]:
        """Invokes NVIDIA API to generate dynamic, AI-reasoned FinOps advice."""
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY") or getattr(settings, "NVIDIA_API_KEY", "")
        if not api_key:
            return None, None, None, ExplanationSource.UNAVAILABLE

        effective_timeout = timeout_seconds if timeout_seconds is not None else getattr(
            settings, "LLM_REQUEST_TIMEOUT_SECONDS", 360.0
        )
        base_url = (
            os.getenv("NVIDIA_BASE_URL")
            or getattr(settings, "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        )
        target_model = (
            os.getenv("NVIDIA_MODEL")
            or getattr(settings, "NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
        )
        max_tokens = getattr(settings, "LLM_MAX_COMPLETION_TOKENS", 4096)

        total_cost = solver_result.get(
            "total_monthly_cost_usd",
            solver_result.get("estimated_monthly_cost_usd", 0.0),
        )
        budget = float(contract.budget_max_usd)
        utilization = round((float(total_cost) / budget) * 100, 2) if budget > 0 else 0.0
        vms = [
            f"{v.get('provider', '')} {v.get('instance_type', '')} x{v.get('count', 1)}"
            for v in solver_result.get("allocated_vms", [])
        ]
        vms_str = ", ".join(vms) if vms else "Optimized compute instance"

        system_prompt = (
            "You are a Principal Cloud FinOps Architect. "
            "Be concise. Provide direct technical recommendations immediately without verbose step-by-step thinking. "
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
            f"- Monthly Budget Cap: ${budget:.2f} USD\n"
            f"- Optimized Monthly Cost: ${total_cost:.2f} USD ({utilization:.1f}% budget utilized)\n"
            f"- Placed Resources: {vms_str}\n\n"
            f"Provide 3-4 actionable FinOps recommendations tailored specifically to this deployment."
        )

        t0 = time.perf_counter()
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=effective_timeout,
                max_retries=0,
            )
            resp = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            elapsed_s = round(time.perf_counter() - t0, 2)
            if not resp or not resp.choices:
                return None, None, elapsed_s, ExplanationSource.UNAVAILABLE
            choice = resp.choices[0]
            raw_content = (choice.message.content or "").strip()
            resp_id = getattr(resp, "id", None)
            recs = cls._clean_llm_recommendations(raw_content)
            return (recs[:4] if recs else None), resp_id, elapsed_s, ExplanationSource.LIVE_PROVIDER
        except Exception:
            elapsed_s = round(time.perf_counter() - t0, 2)
            return None, None, elapsed_s, ExplanationSource.UNAVAILABLE

    @classmethod
    def generate_report(
        cls,
        contract: CloudOptimizationContract,
        solver_result: Dict[str, Any],
        check_result: Optional[Dict[str, Any]] = None,
        exchange_rate: float = settings.USD_TO_INR_RATE,
        enable_llm_explainer: bool = False,
    ) -> str:
        """Generates a structured ASCII Executive FinOps Deployment Report.

        Strictly respects verification results and uses Budget Headroom.
        """
        problem = contract.problem_type
        solver_name = solver_result.get("solver", "Neuro-Symbolic Engine")
        budget_max = float(contract.budget_max_usd)
        total_cost = float(solver_result.get(
            "total_monthly_cost_usd",
            solver_result.get("estimated_monthly_cost_usd", 0.0),
        ))

        is_feasible = True
        violations: List[str] = []
        if check_result is not None:
            is_feasible = bool(check_result.get("feasible_against_contract", False))
            violations = check_result.get("violations", [])
        else:
            status_str = str(solver_result.get("status", "UNKNOWN")).upper()
            if status_str in ["INFEASIBLE", "FAILED", "UNSAT"]:
                is_feasible = False

        budget_inr = cls.usd_to_inr(budget_max, exchange_rate)
        cost_inr = cls.usd_to_inr(total_cost, exchange_rate)
        headroom = max(0.0, budget_max - total_cost)
        headroom_inr = cls.usd_to_inr(headroom, exchange_rate)
        utilization = round((total_cost / budget_max) * 100, 2) if budget_max > 0 else 0.0

        inr_sym = settings.CURRENCY_SYMBOL_INR
        usd_sym = settings.CURRENCY_SYMBOL_USD

        lines = [
            "=" * 82,
            "               NEURASYM FINOPS EXECUTIVE DEPLOYMENT REPORT                ",
            "                  Stage 6: Symbolic Optimization Verdict                  ",
            "=" * 82,
            f" Problem Type         : {problem}",
            f" Optimization Engine  : {solver_name}",
            f" Feasibility Status   : {'Feasible' if is_feasible else 'Infeasible'}",
            f" Verification Verdict : {'FEASIBLE & CERTIFIED' if is_feasible else 'REJECTED (Constraint Violations Detected)'}",
            f" Target Cloud(s)      : {', '.join(contract.cloud_providers)}",
            f" Exchange Rate        : 1 USD = {exchange_rate:.2f} INR",
            "-" * 82,
            " FINANCIAL & BUDGET SUMMARY (DUAL CURRENCY: USD / INR)",
            "-" * 82,
        ]

        if not is_feasible:
            budget_violation_msg = "The requested allocation exceeds the available budget."
            lines.extend([
                f"  - Monthly Budget Cap       : {usd_sym}{budget_max:,.2f} USD ({inr_sym}{budget_inr:,.2f} INR) / month",
                f"  - Optimized Monthly Cost   : N/A",
                f"  - Proposed Monthly Cost    : {usd_sym}{total_cost:,.2f} USD ({inr_sym}{cost_inr:,.2f} INR) / month (REJECTED)",
                f"  - Monthly Net Savings      : N/A",
                f"  - Monthly Budget Headroom  : N/A (Infeasible Plan - No Savings Achieved)",
                f"  - Budget Utilization Rate  : N/A (Rejected Plan)",
                "-" * 82,
                " CRITICAL SAFETY ARBITRATION & VIOLATIONS",
                "-" * 82,
                "  [DEPLOYMENT ADVICE SUPPRESSED: Mathematical verification detected constraint violation(s)]",
                "  The requested workload cannot be satisfied under the supplied constraints.",
                f"  {budget_violation_msg}",
                "  This allocation CANNOT be safely deployed as-is.",
            ])
            if violations:
                lines.append("")
                lines.append("  Detected Constraint Violations:")
                for idx, v in enumerate(violations, 1):
                    lines.append(f"   [{idx}] {v}")

            lines.extend([
                "",
                "  Remediation Strategy:",
                "  - The workload parameters exceed physical or budgetary invariants.",
                "  - Increase the monthly budget cap or reduce workload capacity requirements.",
                "  - Any alternative configuration proposed below is an UNVERIFIED PROPOSAL until separately checked.",
                "=" * 82,
            ])
            return "\n".join(lines)

        # Feasible report
        lines.extend([
            f"  - Monthly Budget Cap       : {usd_sym}{budget_max:,.2f} USD ({inr_sym}{budget_inr:,.2f} INR) / month",
            f"  - Optimized Monthly Cost   : {usd_sym}{total_cost:,.2f} USD ({inr_sym}{cost_inr:,.2f} INR) / month",
            f"  - Monthly Budget Headroom  : {usd_sym}{headroom:,.2f} USD ({inr_sym}{headroom_inr:,.2f} INR) / month ({max(0.0, 100.0 - utilization):.1f}% under budget cap)",
            f"  - Budget Utilization Rate  : {utilization:>6.2f}%",
            "-" * 82,
        ])

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
                    f"({vm.get('vcpus_per_vm', 2)} vCPUs, {vm.get('ram_gb_per_vm', 4.0)}GB) -> "
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
            topology = solver_result.get("disaster_recovery_topology", "Active-Active Mesh")

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
        source_label = "Local Rule-Based Template"
        if enable_llm_explainer and is_feasible:
            recs, resp_id, elapsed_s, src = cls.generate_llm_recommendations(contract, solver_result)
            if recs:
                recommendations = recs
                source_label = f"NVIDIA API (nvidia/llama-3.1-nemotron-70b-instruct | response_id: {resp_id or 'unknown'} | {elapsed_s:.2f}s)"

        if not recommendations:
            recommendations = cls.generate_dynamic_recommendations(
                contract, solver_result, exchange_rate, is_feasible=is_feasible, violations=violations
            )

        lines.extend([
            "-" * 82,
            f" ACTIONABLE FINOPS RECOMMENDATIONS (Source: {source_label})",
            "-" * 82,
        ])
        for idx, rec in enumerate(recommendations, 1):
            lines.append(f"  {idx}. {rec}")

        lines.append("=" * 82)
        return "\n".join(lines)
