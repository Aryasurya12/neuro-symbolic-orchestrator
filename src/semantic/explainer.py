"""SEM-5: Stage 6 FinOps Report Generator & Natural Language Explainer."""

from typing import Any, Dict
from config.settings import settings
from src.semantic.schemas import CloudOptimizationContract


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
    def generate_report(
        cls,
        contract: CloudOptimizationContract,
        solver_result: Dict[str, Any],
        exchange_rate: float = settings.USD_TO_INR_RATE,
    ) -> str:
        """Generates a structured ASCII Executive FinOps Deployment Report

        displaying both USD and INR currency metrics.
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
        ]

        if status.upper() == "INFEASIBLE":
            lines.extend([
                f"  - Monthly Budget Cap       : {usd_sym}{budget_max:,.2f} USD ({inr_sym}{budget_inr:,.2f} INR) / month",
                f"  - Optimized Monthly Cost   : N/A",
                f"  - Monthly Net Savings      : N/A",
                f"  - Budget Utilization Rate  : N/A",
                "-" * 82,
            ])
            
            lines.append(" ACTIONABLE FINOPS RECOMMENDATIONS")
            lines.append("-" * 82)
            lines.append("  The requested workload cannot be satisfied under the supplied constraints.")
            lines.append("  Consider increasing the budget, reducing required resources, or relaxing")
            lines.append("  the relevant constraints.")
            
            if "error_message" in solver_result and solver_result["error_message"]:
                lines.append("")
                lines.append(f"  Reason: {solver_result['error_message']}")

            if "constraint_status" in solver_result:
                cstatus = solver_result["constraint_status"]
                failed = []
                if not cstatus.get("budget_ok", True): failed.append("The requested allocation exceeds the available budget.")
                if not cstatus.get("vcpu_ok", True): failed.append("Insufficient vCPUs available in target cloud/budget.")
                if not cstatus.get("ram_ok", True): failed.append("Insufficient RAM available in target cloud/budget.")
                if not cstatus.get("latency_ok", True): failed.append("Latency constraints cannot be met.")
                if not cstatus.get("sla_ok", True): failed.append("SLA availability constraints cannot be met.")
                
                if failed:
                    lines.append("")
                    lines.append("  Failed Constraints:")
                    for f in failed:
                        lines.append(f"  - {f}")

            lines.append("=" * 82)
            return "\n".join(lines)

        lines.extend([
            f"  - Monthly Budget Cap       : {usd_sym}{budget_max:,.2f} USD ({inr_sym}{budget_inr:,.2f} INR) / month",
            f"  - Optimized Monthly Cost   : {usd_sym}{total_cost:,.2f} USD ({inr_sym}{cost_inr:,.2f} INR) / month",
            f"  - Monthly Net Savings      : {usd_sym}{savings:,.2f} USD ({inr_sym}{savings_inr:,.2f} INR) / month ({max(0.0, 100.0 - utilization):.1f}% under cap)",
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

        lines.extend([
            "-" * 82,
            " ACTIONABLE FINOPS RECOMMENDATIONS",
            "-" * 82,
            "  1. Commit to 1-Year Savings Plans or Reserved Instances for 20-35% additional discount.",
            "  2. Establish CloudWatch/Prometheus anomaly alerts triggered at 85% budget utilization.",
            "  3. Schedule automated rightsizing reviews after 14 days of sustained metric observation.",
            "=" * 82,
        ])

        return "\n".join(lines)
