"""Standardized Development Query Manifest and Ground-Truth Label Definitions for Neurasym.

Defines a balanced 36-query development suite covering 5 query categories:
1. FORMAL_ENGLISH: Well-formed, explicit infrastructure requests in standard English.
2. COLLOQUIAL_HINGLISH: Natural conversational phrasing, Hindi-English code-mixing, informal slang.
3. MISSING_OR_AMBIGUOUS: Vague workloads lacking budgets, CPU counts, or traffic parameters (requires clarification).
4. CONFLICTING_CONSTRAINTS: Mathematically impossible requirements (e.g. 64 vCPUs under $15/mo).
5. UNSUPPORTED_TASKS: Out-of-domain requests (quantum computing, databases, kubernetes storage) requiring clean rejection.

All items in this development file are strictly labeled:
- annotation_source: 'HUMAN_DEVELOPMENT'
- approval_status: 'UNAPPROVED'
- is_approved: False
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ManifestQueryItem:
    """Single benchmark query entry with declared metadata and expected outcome."""
    query_id: str
    scenario_family_id: str
    query_text: str
    category: str  # FORMAL_ENGLISH, COLLOQUIAL_HINGLISH, MISSING_OR_AMBIGUOUS, CONFLICTING_CONSTRAINTS, UNSUPPORTED_TASKS
    intended_archetype: str  # ILP_VM_Allocation, PSO_Continuous_Scaling, Z3_Graph_Disaster_Recovery, NONE
    expected_outcome: str  # FEASIBLE, CLARIFICATION_REQUIRED, INFEASIBLE, UNSUPPORTED
    required_vcpus: Optional[int] = None
    required_ram_gb: Optional[float] = None
    target_bandwidth_mbps: Optional[float] = None
    min_bandwidth_mbps: Optional[float] = None
    target_cpu_pct: Optional[float] = None
    max_cpu_pct: Optional[float] = None
    latency_max_ms: Optional[float] = None
    sla_availability_pct: Optional[float] = None
    budget_max_usd: Optional[float] = None
    cloud_providers: List[str] = field(default_factory=lambda: ["AWS"])
    annotation_source: str = "HUMAN_DEVELOPMENT"
    approval_status: str = "UNAPPROVED"
    is_approved: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 36 Curated Development Queries (Unapproved Draft for Human Review)
DEVELOPMENT_QUERIES: List[ManifestQueryItem] = [
    # -------------------------------------------------------------------------
    # Scenario Family 1: Standard Compute Allocation (VM Knapsack)
    # -------------------------------------------------------------------------
    ManifestQueryItem(
        query_id="Q01_VM_FORMAL_BASIC",
        scenario_family_id="FAM_VM_01_COMPUTE",
        query_text="Deploy an application requiring 8 vCPUs and 16GB RAM for under $300 a month on AWS.",
        category="FORMAL_ENGLISH",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="FEASIBLE",
        required_vcpus=8,
        required_ram_gb=16.0,
        budget_max_usd=300.0,
        cloud_providers=["AWS"],
        notes="Standard formal VM knapsack request.",
    ),
    ManifestQueryItem(
        query_id="Q02_VM_HINGLISH_COMPUTE",
        scenario_family_id="FAM_VM_01_COMPUTE",
        query_text="Bhai AWS pe 8 vCPU aur 16GB RAM ka setup lagade 300 dollar per month ke andar.",
        category="COLLOQUIAL_HINGLISH",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="FEASIBLE",
        required_vcpus=8,
        required_ram_gb=16.0,
        budget_max_usd=300.0,
        cloud_providers=["AWS"],
        notes="Hinglish paraphrase of Q01.",
    ),
    ManifestQueryItem(
        query_id="Q03_VM_HIGH_MEM",
        scenario_family_id="FAM_VM_01_COMPUTE",
        query_text="We need high memory instances totaling 32GB RAM and 4 vCPUs on AWS with budget $250.",
        category="FORMAL_ENGLISH",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="FEASIBLE",
        required_vcpus=4,
        required_ram_gb=32.0,
        budget_max_usd=250.0,
        cloud_providers=["AWS"],
        notes="Memory-weighted VM selection.",
    ),
    ManifestQueryItem(
        query_id="Q04_VM_AZURE_GCP_MULTI",
        scenario_family_id="FAM_VM_01_COMPUTE",
        query_text="Provision 4 vCPUs and 16GB RAM across Azure or GCP under $150 monthly limit.",
        category="FORMAL_ENGLISH",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="FEASIBLE",
        required_vcpus=4,
        required_ram_gb=16.0,
        budget_max_usd=150.0,
        cloud_providers=["Azure", "GCP"],
        notes="Cross-cloud provider selection.",
    ),
    ManifestQueryItem(
        query_id="Q05_VM_INCOMPLETE_BUDGET",
        scenario_family_id="FAM_VM_01_COMPUTE",
        query_text="I want to run a web backend with 8 vCPUs and 16GB RAM on AWS.",
        category="MISSING_OR_AMBIGUOUS",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="CLARIFICATION_REQUIRED",
        required_vcpus=8,
        required_ram_gb=16.0,
        budget_max_usd=None,
        cloud_providers=["AWS"],
        notes="Missing budget constraint; requires confirmation or default policy.",
    ),
    ManifestQueryItem(
        query_id="Q06_VM_IMPOSSIBLE_BUDGET",
        scenario_family_id="FAM_VM_01_COMPUTE",
        query_text="Deploy a massive database requiring 64 vCPUs and 256GB RAM on AWS for under $10 a month.",
        category="CONFLICTING_CONSTRAINTS",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="INFEASIBLE",
        required_vcpus=64,
        required_ram_gb=256.0,
        budget_max_usd=10.0,
        cloud_providers=["AWS"],
        notes="Mathematically impossible budget.",
    ),

    # -------------------------------------------------------------------------
    # Scenario Family 2: Dynamic Continuous Scaling
    # -------------------------------------------------------------------------
    ManifestQueryItem(
        query_id="Q07_SCALE_FORMAL_BASIC",
        scenario_family_id="FAM_SCALE_02_TRAFFIC",
        query_text="Continuous dynamic scaling with target CPU 70% and offered traffic 150 Mbps under $1500 monthly budget.",
        category="FORMAL_ENGLISH",
        intended_archetype="PSO_Continuous_Scaling",
        expected_outcome="FEASIBLE",
        target_bandwidth_mbps=150.0,
        min_bandwidth_mbps=150.0,
        target_cpu_pct=70.0,
        budget_max_usd=1500.0,
        notes="Standard continuous scaling request.",
    ),
    ManifestQueryItem(
        query_id="Q08_SCALE_HINGLISH",
        scenario_family_id="FAM_SCALE_02_TRAFFIC",
        query_text="Traffic 200 Mbps tak aayega, CPU utilization 70% maintain rakhna hai, budget 1500 USD per month hai.",
        category="COLLOQUIAL_HINGLISH",
        intended_archetype="PSO_Continuous_Scaling",
        expected_outcome="FEASIBLE",
        target_bandwidth_mbps=200.0,
        min_bandwidth_mbps=200.0,
        target_cpu_pct=70.0,
        budget_max_usd=1500.0,
        notes="Hinglish scaling request.",
    ),
    ManifestQueryItem(
        query_id="Q09_SCALE_HARD_CEILING",
        scenario_family_id="FAM_SCALE_02_TRAFFIC",
        query_text="Dynamic scaling for 300 Mbps peak workload with target CPU 70% and strict maximum CPU ceiling of 60% under $1500.",
        category="FORMAL_ENGLISH",
        intended_archetype="PSO_Continuous_Scaling",
        expected_outcome="FEASIBLE",
        target_bandwidth_mbps=300.0,
        min_bandwidth_mbps=300.0,
        target_cpu_pct=70.0,
        max_cpu_pct=60.0,
        budget_max_usd=1500.0,
        notes="Explicit hard ceiling lower than target nominal point.",
    ),
    ManifestQueryItem(
        query_id="Q10_SCALE_OVERLOAD_PREVENT",
        scenario_family_id="FAM_SCALE_02_TRAFFIC",
        query_text="Continuous auto-scaling for 100 Mbps traffic stream under $1000 budget with 70% target utilization.",
        category="FORMAL_ENGLISH",
        intended_archetype="PSO_Continuous_Scaling",
        expected_outcome="FEASIBLE",
        target_bandwidth_mbps=100.0,
        min_bandwidth_mbps=100.0,
        target_cpu_pct=70.0,
        budget_max_usd=1000.0,
        notes="Tests that solver provisions at least 2 replicas (133.3% overload prevented).",
    ),
    ManifestQueryItem(
        query_id="Q11_SCALE_MISSING_BW",
        scenario_family_id="FAM_SCALE_02_TRAFFIC",
        query_text="Configure continuous auto-scaling to maintain 70% target CPU on AWS under $1200.",
        category="MISSING_OR_AMBIGUOUS",
        intended_archetype="PSO_Continuous_Scaling",
        expected_outcome="CLARIFICATION_REQUIRED",
        target_bandwidth_mbps=None,
        target_cpu_pct=70.0,
        budget_max_usd=1200.0,
        notes="Missing offered bandwidth traffic parameter.",
    ),
    ManifestQueryItem(
        query_id="Q12_SCALE_IMPOSSIBLE_BUDGET",
        scenario_family_id="FAM_SCALE_02_TRAFFIC",
        query_text="Scale continuous replicas to handle 800 Mbps traffic under strict $30 a month budget cap.",
        category="CONFLICTING_CONSTRAINTS",
        intended_archetype="PSO_Continuous_Scaling",
        expected_outcome="INFEASIBLE",
        target_bandwidth_mbps=800.0,
        min_bandwidth_mbps=800.0,
        budget_max_usd=30.0,
        notes="Impossible scaling budget.",
    ),

    # -------------------------------------------------------------------------
    # Scenario Family 3: Multi-Region Disaster Recovery (Z3 Graph SMT)
    # -------------------------------------------------------------------------
    ManifestQueryItem(
        query_id="Q13_DR_FORMAL_AWS_GCP",
        scenario_family_id="FAM_DR_03_REGION",
        query_text="We need disaster recovery setup across AWS and GCP with 99.99% SLA and max 50ms latency under $600/month.",
        category="FORMAL_ENGLISH",
        intended_archetype="Z3_Graph_Disaster_Recovery",
        expected_outcome="FEASIBLE",
        latency_max_ms=50.0,
        sla_availability_pct=99.99,
        budget_max_usd=600.0,
        cloud_providers=["AWS", "GCP"],
        notes="Standard multi-cloud DR between AWS and GCP.",
    ),
    ManifestQueryItem(
        query_id="Q14_DR_HINGLISH",
        scenario_family_id="FAM_DR_03_REGION",
        query_text="Disaster recovery plan banana hai AWS aur Azure ke beech, latency 50ms se kam aur 99.99% availability under 800 USD.",
        category="COLLOQUIAL_HINGLISH",
        intended_archetype="Z3_Graph_Disaster_Recovery",
        expected_outcome="FEASIBLE",
        latency_max_ms=50.0,
        sla_availability_pct=99.99,
        budget_max_usd=800.0,
        cloud_providers=["AWS", "Azure"],
        notes="Hinglish DR request.",
    ),
    ManifestQueryItem(
        query_id="Q15_DR_EXPLICIT_REGIONS",
        scenario_family_id="FAM_DR_03_REGION",
        query_text="Topological disaster recovery between us-east-1 and us-west-2 with max 70ms latency and $850 budget.",
        category="FORMAL_ENGLISH",
        intended_archetype="Z3_Graph_Disaster_Recovery",
        expected_outcome="FEASIBLE",
        latency_max_ms=70.0,
        sla_availability_pct=99.95,
        budget_max_usd=850.0,
        cloud_providers=["AWS"],
        notes="Single-provider dual-region DR.",
    ),
    ManifestQueryItem(
        query_id="Q16_DR_TIGHT_LATENCY",
        scenario_family_id="FAM_DR_03_REGION",
        query_text="Disaster recovery across independent failure domains with 99.99% SLA and ultra-low latency <= 20ms under $500.",
        category="FORMAL_ENGLISH",
        intended_archetype="Z3_Graph_Disaster_Recovery",
        expected_outcome="FEASIBLE",
        latency_max_ms=20.0,
        sla_availability_pct=99.99,
        budget_max_usd=500.0,
        cloud_providers=["AWS", "Azure"],
        notes="Requires us-east-1 to eastus link (12ms).",
    ),
    ManifestQueryItem(
        query_id="Q17_DR_MISSING_PROVIDERS",
        scenario_family_id="FAM_DR_03_REGION",
        query_text="Design a disaster recovery topology with 99.99% composite availability.",
        category="MISSING_OR_AMBIGUOUS",
        intended_archetype="Z3_Graph_Disaster_Recovery",
        expected_outcome="CLARIFICATION_REQUIRED",
        latency_max_ms=None,
        sla_availability_pct=99.99,
        budget_max_usd=None,
        notes="Missing latency tolerance and budget limits.",
    ),
    ManifestQueryItem(
        query_id="Q18_DR_IMPOSSIBLE_LATENCY",
        scenario_family_id="FAM_DR_03_REGION",
        query_text="Multi-region DR between US East and Europe with max 5ms latency under $1000.",
        category="CONFLICTING_CONSTRAINTS",
        intended_archetype="Z3_Graph_Disaster_Recovery",
        expected_outcome="INFEASIBLE",
        latency_max_ms=5.0,
        sla_availability_pct=99.99,
        budget_max_usd=1000.0,
        notes="Physically impossible speed-of-light inter-continental latency (<5ms).",
    ),

    # -------------------------------------------------------------------------
    # Scenario Family 4: Unsupported Out-of-Domain Tasks
    # -------------------------------------------------------------------------
    ManifestQueryItem(
        query_id="Q19_UNSUPPORTED_QUANTUM",
        scenario_family_id="FAM_OUT_04_UNSUPPORTED",
        query_text="Optimize quantum circuit depth using tensor network state contraction algorithms.",
        category="UNSUPPORTED_TASKS",
        intended_archetype="NONE",
        expected_outcome="UNSUPPORTED",
        notes="Quantum computing domain is out of scope.",
    ),
    ManifestQueryItem(
        query_id="Q20_UNSUPPORTED_SQL",
        scenario_family_id="FAM_OUT_04_UNSUPPORTED",
        query_text="Write a PostgreSQL query with recursive CTE to compute graph reachability in customer tables.",
        category="UNSUPPORTED_TASKS",
        intended_archetype="NONE",
        expected_outcome="UNSUPPORTED",
        notes="SQL query synthesis is out of scope.",
    ),
    ManifestQueryItem(
        query_id="Q21_UNSUPPORTED_K8S_STORAGE",
        scenario_family_id="FAM_OUT_04_UNSUPPORTED",
        query_text="Configure Kubernetes PersistentVolumeClaim with ReadWriteMany Ceph CSI driver.",
        category="UNSUPPORTED_TASKS",
        intended_archetype="NONE",
        expected_outcome="UNSUPPORTED",
        notes="Kubernetes storage manifests are out of scope.",
    ),
    ManifestQueryItem(
        query_id="Q22_UNSUPPORTED_HINGLISH_CHITCHAT",
        scenario_family_id="FAM_OUT_04_UNSUPPORTED",
        query_text="Kal ka cricket match kaisa tha aur Virat Kohli ne kitne run banaye?",
        category="UNSUPPORTED_TASKS",
        intended_archetype="NONE",
        expected_outcome="UNSUPPORTED",
        notes="Chitchat / sports trivia is out of scope.",
    ),
    ManifestQueryItem(
        query_id="Q23_UNSUPPORTED_IMAGE_GEN",
        scenario_family_id="FAM_OUT_04_UNSUPPORTED",
        query_text="Generate a photorealistic image of a futuristic cloud datacenter in neon cyberpunk style.",
        category="UNSUPPORTED_TASKS",
        intended_archetype="NONE",
        expected_outcome="UNSUPPORTED",
        notes="Image generation is out of scope.",
    ),
    ManifestQueryItem(
        query_id="Q24_UNSUPPORTED_CODE_TRANSLATION",
        scenario_family_id="FAM_OUT_04_UNSUPPORTED",
        query_text="Translate this C++20 coroutine template into equivalent Rust async/await code.",
        category="UNSUPPORTED_TASKS",
        intended_archetype="NONE",
        expected_outcome="UNSUPPORTED",
        notes="Code translation is out of scope.",
    ),

    # -------------------------------------------------------------------------
    # Scenario Family 5: Extended Variational & Paraphrase Suite
    # -------------------------------------------------------------------------
    ManifestQueryItem(
        query_id="Q25_VM_INR_CURRENCY",
        scenario_family_id="FAM_VM_05_EXTENDED",
        query_text="Deploy 4 vCPUs and 8GB RAM on AWS with monthly budget Rs 10000.",
        category="COLLOQUIAL_HINGLISH",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="FEASIBLE",
        required_vcpus=4,
        required_ram_gb=8.0,
        budget_max_usd=120.0,
        cloud_providers=["AWS"],
        notes="Rupee currency conversion (Rs 10,000 ~ $120.00).",
    ),
    ManifestQueryItem(
        query_id="Q26_VM_STRICT_AWS_AZURE",
        scenario_family_id="FAM_VM_05_EXTENDED",
        query_text="Allocate 16 vCPUs and 32GB RAM strictly on AWS or Azure under $400/month.",
        category="FORMAL_ENGLISH",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="FEASIBLE",
        required_vcpus=16,
        required_ram_gb=32.0,
        budget_max_usd=400.0,
        cloud_providers=["AWS", "Azure"],
        notes="Multi-node VM allocation.",
    ),
    ManifestQueryItem(
        query_id="Q27_VM_COLLOQUIAL_CHEAP",
        scenario_family_id="FAM_VM_05_EXTENDED",
        query_text="Need cheap VMs for small microservices: 2 vCPUs and 4GB RAM on AWS, budget $50.",
        category="COLLOQUIAL_HINGLISH",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="FEASIBLE",
        required_vcpus=2,
        required_ram_gb=4.0,
        budget_max_usd=50.0,
        cloud_providers=["AWS"],
        notes="Single t3.medium instance matches perfectly.",
    ),
    ManifestQueryItem(
        query_id="Q28_SCALE_COLLOQUIAL_SPIKE",
        scenario_family_id="FAM_SCALE_05_EXTENDED",
        query_text="Handle continuous traffic load of 250 Mbps with 70% CPU target within $1200 monthly spend.",
        category="COLLOQUIAL_HINGLISH",
        intended_archetype="PSO_Continuous_Scaling",
        expected_outcome="FEASIBLE",
        target_bandwidth_mbps=250.0,
        min_bandwidth_mbps=250.0,
        target_cpu_pct=70.0,
        budget_max_usd=1200.0,
        notes="Requires 4 replicas ($200.00/mo).",
    ),
    ManifestQueryItem(
        query_id="Q29_SCALE_MAX_BANDWIDTH",
        scenario_family_id="FAM_SCALE_05_EXTENDED",
        query_text="Continuous scaling up to 1000 Mbps maximum bandwidth with target CPU 70% under $1800.",
        category="FORMAL_ENGLISH",
        intended_archetype="PSO_Continuous_Scaling",
        expected_outcome="FEASIBLE",
        target_bandwidth_mbps=1000.0,
        min_bandwidth_mbps=1000.0,
        target_cpu_pct=70.0,
        budget_max_usd=1800.0,
        notes="Boundary test at 1000 Mbps.",
    ),
    ManifestQueryItem(
        query_id="Q30_DR_THREE_REGIONS_PROMPT",
        scenario_family_id="FAM_DR_05_EXTENDED",
        query_text="Set up active-passive failover between US East and US Central on AWS and GCP with 99.95% availability under $400.",
        category="FORMAL_ENGLISH",
        intended_archetype="Z3_Graph_Disaster_Recovery",
        expected_outcome="FEASIBLE",
        latency_max_ms=50.0,
        sla_availability_pct=99.95,
        budget_max_usd=400.0,
        cloud_providers=["AWS", "GCP"],
        notes="Selects us-east-1 and us-central1 (32ms latency).",
    ),
    ManifestQueryItem(
        query_id="Q31_DR_ZERO_LATENCY_IMPOSSIBLE",
        scenario_family_id="FAM_DR_05_EXTENDED",
        query_text="Disaster recovery across independent data centers with zero latency (0ms) and 100% availability.",
        category="CONFLICTING_CONSTRAINTS",
        intended_archetype="Z3_Graph_Disaster_Recovery",
        expected_outcome="INFEASIBLE",
        latency_max_ms=0.0,
        sla_availability_pct=100.0,
        budget_max_usd=5000.0,
        notes="0ms latency and 100% SLA are physically impossible.",
    ),
    ManifestQueryItem(
        query_id="Q32_VM_CONFLICTING_VCPU_RAM",
        scenario_family_id="FAM_VM_05_EXTENDED",
        query_text="Allocate 128 vCPUs and 512GB RAM for $20 a month total.",
        category="CONFLICTING_CONSTRAINTS",
        intended_archetype="ILP_VM_Allocation",
        expected_outcome="INFEASIBLE",
        required_vcpus=128,
        required_ram_gb=512.0,
        budget_max_usd=20.0,
        notes="Impossible compute scale under $20 budget.",
    ),
    ManifestQueryItem(
        query_id="Q33_AMBIGUOUS_CLOUD_SETUP",
        scenario_family_id="FAM_MISC_05_EXTENDED",
        query_text="Can you please optimize my cloud infrastructure to be faster and cheaper?",
        category="MISSING_OR_AMBIGUOUS",
        intended_archetype="NONE",
        expected_outcome="CLARIFICATION_REQUIRED",
        notes="Completely unspecified parameters; requires clarification interview.",
    ),
    ManifestQueryItem(
        query_id="Q34_AMBIGUOUS_SCALE_REPLICAS",
        scenario_family_id="FAM_SCALE_05_EXTENDED",
        query_text="Scale my replicas up when traffic increases.",
        category="MISSING_OR_AMBIGUOUS",
        intended_archetype="PSO_Continuous_Scaling",
        expected_outcome="CLARIFICATION_REQUIRED",
        notes="Missing target metric, bandwidth, and budget.",
    ),
    ManifestQueryItem(
        query_id="Q35_UNSUPPORTED_DNS_SETUP",
        scenario_family_id="FAM_OUT_05_EXTENDED",
        query_text="Configure Route53 DNS weighted routing policy with health checks for Apex record.",
        category="UNSUPPORTED_TASKS",
        intended_archetype="NONE",
        expected_outcome="UNSUPPORTED",
        notes="DNS records management is outside optimization scope.",
    ),
    ManifestQueryItem(
        query_id="Q36_UNSUPPORTED_AUTH0_OAUTH",
        scenario_family_id="FAM_OUT_05_EXTENDED",
        query_text="Integrate Auth0 OAuth2 PKCE flow in React frontend with refresh token rotation.",
        category="UNSUPPORTED_TASKS",
        intended_archetype="NONE",
        expected_outcome="UNSUPPORTED",
        notes="Authentication flow configuration is out of scope.",
    ),
]


class ManifestManager:
    """Manages reading, writing, and validating benchmark query manifests."""

    @classmethod
    def get_development_manifest(cls) -> List[Dict[str, Any]]:
        """Returns the standard 36 development queries with unapproved status."""
        return [q.to_dict() for q in DEVELOPMENT_QUERIES]

    @classmethod
    def export_development_manifest_json(cls, output_path: str = "data/development_query_manifest.json") -> str:
        """Saves development queries to JSON file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = cls.get_development_manifest()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path

    @classmethod
    def export_sanitized_specification_package(
        cls,
        json_output_path: str = "audit_materials/sanitized_specification_package.json",
        md_output_path: str = "audit_materials/sanitized_specification_package.md",
    ) -> Dict[str, Any]:
        """Exports the complete, sanitized specification package for LLM benchmark dataset authoring.
        
        Contains:
        1. Actual current resource catalog (VM SKUs, specs, pricing).
        2. Disaster recovery topology graph, regional costs, and availability SLA assumptions.
        3. Dynamic continuous scaling formulas, constants, and replica capacity models.
        4. Supported contract fields and schemas.
        5. Default fallback and clarification policies.
        6. Solver optimization limits and verifier tolerances.
        """
        import os
        from src.verifiers.independent_checker import IndependentChecker

        sku_catalog = IndependentChecker.get_sku_catalog()
        regions_graph = IndependentChecker.get_regions_graph()

        # Format SKU Catalog with monthly costs (730 hrs/month)
        catalog_specs = {}
        for sku, meta in sku_catalog.items():
            hr_cost = meta["hourly_cost_usd"]
            mo_cost = round(hr_cost * IndependentChecker.HOURS_PER_MONTH, 2)
            catalog_specs[sku] = {
                "provider": meta["provider"],
                "vcpus": meta["vcpus"],
                "ram_gb": meta["ram_gb"],
                "hourly_cost_usd": hr_cost,
                "monthly_cost_usd": mo_cost,
            }

        # DR Infrastructure Specifications
        dr_specs = {
            "regions": regions_graph,
            "cost_constants": {
                "latency_cost_per_ms_monthly_usd": IndependentChecker.LATENCY_COST_PER_MS,
            },
            "formulas": {
                "monthly_cost_usd": "base_cost(region_a) + base_cost(region_b) + (peer_latency_ms * 0.25)",
                "composite_availability_sla": "1.0 - ((1.0 - (sla_a / 100.0)) * (1.0 - (sla_b / 100.0)))",
            },
            "constraints": {
                "geographic_isolation": "geo(region_a) != geo(region_b) (Must be distinct geographical regions)",
                "distinct_nodes": "region_a != region_b",
            },
        }

        # Continuous Scaling Specifications
        scaling_specs = {
            "operational_bounds": {
                "bandwidth_min_mbps": 100.0,
                "bandwidth_max_mbps": 1000.0,
                "replicas_min": 1,
                "replicas_max": 16,
            },
            "cost_constants": {
                "cost_per_mbps_monthly_usd": IndependentChecker.SCALING_COST_PER_MBPS,
                "cost_per_replica_monthly_usd": IndependentChecker.SCALING_COST_PER_REPLICA,
                "capacity_factor_mbps_per_replica": IndependentChecker.SCALING_CAPACITY_FACTOR_MBPS,
            },
            "formulas": {
                "monthly_cost_usd": "(bandwidth_mbps * 0.08) + (replicas * 45.00)",
                "modeled_cpu_utilization_pct": "(bandwidth_mbps / (replicas * 75.0)) * 100.0",
            },
            "constraints": {
                "physical_saturation_limit": "<= 100.0% CPU",
                "custom_max_ceiling": "<= max_cpu_pct (if specified in contract)",
            },
        }

        # Supported Contract Schemas
        contract_schemas = {
            "ILP_VM_Allocation": {
                "problem_type": "ILP_VM_Allocation",
                "required_fields": ["required_vcpus", "required_ram_gb", "budget_max_usd"],
                "optional_fields": ["cloud_providers (default: ['AWS'])"],
                "description": "Discrete knapsack VM selection satisfying compute and memory under financial budget.",
            },
            "Z3_Graph_Disaster_Recovery": {
                "problem_type": "Z3_Graph_Disaster_Recovery",
                "required_fields": ["latency_max_ms", "sla_availability_pct", "budget_max_usd"],
                "optional_fields": [],
                "description": "Multi-region disaster recovery deployment satisfying inter-region sync latency, composite SLA, and budget.",
            },
            "PSO_Continuous_Scaling": {
                "problem_type": "PSO_Continuous_Scaling",
                "required_fields": ["target_bandwidth_mbps (or min_bandwidth_mbps)", "target_cpu_pct", "budget_max_usd"],
                "optional_fields": ["max_cpu_pct"],
                "description": "Continuous dynamic scaling determining optimal bandwidth allocation and worker replicas.",
            },
        }

        # Policy Definitions
        policies = {
            "missing_budget_policy": "Flag query with expected_outcome='CLARIFICATION_REQUIRED' when budget is omitted or ambiguous.",
            "unsupported_domain_policy": "Flag query with expected_outcome='UNSUPPORTED' for out-of-domain requests (e.g. quantum computing, DB administration, DNS routing, Auth0).",
            "conflicting_constraints_policy": "Flag query with expected_outcome='INFEASIBLE' when requested parameters exceed mathematical or physical catalog capacity.",
            "provider_default_policy": "If no cloud provider is specified for VM allocations, default to ['AWS'].",
        }

        # Verifier & Solver Limits
        limits_and_tolerances = {
            "solver_timeouts_sec": 10.0,
            "cost_tolerance_usd": 0.50,
            "sla_tolerance_pct": 0.0001,
            "latency_tolerance_ms": 0.0,
            "hours_per_month_standard": IndependentChecker.HOURS_PER_MONTH,
        }

        package: Dict[str, Any] = {
            "package_title": "NEURASYM Ground-Truth Specification & Authoring Guide",
            "version": "3.0.0-auditable",
            "status": "APPROVED_SPECIFICATION_FOR_PROMPT_AUTHORING",
            "resource_catalog_vm_skus": catalog_specs,
            "disaster_recovery_infrastructure": dr_specs,
            "continuous_dynamic_scaling": scaling_specs,
            "supported_contract_schemas": contract_schemas,
            "authoring_and_evaluation_policies": policies,
            "verifier_limits_and_tolerances": limits_and_tolerances,
            "development_queries_count": len(DEVELOPMENT_QUERIES),
            "sample_development_queries": [q.to_dict() for q in DEVELOPMENT_QUERIES[:5]],
        }

        # Export JSON
        os.makedirs(os.path.dirname(os.path.abspath(json_output_path)), exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(package, f, indent=2, ensure_ascii=False)

        # Export Markdown
        os.makedirs(os.path.dirname(os.path.abspath(md_output_path)), exist_ok=True)
        md_content = f"""# NEURASYM Ground-Truth Specification & Authoring Guide
**Version:** 3.0.0-auditable  
**Purpose:** Canonical reference package for generating prompt datasets and ground-truth evaluation labels.

---

## 1. Cloud VM SKU Catalog (Monthly Calculated at {IndependentChecker.HOURS_PER_MONTH} hrs/mo)
| SKU Name | Provider | vCPUs | RAM (GB) | Hourly Price ($) | Monthly Price ($) |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for sku, meta in catalog_specs.items():
            md_content += f"| `{sku}` | {meta['provider']} | {meta['vcpus']} | {meta['ram_gb']} GB | ${meta['hourly_cost_usd']:.4f} | ${meta['monthly_cost_usd']:.2f} |\n"

        md_content += f"""
---

## 2. Multi-Region Disaster Recovery Topology
- **Latency Cost Rate:** ${IndependentChecker.LATENCY_COST_PER_MS:.2f} USD per ms per month
- **Composite Availability SLA Formula:** `1 - ((1 - SLA_A/100) * (1 - SLA_B/100))`
- **Cost Formula:** `Cost = BaseCost(A) + BaseCost(B) + (Latency(A,B) * $0.25)`
- **Constraint:** Primary and Secondary regions must be in distinct geographical domains (`Geo(A) != Geo(B)`).

### Regional Nodes:
"""
        for reg_id, node in regions_graph.items():
            md_content += f"- **`{reg_id}`** ({node['provider']}, Geo: `{node['geo']}`): Base Cost = ${node['base_cost_usd']:.2f}/mo, Single Region SLA = {node['sla_pct']}%\n"
            md_content += f"  - Latencies: {node['peer_latencies_ms']}\n"

        md_content += f"""
---

## 3. Dynamic Continuous Scaling (PSO)
- **Bandwidth Domain:** [{scaling_specs['operational_bounds']['bandwidth_min_mbps']}, {scaling_specs['operational_bounds']['bandwidth_max_mbps']}] Mbps
- **Worker Replicas Domain:** [{scaling_specs['operational_bounds']['replicas_min']}, {scaling_specs['operational_bounds']['replicas_max']}] Replicas
- **Pricing Formula:** `Cost = (Bandwidth * $0.08) + (Replicas * $45.00)`
- **Capacity Model:** `Modeled CPU = (Bandwidth / (Replicas * 75.0 Mbps)) * 100%`
- **Saturation Limit:** Maximum hard ceiling $\\le 100.0\\%$.

---

## 4. Supported Problem Archetypes & Contract Schemas
1. **`ILP_VM_Allocation`**:
   - Required: `required_vcpus` (int), `required_ram_gb` (float), `budget_max_usd` (float)
   - Optional: `cloud_providers` (list of str, defaults to `["AWS"]`)
2. **`Z3_Graph_Disaster_Recovery`**:
   - Required: `latency_max_ms` (float), `sla_availability_pct` (float), `budget_max_usd` (float)
3. **`PSO_Continuous_Scaling`**:
   - Required: `target_bandwidth_mbps` (float), `target_cpu_pct` (float), `budget_max_usd` (float)
   - Optional: `max_cpu_pct` (float ceiling)

---

## 5. Evaluation & Authoring Policies
- **Missing / Ambiguous Requirements:** If critical constraints (e.g. budget) are omitted, the query requires clarification (`expected_outcome="CLARIFICATION_REQUIRED"`).
- **Conflicting / Impossible Constraints:** If requirements cannot be satisfied under the physical catalog (e.g. 64 vCPUs for $10/mo), the query must be proven infeasible (`expected_outcome="INFEASIBLE"`).
- **Out-of-Scope / Unsupported Tasks:** Requests involving unsupported domains (e.g. quantum computing, DNS configuration, OAuth integration) must be rejected cleanly (`expected_outcome="UNSUPPORTED"`).
- **Development Queries Status:** All development queries remain marked `UNAPPROVED` with `annotation_source="HUMAN_DEVELOPMENT"` and `is_approved=False`.
"""
        with open(md_output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return package
