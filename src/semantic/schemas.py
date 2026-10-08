from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from config.settings import settings


class InterpretationOutcome(str, Enum):
    READY = "ready"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"
    CONFLICTING_REQUIREMENTS = "conflicting_requirements"


class CloudOptimizationContract(BaseModel):
    """SEM-4: Strict Pydantic V2 Safety Contract governing the handoff from
    natural-language intent understanding (or neural interpretation) to symbolic optimization engines.
    """

    model_config = ConfigDict(extra="forbid")

    problem_type: Literal[
        "ILP_VM_Allocation",
        "PSO_Continuous_Scaling",
        "Z3_Graph_Disaster_Recovery",
    ] = Field(
        ...,
        description="Target optimization problem type",
    )
    cloud_providers: List[Literal["AWS", "Azure", "GCP"]] = Field(
        default=["AWS"],
        description="Target cloud providers for resource placement",
    )
    budget_max_usd: float = Field(
        default=settings.DEFAULT_BUDGET_USD,
        gt=0,
        description="Maximum monthly budget in USD (must be >= $10.00)",
    )
    service_count: int = Field(
        default=1,
        gt=0,
        description="Number of services or application components to allocate",
    )
    required_vcpus: int = Field(
        default=1,
        gt=0,
        description="Total minimum vCPUs required across allocated instances",
    )
    required_ram_gb: float = Field(
        default=1.0,
        gt=0,
        description="Total minimum RAM in Gigabytes required",
    )
    latency_max_ms: float = Field(
        default=100.0,
        gt=0,
        le=1000.0,
        description="Maximum tolerable network latency in milliseconds",
    )
    sla_availability_pct: float = Field(
        default=99.9,
        ge=90.0,
        le=99.999,
        description="Target SLA availability percentage",
    )

    # Continuous Dynamic Scaling specific bounds
    target_bandwidth_mbps: Optional[float] = Field(
        default=None,
        ge=0,
        description="Target continuous network throughput bandwidth in Mbps",
    )
    min_bandwidth_mbps: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum bandwidth threshold in Mbps",
    )
    max_bandwidth_mbps: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum bandwidth threshold in Mbps",
    )
    min_replicas: Optional[int] = Field(
        default=None,
        ge=1,
        le=64,
        description="Minimum worker replica count bound",
    )
    max_replicas: Optional[int] = Field(
        default=None,
        ge=1,
        le=64,
        description="Maximum worker replica count bound",
    )
    target_replicas: Optional[int] = Field(
        default=None,
        ge=1,
        description="Target worker replica count",
    )
    target_cpu_pct: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=100.0,
        description="Target nominal CPU utilization percentage (e.g. 70.0%)",
    )
    max_cpu_pct: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=100.0,
        description="Hard maximum CPU utilization ceiling (e.g. 70.0% or 100.0%)",
    )

    # Multi-Region Disaster Recovery placement parameters
    allowed_regions: Optional[List[str]] = Field(
        default=None,
        description="Explicit allowed cloud regions for placement",
    )
    primary_region: Optional[str] = Field(
        default=None,
        description="User-specified primary region if fixed",
    )
    secondary_region: Optional[str] = Field(
        default=None,
        description="User-specified secondary/failover region if fixed",
    )
    require_multi_region: Optional[bool] = Field(
        default=None,
        description="Whether placement requires distinct geographic regions",
    )
    require_multi_cloud: Optional[bool] = Field(
        default=None,
        description="Whether placement requires distinct cloud providers (e.g. AWS + GCP)",
    )

    # VM Instance Constraints
    instance_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Exact required instance count if specified",
    )
    min_instance_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum instance count threshold",
    )
    vcpus_per_instance: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum vCPUs required per individual instance",
    )
    ram_gb_per_instance: Optional[float] = Field(
        default=None,
        ge=0.5,
        description="Minimum RAM in GB required per individual instance",
    )

    # Structured interpretation outcome and justification metadata
    interpretation_outcome: Literal[
        "ready",
        "needs_clarification",
        "unsupported",
        "conflicting_requirements",
    ] = Field(
        default="ready",
        description="Classification of requirements completeness and feasibility",
    )
    clarification_questions: List[str] = Field(
        default_factory=list,
        description="Questions for user when outcome is needs_clarification",
    )
    unsupported_reasons: List[str] = Field(
        default_factory=list,
        description="Explanation when request is unsupported",
    )
    conflicting_reasons: List[str] = Field(
        default_factory=list,
        description="Explanation when requirements contradict each other",
    )
    extracted_spans: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Exact source text substrings justifying extracted fields",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Semantic and qualitative intent metadata",
    )

    @field_validator("latency_max_ms", mode="before")
    @classmethod
    def validate_latency(cls, v: Any) -> float:
        if v is None:
            return 100.0
        return float(v)

    @field_validator("sla_availability_pct", mode="before")
    @classmethod
    def validate_sla(cls, v: Any) -> float:
        if v is None:
            return 99.9
        return float(v)

    @field_validator("service_count", mode="before")
    @classmethod
    def validate_service_count(cls, v: Any) -> int:
        if v is None:
            return 1
        return int(v)

    @field_validator("required_vcpus", mode="before")
    @classmethod
    def validate_vcpus(cls, v: Any) -> int:
        if v is None:
            return 1
        return int(v)

    @field_validator("required_ram_gb", mode="before")
    @classmethod
    def validate_ram(cls, v: Any) -> float:
        if v is None:
            return 1.0
        return float(v)

    @field_validator("extracted_spans", mode="before")
    @classmethod
    def clean_extracted_spans(cls, v: Any) -> Dict[str, Optional[str]]:
        if isinstance(v, dict):
            return {str(k): (str(val) if val is not None else None) for k, val in v.items()}
        return {}

    @field_validator("budget_max_usd")
    @classmethod
    def validate_budget_min_threshold(cls, v: float) -> float:
        """Enforces minimum viable cloud budget of $10.00."""
        if v < settings.MIN_VIABLE_BUDGET_USD:
            raise ValueError(
                f"budget_max_usd (${v:.2f}) must be at least "
                f"${settings.MIN_VIABLE_BUDGET_USD:.2f} for viable cloud allocation."
            )
        return v
