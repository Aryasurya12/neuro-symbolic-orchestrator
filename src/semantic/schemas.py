"""SEM-4: Pydantic V2 Safety Contract between Semantic and Symbolic layers."""

from typing import List, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from config.settings import settings


class CloudOptimizationContract(BaseModel):
    """SEM-4: Strict Pydantic V2 Safety Contract governing the handoff from

    natural-language intent understanding to symbolic optimization engines.
    """

    model_config = ConfigDict(extra="forbid")

    problem_type: Literal[
        "ILP_VM_Allocation",
        "PSO_Continuous_Scaling",
        "Z3_Graph_Disaster_Recovery",
    ] = Field(
        ...,
        description="Target optimization problem type selected by CARM matcher",
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
