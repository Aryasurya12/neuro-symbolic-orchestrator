"""Configuration settings for the FinOps Neuro-Symbolic Orchestrator (neurasym)."""

import os
from dataclasses import dataclass, field
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables from .env file at project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=ENV_PATH)


@dataclass
class Settings:
    """Centralized configuration settings."""

    DEFAULT_BUDGET_USD: float = 500.0
    SOLVER_TIMEOUT_MS: int = 500
    ENABLE_Z3_GRAPH_OPTIMIZER: bool = True
    CHROMA_DB_PATH: str = "./chroma_db"

    # API Keys & External Integrations
    OPENROUTER_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    OPENROUTER_MODEL: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
    )
    LLM_REQUEST_TIMEOUT_SECONDS: float = 360.0
    LLM_MAX_COMPLETION_TOKENS: int = 4096

    # Currency conversion & formatting settings
    USD_TO_INR_RATE: float = 85.0
    CURRENCY_SYMBOL_INR: str = "₹"
    CURRENCY_SYMBOL_USD: str = "$"

    # Time & Cost Constants
    HOURS_PER_MONTH: float = 730.0
    LATENCY_COST_PER_MS: float = 0.25

    # Continuous PSO Default Hyperparameters & Costs
    DEFAULT_PSO_COST_PER_MBPS_MONTH: float = 0.08
    DEFAULT_PSO_COST_PER_REPLICA_MONTH: float = 45.0
    DEFAULT_PSO_TARGET_CPU_PCT: float = 70.0
    DEFAULT_PSO_NUM_PARTICLES: int = 30
    DEFAULT_PSO_MAX_ITERATIONS: int = 50
    DEFAULT_PSO_INERTIA_WEIGHT: float = 0.729
    DEFAULT_PSO_COGNITIVE_PARAM: float = 1.494
    DEFAULT_PSO_SOCIAL_PARAM: float = 1.494
    DEFAULT_PSO_CPU_PENALTY_WEIGHT: float = 2.5
    DEFAULT_PSO_BUDGET_PENALTY_WEIGHT: float = 50.0
    DEFAULT_PSO_REPLICAS_CAPACITY_FACTOR: float = 75.0

    # Service & Solver SLA Defaults
    DEFAULT_SLA_AVAILABILITY_PCT: float = 99.99
    DEFAULT_MAX_LATENCY_MS: float = 100.0

    # Derived / secondary constants, centralized here for single-source-of-truth.
    SUPPORTED_CLOUD_PROVIDERS: Tuple[str, ...] = field(
        default=("AWS", "Azure", "GCP")
    )
    MIN_VIABLE_BUDGET_USD: float = 10.0
    LOG_LEVEL: str = "INFO"

    def __post_init__(self) -> None:
        if self.DEFAULT_BUDGET_USD <= 0:
            raise ValueError("DEFAULT_BUDGET_USD must be positive.")
        if self.SOLVER_TIMEOUT_MS <= 0:
            raise ValueError("SOLVER_TIMEOUT_MS must be positive.")
        if self.MIN_VIABLE_BUDGET_USD <= 0:
            raise ValueError("MIN_VIABLE_BUDGET_USD must be positive.")
        if self.USD_TO_INR_RATE <= 0:
            raise ValueError("USD_TO_INR_RATE must be positive.")
        if self.HOURS_PER_MONTH <= 0:
            raise ValueError("HOURS_PER_MONTH must be positive.")
        if self.LATENCY_COST_PER_MS < 0:
            raise ValueError("LATENCY_COST_PER_MS cannot be negative.")

settings = Settings()

# Module-level constant exports for direct importing
USD_TO_INR_RATE: float = settings.USD_TO_INR_RATE
OPENROUTER_API_KEY: str = settings.OPENROUTER_API_KEY
OPENROUTER_MODEL: str = settings.OPENROUTER_MODEL

