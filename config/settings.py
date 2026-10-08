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

    # API Keys & External Integrations for Modes 1 & 2 (Groq API exclusively)
    GROQ_API_KEY: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    GROQ_BASE_URL: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    GROQ_MODEL: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # API Keys & Integrations for Mode 4 (Groq Neural Interpretation)
    MODE4_PROVIDER: str = field(
        default_factory=lambda: os.getenv("MODE4_PROVIDER", "Groq")
    )
    OPENROUTER_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    OPENROUTER_BASE_URL: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    OPENROUTER_MODEL: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
    )

    NVIDIA_API_KEY: str = field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY", "")
    )
    NVIDIA_BASE_URL: str = field(
        default_factory=lambda: os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    NVIDIA_MODEL: str = field(
        default_factory=lambda: os.getenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
    )

    LLM_REQUEST_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_COMPLETION_TOKENS: int = 4096

    def get_mode4_provider_config(self) -> Tuple[str, str, str, str, float]:
        """Returns (provider_name, api_key, base_url, model, timeout) for Mode 4.
        
        Strictly preserves provider routing consistency across key, base URL, and model.
        Mode 4 uses Groq API with GROQ_API_KEY, GROQ_BASE_URL, and GROQ_MODEL from .env.
        """
        # Primary provider: Groq
        if self.GROQ_API_KEY or self.MODE4_PROVIDER.lower() == "groq":
            return (
                "Groq",
                self.GROQ_API_KEY,
                self.GROQ_BASE_URL,
                self.GROQ_MODEL,
                self.LLM_REQUEST_TIMEOUT_SECONDS,
            )
        # OpenRouter fallback if explicitly requested
        if self.OPENROUTER_API_KEY or self.MODE4_PROVIDER.lower() == "openrouter":
            return (
                "OpenRouter",
                self.OPENROUTER_API_KEY,
                self.OPENROUTER_BASE_URL,
                self.OPENROUTER_MODEL,
                self.LLM_REQUEST_TIMEOUT_SECONDS,
            )
        # Default fallback route (Groq)
        return (
            "Groq",
            self.GROQ_API_KEY,
            self.GROQ_BASE_URL,
            self.GROQ_MODEL,
            self.LLM_REQUEST_TIMEOUT_SECONDS,
        )

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
GROQ_API_KEY: str = settings.GROQ_API_KEY
GROQ_BASE_URL: str = settings.GROQ_BASE_URL
GROQ_MODEL: str = settings.GROQ_MODEL
NVIDIA_API_KEY: str = settings.NVIDIA_API_KEY
NVIDIA_BASE_URL: str = settings.NVIDIA_BASE_URL
NVIDIA_MODEL: str = settings.NVIDIA_MODEL
OPENROUTER_API_KEY: str = settings.OPENROUTER_API_KEY
OPENROUTER_MODEL: str = settings.OPENROUTER_MODEL
