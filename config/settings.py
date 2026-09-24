"""Configuration settings for the FinOps Neuro-Symbolic Orchestrator (neurasym)."""

import os
from dataclasses import dataclass, field
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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

    # Currency conversion & formatting settings
    USD_TO_INR_RATE: float = 85.0
    CURRENCY_SYMBOL_INR: str = "₹"
    CURRENCY_SYMBOL_USD: str = "$"

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


settings = Settings()

