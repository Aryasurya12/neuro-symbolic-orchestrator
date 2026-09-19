"""SYM-1: Domain Catalog for Phase-2 Discrete Optimization.

This encapsulates the temporary VM catalog derived from the ILP prototype.
It acts as a stand-in for the future Phase-5 pricing database to keep the 
GA domain cleanly separated.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class VMSku:
    """A purchasable VM instance type available for allocation."""
    name: str
    provider: str
    vcpus: int
    ram_gb: float
    hourly_cost_usd: float

    def monthly_cost(self) -> float:
        """Approximate monthly cost assuming a 730-hour month."""
        return round(self.hourly_cost_usd * 730.0, 2)

# Temporary catalog for discrete GA optimization
VM_CATALOG: List[VMSku] = [
    VMSku("t3.medium", "AWS", 2, 4.0, 0.0416),
    VMSku("t3.large", "AWS", 2, 8.0, 0.0832),
    VMSku("t3.xlarge", "AWS", 4, 16.0, 0.1664),
    VMSku("c5.large", "AWS", 2, 4.0, 0.0850),
    VMSku("c5.xlarge", "AWS", 4, 8.0, 0.1700),
    VMSku("m5.large", "AWS", 2, 8.0, 0.0960),
    VMSku("m5.xlarge", "AWS", 4, 16.0, 0.1920),
    VMSku("m5.2xlarge", "AWS", 8, 32.0, 0.3840),
    VMSku("Standard_D4s_v5", "Azure", 4, 16.0, 0.1920),
    VMSku("e2-standard-4", "GCP", 4, 16.0, 0.1340),
]

# Precomputed NumPy arrays for fast vectorized objective evaluation
CATALOG_COSTS = np.array([sku.monthly_cost() for sku in VM_CATALOG], dtype=np.float32)
CATALOG_VCPUS = np.array([sku.vcpus for sku in VM_CATALOG], dtype=np.float32)
CATALOG_RAM = np.array([sku.ram_gb for sku in VM_CATALOG], dtype=np.float32)

def get_provider_mask(providers: List[str]) -> np.ndarray:
    """Returns a boolean mask of catalog items matching the requested providers."""
    return np.array([sku.provider in providers for sku in VM_CATALOG], dtype=bool)
