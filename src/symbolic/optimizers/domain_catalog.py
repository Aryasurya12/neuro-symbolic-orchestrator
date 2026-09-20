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

from src.symbolic.data.repository import CloudDataRepository
from src.symbolic.data.seed import seed_database
import os

class DatabaseBackedCatalog:
    """Provides a vectorized catalog interface to the local SQLite database."""
    def __init__(self):
        # Auto-seed if db missing
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "data", "cloud_finops.db"
        )
        if not os.path.exists(db_path):
            seed_database(db_path)
            
        self.repo = CloudDataRepository()
        self.skus = self.repo.get_skus(active_only=True)
        
        # We wrap in the legacy VMSku to avoid breaking ObjectiveEvaluator's typing,
        # but realistically ObjectiveEvaluator only needs the numpy arrays.
        self.VM_CATALOG = [
            VMSku(
                name=s.sku,
                provider=s.provider_id,
                vcpus=s.vcpus,
                ram_gb=s.ram_gb,
                hourly_cost_usd=s.hourly_price_usd
            ) for s in self.skus
        ]
        
        self.CATALOG_COSTS = np.array([sku.monthly_cost() for sku in self.VM_CATALOG], dtype=np.float32)
        self.CATALOG_VCPUS = np.array([sku.vcpus for sku in self.VM_CATALOG], dtype=np.float32)
        self.CATALOG_RAM = np.array([sku.ram_gb for sku in self.VM_CATALOG], dtype=np.float32)

    def get_provider_mask(self, providers: List[str]) -> np.ndarray:
        return np.array([sku.provider in providers for sku in self.VM_CATALOG], dtype=bool)

# Global singleton for backwards compatibility
_catalog = DatabaseBackedCatalog()
VM_CATALOG = _catalog.VM_CATALOG
CATALOG_COSTS = _catalog.CATALOG_COSTS
CATALOG_VCPUS = _catalog.CATALOG_VCPUS
CATALOG_RAM = _catalog.CATALOG_RAM

def get_provider_mask(providers: List[str]) -> np.ndarray:
    return _catalog.get_provider_mask(providers)
