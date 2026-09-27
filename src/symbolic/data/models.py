from dataclasses import dataclass
from typing import Optional

@dataclass
class Provider:
    id: str
    name: str

@dataclass
class Region:
    id: str
    provider_id: str
    region_code: str
    region_name: str
    geo: str
    base_cost_usd: float
    sla_pct: float

@dataclass
class VMSku:
    id: str
    provider_id: str
    sku: str
    family: str
    vcpus: int
    ram_gb: float
    hourly_price_usd: float
    monthly_price_usd: float
    currency: str
    pricing_source: str
    active: bool

@dataclass
class Benchmark:
    id: str
    provider_id: str
    target: str
    benchmark_name: str
    benchmark_value: float
    benchmark_unit: str
    benchmark_source: str
    measurement_context: Optional[str] = None
