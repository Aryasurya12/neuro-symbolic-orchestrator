from typing import List, Optional
from .database import CloudDatabase
from .models import Provider, Region, VMSku, Benchmark

class CloudDataRepository:
    """Provides a clean abstraction for querying the local SQLite database."""
    
    def __init__(self, db: Optional[CloudDatabase] = None):
        self.db = db or CloudDatabase()

    def get_providers(self) -> List[Provider]:
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM providers ORDER BY id")
            return [Provider(**dict(row)) for row in cursor.fetchall()]

    def get_regions(self, provider_id: Optional[str] = None) -> List[Region]:
        query = "SELECT * FROM regions"
        params = []
        if provider_id:
            query += " WHERE provider_id = ?"
            params.append(provider_id)
        
        with self.db.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [Region(**dict(row)) for row in cursor.fetchall()]

    def get_skus(self, provider_id: Optional[str] = None, active_only: bool = True) -> List[VMSku]:
        query = "SELECT * FROM skus WHERE 1=1"
        params = []
        if provider_id:
            query += " AND provider_id = ?"
            params.append(provider_id)
        if active_only:
            query += " AND active = 1"
            
        with self.db.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [VMSku(**dict(row)) for row in cursor.fetchall()]

    def find_feasible_skus(
        self,
        provider_ids: Optional[List[str]] = None,
        required_vcpus: Optional[int] = None,
        required_ram_gb: Optional[float] = None,
        max_hourly_cost: Optional[float] = None
    ) -> List[VMSku]:
        """Finds SKUs satisfying the given feasibility constraints."""
        query = "SELECT * FROM skus WHERE active = 1"
        params = []
        
        if provider_ids:
            placeholders = ",".join("?" * len(provider_ids))
            query += f" AND provider_id IN ({placeholders})"
            params.extend(provider_ids)
            
        if required_vcpus is not None:
            query += " AND vcpus >= ?"
            params.append(required_vcpus)
            
        if required_ram_gb is not None:
            query += " AND ram_gb >= ?"
            params.append(required_ram_gb)
            
        if max_hourly_cost is not None:
            query += " AND hourly_price_usd <= ?"
            params.append(max_hourly_cost)
            
        with self.db.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [VMSku(**dict(row)) for row in cursor.fetchall()]

    def get_benchmarks(self, target: Optional[str] = None) -> List[Benchmark]:
        query = "SELECT * FROM benchmarks"
        params = []
        if target:
            query += " WHERE target = ?"
            params.append(target)
            
        with self.db.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [Benchmark(**dict(row)) for row in cursor.fetchall()]
