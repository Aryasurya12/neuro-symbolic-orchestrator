import os
import pytest
from src.symbolic.data.database import CloudDatabase
from src.symbolic.data.repository import CloudDataRepository
from src.symbolic.data.seed import seed_database

@pytest.fixture
def repo(tmp_path):
    db_path = tmp_path / "test.db"
    seed_database(str(db_path))
    db = CloudDatabase(str(db_path))
    return CloudDataRepository(db)

def test_get_providers(repo):
    providers = repo.get_providers()
    assert len(providers) == 3
    names = {p.id for p in providers}
    assert names == {"AWS", "Azure", "GCP"}

def test_get_regions(repo):
    regions = repo.get_regions()
    assert len(regions) == 5
    
    aws_regions = repo.get_regions("AWS")
    assert len(aws_regions) == 3

def test_get_skus(repo):
    skus = repo.get_skus()
    assert len(skus) == 10
    
    gcp_skus = repo.get_skus("GCP")
    assert len(gcp_skus) == 1
    assert gcp_skus[0].sku == "e2-standard-4"

def test_find_feasible_skus(repo):
    # All active SKUs
    skus = repo.find_feasible_skus()
    assert len(skus) == 10
    
    # Provider filter
    skus = repo.find_feasible_skus(provider_ids=["AWS", "GCP"])
    assert len(skus) == 9
    
    # Resource filter (vCPUs)
    skus = repo.find_feasible_skus(required_vcpus=4)
    # AWS has 3, Azure has 1, GCP has 1 (Wait, AWS has t3.xlarge, c5.xlarge, m5.xlarge, m5.2xlarge = 4)
    assert len(skus) == 6
    
    # Resource filter (RAM)
    skus = repo.find_feasible_skus(required_ram_gb=16.0)
    assert len(skus) == 5
    
    # Budget filter (hourly)
    skus = repo.find_feasible_skus(max_hourly_cost=0.10)
    assert len(skus) == 4

def test_get_benchmarks(repo):
    benchmarks = repo.get_benchmarks()
    assert len(benchmarks) == 3
    
    aws_bench = repo.get_benchmarks("AWS-m5.xlarge")
    assert len(aws_bench) == 1
    assert aws_bench[0].benchmark_value == 85.0
