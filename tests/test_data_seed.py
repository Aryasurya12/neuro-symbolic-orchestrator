import os
import pytest
from src.symbolic.data.database import CloudDatabase
from src.symbolic.data.seed import seed_database

@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test.db")

def test_deterministic_seeding(temp_db_path):
    # Seed once
    seed_database(temp_db_path)
    
    db = CloudDatabase(temp_db_path)
    with db.get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
        assert count == 3
        
    # Seed again (should not duplicate)
    seed_database(temp_db_path)
    with db.get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
        assert count == 3
        
        count_skus = conn.execute("SELECT COUNT(*) FROM skus").fetchone()[0]
        assert count_skus == 10

def test_seed_data_quality(temp_db_path):
    seed_database(temp_db_path)
    db = CloudDatabase(temp_db_path)
    
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM skus")
        for row in cursor.fetchall():
            # Validate business rules
            assert row["hourly_price_usd"] > 0
            assert row["monthly_price_usd"] > 0
            assert row["vcpus"] > 0
            assert row["ram_gb"] > 0
            # Consistency
            assert abs((row["hourly_price_usd"] * 730.0) - row["monthly_price_usd"]) < 0.01
