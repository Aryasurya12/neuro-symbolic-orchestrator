import os
import pytest
import sqlite3
from src.symbolic.data.database import CloudDatabase

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    db = CloudDatabase(str(db_path))
    db.initialize_schema()
    return db

def test_database_initialization(temp_db):
    assert os.path.exists(temp_db.db_path)
    
def test_tables_exist(temp_db):
    with temp_db.get_connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}
        
    assert "providers" in tables
    assert "regions" in tables
    assert "skus" in tables
    assert "benchmarks" in tables

def test_thread_safety_concept(temp_db):
    # Test connection can be acquired and used
    with temp_db.get_connection() as conn:
        conn.execute("INSERT INTO providers (id, name) VALUES (?, ?)", ("TEST", "Test Provider"))
        conn.commit()
        
    with temp_db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM providers")
        results = cursor.fetchall()
        assert len(results) == 1
        assert results[0]["id"] == "TEST"
