import sqlite3
import os
from contextlib import contextmanager

# Default local database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "cloud_finops.db"
)

class CloudDatabase:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    @contextmanager
    def get_connection(self):
        """Yields a thread-safe connection to the SQLite database."""
        # check_same_thread=False allows us to pass connections around if needed, 
        # but connection-per-operation is safer for thread isolation anyway.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def initialize_schema(self):
        """Creates the necessary tables if they don't exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS providers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS regions (
            id TEXT PRIMARY KEY,
            provider_id TEXT NOT NULL,
            region_code TEXT NOT NULL,
            region_name TEXT NOT NULL,
            geo TEXT NOT NULL,
            base_cost_usd REAL NOT NULL,
            sla_pct REAL NOT NULL,
            FOREIGN KEY (provider_id) REFERENCES providers (id)
        );

        CREATE TABLE IF NOT EXISTS skus (
            id TEXT PRIMARY KEY,
            provider_id TEXT NOT NULL,
            sku TEXT NOT NULL,
            family TEXT NOT NULL,
            vcpus INTEGER NOT NULL,
            ram_gb REAL NOT NULL,
            hourly_price_usd REAL NOT NULL,
            monthly_price_usd REAL NOT NULL,
            currency TEXT NOT NULL,
            pricing_source TEXT NOT NULL,
            active BOOLEAN NOT NULL DEFAULT 1,
            FOREIGN KEY (provider_id) REFERENCES providers (id)
        );

        CREATE TABLE IF NOT EXISTS benchmarks (
            id TEXT PRIMARY KEY,
            provider_id TEXT NOT NULL,
            target TEXT NOT NULL,
            benchmark_name TEXT NOT NULL,
            benchmark_value REAL NOT NULL,
            benchmark_unit TEXT NOT NULL,
            benchmark_source TEXT NOT NULL,
            measurement_context TEXT,
            FOREIGN KEY (provider_id) REFERENCES providers (id)
        );
        """
        with self.get_connection() as conn:
            conn.executescript(schema)
            conn.commit()
