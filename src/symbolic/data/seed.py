from .database import CloudDatabase
import sqlite3

def seed_database(db_path: str = None):
    """Initializes and seeds the database deterministically."""
    db = CloudDatabase(db_path) if db_path else CloudDatabase()
    db.initialize_schema()
    
    with db.get_connection() as conn:
        _seed_providers(conn)
        _seed_regions(conn)
        _seed_skus(conn)
        _seed_benchmarks(conn)
        conn.commit()
        
def _seed_providers(conn: sqlite3.Connection):
    providers = [
        ("AWS", "Amazon Web Services"),
        ("Azure", "Microsoft Azure"),
        ("GCP", "Google Cloud Platform")
    ]
    conn.executemany("INSERT OR IGNORE INTO providers (id, name) VALUES (?, ?)", providers)

def _seed_regions(conn: sqlite3.Connection):
    # From graph_model.py
    regions = [
        ("us-east-1", "AWS", "us-east-1", "US East (N. Virginia)", "US_East", 120.0, 99.95),
        ("us-west-2", "AWS", "us-west-2", "US West (Oregon)", "US_West", 130.0, 99.95),
        ("eu-west-1", "AWS", "eu-west-1", "Europe (Ireland)", "Europe", 140.0, 99.95),
        ("eastus", "Azure", "eastus", "East US", "US_East", 125.0, 99.95),
        ("us-central1", "GCP", "us-central1", "Iowa", "US_Central", 115.0, 99.95)
    ]
    conn.executemany("""
        INSERT OR IGNORE INTO regions (id, provider_id, region_code, region_name, geo, base_cost_usd, sla_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, regions)

def _seed_skus(conn: sqlite3.Connection):
    # Base catalog for backward compatibility with GA tests
    # Note: monthly_price = hourly_price * 730
    skus = [
        ("AWS-t3.medium", "AWS", "t3.medium", "General Purpose", 2, 4.0, 0.0416, 0.0416 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
        ("AWS-t3.large", "AWS", "t3.large", "General Purpose", 2, 8.0, 0.0832, 0.0832 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
        ("AWS-t3.xlarge", "AWS", "t3.xlarge", "General Purpose", 4, 16.0, 0.1664, 0.1664 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
        ("AWS-c5.large", "AWS", "c5.large", "Compute Optimized", 2, 4.0, 0.0850, 0.0850 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
        ("AWS-c5.xlarge", "AWS", "c5.xlarge", "Compute Optimized", 4, 8.0, 0.1700, 0.1700 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
        ("AWS-m5.large", "AWS", "m5.large", "General Purpose", 2, 8.0, 0.0960, 0.0960 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
        ("AWS-m5.xlarge", "AWS", "m5.xlarge", "General Purpose", 4, 16.0, 0.1920, 0.1920 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
        ("AWS-m5.2xlarge", "AWS", "m5.2xlarge", "General Purpose", 8, 32.0, 0.3840, 0.3840 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
        ("Azure-Standard_D4s_v5", "Azure", "Standard_D4s_v5", "General Purpose", 4, 16.0, 0.1920, 0.1920 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
        ("GCP-e2-standard-4", "GCP", "e2-standard-4", "General Purpose", 4, 16.0, 0.1340, 0.1340 * 730.0, "USD", "Sample Benchmark Pricing 2026", True),
    ]
    conn.executemany("""
        INSERT OR IGNORE INTO skus (id, provider_id, sku, family, vcpus, ram_gb, hourly_price_usd, monthly_price_usd, currency, pricing_source, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, skus)

def _seed_benchmarks(conn: sqlite3.Connection):
    benchmarks = [
        ("bench-1", "AWS", "AWS-m5.xlarge", "OPT-BENCH CPU Score", 85.0, "score", "Simulated OPT-BENCH", "General integer workloads"),
        ("bench-2", "Azure", "Azure-Standard_D4s_v5", "OPT-BENCH CPU Score", 82.0, "score", "Simulated OPT-BENCH", "General integer workloads"),
        ("bench-3", "GCP", "GCP-e2-standard-4", "OPT-BENCH CPU Score", 80.0, "score", "Simulated OPT-BENCH", "General integer workloads"),
    ]
    conn.executemany("""
        INSERT OR IGNORE INTO benchmarks (id, provider_id, target, benchmark_name, benchmark_value, benchmark_unit, benchmark_source, measurement_context)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, benchmarks)

if __name__ == "__main__":
    print("Seeding local cloud finops database...")
    seed_database()
    print("Done.")
