# Phase 6 — SYM-5: Local Cloud Pricing & Benchmark Database

## Objective
The purpose of Phase 6 is to replace the hard-coded cloud resource and pricing assumptions with a structured, local, and deterministic SQLite database layer. This ensures that the optimizer runs against a unified, structured catalog of AWS, Azure, and GCP resources, while remaining fully local and free from external billing API dependencies.

## Architecture

```
                    Part A
                       ↓
             SymbolicOptimizationRequest
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
       GA             PSO          Graph-Z3
        └──────────────┼──────────────┘
                       ↑
                       │
             DatabaseBackedCatalog
                       ↑
                       │
              CloudDataRepository
                       ↑
                       │
                    SQLite (data/cloud_finops.db)
```

## Schema
The database uses a normalized schema:
- **`providers`**: Identifies cloud vendors (AWS, Azure, GCP).
- **`regions`**: Stores regions with their geographical zones, base SLA properties, and base costs.
- **`skus`**: Stores Virtual Machine instance types, VCPUs, RAM in GB, and pricing.
- **`benchmarks`**: Stores performance benchmarks (e.g. OPT-BENCH CPU scores) linked to specific SKUs.

## Seed Dataset
- **Providers**: 3 (AWS, Azure, GCP)
- **Regions**: 5 (us-east-1, us-west-2, eu-west-1, eastus, us-central1)
- **SKUs**: 10
- **Benchmarks**: 3

## Pricing Assumptions
- **Primary unit**: Hourly price in USD.
- **Monthly conversion**: Assumes a standard 730-hour month (`monthly_price = hourly_price * 730.0`).
- **Pricing source/reference date**: Prices are "Sample Benchmark Pricing 2026". They are representative and illustrative for algorithm validation, not guaranteed live prices.

## Benchmarking
The system implements a local benchmark storage abstraction. It currently stores simulated "OPT-BENCH CPU Score" values to fulfill the structural requirements of the project. This serves as the OPT-BENCH integration point for future actual datasets.

## Optimizer Integration
The existing symbolic engines (GA and PSO) depend on extremely fast vectorized operations using NumPy. 
To preserve performance without destabilizing Phase 2:
1. `src/symbolic/optimizers/domain_catalog.py` now uses `DatabaseBackedCatalog`.
2. Upon initialization, it queries the SQLite database via `CloudDataRepository` to fetch all active SKUs.
3. It maps these SKUs into memory as the globally accessible NumPy arrays (`CATALOG_COSTS`, `CATALOG_VCPUS`, `CATALOG_RAM`), retaining exact backward compatibility.

## API Integration
The API endpoints were not fundamentally changed in this phase, as the goal was solely to replace the internal data source. The data layer is kept strictly internal to the optimizers.

## Limitations
- **Static Sample Pricing**: The database contains illustrative data. It does not hit live AWS/Azure/GCP billing APIs.
- **No Production Database**: SQLite is used to keep the project self-contained. It is not intended for high-concurrency mutation environments.
- **Limited Benchmark Corpus**: Contains a small subset of simulated OPT-BENCH scores to prove architectural capability.
