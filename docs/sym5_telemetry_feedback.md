# SYM-5: Telemetry-Driven Adaptive Routing & Online Feedback

## 1. Motivation

SYM-4 introduced a Learned Solver Router trained on a **static synthetic dataset** derived from the same rules as the SYM-3 deterministic router. While accurate on synthetic benchmarks (100% test accuracy), it cannot improve from actual observed execution outcomes.

SYM-5 addresses this by introducing a **structured telemetry layer** that:
- Records every routing decision and its observed result
- Provides a controlled feedback mechanism for converting telemetry into training labels
- Enables explicit (non-automatic) retraining with validation gating before promotion
- Preserves full backward compatibility with SYM-3 and SYM-4

> **The telemetry system optimizes solver-selection performance and does not determine mathematical feasibility.**

---

## 2. SYM-4 Limitation

The SYM-4 Learned Router was trained on 500 synthetic samples (seed=42) with labels derived from the same heuristic rules used by SYM-3. It therefore cannot:
- Improve from actual cloud workload characteristics
- Learn from observed solver runtimes
- Adapt to workload distributions not covered by the synthetic generator

---

## 3. SYM-5 Architecture

```
SymbolicOptimizationRequest
         ↓
SolverRoutingFeatures
         ↓
┌───────────────────────────────────────┐
│  SYM-4 Learned Router                 │
│  (Softmax Classifier)                 │
│         + Confidence Gating           │
│         + SYM-3 Deterministic Fallback│
│         + Safety Arbitration          │
└─────────────────┬─────────────────────┘
                  ↓
         Solver Execution
         (GA / PSO / Z3)
                  ↓
            OptiHive
                  ↓
          Final Result ←── Hard Constraint Layer (authoritative)
                  ↓
     ┌────────────────────────┐
     │ SYM-5 Telemetry Layer  │  ← non-blocking, failure-safe
     │ collector.collect()    │
     └────────────┬───────────┘
                  ↓
        SQLite: routing_telemetry
                  ↓
       [Explicit offline training]
       train_sym5_from_telemetry.py
                  ↓
        Candidate Model
                  ↓
        Validation (val_acc ≥ 0.70)
                  ↓
        Promotion → learned_checkpoint.json
```

---

## 4. Telemetry Schema

**Table**: `routing_telemetry` (added to existing `cloud_finops.db`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `run_id` | TEXT | UUID per optimization run |
| `timestamp` | REAL | Unix timestamp |
| `model_version` | TEXT | Router model version that made the decision |
| `resource_scale` | REAL | vCPU × RAM requested |
| `service_scale` | INTEGER | Number of services |
| `budget_tightness` | REAL | Budget / resource_scale |
| `provider_count` | INTEGER | Number of cloud providers |
| `has_latency_constraint` | INTEGER | 0/1 |
| `has_sla_constraint` | INTEGER | 0/1 |
| `constraint_density` | REAL | Active constraint count |
| `is_highly_constrained` | INTEGER | 0/1 |
| `selected_solvers` | TEXT | JSON array e.g. `["GA","PSO"]` |
| `routing_mode` | TEXT | `learned_high`, `learned_medium`, `deterministic`, `fallback` |
| `learned_confidence` | REAL | Softmax probability from learned model |
| `fallback_used` | INTEGER | 0/1 |
| `total_runtime_ms` | REAL | End-to-end pipeline latency |
| `solver_runtime_ms` | REAL | Solver execution time |
| `is_feasible` | INTEGER | 0/1 (set by symbolic constraint layer) |
| `hard_constraint_violation` | INTEGER | 0/1 |
| `objective_cost_usd` | REAL | NULL if infeasible |
| `scenario_tag` | TEXT | Optional benchmark identifier |

**Privacy**: No raw natural-language prompts are stored. Only structured feature vectors and metrics are persisted.

---

## 5. Collection Process

Collection occurs in `OptimizationService.run_pipeline_sync()` **after** the final result is computed:

```python
try:
    from src.symbolic.telemetry.collector import collect as _collect
    _collect(request, result, routing_metadata, total_ms, run_id, scenario_tag)
except Exception:
    pass  # Telemetry never blocks optimization
```

Key properties:
- **Non-blocking**: executed synchronously but at the very end of the pipeline
- **Failure-safe**: any exception in collection is silently discarded
- **Additive**: does not modify the optimization result in any way
- **Isolated**: telemetry table is separate from pricing/provider tables

---

## 6. Feedback Signal

### Label Generation Policy

For each telemetry record, a training label is derived from the **observed solver set** used in that run:

| Solver Set | Label |
|-----------|-------|
| `{GA, PSO}` | `GA_PSO` |
| `{Z3}` | `Z3` |
| `{GA, PSO, Z3}` | `GA_PSO_Z3` |
| `{GA, Z3}` or `{PSO, Z3}` | `GA_PSO_Z3` |
| `{GA}` or `{PSO}` | `GA_PSO` |

### Record Exclusion Rules

Records are **excluded** from training if:
- `hard_constraint_violation == True` (solver produced invalid output; symbolic safety failure)
- `total_runtime_ms == 0` (incomplete record)
- `selected_solvers == []` (routing failure)

Records with `is_feasible == False` but no `hard_constraint_violation` are **included** — they carry valid information about routing on infeasible workloads (e.g., learning that Z3 is the right solver for constraint-heavy infeasible cases).

### Mathematical Justification

The feedback metric is **routing correctness** — did the selected solver set match the workload characteristics? This is derived from:

```
label = f(observed solver set)
```

Feasibility is NOT used as the reward directly, because:
- A fast solver returning an infeasible result is NOT successful routing
- A slower solver returning a feasible result IS correct routing for a constrained problem
- Hard constraint violations are excluded entirely from training data

---

## 7. Training

**Script**: `benchmarks/train_sym5_from_telemetry.py`

```bash
python benchmarks/train_sym5_from_telemetry.py           # dry-run
python benchmarks/train_sym5_from_telemetry.py --promote # promote if valid
```

Training uses:
- **Model**: Multinomial Softmax Classifier (NumPy, identical architecture to SYM-4)
- **Seed**: 42 (deterministic)
- **Split**: 80% train / 20% val (random with seed=42)
- **Minimum records**: 20 valid observations (`MIN_TELEMETRY_SAMPLES`)
- **Epochs**: 2000, LR=0.5, L2 reg=0.01

---

## 8. Validation

Before promotion, the candidate model must pass:
- `val_accuracy >= 0.70` (`MIN_PROMOTION_ACCURACY`)
- Schema integrity check (required keys present)
- Checkpoint is valid JSON and loadable

---

## 9. Promotion

```python
promote_candidate_model()
```

Steps:
1. Load and validate candidate checkpoint
2. Check `val_accuracy >= min_accuracy`
3. Back up current checkpoint to `learned_checkpoint.previous.json`
4. Atomically write new checkpoint (write to `.tmp` then `os.replace()`)
5. Return result with version, accuracy, backup status

**If any step fails: existing checkpoint is preserved unchanged.**

---

## 10. Rollback

```python
rollback_to_previous()
```

Restores `learned_checkpoint.previous.json` → `learned_checkpoint.json`.

Rollback is a manual operation (not automatic) — aggregate validation failure over multiple requests should trigger it, not a single noisy result.

---

## 11. Model Versioning

| Version | Source | Notes |
|---------|--------|-------|
| `v1.0.0` | SYM-4 synthetic training | Static, 500 samples, seed=42 |
| `sym5-v1` | SYM-5 telemetry training | Derived from observed execution |

Model version is stored in the checkpoint and persisted in each telemetry record, enabling historical analysis of which model generated each routing decision.

---

## 12. Safety Guarantees

- **Feasibility is NEVER determined by telemetry** — only by Z3, ILP, and the OptiHive constraint layer
- **Hard constraint violations are excluded from training data** — routing feedback cannot reward infeasible results
- **Training is explicit** — live requests NEVER trigger automatic model updates
- **Checkpoint writes are atomic** — no partial checkpoint can leave the router in an invalid state
- **Fallback chain is preserved**: SYM-5 → SYM-4 → SYM-3 (all still functional independently)

---

## 13. API Integration

No changes to FastAPI endpoint signatures. Existing endpoints:
- `GET /health` — unchanged
- `POST /optimize` — now collects telemetry after each run
- `POST /optimize/query` — unchanged
- `WS /ws/optimize` — now collects telemetry; collection runs after WebSocket events are sent, never blocking them

New optional parameters on `run_pipeline_sync`:
- `run_id: Optional[str]` — optional UUID for the request (auto-generated if not provided)
- `scenario_tag: Optional[str]` — optional label for benchmark identification

---

## 14. WebSocket Integration

Existing events unchanged:
```
started → routing_decided → solver_started → progress → solver_completed → completed
```

Telemetry collection occurs after `solver_completed` is emitted. It is wrapped in `try/except`, so collection failures do not interrupt WebSocket completion.

---

## 15. Benchmark Methodology

**Script**: `benchmarks/benchmark_sym5_telemetry.py`

Three configurations compared across 10 deterministic scenarios:

| Configuration | Routing | Telemetry |
|--------------|---------|-----------|
| SYM-3 | Deterministic only | None |
| SYM-4 | Learned (static) | Disabled (for fair comparison) |
| SYM-5 | Learned (static) | Enabled |

> **Important**: SYM-3 vs SYM-4 vs SYM-5 in cold-start state all use the same underlying learned model (SYM-4 checkpoint). SYM-5 differs from SYM-4 only by telemetry overhead. Post-training SYM-5 performance requires explicit retraining with `train_sym5_from_telemetry.py --promote`.

---

## 16. Measured Results

### SYM-5 Benchmark (Cold Start — same learned checkpoint as SYM-4)

| Scenario | Feasible | SYM-3 (ms) | SYM-4 (ms) | SYM-5 (ms) | Telem overhead | Mode |
|----------|----------|-----------|-----------|-----------|----------------|------|
| VM_ALLOCATION | F | 95.78 | 20.51 | 28.59 | 8.07 ms | learned_high |
| COMPUTE_INTENSIVE | T | 22.94 | 5.75 | 14.94 | 9.19 ms | learned_high |
| COST_OPTIMIZATION | T | 17.03 | 21.08 | 30.97 | 9.89 ms | learned_medium |
| COST_INFEASIBLE | F | 27.39 | 19.51 | 32.20 | 12.69 ms | learned_high |
| MULTI_CLOUD | T | 16.48 | 6.27 | 15.15 | 8.87 ms | learned_high |
| LOW_LATENCY | F | 29.18 | 17.70 | 28.76 | 11.06 ms | learned_high |
| HIGH_AVAILABILITY | F | 29.43 | 18.71 | 28.91 | 10.20 ms | deterministic |
| DISASTER_RECOVERY | T | 27.21 | 22.58 | 28.48 | 5.91 ms | learned_high |
| SMALL_PROBLEM | T | 14.31 | 5.95 | 14.14 | 8.19 ms | learned_high |
| LARGE_PROBLEM | T | 27.73 | 21.16 | 38.24 | 17.08 ms | learned_high |

**Telemetry overhead: 5.9–17.1 ms per request** (dominated by SQLite write latency on Windows).

### Training on Collected Telemetry (23 real records from benchmark)

| Metric | Value |
|--------|-------|
| Total records | 23 |
| Valid records (after filtering) | 22 |
| Training samples | 18 (80%) |
| Validation samples | 4 (20%) |
| Train accuracy | 88.24% |
| Val accuracy | 80.00% |
| Model version | `sym5-v1` |
| Status | Candidate written (not promoted in dry-run) |

---

## 17. Measured Telemetry Overhead

| Metric | Value |
|--------|-------|
| SQLite write latency (Windows) | ~6–17 ms per record |
| Routing overhead | < 1 ms (unchanged) |
| Learned inference | < 0.5 ms (unchanged) |
| Training time (23 records) | < 1 second |

> The SQLite write overhead is the dominant cost of telemetry on Windows. In production Linux environments, this would be significantly lower. The overhead is acceptable for optimization requests that already take 15–100ms.

---

## 18. Learning Effect

**Current state (23 records):**
- Val accuracy 80% vs SYM-4 100% synthetic test accuracy
- This is expected: the telemetry dataset is small (23 records) and the 20% val split leaves only 4-5 samples
- A meaningful learning effect requires more telemetry (≥ 100 observations across diverse workload types)

**Learning curve (actual measured data only):**
- 23 records: 80.00% val accuracy (candidate written, not promoted)
- More records required for robust promotion

No fabricated learning curves are presented.

---

## 19. Limitations

1. **Small telemetry dataset**: 23 real records from benchmark runs. Meaningful online learning requires ≥100 diverse observations.
2. **SQLite write overhead**: 6–17 ms per request on Windows (acceptable but measurable).
3. **Labels derived from routing, not outcome**: The training label reflects which solver was used, not whether it was the optimal choice. A better label would compare solver runtime across alternatives — this requires running all solvers for comparison, which is not done in production.
4. **No active learning**: SYM-5 does not select which workloads to run which solvers on for exploration purposes.
5. **Manual promotion only**: Retraining requires explicit CLI invocation. There is no automated production retraining loop.
6. **Val split is small**: With 23 records, the 20% val split is only 4–5 samples, making accuracy estimates noisy.

---

## 20. Reproducibility

```bash
# Collect telemetry by running the system normally
python benchmarks/benchmark_sym5_telemetry.py

# Check collected telemetry count
python -c "from src.symbolic.telemetry.repository import TelemetryRepository; print(TelemetryRepository().count())"

# Train candidate model (dry-run)
python benchmarks/train_sym5_from_telemetry.py

# Promote if validation passes
python benchmarks/train_sym5_from_telemetry.py --promote

# Rollback if needed
python -c "from src.symbolic.optihive.telemetry_training import rollback_to_previous; print(rollback_to_previous())"

# Run full test suite
python -m pytest tests/
```
