# SYM-4: Learned Adaptive Solver Routing

## 1. Motivation

SYM-3 introduced a **deterministic heuristic routing layer** that selects between GA, PSO, and Z3 based on extracted problem features. While robust, a purely rule-based system cannot generalise across problem typologies it was not explicitly programmed for.

SYM-4 extends SYM-3 with a **lightweight Learned Router** — a NumPy-based Softmax Classifier — that predicts the optimal solver configuration from the extracted feature vector. The learned model is used as the primary routing signal, confidence-gated against the SYM-3 deterministic router, with a final safety arbitration pass ensuring Z3 is always available for hard-constraint verification.

> **Critical**: The learned model is NEVER allowed to override mathematical feasibility. It only recommends which solvers to execute.

---

## 2. SYM-3 Baseline

SYM-3 introduced:
- `SolverRoutingFeatures`: Deterministic feature extraction from `SymbolicOptimizationRequest`
- `SolverSuitabilityScorer`: Heuristic scoring of GA, PSO, and Z3
- `AdaptiveSolverRouter`: Ranks solvers by score and selects based on confidence threshold

SYM-4 builds on top of this, keeping SYM-3 as the verified fallback path.

---

## 3. SYM-4 Architecture

```
SymbolicOptimizationRequest
         ↓
SolverRoutingFeatures (reused from SYM-3)
         ↓
    ┌────────────────────────────────┐
    │   SYM-4 LearnedSolverRouter    │
    │   (Softmax Classifier)         │
    └────────────┬───────────────────┘
                 │
         confidence gate
          ≥ 0.80 → learned_high
     0.60–0.80 → learned_medium (union with SYM-3)
          < 0.60 → deterministic SYM-3 fallback
                 ↓
         Safety Arbitration
     (Z3 added if latency/SLA/highly_constrained)
                 ↓
       Final Solver Selection
```

---

## 4. Feature Vector

Reused directly from SYM-3. The exact ordering (index 0–7):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `resource_scale` | vCPU × RAM requested |
| 1 | `service_scale` | Number of services |
| 2 | `budget_tightness` | Budget / resource_scale |
| 3 | `provider_count` | Number of allowed cloud providers |
| 4 | `has_latency_constraint` | 1.0 if latency_max_ms > 0 else 0.0 |
| 5 | `has_sla_constraint` | 1.0 if sla_availability_pct > 0 else 0.0 |
| 6 | `constraint_density` | Count of active constraint dimensions |
| 7 | `is_highly_constrained` | 1.0 if constraint_density ≥ 2 else 0.0 |

---

## 5. Training Data

**Source**: Synthetic data generated from the same benchmark methodology used in Phase 9 and Phase 10 evaluations.

The data generator (`learned_training.py::generate_synthetic_data`) creates samples across three workload archetypes:

| Archetype | Characteristics | Target Label |
|-----------|----------------|--------------|
| Unconstrained | High resource, no latency/SLA, large budget | `GA_PSO` |
| Highly Constrained | Low resource, tight budget, latency and/or SLA | `Z3` |
| Mixed | Moderate constraints | `GA_PSO_Z3` |

**Label assignment logic** is consistent with the deterministic SYM-3 routing baseline — labels were derived from the same decision rules, not arbitrarily chosen.

> **Disclaimer**: The training dataset is entirely **synthetic/benchmark-derived**. It does not represent real production cloud workload traces.

---

## 6. Train/Validation/Test Split

| Split | Samples | Seed |
|-------|---------|------|
| Training | 500 | 42 |
| Test | 100 | 1337 |

- Splits are scenario-level (not instance-level) via distinct seeds
- No data leakage (distinct seeds generate non-overlapping distributions)
- Fixed seed `42` for training, `1337` for test

---

## 7. Model Architecture

**Softmax Classifier (Multinomial Logistic Regression)**

```
Feature Vector (dim=8)
      ↓
Z-Score Normalization (per-feature mean/std from training set)
      ↓
Linear Layer: X_norm × W + b   (W: 8×3, b: 3)
      ↓
Softmax activation → class probabilities (3 classes)
      ↓
argmax → predicted_solver
max prob → confidence
```

Trained with:
- SGD with L2 regularization (λ=0.01)
- Learning rate: 0.5
- Epochs: 2000

**Why this model?**
- NumPy-only, no additional dependencies
- Transparent and auditable (JSON checkpoint)
- Deterministic inference
- Fast (~0.1ms per inference)
- Suitable for the small, well-structured feature space

---

## 8. Checkpoint Format

Location: `src/symbolic/optihive/learned_checkpoint.json`

```json
{
  "model_version": "v1.0.0",
  "feature_ordering": ["resource_scale", "service_scale", ...],
  "class_labels": ["GA_PSO", "Z3", "GA_PSO_Z3"],
  "W": [[...], ...],
  "b": [...],
  "mean": [...],
  "scale": [...],
  "metadata": {
    "training_samples": 500,
    "test_samples": 100,
    "accuracy": 1.0
  }
}
```

The checkpoint is JSON-only — no pickle, no executable objects.

---

## 9. Confidence Gating

| Confidence Level | Threshold | Action |
|-----------------|-----------|--------|
| `learned_high` | ≥ 0.80 | Use learned recommendation directly |
| `learned_medium` | 0.60–0.79 | Union of learned + deterministic SYM-3 sets |
| `deterministic` | < 0.60 | SYM-3 deterministic routing only |

Thresholds are documented constants:
```python
LEARNED_HIGH_CONFIDENCE = 0.80
LEARNED_MEDIUM_CONFIDENCE = 0.60
```

---

## 10. Deterministic Fallback

SYM-3 routing is always available as a fallback in these conditions:

| Condition | Fallback Trigger |
|-----------|-----------------|
| Checkpoint missing | `FileNotFoundError` on init |
| Checkpoint unreadable | JSON decode error |
| NaN/Inf in feature vector | Normalization guard |
| Inference exception | `RuntimeError` caught by `AdaptiveSolverRouter` |
| Confidence < 0.60 | Routing mode falls back to `deterministic` |
| Routing exception (any) | Falls back to `["GA", "PSO", "Z3"]` (full race) |

The `AdaptiveSolverRouter.__init__` catches all exceptions from `LearnedSolverRouter` and simply sets `self.learned_router = None`, falling back silently.

---

## 11. Safety Arbitration

After solver selection (learned or deterministic), a mandatory safety pass runs:

```python
if features.has_latency_constraint or features.has_sla_constraint or features.is_highly_constrained:
    if "Z3" not in selected_solvers:
        selected_solvers.append("Z3")
```

This ensures that **Z3 is never pruned away from requests requiring hard symbolic constraint verification**.

The learned model is allowed to recommend GA_PSO for unconstrained requests. It is NOT allowed to suppress Z3 for constrained requests.

---

## 12. API Integration

No changes to FastAPI endpoint signatures. The `AdaptiveSolverRouter` is transparently upgraded.

New WebSocket event (additive, does not break existing clients):
```json
{
  "event": "routing_decided",
  "mode": "learned_high",
  "solver": "GA,PSO",
  "confidence": "HIGH",
  "reason": "Learned router high confidence (0.98)"
}
```

Routing metadata is attached to `OptimizationResult.metadata["routing"]`:
```json
{
  "selected_solvers": ["GA", "PSO"],
  "routing_mode": "learned_high",
  "confidence": "HIGH",
  "reason": "..."
}
```

Existing response schemas (`is_feasible`, `best_candidate`, `metrics`) are unchanged.

---

## 13. Benchmark Methodology

Script: `benchmarks/benchmark_sym4_learned_routing.py`

Three pipelines compared across 10 deterministic scenarios (seed=42):

1. **Baseline**: GA race (GA+PSO) + Z3 always, no routing
2. **SYM-3**: Deterministic heuristic routing only (learned router disabled)
3. **SYM-4**: Learned + deterministic hybrid routing

Same 10 scenarios used in SYM-3 benchmark for fair comparison.

---

## 14. Benchmark Results

| Scenario | Feasible | Baseline (ms) | SYM-3 (ms) | SYM-4 (ms) | SYM-4 Mode | SYM-4 Solvers |
|----------|----------|--------------|-----------|-----------|-----------|--------------|
| VM_ALLOCATION | F | 79.97 | 27.02 | 21.36 | learned_high | Z3 |
| COMPUTE_INTENSIVE | T | 22.36 | 6.09 | 7.31 | learned_high | GA,PSO |
| COST_OPTIMIZATION | T | 19.58 | 6.21 | 21.77 | learned_medium | GA,Z3,PSO |
| COST_OPTIMIZATION_INFEAS | F | 20.09 | 18.77 | 21.89 | learned_high | GA,PSO,Z3 |
| MULTI_CLOUD | T | 22.50 | 5.70 | 5.56 | learned_high | GA,PSO |
| LOW_LATENCY | F | 20.73 | 17.01 | 17.12 | learned_high | Z3 |
| HIGH_AVAILABILITY | F | 20.05 | 18.35 | 19.77 | deterministic | Z3 |
| DISASTER_RECOVERY | T | 23.01 | 18.18 | 21.79 | learned_high | GA,PSO,Z3 |
| SMALL_PROBLEM | T | 20.84 | 6.98 | 5.57 | learned_high | GA,PSO |
| LARGE_PROBLEM | T | 20.70 | 18.22 | 25.50 | learned_high | GA,PSO,Z3 |

---

## 15. Accuracy

**Measured test accuracy: 100%** (on 100 synthetic test samples, seed=1337)

> **Important caveat**: This high accuracy is due to the training data being generated from the same rule-based logic as the labels. It does not imply the model would generalise to real-world production workload distributions. The dataset is small and synthetic.

---

## 16. Latency

| Metric | Value |
|--------|-------|
| Feature extraction | < 0.2 ms |
| Learned inference | < 0.5 ms |
| Routing overhead total | < 1 ms |

No significant latency regression introduced vs SYM-3. Model loading occurs at `OptimizationService` instantiation time, not per-request.

---

## 17. Limitations

1. **Synthetic training data only** — the model learns from the same rule-based logic that generates labels; generalisation to unknown workload types is unproven.
2. **Small dataset** — 500 training samples, 100 test samples. This is not production-scale data.
3. **Accuracy may be misleading** — 100% test accuracy reflects label–feature alignment, not true generalisation.
4. **Learned routing can increase latency** — for mixed-confidence scenarios (`learned_medium`), both learned and deterministic solver sets are executed, increasing total execution time (e.g. COST_OPTIMIZATION: SYM-3 6.21ms → SYM-4 21.77ms).
5. **No online learning** — the model is static after training; it does not improve from live optimization results.

---

## 18. Reproducibility

To reproduce the model:

```bash
cd neuro-symbolic-orchestrator
$env:PYTHONPATH="."
python src/symbolic/optihive/learned_training.py
```

To run the benchmark:

```bash
$env:PYTHONPATH="."
python benchmarks/benchmark_sym4_learned_routing.py
```

To run all tests:

```bash
python -m pytest tests/
```

---

## 19. Why Learned Routing Does NOT Replace Symbolic Verification

The learned model predicts **which execution engines to run**. It does not:
- Evaluate whether a candidate is feasible
- Modify Z3 constraints
- Override ILP filtering
- Change budget, vCPU, RAM, or SLA validation

The `ILPSyntacticFilter` and `OptiHiveSelector` remain the authoritative arbiters of feasibility. If the learned router routes an infeasible request to GA/PSO, the resulting candidates will still be filtered out by the downstream constraint layer. The feasibility output of `OptimizationResult.is_feasible` is strictly determined by constraint verification, never by the routing layer.
