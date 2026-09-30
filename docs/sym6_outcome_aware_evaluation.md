# SYM-6: Outcome-Aware Routing Evaluation & Safe Promotion Gate

## 1. Motivation

SYM-5 introduced telemetry collection and trained a candidate model (`sym5-v1`) from real
routing observations. However, SYM-5's training labels were derived from *which solver set
was used*, not from whether that routing decision was actually better or worse than alternative
approaches on the same workload.

SYM-6 introduces a **deterministic, reproducible evaluation framework** that can answer:

> *"Was the selected routing decision actually useful compared with running all solvers?"*

It does this by comparing each routing configuration against a **reference race oracle**
(full GA + PSO + Z3) on identical benchmark workloads, with documented outcome labels and
a multi-condition promotion gate.

---

## 2. Architecture

```
                   ┌──────────────────────────────────────┐
                   │  OFFLINE EVALUATION ORACLE            │
                   │  benchmarks/benchmark_sym6_*.py       │
                   └──────────────┬───────────────────────┘
                                  │
           Identical deterministic scenarios (13 scenarios)
                                  │
          ┌───────────┬───────────┼─────────────┬────────────┐
          │           │           │             │            │
       BASELINE     SYM-3       SYM-4       CANDIDATE   REFERENCE
    (GA+PSO+Z3    (Deterministic (Production  (SYM-5     (Full race
     no routing)   routing only)  learned)    candidate)  oracle)
          │           │           │             │            │
          └───────────┴───────────┼─────────────┘            │
                                  │                           │
                         RoutingOutcome                 RoutingOutcome
                                  │                           │
                         compare_outcome()─────────────────────┘
                                  │
                         OutcomeComparison
                                  │
                       evaluate_promotion_gate()
                                  │
                         PromotionGateResult
                         (eligible_for_promotion: bool)
                                  │
                    Human authorization required
                    (--promote flag on CLI)
```

---

## 3. Reference Race

The **reference race** runs all three solvers (GA, PSO, Z3) and selects the best result
via OptiHive. It is an **offline evaluation oracle** only — it does not replace the
production router and is never invoked during live requests.

Purpose: establish a ground-truth "best available" result for each scenario so that
selective routing configurations can be judged against it.

The reference race passes through all existing hard-constraint validation (ILP filter,
Z3 constraints, OptiHive selection). It does **not** bypass any safety mechanism.

---

## 4. Evaluation Configurations

| Config | Description |
|--------|-------------|
| `BASELINE` | Full GA+PSO+Z3 race, no adaptive routing |
| `SYM-3` | Deterministic scoring only (learned router disabled) |
| `SYM-4` | Production learned checkpoint (`v1.0.0`) |
| `CANDIDATE` | SYM-5 candidate checkpoint (`sym5-v1`) |
| `REFERENCE` | Full race oracle (basis for comparison) |

---

## 5. Outcome Labels

| Label | Condition | Notes |
|-------|-----------|-------|
| `EQUIVALENT` | Both feasible; cost within 5% of reference | Routing is as good as full race |
| `ACCEPTABLE` | Both feasible; cost within 20%, runtime ≤ 3× reference | Suboptimal but serviceable |
| `INFERIOR` | Both feasible; cost > 20% worse OR runtime > 3× reference | Routing produced material degradation |
| `INFEASIBLE` | Candidate infeasible, reference feasible | **Routing failure — always fails gate** |
| `BOTH_INFEASIBLE` | Both infeasible | Not a routing error; workload is inherently infeasible |
| `NOT_EVALUATED` | Reference also infeasible | No comparison basis |
| `FAILED` | Candidate raised an exception | Technical failure |

### Threshold Rationale

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Cost tolerance (EQUIVALENT) | 5% | Reflects real cloud pricing rounding and solver non-determinism |
| Cost degradation limit (ACCEPTABLE) | 20% | Accounts for suboptimal but usable routing choices |
| Runtime limit factor (ACCEPTABLE) | 3× | Selective routing must not be dramatically slower than full race |
| Minimum agreement rate | 80% | Majority of directly comparable scenarios must be EQUIVALENT or ACCEPTABLE |
| Val accuracy threshold | 0.70 | Minimum model accuracy for promotion consideration |

All thresholds are constants in `outcome_evaluation.py` and are configurable per call.

---

## 6. Promotion Gate

The gate evaluates **all five conditions**. ALL must pass for `eligible_for_promotion=True`.

| Condition | Check |
|-----------|-------|
| 1. Validation accuracy | `val_accuracy >= 0.70` |
| 2. Feasibility preservation | Zero `INFEASIBLE` outcomes |
| 3. Hard constraint safety | Zero hard constraint violations |
| 4. Outcome agreement | ≥ 80% of evaluatable scenarios are EQUIVALENT or ACCEPTABLE |
| 5. Checkpoint integrity | Candidate JSON parseable with all required fields |

### Key safety properties

- `eligible_for_promotion = True` **does NOT cause promotion**
- Default behavior is always dry-run
- `--promote` flag on CLI is required for human authorization
- Promotion uses atomic write (`os.replace`) with backup
- If any step fails, existing checkpoint is preserved unchanged

---

## 7. Evaluation Dataset

**Source**: BENCHMARK_SYNTHETIC (deterministic workload parameters)  
**Derivation**: Same scenarios as Phase 9/10 evaluation infrastructure  
**Random seeds**: N/A (all scenarios are deterministically constructed)  
**Size**: 13 scenarios

| Category | Scenarios |
|----------|-----------|
| Feasible | A1_Small, A2_Medium, A3_Large, B1_Comfortable, B2_NearBoundary |
| Infeasible | B3_Tight, C1_Budget_Infeasible, C2_Resource_Infeasible, F1_Boundary |
| Multi-provider | D1_Multi_AWS_Azure, D2_Multi_All |
| Latency/SLA | E1_DR_Feasible, E2_DR_Infeasible |

> **Limitation**: 13 benchmark scenarios cannot support statistically meaningful claims about
> generalization to unseen workloads. Results are valid only for these specific scenarios.

---

## 8. SYM-6 Benchmark Results (Actual — Measured)

### Outcome comparisons

| Config | EQUIVALENT | ACCEPTABLE | INFERIOR | INFEASIBLE | BOTH_INFEASIBLE |
|--------|-----------|-----------|---------|-----------|----------------|
| BASELINE | 9 | 0 | 0 | 0 | 4 |
| SYM-3 | 8 | 0 | 0 | **1** | 4 |
| SYM-4 | 7 | 0 | 0 | **2** | 4 |
| CANDIDATE | 9 | 0 | 0 | 0 | 4 |

**Observations (honest)**:

- BASELINE and CANDIDATE both achieved 9/9 EQUIVALENT on evaluatable scenarios
- SYM-3 had 1 INFEASIBLE outcome (E1_DR_Feasible scenario — deterministic router omitted Z3 at solver-level even though safety arbitration should have included it)
- SYM-4 had 2 INFEASIBLE outcomes (B1_Comfortable, B2_NearBoundary — production learned router selected GA+PSO only for these scenarios; Z3 needed to produce feasible result)
- These infeasible routing outcomes do NOT indicate a hard constraint bypass — the final feasibility determination was still made correctly by the symbolic layer

> **Important**: The INFEASIBLE outcome label means the candidate routing strategy produced an
> infeasible optimization result while the reference race found a feasible solution. This is
> a routing quality failure, not a safety failure — hard constraints were never bypassed.

### Latency summary

| Config | Median (ms) | P95 (ms) | Max (ms) | N |
|--------|------------|---------|---------|---|
| BASELINE | 18.7 | 30.8 | 30.8 | 13 |
| SYM-3 | 14.9 | 32.3 | 32.3 | 13 |
| SYM-4 | 15.2 | 26.3 | 26.3 | 13 |
| CANDIDATE | 18.6 | 29.2 | 29.2 | 13 |
| REFERENCE | 18.5 | 38.0 | 38.0 | 13 |

> **Note**: P95 is computed from 13 samples, which is too small for statistically reliable estimates.
> Do not extrapolate from these figures.

---

## 9. Promotion Gate Result

| Gate Condition | Status | Detail |
|---------------|--------|--------|
| val_accuracy >= 0.70 | **FAIL** | Candidate val_accuracy = 0.40 (dataset: 25 records, val split = 5) |
| Feasibility preserved | PASS | 0 INFEASIBLE outcomes in 13 scenarios |
| Hard constraints | PASS | 0 violations |
| Outcome agreement | PASS | 9/9 (100%) evaluatable scenarios EQUIVALENT |
| Checkpoint integrity | PASS | Candidate file valid |

**Overall: NOT ELIGIBLE FOR PROMOTION**

Reason: val_accuracy 0.40 < 0.70 threshold

### Why val_accuracy is 0.40

The SYM-5 candidate was trained on 25 telemetry records collected from benchmark runs.
The 20% held-out validation split contains only 5 samples. Additional benchmark runs
(from the SYM-6 test suite) added records with a different solver distribution, causing
the validation accuracy to drop from the previously reported 80% to 40%.

This is the correct behaviour — the gate is working as intended. A model with this low
validation accuracy should NOT be promoted, regardless of outcome agreement scores.

**Action required**: Collect more diverse telemetry (≥ 50 records across all routing
modes) then retrain with `train_sym5_from_telemetry.py --promote`.

---

## 10. Production Model Status

```
Production checkpoint : learned_checkpoint.json
Production version    : v1.0.0
Status                : UNCHANGED
```

The production SYM-4 checkpoint was not modified in any way by the SYM-6 evaluation.

---

## 11. CLI Usage

```bash
# Dry-run evaluation (no promotion)
python benchmarks/evaluate_sym6_candidate.py

# Evaluate with promotion (only if gate passes)
python benchmarks/evaluate_sym6_candidate.py --promote

# Use custom candidate checkpoint
python benchmarks/evaluate_sym6_candidate.py --candidate-path /path/to/checkpoint.json
```

---

## 12. Performance Notes

The reference race intentionally runs all three solvers (GA + PSO + Z3). Its latency
(median 18.5ms) is therefore comparable to BASELINE, which also runs all three.

**Offline evaluation runtime ≠ production request latency.**

Production SYM-4 routing (median 15.2ms) is faster than BASELINE/REFERENCE because
selective routing avoids unnecessary solver invocations. The offline evaluation framework
does not change production behavior.

---

## 13. Files Created by SYM-6

| File | Purpose |
|------|---------|
| `src/symbolic/optihive/outcome_evaluation.py` | Outcome labels, comparison, promotion gate |
| `src/symbolic/optihive/eval_runner.py` | Isolated runners for each configuration |
| `benchmarks/benchmark_sym6_outcome_evaluation.py` | Full benchmark orchestrator |
| `benchmarks/evaluate_sym6_candidate.py` | CLI alias |
| `tests/test_sym6_outcome_evaluation.py` | 43 tests |
| `benchmarks/results/sym6_outcome_evaluation.json` | Machine-readable results |
| `benchmarks/results/sym6_outcome_evaluation.csv` | CSV results |
| `docs/sym6_outcome_aware_evaluation.md` | This document |

---

## 14. Limitations

1. **13 scenarios** — insufficient for statistical significance. Treat results as qualitative indicators only.
2. **P95 latency** — requires ≥ 20 samples for meaningful estimates; reported values are approximate.
3. **Outcome labels from routing only** — labels compare candidate vs reference outcome, not across all possible routings. A CANDIDATE labelled EQUIVALENT may still be achieving it via a different solver path.
4. **SYM-3/SYM-4 INFEASIBLE outcomes** — real finding; production router sometimes fails to include Z3 for workloads where Z3 is needed. Not a safety bypass — the symbolic constraint layer determined final feasibility correctly.
5. **Val accuracy instability** — with only 25 training records and a 5-sample validation set, validation accuracy is highly sensitive to distribution shift. More telemetry is required for stable estimates.
6. **No cross-validation** — single 80/20 split used. Would benefit from k-fold CV with larger datasets.
7. **Benchmark-synthetic data only** — no real cloud pricing data used in evaluation workloads.
8. **test_websocket_concurrency skipped** — pre-existing Windows Z3 GC race condition (not introduced by SYM-6).

---

## 15. Next Steps

1. Collect ≥ 50 diverse telemetry records from real/varied optimization requests
2. Retrain candidate: `python benchmarks/train_sym5_from_telemetry.py`
3. Re-run SYM-6 evaluation: `python benchmarks/evaluate_sym6_candidate.py`
4. If gate passes: `python benchmarks/evaluate_sym6_candidate.py --promote`
5. Investigate SYM-3/SYM-4 INFEASIBLE routing cases (GA+PSO selected without Z3 on workloads that need constraint verification)
