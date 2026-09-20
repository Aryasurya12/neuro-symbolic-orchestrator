# Phase 4 — SYM-3: OptiHive-Inspired Candidate Selection

## 1. Objective
The objective of Phase 4 is to introduce an "OptiHive-inspired" orchestration layer over the symbolic engines (GA, PSO, Z3) that acts as a robust selection funnel. This phase strictly enforces that hard constraints dictate physical feasibility (via an ILP syntactic filter) and ranks valid candidates using a deterministic, probabilistic latent-class model (EM Selection).

## 2. Architecture
```
             Optimization Engines
         ┌────────┬────────┬────────┐
         │   GA   │  PSO   │  Z3    │
         └────────┴────────┴────────┘
                   ↓
             Candidate Pool
                   ↓
          ┌──────────────────┐
          │ ILP Filter       │ (Hard Feasibility)
          └──────────────────┘
                   ↓
          Feasible Candidates
                   ↓
          ┌──────────────────┐
          │ EM Selector      │ (Soft Ranking)
          └──────────────────┘
                   ↓
            Final Candidate
```

## 3. ILP Syntactic Filter
Located in `src/symbolic/optihive/ilp_filter.py`.
- **Role:** Fallback mathematical verifier.
- **Enforcement:** Evaluates budget, explicit variables (e.g. `provider`), and internal `ConstraintStatus` metrics of every candidate. If any constraint is demonstrably breached, it explicitly marks the candidate as structurally invalid and appends violations.

## 4. EM Selector
Located in `src/symbolic/optihive/em_selector.py`.
- **Features:** Normalized Cost, Normalized Runtime.
- **Latent Classes:** $K=2$ assuming solutions belong to either an "optimal" or "suboptimal" generative class.
- **Algorithm:** Deterministic Expectation-Maximization (EM).
  - *Initialization*: Seeded random initialization for reproducibility.
  - *E-step*: Calculates responsibility (probability a candidate belongs to a class).
  - *M-step*: Updates class means, covariances, and weights.
  - *Convergence*: Stops when the log-likelihood delta drops below a set tolerance or at max iterations.
- **Hard Constraint Safety:** EM strictly operates *only* on candidates passed by the ILP Syntactic filter. An infeasible candidate cannot mathematically be selected, securing the architectural separation mandated by Phase 3.

## 5. Noise Experiment Results
A mandatory experimental benchmark (`test_noisy_data_experiment` in `test_solver_selection.py`) tracks selection robustness.
- **Noise model:** $\mathcal{N}(0, 5.0)$ injected onto base costs of an optimal candidate vs suboptimal candidates.
- **Trials:** 100
- **Seed:** 42
- **Selection Metric:** Percentage of trials where the EM selector successfully identifies and retains the true reference optimal candidate despite noisy feature variables.
- **Measured Result:** 85.0% (as measured directly in the pytest suite).

## 6. Limitations
- **Small Candidate Pool:** Currently testing operates on $N \le 3$ candidates from the upstream symbolic layer.
- **No Learned OptiHive Model:** This is a deterministic proxy/inspired abstraction, lacking the external data collection required to fit a true global OptiHive neural classifier.
- **Synthetic Evaluation:** Noise profiles are purely synthetic distributions rather than live telemetry drift.

## 7. Future Integration
This orchestration layer finalizes the purely internal logic of Part-B. It provides a robust, standardized `OptimizationResult` return type cleanly primed for Phase-5+ integration (FastAPI routing, WebSocket streaming, and live production ChromaDB lookup mapping).
