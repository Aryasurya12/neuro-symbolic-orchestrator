# SYM-6 Remediation: Deterministic Safety Arbitration Layer

## 1. Discovered SYM-3 / SYM-4 Failures

During the initial SYM-6 outcome-aware evaluation comparing routing configurations against the reference full-race oracle, real feasibility failures were uncovered:

1. **SYM-3 Deterministic Routing Failure**:
   - Workload: `E1_DR_Feasible` (`latency_max_ms=100.0`, `sla_availability_pct=99.9`)
   - Outcome: `INFEASIBLE` (candidate was infeasible while reference race was feasible at $53.00)
   - Failure Mechanism: SYM-3 scoring awarded Z3=90.0, PSO=55.0, GA=20.0. Because the Z3 score was > 70.0 with a margin > 20.0, the deterministic router selected `['Z3']` alone. Z3 alone was executed without candidate generation from heuristic solvers, resulting in an empty candidate pool in OptiHive.

2. **SYM-4 Learned Routing Failure**:
   - Workloads: `B1_Comfortable` and `B2_NearBoundary`
   - Outcome: `INFEASIBLE` (candidate was infeasible while reference race was feasible at $53.00)
   - Failure Mechanism: The learned model predicted `Z3` with >92% confidence for these constrained scenarios. Safety arbitration permitted `['Z3']` alone without ensuring heuristic companion solvers were present to generate multi-service instance allocation candidates.

---

## 2. Reproduction and Root Cause Analysis

### Reproduction Trace

```
Request: E1_DR_Feasible (latency_max_ms=100.0, sla_availability_pct=99.9, providers=['aws', 'azure'])
       ↓
SolverRoutingFeatures (has_latency=True, has_sla=True, is_highly_constrained=True)
       ↓
SYM-3 Deterministic Scoring (Z3=90, PSO=55, GA=20)
       ↓
Router Proposal: ['Z3'] alone
       ↓
Prior Safety Check: "if constrained and 'Z3' not in solvers: add Z3"
       → Skipped ('Z3' was already the only solver)
       ↓
Solver Execution: GraphSteeredZ3Solver only
       ↓
OptiHive Selection: Infeasible / No candidates
```

### Core Architecture Insights
1. **Z3 is a Formal SMT Verifier & Topology Placer**: In the orchestrator architecture, `GraphSteeredZ3Solver` formalizes disaster recovery, cross-region latency, and SLA placement constraints. Heuristic solvers (`GeneticAlgorithm`, `ParticleSwarmOptimization`) search the discrete multi-service instance configuration space.
2. **Missing Safety Guardrail**: Prior routing safety checks only enforced that Z3 must be added if missing from heuristic sets, but never enforced that heuristic candidate generators must accompany Z3 when Z3 was selected alone.
3. **Provider Normalization**: In `InfrastructureGraph`, node providers are capitalized (`"AWS"`, `"Azure"`), whereas incoming requests may supply lowercase provider keys (`["aws", "azure"]`). Case-insensitive comparison was missing in `constraints.py`, `z3_solver.py`, and `ILPSyntacticFilter`.
4. **Conditional Latency/SLA**: Latency and SLA constraints in `z3_solver.py` were asserted unconditionally, including when `latency_max_ms == 0.0`, asserting `lat <= 0.0` which is impossible for distinct physical cloud regions.

---

## 3. Safety Arbitration Architecture

> **"Adaptive routing proposes solver execution; deterministic safety arbitration retains authority over whether a solver required for hard feasibility validation may be omitted."**

The safety arbitration layer (`src/symbolic/optihive/safety_arbitration.py`) is a pure, deterministic, non-ML component positioned after routing proposals and before solver execution:

```
┌─────────────────────────────────┐
│ SymbolicOptimizationRequest     │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ AdaptiveSolverRouter            │
│  - SYM-3 Deterministic Scoring  │
│  - SYM-4 Learned Inference      │
└───────────────┬─────────────────┘
                │ Proposed Solvers (e.g. ['GA', 'PSO'] or ['Z3'])
                ▼
┌─────────────────────────────────┐
│ SafetyArbitrator (Deterministic)│
│  - Rule 1: Topology/SLA → Z3    │
│  - Rule 2: High Constraint → Z3 │
│  - Rule 3: Z3 Alone → +GA,+PSO  │
│  - Rule 4: Empty → Full Race    │
└───────────────┬─────────────────┘
                │ Final Safe Solver Set (e.g. ['GA', 'PSO', 'Z3'])
                ▼
┌─────────────────────────────────┐
│ Solver Execution Engine         │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ OptiHive / ILPSyntacticFilter   │
└─────────────────────────────────┘
```

The learned router's internal state and predictions are untouched; its recommendation is strictly treated as a proposal subject to deterministic verification.

---

## 4. Deterministic Rules Implemented

| Rule ID | Trigger Condition | Action | Rationale |
|---|---|---|---|
| `TOPOLOGY_CONSTRAINT_REQUIRES_Z3` | `latency_max_ms > 0` OR `sla_availability_pct > 0` OR topology/DR problem type | Add `Z3` | Topology and SLA guarantees require formal SMT reasoning. |
| `HIGHLY_CONSTRAINED_REQUIRES_Z3` | `is_highly_constrained == True` (`density >= 2`) | Add `Z3` | Complex constraint interactions require formal SAT/SMT verification. |
| `Z3_ALONE_REQUIRES_HEURISTIC_COMPANION` | `Z3` present AND neither `GA` nor `PSO` present | Add `GA` and `PSO` | Z3 does not generate discrete instance SKU candidates; heuristics must supply candidate pool. |
| `EMPTY_SOLVER_SET_FALLBACK` | Empty proposed set or internal arbitration exception | Fallback to `['GA', 'PSO', 'Z3']` | System must always fail safe to a complete race. |

Canonical deduplication enforces deterministic ordering `['GA', 'PSO', 'Z3']`.

---

## 5. Selective Safety (Preserving Routing Benefits)

Safety arbitration does **not** blindly run all solvers for all workloads:
- **Normal Compute / Sizing**: `['GA', 'PSO']` selected; runtime ~2.6 ms (avoids ~20 ms Z3 SMT solver overhead).
- **Cost Optimization**: `['GA', 'PSO']` selected; runtime ~2.8 ms.
- **Multi-Cloud Workloads**: `['GA', 'PSO']` selected; runtime ~2.7 ms.
- **Topology / Latency / SLA**: `['GA', 'PSO', 'Z3']` selected; runtime ~22–26 ms with 100% feasibility.

---

## 6. Regression & Benchmark Results

### SYM-6 Outcome Evaluation (13 Deterministic Scenarios)

| Configuration | Equivalent | Acceptable | Inferior | Infeasible | Failed | Both Infeasible |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BASELINE** | 9 | 0 | 0 | **0** | 0 | 4 |
| **SYM-3** | 9 | 0 | 0 | **0** | 0 | 4 |
| **SYM-4** | 9 | 0 | 0 | **0** | 0 | 4 |
| **CANDIDATE** | 9 | 0 | 0 | **0** | 0 | 4 |

- `E1_DR_Feasible` on SYM-3: **EQUIVALENT** (+0.0% cost delta, 1.04x runtime ratio)
- `B1_Comfortable` on SYM-4: **EQUIVALENT** (+0.0% cost delta, 1.04x runtime ratio)
- `B2_NearBoundary` on SYM-4: **EQUIVALENT** (+0.0% cost delta, 0.89x runtime ratio)
- Hard constraint violations: **0**

### Performance Across Workload Categories

| Workload Category | Selected Solvers | Feasible | Latency (ms) | Safety Triggered |
|---|---|:---:|:---:|:---:|
| Normal Workload | `['GA', 'PSO']` | True | 2.64 | False |
| Compute-Intensive | `['GA', 'PSO']` | True | 2.60 | False |
| Cost Optimization | `['GA', 'PSO']` | True | 2.82 | False |
| Multi-Cloud Workload | `['GA', 'PSO']` | True | 2.73 | False |
| DR / Topology Workload | `['GA', 'PSO', 'Z3']` | True | 26.31 | True |
| Latency Constrained | `['GA', 'PSO', 'Z3']` | True | 22.32 | True |
| SLA Constrained | `['GA', 'PSO', 'Z3']` | True | 24.55 | True |
| Infeasible Workload | `['GA', 'PSO', 'Z3']` | False | 15.80 | True |

**Summary Statistics:**
- Median Latency: **9.31 ms**
- Maximum Latency: **26.31 ms**
- Safety Trigger Frequency: **4/8 (50.0%)**

---

## 7. Hard-Constraint Safety Guarantees

Safety arbitration does **not** evaluate or alter feasibility decisions:
- `ILPSyntacticFilter` remains strictly authoritative over candidate validity.
- `GraphSteeredZ3Solver` hard assertions remain unaltered.
- Infeasible workloads (`B3_Tight`, `C1_Budget_Infeas`, `E2_DR_Infeasible`, `F1_Boundary`) remain strictly infeasible (`BOTH_INFEASIBLE`).
- Zero hard constraint violations occurred across Phase 9, Phase 10, and SYM-6 regressions.

---

## 8. Remaining Limitations

1. **Static Topology Graph**: `InfrastructureGraph` models 5 canonical cloud regions; dynamic graph discovery or customer-defined VPC endpoints require extending the graph model.
2. **Linear Latency Penalty**: Cost modeling for inter-region transfer latency uses a linear proxy coefficient ($0.25/ms), suitable for optimization tie-breaking but not detailed egress billing.
3. **Production Checkpoint Unpromoted**: Per promotion gate specifications, candidate checkpoints require >=70% validation accuracy; current candidate stands at 40% and remains unpromoted.
