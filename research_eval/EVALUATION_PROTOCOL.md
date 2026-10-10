# Neurasym Deterministic Evaluation Protocol & Scoring Contract

**Standard Version**: 2.0.0 (Research Integrity Milestone)  
**Authors**: Independent AI Research Auditor & Senior Python Research Engineer  
**Date**: October 2026  
**Scope**: Unified 4-Mode Evaluation for Cloud FinOps Optimization (Modes 1–4)  

---

## 1. Core Principles & Evaluation Philosophy

1. **Universal Metric Parity**: All four execution modes (Raw LLM, Schema LLM, Pure Symbolic, Neuro-Symbolic) are evaluated against the exact same mathematical ground truth and behavioral standard.
2. **Decoupled Verification**: The evaluation oracle is strictly independent of the solvers and models under test. No solver total, status flag, or model claim is accepted without independent verification against the frozen catalog and ground-truth requirements.
3. **Multi-Dimensional Orthogonality**: Semantic extraction accuracy, mathematical constraint feasibility, cost accuracy, optimality, and high-level outcome correctness are measured as separate orthogonal dimensions. No single aggregated metric may mask an underlying failure.
4. **No Selective Grading**: Every benchmark run must receive a deterministic evaluation result or be routed to a blinded human adjudication queue. Missing outputs, provider errors, and unparseable prose are never silently converted into arbitrary success or failure.

---

## 2. The Seven Distinct Evaluation Dimensions

```
                                  USER QUERY
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        1. INTERPRETATION FIDELITY              2. OUTCOME CORRECTNESS
     (vCPUs, RAM, SLA, Latency, Budget)       (FEASIBLE, INFEASIBLE,
                   │                           CLARIFICATION, CONFLICT,
                   ▼                           UNSUPPORTED)
        3. CONSTRAINT SATISFACTION                       │
     (Hardware, Network, Cloud Bounds)                   │
                   │                                     │
                   ▼                                     ▼
          4. COST ACCURACY                      6. REFUSAL & CLARIFICATION
     (Recomputed from Frozen Catalog)           (Justified Rationale & Grounding)
                   │                                     │
                   ▼                                     │
          5. OPTIMALITY CERTIFICATION                    │
     (Proved Bound vs Reference Optimum)                 │
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                           7. STRICT END-TO-END SUCCESS
```

### Dimension 1: Semantic Interpretation Fidelity (`interpretation_correct: bool`)
- **Definition**: Did the system correctly extract every explicit user requirement without omission, hallucination, or distortion?
- **Checked Fields**:
  - `cloud_providers`: Exact set equality with requested providers.
  - `required_vcpus`: $\ge$ requested vCPU count.
  - `required_ram_gb`: $\ge$ requested RAM capacity.
  - `budget_max_usd`: Numeric match with stated ceiling.
  - `sla_availability_pct`: Match with stated availability percentage.
  - `latency_max_ms`: Match with stated maximum latency.
  - `target_bandwidth_mbps`, `target_cpu_pct`, `max_cpu_pct`: Exact numerical match for scaling workloads.
- **Rule**: If an extracted field differs from the manifest ground truth (e.g. extracted 48 GB when user asked for 32 GB), `interpretation_correct` is `False`.

---

### Dimension 2: Outcome Correctness (`outcome_correct: bool`)
- **Definition**: Did the system correctly classify the high-level solvability of the task?
- **Categorical Outcomes**:
  1. `FEASIBLE`: Solvable problem with at least one valid allocation plan in the catalog.
  2. `INFEASIBLE`: Provably impossible under the catalog due to budget, hardware, latency, or SLA bounds.
  3. `CLARIFICATION_REQUIRED`: Missing critical specifications (e.g. missing budget, missing workload bandwidth).
  4. `CONFLICTING_REQUIREMENTS`: Contradictory constraints that cannot be satisfied under any configuration (e.g. Target CPU 70% with strict ceiling 60%).
  5. `UNSUPPORTED`: Out-of-domain request (e.g. GPU training clusters, Oracle database migrations, cooking recipes).

---

### Dimension 3: Constraint Satisfaction (`constraints_satisfied: bool`)
- **Definition**: For a proposed allocation plan, do all physical and operational constraints hold under independent recomputation?
- **Verification Invariants**:
  - **VM Allocation**: $\sum \text{vCPU}_i \ge \text{required\_vCPUs}$, $\sum \text{RAM}_i \ge \text{required\_RAM}$, Provider $\in \text{allowed\_providers}$.
  - **Disaster Recovery**: Selected region pair $(r_1, r_2)$ must satisfy $\text{Latency}(r_1, r_2) \le \text{latency\_max\_ms}$ and $1 - (1 - A_1)(1 - A_2) \ge \text{sla\_availability\_pct}$.
  - **Continuous Scaling**: Replicas $R \ge 1$, Traffic $\le R \times 75 \text{ Mbps}$, CPU Utilization $\frac{\text{Traffic}}{R \times 75} \le \text{max\_cpu\_ceiling}$.

---

### Dimension 4: Cost Accuracy (`cost_correct: bool`)
- **Definition**: Does the self-reported monthly cost agree with independent recomputation from the frozen SQLite catalog?
- **Tolerance**:
  $$\text{Absolute Error} = |\text{Reported Cost} - \text{Recomputed Cost}| \le \$0.02$$
  (Accounts for cent rounding of hourly rates across 730 monthly hours).

---

### Dimension 5: Optimality Verification (`optimality_status: str`)
- **Classification**:
  - `PROVED_OPTIMAL`: Exact mathematical solvers (HiGHS MILP, Z3 SMT) with proven optimality bounds within $0.01 tolerance of manifest optimum.
  - `HEURISTIC_FEASIBLE`: Stochastic meta-heuristics (PSO, GA) that discovered a valid feasible point without a mathematical optimality bound.
  - `SUBOPTIMAL`: Feasible allocation that costs more than the reference optimum by $> \$0.01$.
  - `NOT_APPLICABLE`: Infeasible, conflicting, unsupported, or clarification requests.

---

### Dimension 6: Refusal & Clarification Correctness (`refusal_correct: bool`)
- **Definition**: Was a refusal or clarification request mathematically and logically justified?
- **Negative Invariant**: A response is NOT awarded refusal success simply because it contains refusal-like phrasing. The refusal must match an actual `INFEASIBLE`, `CLARIFICATION_REQUIRED`, `CONFLICTING_REQUIREMENTS`, or `UNSUPPORTED` state in the ground-truth benchmark.

---

### Dimension 7: Strict End-to-End Success (`strict_success: int`)
- **Definition**: Strict binary success ($1$ or $0$) requiring complete correctness across all applicable dimensions:
  - For `FEASIBLE` queries: `interpretation_correct == True` AND `outcome_correct == True` AND `constraints_satisfied == True` AND `cost_correct == True`.
  - For `INFEASIBLE` / `CONFLICTING` / `UNSUPPORTED` / `CLARIFICATION_REQUIRED`: `outcome_correct == True` AND `refusal_correct == True`.
- **Core Axiom**: *"A correct answer obtained from a wrong interpretation cannot earn strict success."*

---

## 3. Explicit Scoring Policy Matrix

| Expected Outcome | Mode Output Category | Interpretation Match | Constraints Valid | Cost Accurate | Strict Success | Outcome Label |
|---|---|---|---|---|---|---|
| `FEASIBLE` | Valid Optimal Plan | True | True | True | **1** | `CORRECT` |
| `FEASIBLE` | Valid Plan (Suboptimal) | True | True | True | **0** | `CORRECT_BUT_COSTLIER` |
| `FEASIBLE` | Plan with Wrong Extraction | False | True | True | **0** | `RIGHT_ANSWER_WRONG_READING` |
| `FEASIBLE` | Violates Constraints | Any | False | Any | **0** | `WRONG_PLAN` |
| `FEASIBLE` | Refusal / Clarification | Any | N/A | N/A | **0** | `WRONG_OUTCOME` |
| `INFEASIBLE` | Refusal / Infeasible Notice | True / N/A | N/A | N/A | **1** | `CORRECT_REFUSAL` |
| `INFEASIBLE` | Generates Plan | Any | Any | Any | **0** | `WRONG_OUTCOME` |
| `CONFLICTING` | Refuses Conflicting Spec | True / N/A | N/A | N/A | **1** | `CORRECT_REFUSAL` |
| `CONFLICTING` | Generates Unconstrained Plan| Any | Any | Any | **0** | `WRONG_OUTCOME` |
| `CLARIFICATION` | Requests Clarification | True / N/A | N/A | N/A | **1** | `CORRECT_REFUSAL` |
| `CLARIFICATION` | Hallucinates / Guesses Plan | Any | Any | Any | **0** | `WRONG_OUTCOME` |
| `UNSUPPORTED` | Rejects Out-of-Domain Task | True / N/A | N/A | N/A | **1** | `CORRECT_REFUSAL` |
| `UNSUPPORTED` | Attempts Allocation Plan | Any | Any | Any | **0** | `WRONG_OUTCOME` |
| `ANY` | Unparseable Prose (Mode 1) | N/A | N/A | N/A | **0 / Unresolved** | `UNPARSEABLE` |
| `ANY` | Provider Timeout / 429 / Crash | N/A | N/A | N/A | **0** | `PROVIDER_FAILURE` |

---

## 4. Currency, Billing & Catalog Assumptions

1. **Monthly Multiplier**: Flat $730.0$ hours per month ($365 \times 24 / 12$).
2. **SKU Pricing**: Strictly sourced from frozen `data/cloud_finops.db`.
3. **Data Egress**: Zero egress cost assumption documented as a closed-world benchmark limitation.
4. **Network Latency**: Fixed deterministic distance matrix in SQLite `region_topologies`.
5. **DR SLA**: Independent joint availability: $A_{\text{composite}} = 1 - (1 - A_1)(1 - A_2)$.
