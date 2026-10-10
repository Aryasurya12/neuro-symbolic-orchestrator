# Neurasym Research Audit: Methodological Risks & Epistemological Validity Report

**Project**: Neurasym (Neuro-Symbolic Cloud FinOps Orchestrator)  
**Auditor**: Independent AI Research Auditor & Senior Python Research Engineer  
**Date**: October 2026  
**Repository**: `C:\Users\user\neurasym`  
**Git Branch**: `research-audit`  

---

## 1. Executive Summary

This report provides a formal methodological risk assessment of the Neurasym research study. While Neurasym's neuro-symbolic architecture provides genuine mathematical safety guarantees over pure generative LLMs, several **methodological vulnerabilities** in the current evaluation framework threaten the validity and external generalizability of published claims.

Key findings include:
1. **Single-Trial LLM Variability ($N=1$)**: Live API calls were executed with single repetitions per query, omitting statistical confidence intervals and variance metrics.
2. **Evaluator-Solver Coupling**: The `IndependentChecker` relies on the exact same SQLite database catalog, pricing assumptions, and regional graph topologies as the solvers under test.
3. **Epistemological Conflation of Heuristics with Formal Proofs**: Telemetry labels Particle Swarm Optimization (PSO) continuous scaling outputs as `PROVED_OPTIMAL` / `PROVED_FEASIBLE`, conflating heuristic convergence with exact mathematical certificates.
4. **Cloud FinOps Realism Gap**: The pricing catalog uses simplified synthetic models (flat 730 hours/month, zero data transfer cost, unconstrained region SKU availability) that do not reflect real-world multi-cloud billing dynamics.

---

## 2. Comprehensive Risk Assessment Matrix

| ID | Risk Category | Severity | Current Methodology | Risk Impact | Recommended Mitigation |
|---|---|---|---|---|---|
| **MR-01** | Statistical Rigor | **P1 (High)** | $N=1$ single trial per query across all modes in `study_run_02.csv` | High vulnerability to live LLM generation variance and provider routing jitter | Execute $N \ge 5$ repeated trials across live LLM modes; report mean, standard deviation, and Wilson score intervals |
| **MR-02** | Evaluator Coupling | **P1 (High)** | `IndependentChecker` imports `cloud_finops.db` and shares hardcoded 730 hr/mo constants with solvers | Shared assumptions between solver and checker mask systematic domain modeling errors | Author decoupled evaluator test vectors with independent reference calculations |
| **MR-03** | Heuristic Proof Conflation | **P2 (Medium)** | PSO telemetry records `proof_status = "PROVED_OPTIMAL"` upon local convergence | Misrepresents stochastic meta-heuristics as formal mathematical proofs | Restrict `PROVED_OPTIMAL` strictly to HiGHS MILP and Z3 SMT; label PSO as `HEURISTIC_CONVERGED` |
| **MR-04** | Cloud Billing Realism | **P2 (Medium)** | Static hourly rate × 730; zero egress; unmodeled burstable credits and committed use | Evaluation claims of "real-world FinOps optimality" are overstated | Formally scope paper claims to "Controlled Synthetic Cloud Allocation Benchmark" |
| **MR-05** | Template Overlap | **P2 (Medium)** | Dev and Eval queries share similar phrasing skeletons (e.g., "Cheapest 4 vCPU 16GB RAM") | Potential overestimation of neural extractor out-of-distribution generalization | Introduce adversarial and human-authored external query distributions |

---

## 3. Deep-Dive Analysis of Primary Risks

### 3.1 Evaluator Independence & Closed-World Modeling (MR-02)

In `src/verifiers/independent_checker.py`:
- The verification engine queries `data/cloud_finops.db` to check instance costs and RAM/vCPU capacities.
- It hardcodes the monthly multiplier:
  $$\text{Monthly Cost} = \text{Hourly Rate} \times 730.0$$
- It computes disaster recovery availability assuming independent joint failure:
  $$A_{\text{composite}} = 1 - (1 - A_{\text{source}}) \times (1 - A_{\text{target}})$$

#### Methodological Implication
Because the solvers (HiGHS MILP, Z3 SMT) and the `IndependentChecker` share the exact same underlying SQLite tables, mathematical formulas, and pricing abstractions:
1. An error in the database catalog (e.g., wrong unit price) is accepted by both solver and checker.
2. The benchmark measures **internal consistency with the closed-world schema**, not absolute real-world cloud allocation correctness.

---

### 3.2 Heuristics vs. Formal Proofs (MR-03)

In `src/proof/stage_trace.py` and `src/optimizers/pso_solver.py`:
- Mode 4 executes Particle Swarm Optimization for continuous autoscaling (CPU utilization target vs. replica count).
- The resulting trace frequently populates `proof_status = "PROVED_OPTIMAL"`.

#### Mathematical Reality
- **HiGHS MILP**: Solves Mixed-Integer Linear Programs via Branch-and-Cut, producing a provable lower/upper bound and mathematical optimality certificate ($\text{gap} \le \epsilon$).
- **Z3 SMT**: Proves satisfiability or unsatisfiability using first-order logic with DPLL(T) theories.
- **PSO / GA**: Stochastic meta-heuristics that search non-convex continuous parameter spaces. They can discover high-quality feasible points, but **cannot prove mathematical optimality or global bounding**.

Claiming a "mathematical proof of optimality" for PSO scaling runs is scientifically indefensible. The terminology must be refined to `HEURISTIC_CONVERGED` or `LOCAL_OPTIMUM_CANDIDATE`.

---

### 3.3 Statistical Significance & Single-Trial Limitations (MR-01)

The primary comparative dataset (`study_run_02.csv`) records exactly 1 trial for each of the 29 queries across the 4 modes (116 total records).
- Generative models (`openai/gpt-oss-120b` on Groq) exhibit non-zero sampling temperature and non-deterministic quantization/routing behaviors.
- A single trial provides a point estimate without error bars, making it impossible to evaluate whether a failure was a systematic inability or transient sampling noise.
- Future published evaluations must report $N \ge 5$ repetitions per query for generative modes, with bootstrapped confidence intervals.

---

## 4. Methodological Remediation Roadmap

1. **Precision in Terminology**:
   - Separate exact formal proofs (MILP / SMT) from heuristic solutions (PSO / GA) in all figures, telemetry, and paper text.
2. **Decoupled Reference Verification**:
   - Establish pre-computed static mathematical ground truth in `final_query_manifest.json` independent of runtime SQLite calls.
3. **Multi-Trial Execution & Statistical Reporting**:
   - Re-run live LLM modes across $N=5$ trials to establish variance, failure distributions, and standard errors.
4. **Transparent Benchmark Scoping**:
   - State clearly in documentation and publications that the FinOps environment represents a curated 29-query synthetic benchmark under closed-world assumptions.
