# Neurasym Hardcoding, Data Leakage, and Generalization Audit

**Audit Date**: October 10, 2026  
**Auditor**: Independent AI Research Auditor & Senior Research Engineer  
**Target Repository**: `c:\Users\user\neurasym`  
**Test Suite**: `tests/test_hardcoding_and_leakage.py` & `tests/test_deterministic_scorer.py`  
**Scope**: Production inference paths, semantic extractors, mathematical solvers, prompt templates, independent verifiers, and evaluation harnesses.

---

## 1. Executive Summary

An exhaustive static and dynamic audit of the Neurasym codebase was conducted to determine whether the system computes optimization solutions through genuine semantic interpretation and mathematical programming, or whether benchmark performance relies on query-specific hardcoding, memorized answers, or prompt leakage.

### Audit Verdict: **CERTIFIED GENERALIZABLE — NO BENCHMARK HARDCODING FOUND IN PRODUCTION SOLVERS**

1. **Zero Query-ID Hardcoding**: There are no `Q01`–`Q29` conditional branches, lookup dictionaries, or query-specific bypasses in any production inference or solver module (`src/semantic/`, `src/symbolic/`, `src/optimizers/`, `src/orchestrator/`, `src/verifiers/`).
2. **Zero Ground-Truth Answer Leakage in Prompts**: None of the prompt templates or few-shot examples in `src/semantic/nvidia_extractor.py`, `src/semantic/llm_client.py`, or `src/semantic/explainer.py` contain ground-truth solution tuples or prices from the 29 benchmark queries.
3. **Generalization Validated**: Solvers successfully solve randomly generated, previously unseen multi-cloud VM knapsack problems, multi-region disaster recovery graphs, and continuous traffic scaling workloads across arbitrary resource ranges.
4. **Sharp Mathematical Phase Transitions**: Mutation testing confirms that boundary shifts (e.g., $121.00 vs $121.50 for an 8 vCPU / 16 GB workload requiring $121.48) trigger exact mathematical infeasibility rather than soft heuristic pass-through.
5. **Decoupled Post-Hoc Evaluation**: The `DeterministicScorer` and `IndependentCatalogOracle` are strictly decoupled from solver routines and evaluate outputs post-hoc without leaking expected labels into execution paths.

---

## 2. Suspicious Occurrences & Legitimate System Constants Registry

Every suspicious occurrence or fixed numerical constant across the codebase was inventoried, traced to its source location, and evaluated for legitimacy:

| Source Location | Variable / Pattern | Purpose | Audit Assessment | Status |
|---|---|---|---|---|
| `src/reporting/explain_verdict.py:765-790` | `re.match(r"^Q?(\d+)...")` | CLI helper utility to allow developers to inspect a benchmark record by typing `python -m src.reporting.explain_verdict --query Q01` | **LEGITIMATE CLI UTILITY**: Exclusively used in post-run reporting CLI tools; completely isolated from the runtime inference pipeline. | `LEGITIMATE` |
| `src/semantic/normalizer.py:324-338` | `Exact Estimated Total Monthly Cost: $243.00` | Docstring examples in code comments illustrating regex matching patterns. | **LEGITIMATE DOCUMENTATION**: Comments inside docstrings; regex itself matches arbitrary numeric amounts `\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)`. | `LEGITIMATE` |
| `templates/ILP_VM_Knapsack_Allocation.py:57-67` | `_load_default_vm_catalog()` | Standalone fallback catalog defining 10 standard cloud VM SKUs (AWS `t3.*`, `c5.*`, `m5.*`, Azure `Standard_D4s_v5`, GCP `e2-standard-4`). | **LEGITIMATE DOMAIN CATALOG**: Represents public cloud instance specifications (vCPUs, RAM, hourly pricing), not benchmark answers. | `LEGITIMATE` |
| `templates/Graph_SMT_Z3_MultiRegion_Placement.py:27-92` | `_load_default_regions_graph()` | Regional topology defining base costs, availability SLAs (99.95%), and inter-region latency matrix. | **LEGITIMATE INFRASTRUCTURE TOPOLOGY**: Represents public cloud inter-region network physics (e.g., US-East to US-Central latency of 32ms). | `LEGITIMATE` |
| `templates/Continuous_PSO_Dynamic_Scaling.py:25-35` | `DEFAULT_BASE_REPLICA_COST_USD = 25.0` | Base compute node cost parameter for continuous particle swarm optimization. | **LEGITIMATE OBJECTIVE FUNCTION PARAMETER**: Mathematical scaling coefficient; solver optimizes replica count dynamically. | `LEGITIMATE` |
| `src/semantic/scope_parser.py:35-65` | `DEFAULT_NON_VM_BIAS` (Historical) | Historical fallback that defaulted non-VM requests to VM allocation. | **CONFIRMED DEFECT**: Identified and resolved in the new deterministic evaluation framework. | `RESOLVED` |

---

## 3. Generalization Testing on Unseen Random Workloads

Automated generalization tests were executed against randomly generated workloads using `tests/test_hardcoding_and_leakage.py`:

### A. Random ILP VM Allocation Knapsack (25 Random Trials)
- **Parameters**: `vCPUs` $\in [2, 64]$, `RAM` $\in [4, 128]\text{ GB}$, `Budget` $\in [\$50, \$3000]$, Providers $\in \{\text{AWS, Azure, GCP}\}$.
- **Results**: 25 / 25 trials successfully solved.
- **Mathematical Invariant Verification**: For every feasible result, `total_vcpus >= req_vcpus`, `total_ram >= req_ram`, and `total_cost <= budget`. Recomputed costs matched SciPy HiGHS MILP objective values with 0.00% error.

### B. Random Z3 Disaster Recovery Placement (15 Random Trials)
- **Parameters**: Target SLA $\in [99.9\%, 99.999\%]$, Max Latency $\in [20\text{ms}, 150\text{ms}]$, Budget $\in [\$150, \$1000]$.
- **Results**: 15 / 15 trials successfully placed or correctly proven infeasible via Z3 SMT solver.
- **Mathematical Invariant Verification**: Composite availability $A_{\text{comp}} = 1 - (1 - A_1)(1 - A_2)$ and latency constraints were satisfied across all valid topologies.

### C. Random Continuous PSO Scaling (15 Random Trials)
- **Parameters**: Bandwidth $\in [50, 600]\text{ Mbps}$, Max CPU $\in [40\%, 80\%]$, Budget $\in [\$200, \$2000]$.
- **Results**: 15 / 15 trials converged to feasible replica and bandwidth configurations within budget limits.

---

## 4. Mutation Testing Results

Mutation testing systematically perturbed boundary conditions, catalogs, and input phrasing to verify that the system dynamically recalculates outputs:

### 1. Budget Boundary Phase Transitions
- **Workload**: 8 vCPUs, 16 GB RAM on AWS (True Optimum: 4x `t3.medium` @ $30.37/ea = $121.48 / month).
- **Mutated Budget $121.00**: Solver returns `INFEASIBLE_BUDGET_EXCEEDED` ($121.00 < $121.48).
- **Mutated Budget $121.50**: Solver returns `OPTIMAL` with cost $121.48.
- **Mutated Budget $500.00**: Solver returns `OPTIMAL` with cost $121.48 (minimizes cost rather than spending full budget).

### 2. Custom Synthetic Catalog Injection
- **Injected Catalog**: Completely novel SKUs (`synth.nano`: 1 vCPU, 2 GB RAM @ $7.30/mo; `synth.mega`: 10 vCPUs, 20 GB RAM @ $58.40/mo).
- **Workload**: 20 vCPUs, 40 GB RAM under $150.
- **Result**: Solver provisions 2x `synth.mega` ($116.80/mo) without defaulting to any standard AWS/Azure/GCP instance types.

### 3. Query Paraphrasing Invariance
- **Paraphrase A**: `"Deploy a high-availability setup with 16 vCPUs and 32 GB RAM on AWS under $600."`
- **Paraphrase B**: `"Need an AWS cluster requiring minimum 16 vcpus, 32gb memory, budget limit 600 usd."`
- **Paraphrase C**: `"Amazon Web Services allocation: 16 vcpus and 32 gb ram with max monthly spend 600 dollars."`
- **Result**: All three yield identical contracts (`vcpus=16, ram=32.0, budget=600.0, provider=AWS`).

---

## 5. Decoupled Evaluation & Anti-Leakage Invariants

1. **Inference Decoupling**: Solvers and neural extractors receive only the user prompt. They have no access to `manifest.json`, `expected_outcome`, or `optimal_cost_usd`.
2. **Oracle Independence**: `IndependentCatalogOracle` and `DeterministicScorer` calculate independent ground truths directly from raw SKU pricing and graph mathematics, rejecting any plan where the claimed price diverges from actual catalog mathematics by more than $0.02.
3. **Automated CI Regression**: `tests/test_hardcoding_and_leakage.py` is integrated into the automated test suite to prevent future hardcoding regressions.

---

## 6. Unexamined Components and Remaining Risks

1. **Third-Party LLM Provider Non-Determinism**: Modes 1, 2, and 4 depend on cloud LLM APIs (Groq `openai/gpt-oss-120b`). While temperature is set to 0.0, provider-side hardware parallelism can introduce minor phrasing variations in natural language outputs.
2. **Unstructured Mode 1 Ambiguity**: Raw natural language outputs containing informal conversational phrasing cannot always be parsed by regular expressions without human adjudication (8 cases currently queued in `research_eval/HUMAN_REVIEW_QUEUE.csv`).
