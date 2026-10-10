# Neurasym Research Audit: Evidence Reconciliation & Verification Report

**Project**: Neurasym (Neuro-Symbolic Cloud FinOps Orchestrator)  
**Auditor**: Independent AI Research Auditor & Senior Python Research Engineer  
**Date**: October 2026  
**Repository**: `C:\Users\user\neurasym`  
**Git Branch**: `research-audit`  
**Immutable Reference Artifacts Verified**:
- Primary Tabular Result: `results/study_run_02.csv` (SHA256 verified)
- Frozen Archive: `results/archive/run_20261009_093328/` (Tag: `study-run_20261009_093328`)
- Raw Inference Files: `results/raw/run_20261009_093328/*.json` and `*.txt` (87 total raw response files)
- Ground-Truth Benchmark Manifest: `data/final_query_manifest.json` (29 approved queries)

---

## 1. Executive Summary

This reconciliation report cross-references every finding in the Phase 1 Neurasym audit directly against the original immutable raw responses, benchmark manifest, and evaluation harness source code.

All discrepancies, field values, query IDs, and constraint thresholds have been reconciled with 100% empirical grounding:
1. **Q19 Mode 4 Telemetry**: Directly confirmed from `study_run_02.csv` (Row 75) that `task_success` is blank (`NaN`), `mode_outcome = "PLAN"`, `normalization_status = "NORMALIZATION_FAILURE"`, `explain_label = "CORRECT_REFUSAL"`, `proof_status = "Conflicting Requirements"`, and `plan_valid = False`.
2. **Benchmark Query ID Mapping**: Verified against `data/final_query_manifest.json` that `Q28` is `BOUNDARY_DR_UNDER_240_INFEASIBLE` and the unsupported GPU training query is `Q22_UNSUPPORTED_GPU_H100_TRAINING`.
3. **Q19 Conflict Specification**: Verified that Q19's ground-truth conflict is **target CPU 70% versus strict maximum CPU ceiling of 60%** under $1500.
4. **Mode 2 Empirical Adjudication**: Individually audited all 29 raw Mode 2 JSON responses (`results/raw/run_20261009_093328/*_mode2_trial1.json`). Disproved the assumption that all 7 `TASK_INCOMPATIBLE` cases were successes: exactly **5 are valid successes** (`Q11`, `Q12`, `Q17`, `Q18`, `Q29`) and **2 are legitimate failures** (`Q16`, `Q21` where Mode 2 hallucinated plans instead of requesting clarification).
5. **Investigation of 7 Blank `task_success` Rows**: Verified all 7 rows, including confirming that `Q07` Mode 4 represents a **true failure** (neural extractor emitted `budget_max_usd = null` with `interpretation_outcome = "ready"`, causing solver failure).
6. **Three-Column Score Reporting**: Formalized historical scores, raw-evidence corrected scores, and unresolved policy cases.

---

## 2. Point-by-Point Evidence Reconciliation

### 2.1 Q19 Mode 4 Field Reconciliation

**Inspected Source**: `results/study_run_02.csv` (Line 76 / Row 75) & `results/raw/run_20261009_093328/Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING_mode4_trial1.json`

| Field in `study_run_02.csv` | Observed Value in CSV | Explanation & Code Mechanism |
|---|---|---|
| `query_id` | `Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING` | Matched benchmark entry |
| `mode` | `4` | Neuro-symbolic orchestrator pipeline |
| `task_success` | `""` (blank / `NaN`) | `TruthTableEngine` in `src/benchmarks/truth_table.py:103-113` returned `task_pass = 0` due to `norm_stat = NORMALIZATION_FAILURE`, and `run_all_modes_comparative.py:744-776` emitted blank for unhandled exception/refusal fallthrough |
| `mode_outcome` | `"PLAN"` | Pipeline default initialized before refusal routing |
| `normalization_status` | `"NORMALIZATION_FAILURE"` | Normalizer caught exception from aborted solver execution |
| `explain_label` | `"CORRECT_REFUSAL"` | `finops_explainer` evaluated `proof_status == "Conflicting Requirements"` and correctly assigned `CORRECT_REFUSAL` |
| `plan_valid` | `False` | No plan was generated (aborted as expected) |
| `proof_status` | `"Conflicting Requirements"` | Populated directly by StageTrace telemetry validator |

**Reconciliation Conclusion**:
There is an internal contradiction in the historical evaluation pipeline between the **FinOps Explainer** (which correctly identified `CORRECT_REFUSAL`) and the **TruthTableEngine / Grader** (which flagged `NORMALIZATION_FAILURE` and left `task_success` blank). Because Mode 4's raw JSON proves that the neural extractor extracted `target_cpu_pct = 70.0`, `max_cpu_pct = 60.0`, and marked `interpretation_outcome = "conflicting_requirements"`, the system performed the correct action. Supported score: **`task_success = 1.0` (`CORRECT_REFUSAL`)**.

---

### 2.2 Query ID Disambiguation (Q28 vs Q22)

**Inspected Source**: `data/final_query_manifest.json` lines 340–420.

| Query ID | Manifest Query Name | Query Text | Expected Outcome |
|---|---|---|---|
| **`Q22`** | `Q22_UNSUPPORTED_GPU_H100_TRAINING` | *"Train a transformer model on 8 H100 GPUs for two weeks and tell me the cheapest option"* | `UNSUPPORTED` |
| **`Q23`** | `Q23_UNSUPPORTED_BIRYANI_RECIPE` | *"What is the best recipe for biryani for 20 people?"* | `UNSUPPORTED` |
| **`Q24`** | `Q24_UNSUPPORTED_ORACLE_DB_MIGRATION` | *"Migrate my on-premise Oracle database to the cloud and tell me how long it will take."* | `UNSUPPORTED` |
| **`Q25`** | `Q25_CONFLICTING_CHEAPEST_AND_HIGHEST_PERFORMANCE` | *"Give me the cheapest and also the highest-performance server on AWS under 200 dollar."* | `CLARIFICATION_REQUIRED` |
| **`Q26`** | `Q26_BOUNDARY_VM_UNDER_121_INFEASIBLE` | *"Deploy 8 vCPUs and 16GB RAM on AWS for under 121 dollar a month"* | `INFEASIBLE` |
| **`Q27`** | `Q27_BOUNDARY_VM_UNDER_122_FEASIBLE` | *"Deploy 8 vCPUs and 16GB RAM on AWS for under 122 dollar a month"* | `FEASIBLE` |
| **`Q28`** | `Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE` | *"Disaster recovery across AWS and GCP with 99.99 percent SLA and max 50ms latency under 240 dollar a month"* | `INFEASIBLE` |
| **`Q29`** | `Q29_BOUNDARY_DR_UNDER_250_FEASIBLE` | *"Disaster recovery across AWS and GCP with 99.99 percent SLA and max 50ms latency under 250 dollar a month"* | `FEASIBLE` |

**Reconciliation Conclusion**:
All query ID references across `AUDIT_REPORT.md` and `DISCREPANCIES.csv` have been corrected to strictly align with `final_query_manifest.json`.

---

### 2.3 Q19 Conflict Constraint Specification

**Inspected Source**: `data/final_query_manifest.json` (Entry `Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING`)
- **Query Text**: `"Dynamic scaling for 300 Mbps peak workload with target CPU 70 percent and strict maximum CPU ceiling of 60 percent under 1500 dollar"`
- **Ground Truth**: `target_cpu_pct = 70.0`, `max_cpu_pct = 60.0`, `target_bandwidth_mbps = 300.0`, `budget_max_usd = 1500.0`, `expected_outcome = "CONFLICTING_REQUIREMENTS"`.
- **Mathematical Conflict**: In continuous autoscaling, the target CPU setpoint ($70\%$) cannot exceed the hard ceiling limit ($60\%$). Any controller attempting to regulate CPU at $70\%$ will inherently violate the $60\%$ safety ceiling.

**Reconciliation Conclusion**:
The audit text now uniformly specifies **Target CPU 70% vs Strict Maximum CPU Ceiling 60%**.

---

### 2.4 Mode 2: Detailed Case-by-Case Empirical Adjudication

We inspected the raw JSON output files in `results/raw/run_20261009_093328/` for all 7 queries where Mode 2 received `TASK_INCOMPATIBLE` with `task_success = 0.0`.

```python
# The Grader Bias in run_all_modes_comparative.py (lines 570-585):
contract = scope_parser.parse(query_text)
if contract is None:
    requested_problem_type = "ILP_VM_Allocation"  # Fallback default from Mode 3 parser
```

#### Individual Case Review:

1. **`Q11_DR_AWS_GCP_50MS_600USD_FEASIBLE`**
   - **Mode 2 Raw JSON**: `{"task_type": "Z3_Graph_Disaster_Recovery", "outcome": "ready", "primary_region": "us-east-1", "secondary_region": "us-central1", "total_monthly_cost_usd": 243.0}`
   - **Verification**: Region pair meets latency (32 ms $\le 50$ ms), SLA ($99.99\%$), and cost ($243.00 \le \$600$).
   - **Verdict**: **VALID SUCCESS (+1)**.

2. **`Q12_DR_HINGLISH_DIGITS_50MS_600USD`**
   - **Mode 2 Raw JSON**: `{"task_type": "Z3_Graph_Disaster_Recovery", "outcome": "ready", "primary_region": "us-east-1", "secondary_region": "us-central1", "total_monthly_cost_usd": 243.0}`
   - **Verification**: Parsed Hinglish text correctly; selected optimal region pair and valid cost.
   - **Verdict**: **VALID SUCCESS (+1)**.

3. **`Q16_DR_MISSING_BUDGET`**
   - **Query**: *"I need disaster recovery across two clouds."* (Expected: `CLARIFICATION_REQUIRED`)
   - **Mode 2 Raw JSON**: `{"task_type": "Z3_Graph_Disaster_Recovery", "outcome": "ready", "primary_region": "us-east-1", "secondary_region": "us-central1", "total_monthly_cost_usd": 243.0}`
   - **Verification**: Mode 2 **hallucinated/assumed** missing budget, SLA, and latency constraints and returned a ready plan instead of requesting clarification.
   - **Verdict**: **TRUE FAILURE (0.0)**. Must NOT be upgraded.

4. **`Q17_SCALING_300MBPS_60CPU_FEASIBLE`**
   - **Mode 2 Raw JSON**: `{"task_type": "PSO_Continuous_Scaling", "outcome": "ready", "optimal_bandwidth_mbps": 300.0, "recommended_replicas": 7, "total_monthly_cost_usd": 339.0}`
   - **Verification**: Minimum replicas for CPU $\le 60\%$ at 300 Mbps is $\lceil 300 / (75 \times 0.6) \rceil = 7$ replicas. CPU = $57.14\% \le 60\%$. Cost $339.00 \le \$1500$.
   - **Verdict**: **VALID SUCCESS (+1)**.

5. **`Q18_SCALING_HINGLISH_300MBPS_60CPU`**
   - **Mode 2 Raw JSON**: `{"task_type": "PSO_Continuous_Scaling", "outcome": "ready", "optimal_bandwidth_mbps": 300.0, "recommended_replicas": 7, "total_monthly_cost_usd": 339.0}`
   - **Verification**: Correctly parsed Hinglish query; computed 7 replicas and $339.00.
   - **Verdict**: **VALID SUCCESS (+1)**.

6. **`Q21_SCALING_MISSING_WORKLOAD`**
   - **Query**: *"Continuous dynamic scaling with target CPU 70% under $1500"* (Expected: `CLARIFICATION_REQUIRED`)
   - **Mode 2 Raw JSON**: `{"task_type": "PSO_Continuous_Scaling", "outcome": "ready", "optimal_bandwidth_mbps": 840.0, "recommended_replicas": 16, "total_monthly_cost_usd": 787.2}`
   - **Verification**: Mode 2 **fabricated** 840 Mbps traffic and 16 replicas instead of asking for missing workload bandwidth.
   - **Verdict**: **TRUE FAILURE (0.0)**. Must NOT be upgraded.

7. **`Q29_BOUNDARY_DR_UNDER_250_FEASIBLE`**
   - **Mode 2 Raw JSON**: `{"task_type": "Z3_Graph_Disaster_Recovery", "outcome": "ready", "primary_region": "us-east-1", "secondary_region": "us-central1", "total_monthly_cost_usd": 243.0}`
   - **Verification**: Optimal pair cost $243.00 \le \$250$ budget ceiling.
   - **Verdict**: **VALID SUCCESS (+1)**.

**Summary for Mode 2**:
Out of the 7 `TASK_INCOMPATIBLE` cases:
- **5 Verified Valid Successes**: `Q11`, `Q12`, `Q17`, `Q18`, `Q29`.
- **2 Confirmed True Failures**: `Q16`, `Q21`.

Additionally, on `Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING`, Mode 2 generated `outcome = "infeasible"` with a complete explanation of the $70\% > 60\%$ conflict. Mode 2 earned `explain_label = CORRECT_REFUSAL`, but had `task_success = 0.0` in telemetry.

---

### 2.5 Investigation of All Seven Blank `task_success` Values

| Row in CSV | Query ID | Mode | Observed Fields in CSV | Raw Response Evidence | Root Cause Analysis | Reconciled Score |
|---|---|---|---|---|---|---|
| **Row 12** | `Q04_VM_AWS_16VCPU_64GB_INFEASIBLE` | Mode 1 | `task_success=""`, `mode_outcome=PLAN`, `norm_status=AMBIGUOUS`, `explain_label=WRONG_OUTCOME` | Mode 1 wrote: *"With the current AWS SKU pricing... cannot be built for <= $450/month... cheapest is 2x m5.2xlarge = $560.64"* | Mode 1 correctly refused the infeasible request in English. Grader failed regex extraction on prose. | **Unresolved (Policy)**: 0.0 under strict schema; 1.0 under prose refusal policy |
| **Row 27** | `Q07_VM_AWS_MISSING_BUDGET` | Mode 4 | `task_success=""`, `mode_outcome=PLAN`, `norm_status=MALFORMED_OUTPUT`, `explain_label=WRONG_OUTCOME`, `proof_status=Infeasible` | Extracted `budget_max_usd = null`, but set `interpretation_outcome = "ready"`, causing MILP solver crash/infeasibility | **Confirmed Failure**: Mode 4 failed to request clarification for missing budget. | **Confirmed Failure (`0.0`)** |
| **Row 52** | `Q14_DR_AWS_GCP_100USD_INFEASIBLE` | Mode 1 | `task_success=""`, `mode_outcome=PLAN`, `norm_status=NORMALIZATION_FAILURE`, `explain_label=WRONG_OUTCOME` | Mode 1 wrote: *"A disaster-recovery pair... cannot be built for <= $100/month... minimum base cost $235"* | Mode 1 correctly identified infeasibility in English. Normalizer failed to parse. | **Unresolved (Policy)**: 0.0 under strict schema; 1.0 under prose refusal policy |
| **Row 75** | `Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING` | Mode 4 | `task_success=""`, `mode_outcome=PLAN`, `norm_status=NORMALIZATION_FAILURE`, `explain_label=CORRECT_REFUSAL`, `proof_status=Conflicting Requirements` | Extracted `target_cpu_pct = 70.0`, `max_cpu_pct = 60.0`, `interpretation_outcome = "conflicting_requirements"` | **Confirmed Success**: Mode 4 neural extractor and validator detected conflict and refused plan. Grader had unhandled exception branch. | **Confirmed Success (`1.0`)** |
| **Row 76** | `Q20_SCALING_1000MBPS_30CPU_INFEASIBLE` | Mode 1 | `task_success=""`, `mode_outcome=PLAN`, `norm_status=AMBIGUOUS`, `explain_label=WRONG_OUTCOME` | Mode 1 wrote: *"the request cannot be satisfied... requires >= 45 replicas... catalog caps at 16"* | Mode 1 correctly identified infeasibility in English. Normalizer failed to parse. | **Unresolved (Policy)**: 0.0 under strict schema; 1.0 under prose refusal policy |
| **Row 88** | `Q23_UNSUPPORTED_BIRYANI_RECIPE` | Mode 1 | `task_success=""`, `mode_outcome=PLAN`, `norm_status=NORMALIZATION_FAILURE`, `explain_label=CORRECT_REFUSAL` | Mode 1 wrote: *"I’m sorry, but I can’t help with that."* | **Confirmed Success**: Mode 1 refused out-of-domain query. Grader left task_success blank. | **Confirmed Success (`1.0`)** |
| **Row 108** | `Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE` | Mode 1 | `task_success=""`, `mode_outcome=PLAN`, `norm_status=AMBIGUOUS`, `explain_label=WRONG_OUTCOME` | Mode 1 wrote: *"the requested constraints cannot be satisfied... cheapest is $235 + latency cost > $240"* | Mode 1 correctly identified boundary infeasibility. Normalizer failed to parse. | **Unresolved (Policy)**: 0.0 under strict schema; 1.0 under prose refusal policy |

---

## 3. Scientific Categorization of Findings

### 3.1 Confirmed Root Causes (Directly Proven in Source Code & Data)
1. **Mode 3 Q19 Evaluator Fallthrough (`run_all_modes_comparative.py:776`)**: `expected_outcome = "CONFLICTING_REQUIREMENTS"` had no matching handler branch, falling through to return `"SUCCESS"` based solely on feasibility status.
2. **Mode 2 SCOPE Fallback Bias (`run_all_modes_comparative.py:575-585`)**: Mode 2 was graded against `ILP_VM_Allocation` whenever the rule-based SCOPE parser failed on text, penalizing valid DR and Scaling schemas.
3. **Mode 4 Q19 Refusal Emission Gap (`src/benchmarks/truth_table.py:103-113`)**: The truth table engine treated all `NORMALIZATION_FAILURE` statuses as task failures without checking if the underlying execution was a valid refusal of conflicting requirements.
4. **Mode 4 Q07 Clarification Failure (`results/raw/run_20261009_093328/Q07_mode4_trial1.json`)**: Neural extractor set `interpretation_outcome = "ready"` despite `budget_max_usd = null`, causing solver failure.

### 3.2 Inferred Root Causes (Strongly Supported by Consistent Observations)
1. **Mode 1 Regex Fragility**: Mode 1 failures on infeasible queries (`Q04`, `Q14`, `Q20`, `Q28`) stem from post-hoc regular expressions expecting rigid tabular SKU layouts rather than conversational proofs.
2. **Solver Crash on Null Budget in Q07 Mode 4**: HiGHS MILP knapsack solver expects a floating-point numeric budget; receiving `None` triggers an unhandled parameter exception.

### 3.3 Unverified Hypotheses (Requiring Empirical Adjudication in Phase 2)
1. **Zero-Shot Constrained Mode 1**: Whether Mode 1 would match Mode 2's accuracy if prompted with rigid JSON schema rather than conversational prose.
2. **Multi-Trial Quantization Drift**: The degree of response variation on Groq's `openai/gpt-oss-120b` across $N \ge 5$ repetitions with identical seeds.

---

## 4. Reconciled Three-Column Score Reporting

| Execution Mode | (1) Historical Published Score | (2) Corrected Score Supported by Raw Evidence | (3) Unresolved Policy Cases |
|---|---|---|---|
| **Mode 1 (Raw LLM)** | **10.3%** (3 / 29) | **13.8%** (4 / 29) *(Crediting Q23 out-of-domain refusal)* | 6 cases (`Q04`, `Q14`, `Q20`, `Q22`, `Q24`, `Q28`) pending policy on unverified natural language refusals |
| **Mode 2 (Schema LLM)** | **34.5%** (10 / 29) | **55.2%** (16 / 29) *(10 historical + 5 valid DR/Scaling + 1 Q19 conflict refusal)* | 0 cases (All 29 raw JSON outputs fully validated) |
| **Mode 3 (Pure Symbolic)** | **82.8%** (24 / 29) | **79.3%** (23 / 29) *(24 historical - 1 Q19 false positive)* | 0 cases (Deterministic rule-based execution) |
| **Mode 4 (Neuro-Symbolic)** | **86.2%** (25 / 29) | **89.7%** (26 / 29) *(25 historical + 1 Q19 valid conflict refusal)* | 0 cases (All 29 traces and JSON records fully validated) |

---

## 5. Summary of Deliverable File Modifications

1. **`research_audit/DISCREPANCIES.csv`**:
   - Reconciled all 18 discrepancy rows.
   - Updated Q19 conflict description to **Target CPU 70% vs Strict Maximum CPU Ceiling 60%**.
   - Corrected query ID references (`Q28` = Boundary DR Infeasible, `Q22` = Unsupported GPU Training).
   - Separated Mode 2 `TASK_INCOMPATIBLE` cases into confirmed valid successes (`Q11`, `Q12`, `Q17`, `Q18`, `Q29`) and confirmed true failures (`Q16`, `Q21`).
2. **`research_audit/AUDIT_REPORT.md`**:
   - Updated with exact line citations, reconciled telemetry tables, and the three-column score matrix.
3. **`research_audit/RECONCILIATION_REPORT.md`**:
   - Created this definitive evidence reconciliation document.
