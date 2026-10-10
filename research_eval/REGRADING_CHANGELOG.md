# Neurasym Complete 116-Record Regrading Changelog

**Evaluation Protocol**: Deterministic Evaluation Protocol v2.1.0
**Total Records**: 116 (29 benchmark queries × 4 execution modes)

This document provides record-by-record justification for every score retention or alteration across the benchmark.

## Mode 1: Raw LLM

| Query ID | Expected Outcome | Hist Score | Regr Score | Hist Label | Regr Label | Regrading Justification |
|---|---|---|---|---|---|---|
| `Q01_VM_AWS_8VCPU_16GB_FEASIBLE` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_PLAN` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |
| `Q02_VM_HINGLISH_8VCPU_16GB` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q03_VM_HINGLISH_WORDS_8VCPU_16GB` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q04_VM_AWS_16VCPU_64GB_INFEASIBLE` | `INFEASIBLE` | `BLANK` | `1.0` | `WRONG_OUTCOME` | `CORRECT_REFUSAL` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT_REFUSAL) previously penalized by flawed regex/truth-table engine. |
| `Q05_VM_HINGLISH_16VCPU_64GB_INFEASIBLE` | `INFEASIBLE` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected INFEASIBLE, but system generated an unconstrained allocation plan.). |
| `Q06_VM_AWS_64VCPU_256GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q07_VM_AWS_MISSING_BUDGET` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q08_VM_AWS_MISSING_SPECS` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q09_VM_AZURE_GCP_4VCPU_8GB` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_PLAN` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |
| `Q10_VM_HINGLISH_GCP_AZURE_8VCPU_32GB` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_PLAN` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |
| `Q11_DR_AWS_GCP_50MS_600USD_FEASIBLE` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_PLAN` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |
| `Q12_DR_HINGLISH_DIGITS_50MS_600USD` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_PLAN` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |
| `Q13_DR_HINGLISH_WORDS_50MS_600USD` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |
| `Q14_DR_AWS_GCP_100USD_INFEASIBLE` | `INFEASIBLE` | `BLANK` | `1.0` | `WRONG_OUTCOME` | `CORRECT_REFUSAL` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT_REFUSAL) previously penalized by flawed regex/truth-table engine. |
| `Q15_DR_AWS_GCP_10MS_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q16_DR_MISSING_BUDGET` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q17_SCALING_300MBPS_60CPU_FEASIBLE` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_PLAN` | `WRONG_PLAN` | Score retained (0.0): Confirmed failure (Invalid or missing replica count in scaling plan.). |
| `Q18_SCALING_HINGLISH_300MBPS_60CPU` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_PLAN` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |
| `Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING` | `CONFLICTING_REQUIREMENTS` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CONFLICTING_REQUIREMENTS, but system generated an unconstrained allocation plan.). |
| `Q20_SCALING_1000MBPS_30CPU_INFEASIBLE` | `INFEASIBLE` | `BLANK` | `1.0` | `WRONG_OUTCOME` | `CORRECT_REFUSAL` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT_REFUSAL) previously penalized by flawed regex/truth-table engine. |
| `Q21_SCALING_MISSING_WORKLOAD` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q22_UNSUPPORTED_GPU_H100_TRAINING` | `UNSUPPORTED` | `0.0` | `0.0` | `CORRECT_REFUSAL` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected UNSUPPORTED, but system generated an unconstrained allocation plan.). |
| `Q23_UNSUPPORTED_BIRYANI_RECIPE` | `UNSUPPORTED` | `BLANK` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT_REFUSAL) previously penalized by flawed regex/truth-table engine. |
| `Q24_UNSUPPORTED_ORACLE_DB_MIGRATION` | `UNSUPPORTED` | `0.0` | `0.0` | `CORRECT_REFUSAL` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected UNSUPPORTED, but system generated an unconstrained allocation plan.). |
| `Q25_CONFLICTING_CHEAPEST_AND_HIGHEST_PERFORMANCE` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q26_BOUNDARY_VM_UNDER_121_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q27_BOUNDARY_VM_UNDER_122_FEASIBLE` | `FEASIBLE` | `0.0` | `0.0` | `UNPARSEABLE` | `WRONG_PLAN` | Score retained (0.0): Confirmed failure (Cost exceeds budget: $768.84 > $122.00.). |
| `Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE` | `INFEASIBLE` | `BLANK` | `1.0` | `WRONG_OUTCOME` | `CORRECT_REFUSAL` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT_REFUSAL) previously penalized by flawed regex/truth-table engine. |
| `Q29_BOUNDARY_DR_UNDER_250_FEASIBLE` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |


## Mode 2: Schema LLM

| Query ID | Expected Outcome | Hist Score | Regr Score | Hist Label | Regr Label | Regrading Justification |
|---|---|---|---|---|---|---|
| `Q01_VM_AWS_8VCPU_16GB_FEASIBLE` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q02_VM_HINGLISH_8VCPU_16GB` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q03_VM_HINGLISH_WORDS_8VCPU_16GB` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q04_VM_AWS_16VCPU_64GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q05_VM_HINGLISH_16VCPU_64GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q06_VM_AWS_64VCPU_256GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q07_VM_AWS_MISSING_BUDGET` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Refusal category mismatch: expected CLARIFICATION_REQUIRED, received infeasible (No AWS VM SKU matches exactly 8 vCPUs and 16GB RAM. Closest options are m5.2xlarge (8 vCPUs, 32GB) or combinations of smaller instances, but a single server with the specified specs is unavailable.).). |
| `Q08_VM_AWS_MISSING_SPECS` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q09_VM_AZURE_GCP_4VCPU_8GB` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q10_VM_HINGLISH_GCP_AZURE_8VCPU_32GB` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q11_DR_AWS_GCP_50MS_600USD_FEASIBLE` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q12_DR_HINGLISH_DIGITS_50MS_600USD` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q13_DR_HINGLISH_WORDS_50MS_600USD` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |
| `Q14_DR_AWS_GCP_100USD_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q15_DR_AWS_GCP_10MS_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q16_DR_MISSING_BUDGET` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q17_SCALING_300MBPS_60CPU_FEASIBLE` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q18_SCALING_HINGLISH_300MBPS_60CPU` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |
| `Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING` | `CONFLICTING_REQUIREMENTS` | `0.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT_REFUSAL) previously penalized by flawed regex/truth-table engine. |
| `Q20_SCALING_1000MBPS_30CPU_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q21_SCALING_MISSING_WORKLOAD` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q22_UNSUPPORTED_GPU_H100_TRAINING` | `UNSUPPORTED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q23_UNSUPPORTED_BIRYANI_RECIPE` | `UNSUPPORTED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q24_UNSUPPORTED_ORACLE_DB_MIGRATION` | `UNSUPPORTED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Refusal category mismatch: expected UNSUPPORTED, received needs_clarification (The request asks for migration planning and duration estimation, which is outside the supported optimization tasks (VM allocation, disaster recovery, or continuous scaling). Please provide specific requirements such as required compute resources, region preferences, or scaling parameters.).). |
| `Q25_CONFLICTING_CHEAPEST_AND_HIGHEST_PERFORMANCE` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q26_BOUNDARY_VM_UNDER_121_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q27_BOUNDARY_VM_UNDER_122_FEASIBLE` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (System refused or asked for clarification when a valid feasible plan exists.). |
| `Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q29_BOUNDARY_DR_UNDER_250_FEASIBLE` | `FEASIBLE` | `0.0` | `1.0` | `WRONG_PLAN` | `CORRECT` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT) previously penalized by flawed regex/truth-table engine. |


## Mode 3: Pure Symbolic

| Query ID | Expected Outcome | Hist Score | Regr Score | Hist Label | Regr Label | Regrading Justification |
|---|---|---|---|---|---|---|
| `Q01_VM_AWS_8VCPU_16GB_FEASIBLE` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q02_VM_HINGLISH_8VCPU_16GB` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q03_VM_HINGLISH_WORDS_8VCPU_16GB` | `FEASIBLE` | `1.0` | `0.0` | `RIGHT_ANSWER_WRONG_READING` | `WRONG_PLAN` | Downgraded (1.0 -> 0.0): Corrected historical evaluator loophole (WRONG_PLAN: Insufficient vCPUs: provided 2 < required 8.; Insufficient RAM: provided 4.0 GB < required 16.0 GB.). |
| `Q04_VM_AWS_16VCPU_64GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q05_VM_HINGLISH_16VCPU_64GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q06_VM_AWS_64VCPU_256GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q07_VM_AWS_MISSING_BUDGET` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q08_VM_AWS_MISSING_SPECS` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q09_VM_AZURE_GCP_4VCPU_8GB` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q10_VM_HINGLISH_GCP_AZURE_8VCPU_32GB` | `FEASIBLE` | `1.0` | `0.0` | `RIGHT_ANSWER_WRONG_READING` | `WRONG_PLAN` | Downgraded (1.0 -> 0.0): Corrected historical evaluator loophole (WRONG_PLAN: Insufficient vCPUs: provided 4 < required 8.; Insufficient RAM: provided 16.0 GB < required 32.0 GB.). |
| `Q11_DR_AWS_GCP_50MS_600USD_FEASIBLE` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q12_DR_HINGLISH_DIGITS_50MS_600USD` | `FEASIBLE` | `1.0` | `1.0` | `RIGHT_ANSWER_WRONG_READING` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q13_DR_HINGLISH_WORDS_50MS_600USD` | `FEASIBLE` | `1.0` | `0.0` | `RIGHT_ANSWER_WRONG_READING` | `RIGHT_ANSWER_WRONG_READING` | Downgraded (1.0 -> 0.0): Corrected historical evaluator loophole (RIGHT_ANSWER_WRONG_READING: NONE). |
| `Q14_DR_AWS_GCP_100USD_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q15_DR_AWS_GCP_10MS_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q16_DR_MISSING_BUDGET` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q17_SCALING_300MBPS_60CPU_FEASIBLE` | `FEASIBLE` | `1.0` | `0.0` | `RIGHT_ANSWER_WRONG_READING` | `WRONG_PLAN` | Downgraded (1.0 -> 0.0): Corrected historical evaluator loophole (WRONG_PLAN: Invalid or missing replica count in scaling plan.). |
| `Q18_SCALING_HINGLISH_300MBPS_60CPU` | `FEASIBLE` | `1.0` | `0.0` | `RIGHT_ANSWER_WRONG_READING` | `WRONG_PLAN` | Downgraded (1.0 -> 0.0): Corrected historical evaluator loophole (WRONG_PLAN: Invalid or missing replica count in scaling plan.). |
| `Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING` | `CONFLICTING_REQUIREMENTS` | `1.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Downgraded (1.0 -> 0.0): Corrected historical evaluator loophole (WRONG_OUTCOME: Expected CONFLICTING_REQUIREMENTS, but system generated an unconstrained allocation plan.). |
| `Q20_SCALING_1000MBPS_30CPU_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q21_SCALING_MISSING_WORKLOAD` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q22_UNSUPPORTED_GPU_H100_TRAINING` | `UNSUPPORTED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q23_UNSUPPORTED_BIRYANI_RECIPE` | `UNSUPPORTED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q24_UNSUPPORTED_ORACLE_DB_MIGRATION` | `UNSUPPORTED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q25_CONFLICTING_CHEAPEST_AND_HIGHEST_PERFORMANCE` | `CLARIFICATION_REQUIRED` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Score retained (0.0): Confirmed failure (Expected CLARIFICATION_REQUIRED, but system generated an unconstrained allocation plan.). |
| `Q26_BOUNDARY_VM_UNDER_121_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q27_BOUNDARY_VM_UNDER_122_FEASIBLE` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q29_BOUNDARY_DR_UNDER_250_FEASIBLE` | `FEASIBLE` | `1.0` | `1.0` | `RIGHT_ANSWER_WRONG_READING` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |


## Mode 4: Neuro-Symbolic

| Query ID | Expected Outcome | Hist Score | Regr Score | Hist Label | Regr Label | Regrading Justification |
|---|---|---|---|---|---|---|
| `Q01_VM_AWS_8VCPU_16GB_FEASIBLE` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q02_VM_HINGLISH_8VCPU_16GB` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q03_VM_HINGLISH_WORDS_8VCPU_16GB` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q04_VM_AWS_16VCPU_64GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q05_VM_HINGLISH_16VCPU_64GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q06_VM_AWS_64VCPU_256GB_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q07_VM_AWS_MISSING_BUDGET` | `CLARIFICATION_REQUIRED` | `BLANK` | `0.0` | `WRONG_OUTCOME` | `WRONG_OUTCOME` | Downgraded (1.0 -> 0.0): Corrected historical evaluator loophole (WRONG_OUTCOME: Refusal category mismatch: expected CLARIFICATION_REQUIRED, received infeasible (infeasible).). |
| `Q08_VM_AWS_MISSING_SPECS` | `CLARIFICATION_REQUIRED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q09_VM_AZURE_GCP_4VCPU_8GB` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q10_VM_HINGLISH_GCP_AZURE_8VCPU_32GB` | `FEASIBLE` | `1.0` | `0.0` | `RIGHT_ANSWER_WRONG_READING` | `RIGHT_ANSWER_WRONG_READING` | Downgraded (1.0 -> 0.0): Corrected historical evaluator loophole (RIGHT_ANSWER_WRONG_READING: NONE). |
| `Q11_DR_AWS_GCP_50MS_600USD_FEASIBLE` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q12_DR_HINGLISH_DIGITS_50MS_600USD` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q13_DR_HINGLISH_WORDS_50MS_600USD` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q14_DR_AWS_GCP_100USD_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q15_DR_AWS_GCP_10MS_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q16_DR_MISSING_BUDGET` | `CLARIFICATION_REQUIRED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q17_SCALING_300MBPS_60CPU_FEASIBLE` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_PLAN` | Score retained (0.0): Confirmed failure (Invalid or missing replica count in scaling plan.). |
| `Q18_SCALING_HINGLISH_300MBPS_60CPU` | `FEASIBLE` | `0.0` | `0.0` | `WRONG_OUTCOME` | `WRONG_PLAN` | Score retained (0.0): Confirmed failure (Invalid or missing replica count in scaling plan.). |
| `Q19_SCALING_CONFLICTING_CPU_TARGET_CEILING` | `CONFLICTING_REQUIREMENTS` | `BLANK` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Upgraded (0.0 -> 1.0): Valid outcome (CORRECT_REFUSAL) previously penalized by flawed regex/truth-table engine. |
| `Q20_SCALING_1000MBPS_30CPU_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q21_SCALING_MISSING_WORKLOAD` | `CLARIFICATION_REQUIRED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q22_UNSUPPORTED_GPU_H100_TRAINING` | `UNSUPPORTED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q23_UNSUPPORTED_BIRYANI_RECIPE` | `UNSUPPORTED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q24_UNSUPPORTED_ORACLE_DB_MIGRATION` | `UNSUPPORTED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q25_CONFLICTING_CHEAPEST_AND_HIGHEST_PERFORMANCE` | `CLARIFICATION_REQUIRED` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q26_BOUNDARY_VM_UNDER_121_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q27_BOUNDARY_VM_UNDER_122_FEASIBLE` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE` | `INFEASIBLE` | `1.0` | `1.0` | `CORRECT_REFUSAL` | `CORRECT_REFUSAL` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |
| `Q29_BOUNDARY_DR_UNDER_250_FEASIBLE` | `FEASIBLE` | `1.0` | `1.0` | `CORRECT` | `CORRECT` | Score retained (1.0): Verified constraints satisfied, correct recomputed cost, and valid outcome. |


