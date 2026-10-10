# Neurasym Blind Human Adjudication Queue

This queue documents cases where raw natural language outputs (Mode 1) cannot be fully verified by structured schema parsers and require blinded human review.

| Review ID | Query ID | Expected Outcome | Regraded Status | Prose Snippet |
|---|---|---|---|---|
| `REV_M1_Q04_VM_AWS_16VCPU_64GB_INFEASIBLE` | `Q04_VM_AWS_16VCPU_64GB_INFEASIBLE` | `INFEASIBLE` | `CORRECT_REFUSAL` | *"**Result:** With the current AWS SKU pricing in the catalog, a deployment that provides **16 vCPUs + 64 GB RAM** cannot ..."* |
| `REV_M1_Q06_VM_AWS_64VCPU_256GB_INFEASIBLE` | `Q06_VM_AWS_64VCPU_256GB_INFEASIBLE` | `INFEASIBLE` | `CORRECT_REFUSAL` | *"**Result:** The requested workload (64 vCPUs + 256 GB RAM) cannot be satisfied on AWS within a $10 / month budget using ..."* |
| `REV_M1_Q14_DR_AWS_GCP_100USD_INFEASIBLE` | `Q14_DR_AWS_GCP_100USD_INFEASIBLE` | `INFEASIBLE` | `CORRECT_REFUSAL` | *"**Result:** A disaster‑recovery pair that satisfies **both** the 99.99 % combined SLA **and** the ≤ 50 ms latency requir..."* |
| `REV_M1_Q15_DR_AWS_GCP_10MS_INFEASIBLE` | `Q15_DR_AWS_GCP_10MS_INFEASIBLE` | `INFEASIBLE` | `CORRECT_REFUSAL` | *"**Result – the request cannot be satisfied with the given catalog**  \| Requirement \| What the catalog can provide \| Why ..."* |
| `REV_M1_Q20_SCALING_1000MBPS_30CPU_INFEASIBLE` | `Q20_SCALING_1000MBPS_30CPU_INFEASIBLE` | `INFEASIBLE` | `CORRECT_REFUSAL` | *"**Result – the request cannot be satisfied with the catalog constraints**  \| Requirement \| What the catalog can deliver ..."* |
| `REV_M1_Q23_UNSUPPORTED_BIRYANI_RECIPE` | `Q23_UNSUPPORTED_BIRYANI_RECIPE` | `UNSUPPORTED` | `CORRECT_REFUSAL` | *"I’m sorry, but I can’t help with that...."* |
| `REV_M1_Q26_BOUNDARY_VM_UNDER_121_INFEASIBLE` | `Q26_BOUNDARY_VM_UNDER_121_INFEASIBLE` | `INFEASIBLE` | `CORRECT_REFUSAL` | *"**Result:** It isn’t possible to meet the requirement “8 vCPUs + 16 GB RAM on AWS for ≤ $121 / month” with the catalog y..."* |
| `REV_M1_Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE` | `Q28_BOUNDARY_DR_UNDER_240_INFEASIBLE` | `INFEASIBLE` | `CORRECT_REFUSAL` | *"**Result – the requested constraints cannot be satisfied with the catalog data**  \| Requirement \| What the catalog force..."* |
