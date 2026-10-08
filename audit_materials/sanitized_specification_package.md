# NEURASYM Ground-Truth Specification & Authoring Guide
**Version:** 3.0.0-auditable  
**Purpose:** Canonical reference package for generating prompt datasets and ground-truth evaluation labels.

---

## 1. Cloud VM SKU Catalog (Monthly Calculated at 730.0 hrs/mo)
| SKU Name | Provider | vCPUs | RAM (GB) | Hourly Price ($) | Monthly Price ($) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `t3.medium` | AWS | 2 | 4.0 GB | $0.0416 | $30.37 |
| `t3.large` | AWS | 2 | 8.0 GB | $0.0832 | $60.74 |
| `t3.xlarge` | AWS | 4 | 16.0 GB | $0.1664 | $121.47 |
| `c5.large` | AWS | 2 | 4.0 GB | $0.0850 | $62.05 |
| `c5.xlarge` | AWS | 4 | 8.0 GB | $0.1700 | $124.10 |
| `m5.large` | AWS | 2 | 8.0 GB | $0.0960 | $70.08 |
| `m5.xlarge` | AWS | 4 | 16.0 GB | $0.1920 | $140.16 |
| `m5.2xlarge` | AWS | 8 | 32.0 GB | $0.3840 | $280.32 |
| `Standard_D4s_v5` | Azure | 4 | 16.0 GB | $0.1920 | $140.16 |
| `e2-standard-4` | GCP | 4 | 16.0 GB | $0.1340 | $97.82 |

---

## 2. Multi-Region Disaster Recovery Topology
- **Latency Cost Rate:** $0.25 USD per ms per month
- **Composite Availability SLA Formula:** `1 - ((1 - SLA_A/100) * (1 - SLA_B/100))`
- **Cost Formula:** `Cost = BaseCost(A) + BaseCost(B) + (Latency(A,B) * $0.25)`
- **Constraint:** Primary and Secondary regions must be in distinct geographical domains (`Geo(A) != Geo(B)`).

### Regional Nodes:
- **`us-east-1`** (AWS, Geo: `US_East`): Base Cost = $120.00/mo, Single Region SLA = 99.95%
  - Latencies: {'us-west-2': 65.0, 'eu-west-1': 85.0, 'eastus': 12.0, 'us-central1': 32.0}
- **`us-west-2`** (AWS, Geo: `US_West`): Base Cost = $130.00/mo, Single Region SLA = 99.95%
  - Latencies: {'us-east-1': 65.0, 'eu-west-1': 135.0, 'eastus': 70.0, 'us-central1': 42.0}
- **`eu-west-1`** (AWS, Geo: `Europe`): Base Cost = $140.00/mo, Single Region SLA = 99.95%
  - Latencies: {'us-east-1': 85.0, 'us-west-2': 135.0, 'eastus': 90.0, 'us-central1': 105.0}
- **`eastus`** (Azure, Geo: `US_East`): Base Cost = $125.00/mo, Single Region SLA = 99.95%
  - Latencies: {'us-east-1': 12.0, 'us-west-2': 70.0, 'eu-west-1': 90.0, 'us-central1': 28.0}
- **`us-central1`** (GCP, Geo: `US_Central`): Base Cost = $115.00/mo, Single Region SLA = 99.95%
  - Latencies: {'us-east-1': 32.0, 'us-west-2': 42.0, 'eu-west-1': 105.0, 'eastus': 28.0}

---

## 3. Dynamic Continuous Scaling (PSO)
- **Bandwidth Domain:** [100.0, 1000.0] Mbps
- **Worker Replicas Domain:** [1, 16] Replicas
- **Pricing Formula:** `Cost = (Bandwidth * $0.08) + (Replicas * $45.00)`
- **Capacity Model:** `Modeled CPU = (Bandwidth / (Replicas * 75.0 Mbps)) * 100%`
- **Saturation Limit:** Maximum hard ceiling $\le 100.0\%$.

---

## 4. Supported Problem Archetypes & Contract Schemas
1. **`ILP_VM_Allocation`**:
   - Required: `required_vcpus` (int), `required_ram_gb` (float), `budget_max_usd` (float)
   - Optional: `cloud_providers` (list of str, defaults to `["AWS"]`)
2. **`Z3_Graph_Disaster_Recovery`**:
   - Required: `latency_max_ms` (float), `sla_availability_pct` (float), `budget_max_usd` (float)
3. **`PSO_Continuous_Scaling`**:
   - Required: `target_bandwidth_mbps` (float), `target_cpu_pct` (float), `budget_max_usd` (float)
   - Optional: `max_cpu_pct` (float ceiling)

---

## 5. Evaluation & Authoring Policies
- **Missing / Ambiguous Requirements:** If critical constraints (e.g. budget) are omitted, the query requires clarification (`expected_outcome="CLARIFICATION_REQUIRED"`).
- **Conflicting / Impossible Constraints:** If requirements cannot be satisfied under the physical catalog (e.g. 64 vCPUs for $10/mo), the query must be proven infeasible (`expected_outcome="INFEASIBLE"`).
- **Out-of-Scope / Unsupported Tasks:** Requests involving unsupported domains (e.g. quantum computing, DNS configuration, OAuth integration) must be rejected cleanly (`expected_outcome="UNSUPPORTED"`).
- **Development Queries Status:** All development queries remain marked `UNAPPROVED` with `annotation_source="AI_DRAFT/DEVELOPMENT"` and `is_approved=False`.
