# Neurasym: Declared Task Model Specification

## 1. Overview & Architectural Principles

Neurasym operates across four operational execution modes (Mode 1: Raw LLM, Mode 2: Schema LLM, Mode 3: Pure Symbolic, Mode 4: Neuro-Symbolic). To ensure consistent, mode-blind verification and eliminate contradictory verdicts, all four modes share a single **Declared Task Model** governing the interpretation of requirements, constraints, units, and catalog boundaries.

---

## 2. Supported Optimization Archetypes

### Archetype 1: Knapsack VM Allocation (`ILP_VM_Allocation`)

* **Semantic Meaning**: Discrete selection and integer placement of virtual machine instance SKUs to satisfy aggregate compute and memory requirements within a monthly budget cap.
* **Requirements & Units**:
  * `required_vcpus`: Integer count ($k \ge 1$) of minimum aggregate virtual CPUs required.
  * `required_ram_gb`: Continuous float ($m \ge 0.5\text{ GB}$) of minimum aggregate RAM required.
  * `budget_max_usd`: Monthly budget ceiling in USD (minimum viable threshold: $\$10.00/\text{month}$).
  * `cloud_providers`: Allowed cloud provider set ($\subseteq \{\text{"AWS"}, \text{"Azure"}, \text{"GCP"}\}$).
  * `service_count`: Integer count of discrete microservices/tiers to allocate.
* **Billing Period**: Monthly, standardized to **730.0 hours per month** ($Cost = \sum_i \text{hourly\_price}_i \times 730 \times \text{quantity}_i$).
* **Hard Constraints**:
  1. $\sum_i \text{vcpus}_i \times \text{quantity}_i \ge \text{required\_vcpus}$
  2. $\sum_i \text{ram\_gb}_i \times \text{quantity}_i \ge \text{required\_ram\_gb}$
  3. $\sum_i \text{monthly\_cost}_i \le \text{budget\_max\_usd}$
  4. $\forall i: \text{quantity}_i \in \mathbb{Z}^+$ (strictly positive integer count; zero/negative or fractional quantities are rejected)
  5. $\forall i: \text{SKU}_i \in \text{Catalog}$ and $\text{provider}(\text{SKU}_i) \in \text{cloud\_providers}$
* **Soft Preferences**: Minimize total monthly cost; favor single-provider co-location unless multi-cloud is requested.
* **Catalog Ground Truth & Substantiation**:
  * *Substantiated*: Seeded SKU definitions (vCPUs, RAM, hourly price cards) derived from real public cloud rate cards (AWS EC2, Azure VMs, GCP Compute Engine).
  * *Boundary Limitation*: Seeded pricing reflects public on-demand list prices. It is a static seeded snapshot, not live dynamic spot auction telemetry.

---

### Archetype 2: Continuous Dynamic Scaling (`PSO_Continuous_Scaling`)

* **Semantic Meaning**: Sizing continuous workload throughput bandwidth and discrete worker replicas to maintain operational CPU utilization within hard capacity and target ceiling constraints within budget.
* **Workload Semantics & Offered Demand**:
  * **Scaling Bandwidth ($B$)**: Represents the **user's offered workload demand** (or minimum required throughput threshold in Mbps).
  * **Critical Rule**: An optimizer is **NOT permitted to arbitrarily reduce bandwidth below the user's required offered workload** simply to make cost smaller. If the user requests 100 Mbps or 500 Mbps, the system must provision at least that offered workload.
* **Requirements & Units**:
  * `bandwidth_mbps` / `target_bandwidth_mbps`: Offered workload throughput in Megabits per second ($\text{Mbps} \in [100.0, 1000.0]$).
  * `recommended_replicas` / `replicas`: Integer worker replica count ($R \in \{1, 2, \dots, 16\}$).
  * `target_cpu_pct` vs `max_cpu_pct`:
    * `cpu_target_pct` / `target_cpu_pct`: The **desired nominal operating point** (e.g., $70.0\%$), used as the optimization setpoint / penalty center.
    * `cpu_max_pct` / `max_cpu_pct`: The **hard maximum CPU utilization ceiling** (e.g., $70.0\%$ or $100.0\%$).
    * **Disambiguation Policy**: If a query states "target CPU 70%", $70.0\%$ is treated as the operational target with a hard physical saturation ceiling of $100.0\%$. If a query states "max CPU 70%" or "ceiling 70%", $70.0\%$ becomes a hard inequality constraint ($U \le 70.0\%$).
* **Capacity & Cost Formulation**:
  * **Modeled Replica Capacity**: Each discrete worker replica provides **$75.0\text{ Mbps}$** of throughput capacity.
  * **Modeled CPU Utilization**:
    $$U = \left( \frac{B}{R \times 75.0\text{ Mbps}} \right) \times 100\%$$
  * **Monthly Infrastructure Cost**:
    $$\text{Cost} = (B \times \$0.08/\text{Mbps}) + (R \times \$45.00/\text{replica})$$
* **Hard Constraints**:
  1. $R \in \mathbb{Z}^+$ and $1 \le R \le 16$ (deployable integer replicas evaluated after conversion; no pre-rounded cost reuse).
  2. $100.0 \le B \le 1000.0\text{ Mbps}$
  3. $U \le 100.0\%$ (Physical capacity overload constraint: no system may operate above 100% capacity; e.g. 100 Mbps on 1 replica $\rightarrow 133.33\% \rightarrow \text{FAIL}$).
  4. $U \le \text{max\_cpu\_pct}$ (when an explicit hard ceiling is declared).
  5. $\text{Cost} \le \text{budget\_max\_usd}$.
* **Model Labeling**:
  * *Substantiated*: Synthetic dynamic pricing formula ($\$0.08/\text{Mbps} + \$45.00/\text{replica}$) and synthetic linear capacity model ($75\text{ Mbps/replica}$).
  * *Boundary Limitation*: These parameters represent a standardized synthetic queueing model for benchmark comparison, not empirical profiling of a live production microservice.

---

### Archetype 3: Multi-Region Disaster Recovery (`Z3_Graph_Disaster_Recovery`)

* **Semantic Meaning**: SMT graph topology placement of primary and secondary failover regions satisfying failure domain disjointness, maximum network latency, and composite availability SLA within budget.
* **Requirements & Units**:
  * `primary_region`: Identifier of the primary compute region (e.g., `us-east-1`, `eastus`, `us-central1`).
  * `secondary_region`: Identifier of the failover compute region (e.g., `us-west-2`, `eu-west-1`).
  * `latency_max_ms`: Maximum tolerable synchronous cross-region network latency in milliseconds ($\text{ms} \le 1000.0$).
  * `sla_availability_pct`: Target composite availability SLA percentage ($\text{SLA} \in [90.0\%, 99.999\%]$).
  * `budget_max_usd`: Monthly budget cap in USD.
* **Cost & SLA Formulations**:
  * **Composite Availability**:
    $$\text{SLA}_{\text{composite}} = \left[ 1 - \left(1 - \frac{\text{SLA}_A}{100}\right) \times \left(1 - \frac{\text{SLA}_B}{100}\right) \right] \times 100\%$$
    *(Each individual regional node has a baseline SLA of $99.95\%$; dual disjoint regions yield $99.999975\%$)*.
  * **Monthly Topology Cost**:
    $$\text{Cost} = \text{base\_cost}(A) + \text{base\_cost}(B) + (\text{latency\_ms}(A, B) \times \$0.25/\text{ms})$$
* **Hard Constraints**:
  1. $A, B \in \text{InfrastructureGraph}$ (both regions must exist as known topological nodes).
  2. $A \ne B$ and $\text{Disjoint}(A, B)$ (must not share identical geographic and provider fault domains).
  3. $\text{latency\_ms}(A, B) \le \text{latency\_max\_ms}$.
  4. $\text{SLA}_{\text{composite}} \ge \text{sla\_availability\_pct}$.
  5. $\text{Cost} \le \text{budget\_max\_usd}$.
  6. $\text{provider}(A), \text{provider}(B) \in \text{cloud\_providers}$.
* **Model Labeling**:
  * *Substantiated*: Fixed regional latency matrix graph, baseline regional costs, and independent failure domain definitions.
  * *Boundary Limitation*: Cross-region latencies and SLAs are synthetic benchmark constants, not live ICMP ping measurements or real-time CloudWatch metrics.

---

## 3. Financial Terminology: Budget Headroom vs Net Savings

* **Budget Headroom**: Defined as $\text{Budget Cap} - \text{Candidate Cost}$. It represents remaining unallocated budget under the user's ceiling. All Neurasym reports must use the term "Budget Headroom" when evaluating a candidate against a standalone budget.
* **Net Savings**: Defined strictly as $\text{Baseline Cost} - \text{Candidate Cost}$ when an explicit, measured prior expenditure baseline is provided. Neurasym does not fabricate baseline savings when no prior architecture baseline was declared.

---

## 4. Mode-Blind Verification Invariant

The `IndependentChecker` evaluates all candidate outputs strictly using:
1. Normalized Decision Vector
2. Declared Workload Requirements
3. Ground-Truth Catalog Snapshot
4. Declared Numeric Tolerances ($\pm \$0.50$ cost discrepancy tolerance)

The checker is completely blind to whether a decision originated from Mode 1 (Prose), Mode 2 (JSON Schema), Mode 3 (Symbolic), or Mode 4 (Neuro-Symbolic). Identical allocations evaluated against identical requirements produce identical verdicts.
