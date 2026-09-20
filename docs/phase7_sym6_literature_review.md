# Phase 7 — SYM-6: Literature Review & Research Traceability

## 1. Research Motivation
Cloud FinOps optimization presents a highly constrained, multi-objective problem space. While Large Language Models (LLMs) excel at semantic interpretation (translating human business intent into structured requirements), they struggle with deterministic constraint solving, arithmetic feasibility, and multi-objective optimization at scale. 

Conversely, symbolic engines (Genetic Algorithms, Particle Swarm, Z3 SMT solvers) provide strict mathematical guarantees but cannot interpret natural language. 

The **Neuro-Symbolic AI Orchestrator** bridges this gap. It separates concerns:
- **Part A (Semantic Layer)** interprets intent and formulates a `CloudOptimizationContract`.
- **Part B (Symbolic Layer)** executes that contract using a pipeline of heuristic and formal solvers, ensuring constraints are strictly met before applying soft preferences.

## 2. Literature Mapping

| Research Work | Year | Core Idea | Project Layer | Adopted? | Adaptation |
|---|---:|---|---|---|---|
| OptiHive | 2024 | Candidate selection & routing | Phase 4 (Selection) | Inspired / Conceptual | Implements an EM-inspired latent class selector for routing between solvers, but not the full neural Hive architecture. |
| SAGE-GNN | 2023 | Graph Neural Networks for placement | Phase 3 (Graph-Z3) | Conceptual only | Graph-based topological steering is implemented heuristically to guide the Z3 solver, not via a trained SAGE-GNN. |
| AutoCO | 2023 | Automated Cloud Optimization | Phase 5 (API) | Conceptual relation | The project shares the goal of automated, API-driven configuration, but utilizes neuro-symbolic partitioning rather than pure ML models. |
| HeurAgenix | 2024 | Agentic heuristic generation | Phase 1/5 | Conceptual only | Separates semantic agents (Part A) from deterministic execution (Part B), drawing on the philosophy of modular agentic heuristics. |
| LLM-Based TSP Heuristics | 2023 | LLMs failing at strict TSP constraints | Part A vs Part B | Core Motivation | Directly motivated the architecture: LLMs cannot reliably do math/constraints, so they must be separated from numerical execution. |

## 3. OptiHive
**Concept**: OptiHive proposes intelligent routing and selection mechanisms to evaluate which optimization engine or heuristic best suits a given problem instance based on latent features.
**Adaptation**: The project implements an *OptiHive-inspired* latent-class Expectation-Maximization (EM) selector. It routes feasible candidates based on simulated historical convergence metadata (latency, cost, SLA) rather than a deep neural network, demonstrating the architectural concept of candidate selection.

## 4. Graph-Based Optimization / SAGE-GNN
**Concept**: SAGE-GNN models use graph embeddings to optimize placement across interconnected cloud nodes, leveraging topological features.
**Adaptation**: Phase 3 implements a *deterministic graph-steering layer*. It uses topological heuristics (peer latencies, regions) to prioritize search spaces for the Z3 SMT solver. It is **not** a trained GNN. The architecture demonstrates how graph data can steer formal constraints, acting as a placeholder for a future GNN.

## 5. AutoCO
**Concept**: AutoCO focuses on automating cloud deployment configurations and resource tuning via ML models.
**Adaptation**: The project's FastAPI orchestration gateway conceptually aligns with AutoCO by automating the pipeline from user request to deployed configuration. The neuro-symbolic approach diverges by relying on deterministic solvers for the actual resource selection rather than black-box ML.

## 6. HeurAgenix
**Concept**: Generating heuristics and optimization strategies using agent-based models.
**Adaptation**: The separation of Part A (Semantic Agent) and Part B (Deterministic Engines) embraces the HeurAgenix philosophy of modular, agentic problem-solving, though Part B relies on traditional metaheuristics rather than LLM-generated algorithms.

## 7. LLM-Based TSP Heuristics
**Concept**: Recent studies highlight that while LLMs can suggest heuristic strategies for the Traveling Salesperson Problem (TSP) and bin-packing, they fail at strict constraint adherence.
**Adaptation**: This literature is the foundational justification for the project's strict architecture: soft semantic processing (LLMs) *must* be separated from hard symbolic execution (Z3/GA/PSO).

## 8. GA / PSO
**Concept**: Metaheuristic algorithms (Genetic Algorithm, Particle Swarm Optimization) for navigating large discrete search spaces.
**Adaptation**: Directly implemented. GA is highly suitable for the discrete selection of VMSkus, while PSO is traditionally used for continuous spaces but was adapted here for discrete/mixed integer validation.

## 9. SMT / Z3
**Concept**: Satisfiability Modulo Theories (SMT) for formal verification and constraint solving.
**Adaptation**: Directly implemented. Used as the formal validation layer to strictly prove whether a proposed cloud configuration meets all budget, VCPU, and RAM constraints.

## 10. Research Gap
This project addresses the gap between **semantic intent** and **deterministic cloud resource provisioning**. While pure LLM approaches hallucinate constraints, and pure symbolic systems are too rigid for business users, this architecture combines natural language contract formulation with strict, multi-engine symbolic execution and latent candidate routing.

## 11. Implementation Gap
To maintain a feasible academic scope, the following research elements remain simplified:
- **No Trained GNNs**: Graph steering is heuristic.
- **No Deep Neural Selection**: OptiHive routing uses classical EM clustering.
- **Simulated Benchmarks**: OPT-BENCH scores are stored in a relational database rather than dynamically executed.
- **Synthetic Pricing**: The local SQLite database uses static, illustrative 2026 pricing data rather than live AWS/Azure billing APIs.
