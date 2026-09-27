# Phase 7 — SYM-6: Research Traceability Mapping

This document maps the implemented project components to the broader literature concepts they draw from, clarifying the boundaries between published research and this specific engineering implementation.

## Part A (Semantic Layer)
- **Literature Concept**: CARM (Cloud Application Resource Mapping), LLM-based prompt engineering for intent translation.
- **Implementation**: Handled by Shreeya. Natural language models interpret intent and produce a structured `CloudOptimizationContract`.
- **Deviation**: Uses a simplified contract schema rather than full Kubernetes/Terraform definitions.

## GA (Genetic Algorithm)
- **Literature Concept**: Discrete Metaheuristic Optimization.
- **Implementation**: Vectorized GA using NumPy (`src/symbolic/optimizers/genetic_algorithm.py`). Evaluates candidate SKUs against cost, vCPU, and RAM objectives.
- **Deviation**: Heavily vectorized for speed rather than object-oriented. Hard constraint masking limits crossover space.

## PSO (Particle Swarm Optimization)
- **Literature Concept**: Swarm Intelligence for continuous/mixed search spaces.
- **Implementation**: Vectorized PSO (`src/symbolic/optimizers/particle_swarm.py`). Adapts continuous particle positions back to discrete SKU indices.
- **Deviation**: Primarily continuous, meaning boundary conditions require forced rounding to discrete SKUs, which can degrade strict convergence guarantees on highly fragmented sets.

## Graph Steering
- **Literature Concept**: Graph-based optimization and SAGE-GNN (Graph Neural Networks).
- **Implementation**: Deterministic topological heuristics (`src/symbolic/solvers/graph_model.py` and `graph_steering.py`). Guides Z3 solver bounds based on peer latencies and base region costs.
- **Deviation**: Completely deterministic. **No GNN is trained or executed.** SAGE-GNN is a conceptual aspiration for future work.

## Z3 Solver
- **Literature Concept**: Satisfiability Modulo Theories (SMT) and formal constraint solving.
- **Implementation**: Z3 bindings (`src/symbolic/solvers/z3_solver.py`) mathematically prove whether constraints (e.g., budget, SLA) can be satisfied.
- **Deviation**: Standard usage of `z3-solver`. Solves integer linear representations of the problem rather than complex non-linear combinations.

## OptiHive-Inspired Selector
- **Literature Concept**: OptiHive (Latent/candidate-selection routing for solvers).
- **Implementation**: EM-based (Expectation-Maximization) clustering (`src/symbolic/optihive/em_selector.py`) selects between the outputs of GA, PSO, and Z3.
- **Deviation**: Uses classical machine learning (Gaussian Mixture Models / EM) over historically simulated convergence metadata rather than a deep neural network routing architecture.

## SQLite FinOps Catalog
- **Literature Concept**: Cloud resource pricing optimization databases.
- **Implementation**: Local SQLite repository (`src/symbolic/data/`).
- **Deviation**: Contains static 2026 illustrative sample data. Does not scrape live AWS/Azure APIs. Simulated OPT-BENCH scores are used.

## FastAPI / WebSocket
- **Literature Concept**: Automated Orchestration Gateways (e.g., AutoCO).
- **Implementation**: Asynchronous API gateway (`src/api/`).
- **Deviation**: Pure engineering wrapper to expose the symbolic pipeline. Does not incorporate active ML feedback loops in the endpoints.
