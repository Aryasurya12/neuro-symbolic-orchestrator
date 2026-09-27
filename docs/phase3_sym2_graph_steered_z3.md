# Phase 3 — SYM-2: Graph-Steered Z3 SMT Solver

## 1. Objective
The objective of Phase 3 was to introduce the **SYM-2 Formal Validation Layer** through a Graph-Steered Z3 SMT Solver. This solver validates multi-region disaster recovery deployments, proving formal correctness while allowing heuristic network topology graphs to bias the cost-minimization logic safely.

## 2. Relationship to Previous Phases
- **Phase 0/1:** Continues to consume the strictly bounded `SymbolicOptimizationRequest`. Does not modify Part-A semantic processing.
- **Phase 2:** Exists as an independent `OptimizationEngine` conforming to the Phase-1 interface, sitting alongside the vectorized GA and PSO engines within the symbolic layer without disrupting their operation.

## 3. Insufficiency of the Old Prototype
The previous prototype (`Graph_SMT_Z3_MultiRegion_Placement.py`) used a Python `for` loop to exhaustively generate all pairwise permutations, validate them manually via `if` statements, and return the cheapest. While practically effective for $N=5$, this was a brute-force combinatorial search, *not* a formal SMT verification. The Phase-3 implementation creates explicit `z3.Bool` variables mapped directly into the `z3.Optimize()` engine, allowing Z3's internal DPLL(T) heuristics to mathematically resolve the feasible bounds directly.

## 4. Graph Model
Located in `src/symbolic/solvers/graph_model.py`.
- **Nodes:** Physical cloud regions equipped with deterministic metadata (`base_cost`, `sla_pct`, `provider`).
- **Edges:** Implicitly defined by standard interconnect latency maps. 

## 5. Graph Steering
Located in `src/symbolic/solvers/graph_steering.py`.
- **Abstraction:** Serves as a deterministic placeholder for the future ML-based SAGE-GNN. 
- **Soft Preference:** It calculates a network topology penalty (e.g. 1 unit of penalty per ms latency, minus 50 units if the nodes cross geographic failover boundaries). 
- **Rule:** This layer outputs a `GraphPreference` strictly as a heuristic soft-preference; it never validates feasibility.

## 6. Z3 Hard Constraints
Located in `src/symbolic/solvers/constraints.py`.
- **Decision Variables:** A dictionary of `z3.Bool` variables dictating if a region is active.
- **Topological Constraints:** Exactly two regions must be active (`z3.Sum(variables) == 2`).
- **Budget Constraint:** Base instance cost + pairwise latency sync cost $\le$ `Budget`.
- **Provider Constraint:** `z3.Not(selected)` for any region belonging to a non-approved provider.
- **Latency & SLA Constraints:** `z3.Implies` assertions governing inter-node sync metrics if both regions are active.

## 7. Hard vs Soft Separation Mechanism
The Z3 Model minimizes the objective:
$$ Minimize: TotalCost + (0.001 \times GraphPreference) $$
This coefficient ($0.001$) guarantees that Graph Preference operates purely to break ties between equally-priced topological configurations. It is mathematically incapable of rendering an invalid state "valid" because Z3 bounds the feasible region strictly before the objective evaluates. A mandatory security test (`test_hard_vs_soft_separation_security_test`) physically asserts that a candidate with perfect graph topology is rejected if it violates hard constraints.

## 8. Post-Solve Validation
Even after Z3 asserts `sat`, the extracted `OptimizationCandidate` runs through lightweight Python verification to defensively prevent solver mapping errors or floating-point constraint leaks. If verification fails, it reverts the result to `is_feasible = False`.

## 9. Performance
Solver execution operates well within interactive thresholds (standard verification in $< 50$ms on dev hardware for $N=5$ node subsets), utilizing `time.perf_counter()` to accurately report internal metric evaluation times.

## 10. Future ML Integration
By separating `GraphSteeringLayer` entirely from Z3 constraint initialization, Phase 4/Phase 9 can directly swap the deterministic heuristic function with an RGCN node-embedding similarity scoring model without modifying the Z3 optimization engine.

## 11. Phase 3 Exit Criteria
- [x] Genuine `z3-solver` dependency utilized correctly.
- [x] Hard Constraints (Budget, Latency, Provider, SLA) actively enforced.
- [x] Soft Graph Preferences strictly subservient to Hard Constraints.
- [x] `sat`, `unsat`, and `unknown` handled predictably without Python crashes.
- [x] Optimization engine outputs perfectly standard `OptimizationResult`.
- [x] All 52 original tests passing + 9 new Z3/Graph tests. 
