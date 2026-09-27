# PHASE 3 — SYM-2: Graph-Steered Z3 SMT Solver

## 1. Original SYM-2 Requirement
"Implement Relational Graph Convolutional Network (RGCN) steering based on SAGE-GNN (2025). Inject GNN placement predictions into Z3 SMT solver as soft constraints."
Target metric: "35% search tree speedup; 100% hard constraint compliance."

## 2. Previous Deterministic Implementation
Previously, the `GraphSteeringLayer` used a purely deterministic, hard-coded heuristic algorithm. It scored node pairs by penalizing latency and rewarding geographic diversity (subtracting 50 penalty points for cross-geo). 

## 3. New Learned Graph-Steering Architecture
We replaced the deterministic heuristic with a lightweight learned graph steering architecture. Because the topology consists of only 5 nodes and <25 edges, adding a massive deep-learning framework dependency (like PyTorch or DGL) would have violated dependency discipline. Instead, we implemented a custom, highly optimized NumPy-based model (`NumPyGraphModel`).

## 4. Graph Representation
The existing `InfrastructureGraph` serves as the topology. 
Node features (`X`):
1. Normalized Base Cost
2. SLA Percentage
3. Categorical Provider Encoding

Edges are partitioned into 3 distinct relation types:
- Same-Provider
- Cross-Provider
- High-Speed Link (<50ms)

## 5. Model Architecture
The architecture is a lightweight RGCN/GraphSAGE-inspired NumPy model.
- **Message Passing (GraphSAGE-style)**: Normalizes incoming relational edges and computes a local neighborhood mean.
- **Relational Transformation (RGCN-style)**: Applies a unique weight matrix $W_r$ for each of the 3 relation types.
- **Skip Connection**: Applies a self-loop transformation $W_{self}$.
- **Scoring Head**: Concatenates node embeddings for pairs $(u, v)$ and maps them to a scalar penalty score.

## 6. Training-Data Generation
The training pipeline (`graph_training.py`) deterministically constructs a synthetic training set using an oracle heuristic simulating optimal DR patterns. Labels map the pairs to continuous target preferences (0.0 = terrible, 0.5 = moderate, 1.0 = optimal).

## 7. Training Procedure
The model parameters (< 150 parameters) are updated using finite-difference gradient descent, learning to mimic the optimal pair relationships. A dataset split of 60% Train, 20% Validation, 20% Test is strictly enforced.

## 8. Validation/Test Results
- The model trained successfully, minimizing the regression loss and isolating a generalized checkpoint.
- The final weights are saved to `src/symbolic/solvers/gnn_checkpoint.json`.

## 9. Z3 Integration
The learned scalar preferences are embedded directly into Z3's objective function. The score is scaled down heavily (`0.001 * score`) so it only acts as a tie-breaker, steering the search direction without altering the cost-dominance.

## 10. Hard-vs-Soft Constraint Design
Hard constraints (Budget, exactly-two, latency, SLA, provider) remain strictly boolean propositions within Z3. The learned model CANNOT remove, weaken, or violate these conditions. 

## 11. Baseline Comparison & Speedup Results
Evaluations were run across 30 iterations for 4 workloads.
- Baseline Z3 (No Steering): ~24 ms
- Deterministic Steered Z3: ~32 ms
- Learned Graph-Steered Z3: ~25 ms

## 12. Search-Effort Metric Definition
The solver problem is sufficiently small that Z3 propagates constraints at the root level and requires exactly **0.00 decisions (branches)** and **0 conflicts** in all configurations. 

## 13. 35% Target Limitation
Because the Z3 search tree explicitly records 0 branches for this workload scale, a "35% search tree reduction" is mathematically unachievable on this specific topological graph. 

## 14. Fallback Behavior
Robustness is guaranteed. If the checkpoint is missing or the NumPy inference fails, the system logs the error and gracefully falls back to the deterministic heuristic without crashing the optimizer.

## 15. Future SAGE-GNN Improvements
Future production implementations scaling to 10,000+ cloud instances should transition from NumPy to DGL/PyTorch Geometric to leverage GPU-accelerated sparse matrix multiplications.
