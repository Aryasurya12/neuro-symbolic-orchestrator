# Phase 1 — Symbolic Foundation & Integration Contract

## 1. Purpose
The purpose of Phase 1 is to establish a clear, strongly typed symbolic-layer foundation (`src/symbolic`) that allows future Part-B optimization engines (GA, PSO, Z3, OptiHive) to plug in cleanly, without disrupting the existing semantic parsing from Part-A (Shreeya's work).

## 2. Relationship to Phase 0
Phase 1 builds strictly on the Phase 0 baseline documented in `docs/phase0_baseline.md`. No previous tests or interactive CLI logic have been modified or broken. The existing prototype templates remain untouched.

## 3. Part-A → Part-B Boundary
The integration boundary is defined natively by the semantic layer. The `CloudOptimizationContract` continues to serve as the authoritative handshake boundary. We treat this contract as read-only on the Part-B side, using an adapter pattern to accept it into the symbolic layer.

## 4. Exact CloudOptimizationContract Used
Inspected from `src/semantic/schemas.py`:
- `problem_type` (Literal)
- `cloud_providers` (List of Literal)
- `budget_max_usd` (float)
- `service_count` (int)
- `required_vcpus` (int)
- `required_ram_gb` (float)
- `latency_max_ms` (float)
- `sla_availability_pct` (float)

## 5. SymbolicOptimizationRequest
Defined in `src/symbolic/models.py`. 
Normalizes the request inside the symbolic boundaries. It contains identically named fields to the `CloudOptimizationContract` but uses standard typed annotations (`str`, `List[str]`) for solver ease, separating the Pydantic intent parsing structure from algorithmic parameters.

## 6. OptimizationCandidate
Defined in `src/symbolic/models.py`.
Represents a generic solver candidate. Fields include:
- `decision_variables`: Dict of selected resources/allocations.
- `objective_cost_usd`: Float metric representing fitness.
- `is_feasible`: Boolean indicating constraint compliance.
- `constraint_status`: Detailed `ConstraintStatus` structure.
- `metadata`: Flexible dictionary for algorithm-specific state.

## 7. ConstraintStatus
Defined in `src/symbolic/models.py`.
Stores structured feasibility mapping for future Z3 hard-constraint compliance matching:
- `is_feasible`, `budget_ok`, `vcpu_ok`, `ram_ok`, `latency_ok`, `sla_ok` (booleans)
- `details` (string)

## 8. OptimizationMetrics
Defined in `src/symbolic/models.py`.
Common algorithmic execution tracking for Phase 5 (OptiHive) evaluation:
- `runtime_ms` (float)
- `iterations` (int, optional)
- `evaluations` (int, optional)

## 9. OptimizationResult
Defined in `src/symbolic/models.py`.
Standardized output structure for any solver:
- `best_candidate` (OptimizationCandidate, optional)
- `is_feasible` (bool)
- `metrics` (OptimizationMetrics)
- `solver_name` (str)
- `error_message` (str, optional)

## 10. OptimizationEngine Interface
Defined in `src/symbolic/interfaces.py`.
Provides a generic `Protocol` with the signature:
`def solve(self, request: SymbolicOptimizationRequest) -> OptimizationResult:`
All future optimizations (GA, PSO, Z3) will implement this duck-typed interface, making downstream reporting and API interactions uniform.

## 11. Contract Adapter
Defined in `src/symbolic/adapters.py`.
Implements `from_contract(contract: CloudOptimizationContract) -> SymbolicOptimizationRequest` to perform standard translation, preserving semantic meaning without executing NLP parsing again.

## 12. Data Flow
```text
CloudOptimizationContract (From Part A)
          ↓ (Adapter)
SymbolicOptimizationRequest
          ↓ (Engine Interface)
      [GA / PSO / Z3]
          ↓
  OptimizationResult
          ↓ (Future)
      OptiHive Filter
```

## 13. Why the Interfaces are Algorithm-Agnostic
They use flexible decision variable dictionaries and a standard metrics payload. This allows discrete combinatorial solvers (ILP/Z3) and continuous approximation models (PSO) to conform to the same interface without dictating internal state.

## 14. What Phase 1 Intentionally Does NOT Implement
- No actual GA, PSO, or Z3 solvers exist.
- OptiHive logic and Database querying are unwritten.
- The `main.py` CLI pipeline is intentionally NOT connected to these new symbolic models yet, to preserve baseline execution stability until the actual engines are written.

## 15. Test Results
`python -m pytest tests/`
41 tests passing (35 legacy baseline semantic tests + 6 new symbolic interface/model tests).

## 16. Phase 1 Exit Criteria
- [x] CloudOptimizationContract verified.
- [x] Part A intact.
- [x] Symbolic package created.
- [x] Required models and interface established.
- [x] Adapter to map between Part A and Part B defined.
- [x] 100% test passing (Originals + New).
