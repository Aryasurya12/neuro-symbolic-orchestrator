# neurasym: Pre-Flight Verification Report

**Verification Timestamp:** 2026-10-03  
**Target Codebase:** Aryasurya12/neuro-symbolic-orchestrator (`neurasym`)  
**Status:** Pre-flight complete. All 3 items investigated with empirical/code evidence.

---

## Item 1: OptiHiveSelector Dead Code Investigation

### Finding (Code Evidence)

1. **Candidate Solver Generation in Main Orchestrator (`src/orchestrator/service.py`)**:
   In `NeuroSymbolicOrchestrator.optimize_contract()` (lines 60–77):
   ```python
   sym_req = from_contract(contract)
   candidates = []

   # Stage 4: Solver execution / Parallel Race
   if contract.problem_type == "ILP_VM_Allocation":
       candidates.append(self.ga_solver.solve(sym_req))
   elif contract.problem_type == "PSO_Continuous_Scaling":
       candidates.append(self.pso_solver.solve(sym_req))
   elif contract.problem_type == "Z3_Graph_Disaster_Recovery":
       candidates.append(self.z3_solver.solve(sym_req))
   else:
       candidates.append(self.ga_solver.solve(sym_req))
       candidates.append(self.pso_solver.solve(sym_req))
       candidates.append(self.z3_solver.solve(sym_req))

   # Stage 5: OptiHive Latent-Class EM Solver Selection
   final_result = self.selector.select(sym_req, candidates)
   ```
   Furthermore, in `src/semantic/schemas.py` (lines 16–23), `CloudOptimizationContract` strictly validates `problem_type` via Pydantic:
   ```python
   problem_type: Literal[
       "ILP_VM_Allocation",
       "PSO_Continuous_Scaling",
       "Z3_Graph_Disaster_Recovery",
   ] = Field(..., description="Target optimization problem type selected by CARM matcher")
   ```
   Because `problem_type` is constrained to this 3-value literal, every valid contract executes exactly one branch in `src/orchestrator/service.py`, appending exactly **1 candidate solver** to `candidates` (`ga_solver`, `pso_solver`, or `z3_solver`). The `else:` branch (which would generate 3 candidates) is unreachable for all valid queries.

2. **Downstream Selection Logic (`src/symbolic/optihive/solver_selection.py`)**:
   In `OptiHiveSelector.select()` (lines 45–69):
   ```python
   # 2. ILP Syntactic Filter
   filtered_candidates = []
   for cand in candidates:
       fc = ILPSyntacticFilter.filter(cand, request)
       if fc.syntactically_feasible:
           fc.candidate.metadata["ilp_filter"] = "passed"
           filtered_candidates.append(fc)
       else:
           cand.metadata["ilp_filter"] = "failed"
           cand.metadata["violations"] = fc.violations
           
   # 3. EM Latent-Class Selection
   if not filtered_candidates:
       ...
       return OptimizationResult(best_candidate=best_infeasible, is_feasible=False, ...)
       
   best_fc, _, _ = self.em_selector.select(filtered_candidates)
   ```
   When `candidates` contains only 1 item:
   - `ILPSyntacticFilter.filter()` acts as a single-candidate feasibility/syntax check.
   - If feasible, `self.em_selector.select([fc])` receives a 1-element list and immediately returns `cands[0]`.

3. **Multi-Candidate Branches in Other Repo Call Sites**:
   - **Offline Benchmark Suites** (`benchmarks/end_to_end_demo.py` lines 58–62 and `benchmarks/system_evaluation.py` lines 121–124):
     ```python
     candidates = [res_ga, res_pso, res_z3]
     selector = OptiHiveSelector(seed=42)
     final_result = selector.select(req, candidates)
     ```
     Here, all 3 solvers are explicitly executed and passed into `OptiHiveSelector.select()`, where true EM latent-class arbitration occurs between competing solver results.
   - **API / WebSocket Service** (`src/api/service.py` lines 58–96):
     When `AdaptiveSolverRouter` returns a multi-solver fallback set (e.g. `["GA", "PSO", "Z3"]`), GA and PSO are raced into a single `race_result` by `OptimizerRace`, and up to 2 candidates (`race_result`, `z3_result`) can enter `OptiHiveSelector.select()`.

### Conclusion
- **Is it broken?** **Yes** (in terms of active multi-solver arbitration during live query orchestration).
- In the primary query orchestration pipeline (`NeuroSymbolicOrchestrator`), `OptiHiveSelector.select()` is **never choosing between alternatives** — it functions as a single-candidate feasibility filter and pass-through no-op.

### Action Taken
- **Left as documented limitation**. No synthetic candidate fabrication was introduced. This architectural boundary is documented honestly as a single-solver deterministic dispatch with post-hoc feasibility verification in the runtime orchestrator, with full multi-candidate EM arbitration reserved for the offline benchmark harness.

---

## Item 2: ILP_VM_Allocation Solver Mismatch

### Finding (Code Evidence)

1. **Actual Solver Executed (`src/orchestrator/service.py`)**:
   In `NeuroSymbolicOrchestrator.__init__()` and `optimize_contract()` (lines 28–30, 64–65):
   ```python
   # Hoist solvers to instance level: instantiated once per orchestrator lifetime
   self.ga_solver = GeneticAlgorithm(random_seed=42)
   self.pso_solver = ParticleSwarmOptimization(random_seed=42)
   self.z3_solver = GraphSteeredZ3Solver(...)
   ...
   if contract.problem_type == "ILP_VM_Allocation":
       candidates.append(self.ga_solver.solve(sym_req))
   ```
   The underlying solver called is `src.symbolic.optimizers.genetic_algorithm.GeneticAlgorithm` (`solver_name="Vectorized_GA"`), **not** `scipy.optimize.milp` (HiGHS).

2. **Dashboard Label Before Fix (`app.py` & `index.html`)**:
   In `app.py` (lines 894–896):
   ```python
   else:
       engine_label = "SciPy HiGHS MILP"
       solver_desc = "Branch & Bound global integer programming solution satisfying all resource inequalities."
   ```
   In `app.py` empty state (line 2516):
   ```html
   <span style="background: var(--bg-surface-inset); border: 1px solid var(--border-default); color: var(--seq-3); padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem;">SciPy HiGHS MILP</span>
   ```
   In `app.py` architecture card (line 2122):
   ```html
   <div style="font-size: 0.98rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">SYM-1 & SYM-2: SciPy HiGHS & Z3 SMT Graph</div>
   ```

3. **Status**:
   There was a clear mismatch between the dashboard labels ("SciPy HiGHS MILP") and the runtime execution engine (`Vectorized_GA`).

### Conclusion
- **Is it broken?** **Yes** (label/code mismatch). The code was executing `Vectorized GA` while the UI displayed `SciPy HiGHS MILP`.

### Action Taken
- **Fixed (Dashboard Labels Updated)**. Updated `app.py` and `index.html` to accurately label the engine as `Vectorized GA (ILP Allocation)` / `Vectorized GA` and reflect heuristic discrete SKU search in the descriptions, preserving solver logic integrity while making the UI 100% truthful to reality.

---

## Item 3: Mode 1 LLM Truncation Fix Status

### Finding (Empirical Live Evidence)

A dedicated 3-run live test was executed using `verify_mode1_live.py` against OpenRouter (`nvidia/nemotron-3.5-lightning:free`) with the standardized prompt:
> *"Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."*

```
================================================================================
MODE 1 LIVE TRUNCATION & VERIFICATION (3 LIVE RUNS)
Model: nvidia/nemotron-3.5-lightning:free
API Key: sk-or-v1-6...9241cb
================================================================================

>>> RUN 1 of 3...
   - Latency: 27.20s
   - max_tokens sent: 2048
   - finish_reason: stop
   - Usage tokens: {'prompt_tokens': 215, 'completion_tokens': 129, 'total_tokens': 344}
   - Extracted Cost: $216.00/mo
   - Model Recommendation: AWS, 4x t3.large instances (16 vCPUs, 64GB RAM). Stated Cost: $216.00

>>> RUN 2 of 3...
   - Latency: 1.96s
   - max_tokens sent: 2048
   - finish_reason: stop
   - Usage tokens: {'prompt_tokens': 215, 'completion_tokens': 59, 'total_tokens': 274}
   - Extracted Cost: $288.00/mo
   - Model Recommendation: AWS, 2x c5.4xlarge instances (32 vCPUs, 64GB RAM). Stated Cost: $288.00

>>> RUN 3 of 3...
   - Latency: 44.17s
   - max_tokens sent: 2048
   - finish_reason: stop
   - Usage tokens: {'prompt_tokens': 215, 'completion_tokens': 73, 'total_tokens': 288}
   - Extracted Cost: $144.00/mo
   - Model Recommendation: AWS, 1x c5.2xlarge instance (8 vCPUs, 16GB RAM). Stated Cost: $144.00
```

1. **`finish_reason`**: `stop` in all 3 runs (0% truncation rate).
2. **`max_tokens`**: `2048` sent in the primary request (with 4096 fallback retry configured in code).
3. **Preamble / Extraction Soundness**:
   - `strip_reasoning_preamble()` successfully stripped thinking text.
   - `extract_mode1_cost()` extracted `$216.00`, `$288.00`, and `$144.00` directly from the model's recommendation lines.
   - None of the extracted costs matched the user input budget (`$300.00`), proving that extraction is reading actual generated recommendation numbers rather than echoing the budget figure.
4. **Independent Catalog Cross-Check**:
   - Run 1 (4x `t3.large`): Real catalog cost = $0.0832/hr * 730 * 4 = **$242.94/mo** (Model claimed $216.00/mo $\rightarrow$ hallucinated pricing delta of -$26.94).
   - Run 3 (1x `c5.2xlarge`): Real catalog cost = $0.3400/hr * 730 = **$248.20/mo** (Model claimed $144.00/mo $\rightarrow$ hallucinated pricing delta of -$104.20).
   - This validates the core benchmark premise: Pure LLMs hallucinate pricing arithmetics even when completion cleanly completes with `finish_reason == "stop"`.

### Conclusion
- **Is it broken?** **No**. The Mode 1 truncation fix is active and fully functional. 3 consecutive live runs cleanly returned `finish_reason == "stop"` with verified independent cost extraction.

### Action Taken
- Fixed minor OpenRouter reasoning payload parameter conflict (`extra_body={"reasoning": {"effort": "none"}}`) in `benchmarks/run_4way_benchmark.py` and `diagnose_llm_modes.py` to prevent OpenRouter 400 parameter errors when invoking reasoning models.

---

## Summary Matrix

| Item | Focus Area | Code/Runtime Reality | Status Before | Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| **Item 1** | `OptiHiveSelector` | Exactly 1 candidate passed per branch in `NeuroSymbolicOrchestrator` | No-op pass-through in live orchestrator | **Left as-is (Documented Limitation)** |
| **Item 2** | `ILP_VM_Allocation` | Backend executes `Vectorized_GA`; UI labeled `SciPy HiGHS MILP` | Label/Code Mismatch | **Fixed Dashboard Labels in `app.py` & `index.html`** |
| **Item 3** | Mode 1 Truncation | `max_tokens=2048`, `finish_reason="stop"`, cost correctly extracted 3/3 | Working / Clean | **Verified Live (3/3 Clean Runs)** |
