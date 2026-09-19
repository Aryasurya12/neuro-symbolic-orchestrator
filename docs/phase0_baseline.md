# Phase 0 — Baseline & Branch Freeze

## 1. Purpose
The purpose of Phase 0 is to establish a trustworthy, clean, and verified starting point on the `arya-part-b` branch before implementing any symbolic optimization modules. This ensures Shreeya's Part-A implementation (Semantic / LLM layer) remains functional and that the exact integration boundaries are documented.

## 2. Branch State
The repository uses the following branching model:
- `main`: Production/stable branch.
- `shreeya-part-a`: Development branch for the Neural/Semantic layer.
- `arya-part-b`: **Current active branch**; the dedicated development branch for Arya's Symbolic Metaheuristics & Validation work (Part B). 
The branch has been verified as clean and directly based on the expected starting state.

## 3. Current Repository Structure
The repository structure differs slightly from the strictly expected template. Noticeably, `requirements.txt` is missing from the root, and there's an unexpectedly nested `neuro-symbolic-orchestrator/README.md`.

```text
.
├── .gitattributes
├── .gitignore
├── config/
│   ├── __init__.py
│   └── settings.py
├── main.py
├── neuro-symbolic-orchestrator/
│   └── README.md
├── src/
│   ├── __init__.py
│   └── semantic/
│       ├── __init__.py
│       ├── carm_matcher.py
│       ├── explainer.py
│       ├── schemas.py
│       └── scope_parser.py
├── templates/
│   ├── __init__.py
│   ├── Continuous_PSO_Dynamic_Scaling.py
│   ├── Graph_SMT_Z3_MultiRegion_Placement.py
│   └── ILP_VM_Knapsack_Allocation.py
└── tests/
    └── test_semantic.py
```

## 4. Existing Part-A Components
- **SCOPE parser (`src/semantic/scope_parser.py`)**: Disentangles natural language intents into sets of string constraints (e.g., `"Budget_Limit_Max"`, `"Continuous_Bandwidth_Range"`). Extracts key parameters (budget, vCPUs, RAM, latency) and supports dual currency parsing (USD/INR).
- **CARM matcher (`src/semantic/carm_matcher.py`)**: Maps the extracted constraints against predefined symbolic optimization archetypes using Jaccard Similarity.
- **Pydantic schema (`src/semantic/schemas.py`)**: Defines the strict boundary contract (`CloudOptimizationContract`).
- **Explainer (`src/semantic/explainer.py`)**: Takes the raw solver output and the original contract, transforming them into a dual-currency (USD/INR) ASCII executive FinOps report.

## 5. Existing Prototype Components
- **PSO template (`templates/Continuous_PSO_Dynamic_Scaling.py`)**: A continuous optimization prototype using basic numpy loops to search for bandwidth sizing and replica counts.
- **graph/SMT placement template (`templates/Graph_SMT_Z3_MultiRegion_Placement.py`)**: Currently a Python-side combinatorial evaluation script checking region pairs against latency/SLA limits, not a true formal Z3 model.
- **ILP allocation template (`templates/ILP_VM_Knapsack_Allocation.py`)**: Uses SciPy's `milp` solver (if available) and falls back to a custom branch-and-bound numpy loop to find optimal VM combinations.

## 6. Current Data Contract
The absolute integration boundary between Part A and Part B is defined in `CloudOptimizationContract`. 

```python
class CloudOptimizationContract(BaseModel):
    problem_type: Literal[
        "ILP_VM_Allocation",
        "PSO_Continuous_Scaling",
        "Z3_Graph_Disaster_Recovery",
    ]
    cloud_providers: List[Literal["AWS", "Azure", "GCP"]] = ["AWS"]
    budget_max_usd: float  # (Must be >= 10.0)
    service_count: int
    required_vcpus: int
    required_ram_gb: float
    latency_max_ms: float  # (Must be <= 1000.0)
    sla_availability_pct: float  # (Between 90.0 and 99.999)
```

**JSON Example:**
```json
{
  "problem_type": "ILP_VM_Allocation",
  "cloud_providers": ["AWS", "Azure"],
  "budget_max_usd": 300.0,
  "service_count": 5,
  "required_vcpus": 4,
  "required_ram_gb": 16.0,
  "latency_max_ms": 50.0,
  "sla_availability_pct": 99.9
}
```

## 7. Current Execution Flow
The runtime operates as an interactive CLI loop inside `main.py`:
1. `User input` (via standard input)
2. `SCOPEParser.parse_query_to_contract()` extracts constraints and builds the `CloudOptimizationContract`.
3. `CARMMatcher` assigns a template based on extracted features.
4. Execution routes dynamically to one of the prototype functions in `templates/`.
5. The solver's dictionary result is passed to `FinOpsExplainer.generate_report()` to output an ASCII summary.

## 8. Dependency Inventory
There is currently **no `requirements.txt` or `pyproject.toml`** file at the repository root. Based on the imports, the active required dependencies are:
- `pydantic` (for `CloudOptimizationContract`)
- `numpy` (for PSO and ILP prototypes)
- `scipy` (optional fallback for the ILP prototype)
- `pytest` (development/testing)

## 9. Baseline Test Results
Tests ran via `python -m pytest tests/` on 2026-09-20:
- **Total tests**: 35
- **Passed**: 35
- **Failed**: 0
- **Skipped**: 0

*Note: Initially `pytest tests/` failed due to `PYTHONPATH` not including the root module, resolving correctly using `python -m pytest`.*

## 10. Baseline Execution Result
The application successfully starts via `python main.py`. It presents an interactive CLI loop (`💬 Enter Cloud Request:`). Inputting requests generates appropriate CARM matches and solver metrics.

## 11. Known Limitations / Technical Debt
- **Missing `requirements.txt`**: Dependency management is not defined at the project root.
- **Misplaced `README.md`**: Found inside a redundant `neuro-symbolic-orchestrator` subdirectory.
- **No True Z3 Model**: `Graph_SMT_Z3_MultiRegion_Placement.py` only implements Python enumeration, not an actual SMT boolean constraint formulation.
- **Prototype Optimizers**: PSO and ILP templates are naive implementations lacking the advanced vectorized logic intended for SYM-1.
- **No Database**: Cloud providers and SKUs are currently hard-coded directly into the template files.

## 12. Phase 0 Exit Criteria
- [x] Branch is confirmed as `arya-part-b`.
- [x] Repository fully inspected and structure documented.
- [x] `CloudOptimizationContract` verified and logged.
- [x] Baseline execution and test suite run (35/35 passing).
- [x] Phase 0 baseline committed professionally.
