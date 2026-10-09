# NEURASYM Live Study Run Archive: `run_20261009_093328`

## Execution Metadata
- **Run ID**: `run_20261009_093328`
- **Timestamp (UTC)**: `2026-10-09T09:33:28.099094+00:00`
- **Manifest File**: `data/final_query_manifest.json`
- **Manifest SHA-256**: `2a0666c687664c0bdae3acfb4af4045f9d1a8fb46b09c098b37e4eae86fe3874`
- **Code Commit**: `f437679`
- **Model**: `openai/gpt-oss-120b` (Groq API)
- **Resolved Providers**:
  - Mode 1: `Groq API (openai/gpt-oss-120b)`
  - Mode 2: `Groq API (openai/gpt-oss-120b)`
  - Mode 3: `Pure Local Symbolic (SCOPE + HiGHS/Z3/PSO)`
  - Mode 4: `Groq API (openai/gpt-oss-120b) + Local Solvers`
- **Is Mock Provider**: `False`
- **Number of Manifest Queries ($T$)**: `29` (15 `dev` / 14 `eval`)
- **Total Output Rows ($T \times 4$)**: `116`

## Assertion Verification Results (All Pass)
- **Assertion a) Row Count**: `PASS` (116 rows == 29 * 4)
- **Assertion b) Tuple Uniqueness & Completeness**: `PASS` (Exactly 1 row per query_id x mode x trial, 0 missing, 0 duplicate)
- **Assertion c) Live Execution Verification (`is_mock == False`)**: `PASS` (0 mock rows)
- **Assertion d) Raw Response Persistence**: `PASS` (87/87 Mode 1, 2, 4 raw files present and non-empty)
- **Assertion e) Provider Verification (`provider != MockProvider`)**: `PASS` (0 MockProvider rows)
- **Assertion f) Groq Request Count**: `PASS` (201 live calls recorded, exceeded 3*T threshold)
- **Assertion g) Ground Truth Integrity**: `PASS` (Every row has expected outcome and optimal cost or explicit non-cost status)

## Archived Artifacts in This Folder
1. `study_run_02.csv`: Full 60-column tabular benchmark output.
2. `results_presentation.csv`: 116-row standardized presentation dataset.
3. `results_side_by_side.csv`: 29-row side-by-side mode comparison matrix.
4. `results_summary.csv`: Aggregated accuracy and latency summary table.
5. `final_query_manifest.json`: Snapshot of the exact 29-query manifest evaluated.
6. `raw/`: All 87 raw LLM responses (prose text & JSON objects).
7. `console_run.log`: Full verbose console execution log.
