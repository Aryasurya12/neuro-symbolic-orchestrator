# Neurasym: honest four-mode execution pack

Reviewed repository: https://github.com/Aryasurya12/neuro-symbolic-orchestrator
Commit: `0bfd3161816d2fb14e19a0836fa2d4e29c7a0fce`.

This package adds `benchmarks/audit/` and `audit_materials/` to that checkout. The existing application, solvers, parser, dashboard, legacy benchmark and proof engine are unchanged. Run the NEW harness for evidence; the old dashboard still contains misleading comparison logic. Do not present its labels as measured results.

The attached execution prompt was read. The uploaded ZIP was unavailable. The separately named `neurasym_project_context_for_second_opinion.md` was not supplied; its contents are not assumed.

## What has actually been verified

- Ten independent-checker unit tests pass.
- Six local integration executions, covering VM allocation, disaster recovery and scaling for C/D, are included under `audit_materials/verified_smoke/`.
- Both VM outputs and both DR outputs passed their development checks. Both scaling outputs failed the CPU ceiling: 100 Mbps / (1 replica × 75 Mbps capacity) = 133.33%, above 70%. These are **developer fixtures, not user-authored ground truth, held-out queries, or benchmark accuracy evidence**.
- Resume skips completed rows. Recovery from a durable artifact saved before a JSONL append and a partial final JSONL line was tested without rerunning completed calls.
- A/B transport was tested using mocks: raw response preservation, no SDK retries, structured parsing, malformed JSON and truncation handling. **A/B were not executed against a live provider: no OpenRouter credential is configured here.**
- The proposed 15-query evaluation draft has NOT been executed or labeled. You must write its ground truth before freezing.

## Blocking findings in the existing code

| Location | Actual behavior | Implication |
|---|---|---|
| `benchmarks/run_4way_benchmark.py:compute_calibrated_baseline_metrics` and missing-key branches | Derives fictional latency/cost errors from query length, hashes and budget multipliers | Synthetic numbers cannot be measured baselines |
| `run_mode_1_pure_llm` | Always returns “Unverified / Hallucinated”; adds a risk even when a cost is found | Correct answers cannot earn a neutral success result |
| `run_mode_2_structured_llm` | Checks selected arithmetic, not the complete user requirement; uses `predicted_cost * 1.35` when catalog cost is unavailable | Not an independent or complete checker; unknown cost is fabricated |
| A/B prompts in legacy benchmark | Do not include the catalog used for grading; A uses temperature 0.2 and B uses 0.0 | Catalog knowledge and sampling settings confound the comparison |
| `run_mode_3_pure_symbolic` | Uses SCOPE parsing and templates but labels NLU 0% and math 100%, irrespective of the result | The labels contradict the executed computation |
| `src/optimizers/raw_symbolic_runner.py:run_pure_symbolic_raw` | Intentionally parses text as a numeric vector, then unconditionally raises even if parsing succeeded | The dashboard's raw-text baseline is a forced failure; it does not test solver capability |
| `run_mode_4_neurasym` | Calls `process_query`, stops timer, then parses and optimizes AGAIN for reported metrics | Timing and allocation may come from different stochastic executions |
| Same Mode 4 wrapper | Fixed 100% NLU/feasibility and zero-violation labels | Mode 4 is not allowed an honest grade |
| `app.py` comparison path | 3-second LLM timeouts; fabricated fallback costs; A/B `is_feasible=False`; Mode 4 candidate fills missing achieved values from requested values; fixed/clamped race-bar widths | “source: live” does not establish measured performance; do not use these charts |
| `src/verifiers/proof_engine.py:compute_optimality_certificate` | Treats solver names/statuses and heuristic FEASIBLE as evidence of optimality; can produce zero gap without a bound | Feasibility is not an optimality proof; a solver name or SAT label is insufficient |
| `src/symbolic/optimizers/objective.py` | GA checks budget/CPU/RAM, treats SLA/latency as placeholders; PSO feasibility is budget-only | Important constraints can be silently ignored |
| `src/semantic/schemas.py` / adapters | No explicit scaling bandwidth/CPU target fields, region residency, GPU, exact VM-count semantics | Must not grade only the extracted contract: it may have lost the requirement |
| `templates/ILP_VM_Knapsack_Allocation.py` | Fallback explores bounded one-/two-SKU combinations, despite “Exact Branch and Bound” naming | Not a general optimality certificate |
| `src/orchestrator/service.py` | Normally produces one solver candidate per valid archetype | No live multi-solver arbitration evidence from these branches |

Mode 4's inspected orchestrator and dashboard use `parse_query_to_contract`, a LOCAL rule/heuristic parser. A separate `parse_query_hybrid`/Nemotron fallback exists but is not the path above. Thus local D makes no API call, but the statement “LLM extraction plus deterministic solver” misdescribes this execution. GA and PSO are randomized; seed 42 at object initialization does not make repeated calls on a reused RNG identical. Graph steering may still use learned weights; absence of LLM extraction does not imply the whole project has no learned component.

The repository uses seeded SAMPLE benchmark pricing, a fixed regional graph and a synthetic capacity model. These support real program execution against a controlled catalog, not claims of current cloud pricing, measured production latency or guaranteed real-world uptime. Composite DR availability assumes independent failures.

## Defensible experimental definitions

A = plain-language LLM answer, with catalog and task supplied.
B = same model, catalog, task, temperature and token budget; JSON schema prompt plus Pydantic validation. This is prompt-guided JSON, not provider-enforced constrained decoding.
C = the existing local rule parser plus traditional templates, explicitly labeled **rule-based symbolic baseline**. The harness does not call the forced-failure raw runner.
D = local parser, existing orchestrator optimization and explainer, executed ONCE.

This C is not a bare solver. A bare solver has no natural-language interface: record that limitation as a capability limitation, not a 100%-failure optimization benchmark. A separate human-structured-input solver experiment is appropriate, but it is NOT implemented here and must be reported separately, since it removes interpretation work. C/D also differ in solver implementations/objectives, so this is a comparison of systems, not a clean isolation of the effect of adding a neural parser. A strong extra ablation would hold the solver constant and vary only parsing; do that later rather than mislabel this study.

The harness creates a fresh process per run and sets C's PSO and D's GA/PSO seed to `base_seed + trial_num`. This makes resume independent of execution order and explores algorithmic variation across trials. It is a declared harness policy, not the dashboard's persistent-RNG lifecycle. All modes use cold end-to-end timing, including process/import/initialization costs; D includes explanation. Grading is excluded. Do not compare these times with the old warm solver-only measurements.

## Shared checker and schema

`checker.check(output, truth, catalog)` receives no mode identifier and imports no solver/parser code. It recalculates resources/cost from concrete allocations and checks human constraints, independent of solver totals/status certificates. It supports VM, DR, scaling, and explicit infeasible/clarification/unsupported outcomes. The catalog snapshot is visible equally to A/B; C/D read the same frozen underlying catalog. User truth is never included in LLM messages or supplied to the solver.

- Write expected outcome, constraints and rationale per query. Do not copy Mode 4's extracted contract or output into truth.
- For conflicting/missing/out-of-domain queries, define acceptable behavior before running. A crash is never successful abstention.
- For feasible requests, equivalent valid allocations can pass; an exact allocation match is not required.
- Cost error is absolute claimed-minus-recomputed cost divided by recomputed cost, ×100. Cost accuracy and optimality are different. Optional human `optimal_cost_usd` and `optimality_tolerance_usd` add an optimality check; otherwise no optimality claim is made.
- The default absolute monetary tolerance is $0.02; override explicitly per query if appropriate before freeze. VM unit prices are rounded to cents before multiplication, matching the catalog's monthly method.
- Truth supports ONLY the constraints documented in `checker.FIELDS`. Unknown fields make freeze fail. Extend/test the checker BEFORE freezing if you need another supported constraint. Unrepresentable requests can instead have a human-authored unsupported outcome.

The requested JSONL field structure is retained, with explicit nullable fields:
- `nlu_success=null`: not separately measured. JSON validity does not prove understanding.
- `passed=null`: only while plain prose awaits transcription; never counted as success/failure.
- `constraint_satisfied=null`: not applicable for abstention/execution error.
- `cost_error_pct=null`: undefined or not applicable.
- `latency_ms=null`: an interrupted call has no known completion time.

A forced Boolean for unknown facts would invent evidence. Complete A's transcription before presenting accuracy. Human reviewers transcribe final stated allocations/status/cost WITHOUT correcting the answer. The exported review form withholds truth and grades, though this is not perfect blinding: prose reveals the condition. Use another reviewer if possible. For vague/unparseable prose, record an empty allocation and `self_reported_status='unparseable'`; do not infer an unstated plan or cost. Reviewer decisions and timestamps are appended separately; original records remain unchanged.

Full API response, finish reason, usage, messages, solver result, extracted parameters/contract and explanation are retained in per-run artifacts. A/B inference uses one request per scheduled trial, SDK retries disabled, no automatic model switch or truncation retry. 429s/timeouts/errors are logged as execution failures, with availability reported separately; do not call them hallucinations. Provider-internal routing is outside the SDK retry policy and is not fully controlled.

`manifest.json` freezes query text, human labels, catalog, trial plan, model/settings, code hashes and package versions. `results.jsonl` is flushed and fsynced after every result, with durable per-run artifacts. `events.jsonl` logs starts. A call interrupted without a final artifact is marked unknown/interrupted, not silently retried. Inference already completed remotely during a local crash cannot be guaranteed recoverable. `max-api-calls` is a cumulative harness-call ceiling for this experiment, not an account-wide daily quota meter.

## Proposed count and tonight's schedule

The draft is a 15-query balanced STRESS PANEL: three each of formal English, Hinglish, missing/ambiguous fields, conflicts, and out-of-archetype requests. Supported queries cover all three archetypes. Five query IDs are preselected for one extra trial (one per category): 20 runs per mode, 40 total A/B calls. This deliberately high stress share is not an estimate of real usage frequency; report category-specific results and label the weighting. If representative usage accuracy is required, use a separate independently sampled workload with declared weights.

At your observed 193–287 seconds/call, sequential estimates are:

| Plan | A/B calls | API time alone |
|---|---:|---:|
| 10 queries, one trial each | 20 | 64–96 min |
| 15 queries, one trial each | 30 | 97–144 min |
| 15 queries + five repeated once | 40 | 129–191 min |
| 150 queries, one trial each | 300 | 16–24 hours |

These are planning estimates based on YOUR observations, not benchmark measurements. New catalog-rich prompts can change latency. Reserve time for labeling, transport errors and human transcription.

Recommended relative schedule from when you start:
1. 0–15 min: verify exact checkout, remaining API quota and credentials. Run unit tests. Check one or two separate DEVELOPMENT prompts if changing model/settings; freeze only after calibration. Never tune settings on held-out outcomes.
2. 15–60 min: YOU write/review labels and missing-field policies. Freeze the 15-query set, five repeats, code, catalog, sampling and token policy. If labeling takes longer, cut query count BEFORE freeze, not based on output.
3. 60–75 min: run C/D on the shared set and inspect execution/log integrity. Do not fix a solver failure in-place after freeze. A fix requires a new version and a new declared experiment, retaining old results.
4. 75–266 min: run A/B, trial 1 first, then preselected repeats, sequentially. 40 calls may take 129–191 minutes. With timeouts at 360 seconds, budget can be longer.
5. 266–311 min: transcribe prose, review failures, produce summaries/slides. Allow 30–45 minutes rather than pretending review is free.

Roughly 4–5+ hours from setup start is a realistic complete pilot window, assuming quota remains and no major integration failure. Submission was 6 October at 23:44 IST; illustrative completion would be roughly 04:00–05:00 IST if starting then, not a promise. Actual remaining time/deadline has not been supplied. For under three hours, freeze 10 queries × one trial, omit repeats, and explicitly say consistency was not evaluated. For under two hours, a complete four-mode study is unlikely: prioritize audited C/D plus a small matched A/B pilot and report missing runs. Never substitute estimated A/B outputs.

For C/D, start with the same 15 × 2 maximum planned trials. If annotation time allows, predeclare 30–50 total queries with three seeded trials for a C/D-only exploratory set before any results; do not call the larger C/D denominator a four-way comparison. The smoke's 0.6–1.0 second cold runs suggest execution cost is modest, but measure harder workloads/timeouts before extrapolating. Labeling and review are the bottleneck; 150 low-quality labels are worse than 30 defensible ones.

For tonight, keep the SAME already-working model for the paired pilot. Changing to a non-reasoning model helps build a later larger experiment but changes capability as well as reasoning overhead; it does NOT cleanly isolate that confound. Run it as a separately labeled model condition; keep Nemotron examples separate and preselected, not cherry-picked. Buying credits can raise request capacity but does not guarantee faster responses or provider availability. OpenRouter's official FAQ currently describes 50 free requests/day below its purchase threshold and 1,000/day after at least $10 credits purchased. Its limits documentation exposes the current UTC-day request counter via `/api/v1/key`; verify your account before deciding. UTC midnight corresponds to 05:30 IST, not midnight IST.

Official references checked for planning:
- https://openrouter.ai/docs/faq
- https://openrouter.ai/docs/api_reference/limits

## Run commands (from this repository root)

Python 3.10+ required; validated here on Python 3.12. Use a fresh environment on your machine if desired.

```powershell
python -m pip install -r requirements.txt
python -m unittest benchmarks.audit.test_checker -v
```

Edit `audit_materials/queries_DRAFT.json`: fill every `truth`, write `ground_truth_author`, and change `approved_by_user` to true ONLY after you have reviewed your own labels. Save a separate file, e.g. `audit_materials/queries_APPROVED.json`.

Example VM truth (illustration only, not a label for any draft query):
```json
{
  "expected_status": "feasible",
  "archetype": "vm",
  "constraints": {
    "providers": ["AWS"],
    "min_vcpus": 4,
    "min_ram_gb": 16,
    "budget_max_usd": 300
  },
  "cost_tolerance_usd": 0.02,
  "rationale": "These are the explicit total requirements in this illustrative request."
}
```
For abstention, use e.g. `{"expected_status":"clarification","rationale":"The monthly budget and currency are unspecified; our predeclared policy requires clarification."}`. Other statuses are `infeasible` and `unsupported`. Review whether clarification is actually necessary in that query; do not make every missing field a failure by fiat.

Set your locally verified model ID and API key in your shell/environment. Do not include secrets in artifacts. For PowerShell:
```powershell
$env:OPENROUTER_MODEL = "YOUR_EXACT_WORKING_MODEL_ID"
$env:OPENROUTER_API_KEY = "YOUR_API_KEY"
python -m benchmarks.audit.run freeze --dataset audit_materials/queries_APPROVED.json --out audit_run --model $env:OPENROUTER_MODEL --max-tokens 4096 --timeout 360
python -m benchmarks.audit.run run --out audit_run --configs CD
python -m benchmarks.audit.run run --out audit_run --configs AB --max-api-calls 40
```

Choose the token limit based on development calibration BEFORE freezing; the 4096 default is not a guarantee against reasoning truncation. The harness does not change reasoning settings silently. It does not enforce a native provider JSON grammar for B.

Run the identical command to resume. `--max-runs N` voluntarily pauses after N newly executed jobs. Trial-1 jobs come before repeats. If fewer than 40 account requests remain, use the available count as the ceiling and keep the incomplete rows visible. Raise that cumulative ceiling on a later session only after checking actual quota. A preexisting lock prevents concurrent writers; after a crash, verify the old process is stopped before deleting `audit_run/RUNNING.lock`.

```powershell
python -m benchmarks.audit.run review export --out audit_run --file prose_review.json
# Reviewer fills reviewer + answer for EACH exported record, without seeing truth.
python -m benchmarks.audit.run review import --out audit_run --file prose_review.json
python -m benchmarks.audit.run summary --out audit_run
```

An answer transcription has the identical normalized shape used by B/C/D:
```json
{"allocated_resources":{"vms":[{"provider":"AWS","instance_type":"t3.medium","count":1}]},"claimed_cost_usd":30.37,"self_reported_status":"feasible"}
```

`summary.json` reports the planned common query set, completed/missing/pending cases, primary trial-1 accuracy, category counts, cost errors, cold latency median/range, repeated-output consistency, and individual failure reasons. It withholds primary accuracy if cases are missing or pending. Repetitions are not new independent queries. For a small curated stress panel the Wilson interval is only illustrative binomial uncertainty, not population coverage. For example 13/15 = 86.7%, with an approximate Wilson 95% interval of 62.1–96.3%; do not present this as a precise generalization claim. Analyze paired correctness differences on the SAME queries, not independent-proportion tests; an exact paired test is more appropriate later, with enough discordant cases.

The generic summary does not establish significance, production reliability, optimality, or a guaranteed winner. Included smoke artifacts remain separate from the final evaluation.
