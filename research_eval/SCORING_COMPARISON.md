# Neurasym Legacy Regrading & Scoring Comparison Report

**Evaluation Harness**: Deterministic Evaluation Protocol v2.1.0
**Total Records Evaluated**: 116 rows across Modes 1–4

## 1. Executive Summary: Mode Pass-Rate Comparison

| Mode | Historical Published Accuracy | Regraded Strict Accuracy | Delta | Human Review Cases |
|---|---|---|---|---|
| **Mode 1 (Raw LLM)** | **3/29 (10.3%)** | **10/29 (34.5%)** | **+7 (+24.1%)** | 8 cases |
| **Mode 2 (Schema LLM)** | **10/29 (34.5%)** | **21/29 (72.4%)** | **+11 (+37.9%)** | 0 cases |
| **Mode 3 (Pure Symbolic)** | **24/29 (82.8%)** | **18/29 (62.1%)** | **-6 (-20.7%)** | 0 cases |
| **Mode 4 (Neuro-Symbolic)** | **25/29 (86.2%)** | **25/29 (86.2%)** | **0 (+0.0%)** | 0 cases |

---

## 2. Detailed Breakdown of Material Scoring Changes

### Mode 1 (Raw LLM)
- **Historical score**: 3 / 29 (10.3%)
- **Regraded score**: 10 / 29 (34.5%)
- **Analysis**: Mode 1 generated valid natural language proofs of infeasibility on queries Q04, Q05, Q06, Q14, Q15, Q20, Q26, Q28 and valid refusals on Q22, Q23, Q24. In the historical pipeline, rigid regexes failed on prose and emitted blank/0 scores. All 8 prose refusal cases have been routed to a blinded human adjudication package (`UNRESOLVED_ADJUDICATION.csv`).

### Mode 2 (Schema LLM)
- **Historical score**: 10 / 29 (34.5%)
- **Regraded score**: 21 / 29 (72.4%)
- **Analysis**: Fixed the critical SCOPE parser fallback default bias. Valid Disaster Recovery schemas (Q11, Q12, Q29) and Scaling schemas (Q17, Q18) are now correctly credited as verified successes. Q16 and Q21 remain legitimate failures because Mode 2 hallucinated plans instead of asking for clarification.

### Mode 3 (Pure Symbolic)
- **Historical score**: 24 / 29 (82.8%)
- **Regraded score**: 18 / 29 (62.1%)
- **Analysis**: Corrected the false-positive scoring inversion on Q19 where Mode 3 stripped the conflicting ceiling, generated an unconstrained plan, and was awarded success by an evaluator fallthrough bug. Furthermore, pure symbolic regex extraction failures on Hindi words (Q03, Q10, Q13) and missing scaling telemetry (Q17, Q18) are strictly marked as failures (0.0) rather than being credited under RIGHT_ANSWER_WRONG_READING.

### Mode 4 (Neuro-Symbolic)
- **Historical score**: 25 / 29 (86.2%)
- **Regraded score**: 25 / 29 (86.2%)
- **Analysis**: Mode 4 correctly detected and refused the Q19 CPU ceiling conflict (+1). However, Q10 is regraded to 0.0 due to extracting 48 GB RAM instead of 32 GB ('arthees GB' in Hinglish misread as 48 GB). Q07 remains a confirmed failure (0.0) due to null budget extraction, and Q17/Q18 remain failures due to unwarranted clarification requests.

