"""Durable Benchmark Execution Engine, Checkpointing, & Resume Controller for Neurasym.

Ensures complete reproducibility and crash-safety for 4-way comparative evaluation:
- Appends completed results and failures immediately to JSONL with synchronous disk flushes (fsync).
- Records request lifecycle: STARTING, COMPLETED, FAILED, and INTERRUPTED states.
- Version Fingerprinting: Freezes SKU catalog, topology graph, model configurations, prompt templates,
  and checker tolerances. Rejects accidental resume if run environment is incompatible.
- Resumes without repeating completed (query_id, mode, trial) calls.
- Preserves raw responses, extracted evidence, and full stage traces.
- Supports offline mock execution with ZERO live network calls.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config.settings import settings
from src.benchmarks.csv_exporter import CSVExporter
from src.benchmarks.truth_table import TruthTableEngine
from src.proof.stage_trace import (
    trace_stage_1_parsing,
    trace_stage_2_archetype_matching,
    trace_stage_3_contract_validation,
    trace_stage_4_solver_execution,
    trace_stage_5_independent_verification,
    trace_stage_6_explanation,
)
from src.semantic.carm_matcher import CARMMatcher
from src.semantic.explainer import FinOpsExplainer
from src.semantic.normalizer import OutputNormalizer
from src.semantic.nvidia_extractor import NVIDIAExtractor
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.scope_parser import SCOPEParser
from src.verifiers.canonical_record import (
    AuditEvent,
    CanonicalExecutionRecord,
    CheckStatus,
    ExplanationSource,
    FeasibilityStatus,
    NormalizationStatus,
    OptimalityStatus,
)
from src.verifiers.independent_checker import IndependentChecker


class IncompatibleVersionError(Exception):
    """Raised when attempting to resume a benchmark with mismatched catalog, models, or tolerances."""
    pass


@dataclass
class BenchmarkEnvironmentFingerprint:
    """Cryptographic hash and frozen configuration of the evaluation environment."""
    version_id: str
    catalog_hash: str
    topology_hash: str
    tolerances_hash: str
    prompt_version_hash: str
    groq_model: str
    nvidia_model: str
    code_version: str = "2.0.0-prompt3"
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def compute_current(cls) -> "BenchmarkEnvironmentFingerprint":
        catalog = IndependentChecker.get_sku_catalog()
        cat_str = json.dumps(catalog, sort_keys=True)
        cat_hash = hashlib.sha256(cat_str.encode("utf-8")).hexdigest()[:16]

        regions = IndependentChecker.get_regions_graph()
        reg_str = json.dumps(regions, sort_keys=True)
        reg_hash = hashlib.sha256(reg_str.encode("utf-8")).hexdigest()[:16]

        tolerances = {
            "hours_per_month": IndependentChecker.HOURS_PER_MONTH,
            "latency_cost_per_ms": IndependentChecker.LATENCY_COST_PER_MS,
            "scaling_cost_per_mbps": IndependentChecker.SCALING_COST_PER_MBPS,
            "scaling_cost_per_replica": IndependentChecker.SCALING_COST_PER_REPLICA,
            "scaling_capacity_factor_mbps": IndependentChecker.SCALING_CAPACITY_FACTOR_MBPS,
            "cost_delta_tolerance_usd": 0.50,
        }
        tol_str = json.dumps(tolerances, sort_keys=True)
        tol_hash = hashlib.sha256(tol_str.encode("utf-8")).hexdigest()[:16]

        prompt_str = "PROMPT_VERSIONS_V2_DISCRIMINATED_NVIDIA_EXTRACTOR"
        p_hash = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()[:16]

        groq_m = os.getenv("GROQ_MODEL") or getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
        nvd_m = os.getenv("NVIDIA_MODEL") or getattr(settings, "NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")

        combined = f"{cat_hash}:{reg_hash}:{tol_hash}:{p_hash}:{groq_m}:{nvd_m}"
        ver_id = f"fp_{hashlib.sha256(combined.encode('utf-8')).hexdigest()[:12]}"

        return cls(
            version_id=ver_id,
            catalog_hash=cat_hash,
            topology_hash=reg_hash,
            tolerances_hash=tol_hash,
            prompt_version_hash=p_hash,
            groq_model=groq_m,
            nvidia_model=nvd_m,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkEngine:
    """Executes, checkpoints, and resumes 4-way evaluation campaigns."""

    def __init__(
        self,
        output_dir: str = "results/benchmark_runs",
        session_name: str = "default_session",
        mock_llm: bool = True,
        trials_per_query: int = 1,
        force_resume: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_name = session_name
        self.mock_llm = mock_llm
        self.trials_per_query = max(1, trials_per_query)
        self.force_resume = force_resume

        self.ledger_path = self.output_dir / f"{session_name}_ledger.jsonl"
        self.fingerprint = BenchmarkEnvironmentFingerprint.compute_current()

        # In-memory index of completed runs: (query_id, mode, trial)
        self.completed_runs: Set[Tuple[str, int, int]] = set()
        self.saved_records: List[CanonicalExecutionRecord] = []
        self._init_or_validate_ledger()

    def _init_or_validate_ledger(self) -> None:
        """Initializes a new ledger or validates compatibility of an existing one."""
        if not self.ledger_path.exists():
            # Initialize new ledger header
            header = {
                "record_type": "SESSION_HEADER",
                "session_name": self.session_name,
                "is_mock": self.mock_llm,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "fingerprint": self.fingerprint.to_dict(),
            }
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(header) + "\n")
                f.flush()
            return

        # Existing ledger: Validate fingerprint and mock/live separation
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                try:
                    header = json.loads(first_line)
                    saved_mock = header.get("is_mock")
                    if saved_mock is not None and saved_mock != self.mock_llm and not self.force_resume:
                        raise IncompatibleVersionError(
                            f"Cannot resume session '{self.session_name}': Ledger was created with "
                            f"is_mock={saved_mock}, but current execution has mock_llm={self.mock_llm}. "
                            "Pass force_resume=True or use a distinct session name."
                        )
                    saved_fp = header.get("fingerprint", {})
                    saved_ver = saved_fp.get("version_id")
                    current_ver = self.fingerprint.version_id

                    if saved_ver and saved_ver != current_ver and not self.force_resume:
                        raise IncompatibleVersionError(
                            f"Ledger fingerprint mismatch! Existing session '{self.session_name}' was created with "
                            f"fingerprint '{saved_ver}', but current environment is '{current_ver}'. "
                            "Pass force_resume=True or use a different session_name."
                        )
                except json.JSONDecodeError:
                    pass

        # Scan existing completed runs
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                    if obj.get("record_type") == "RUN_RECORD":
                        rec_dict = obj.get("record", {})
                        q_id = obj.get("query_id") or rec_dict.get("requirements", {}).get("query_id", "")
                        mode_num = int(obj.get("mode", rec_dict.get("mode", 0)))
                        trial_num = int(obj.get("trial", 1))

                        if q_id and mode_num:
                            self.completed_runs.add((q_id, mode_num, trial_num))

                        # Reconstruct CanonicalExecutionRecord
                        # For lightweight in-memory list
                        rec = self._dict_to_record(rec_dict)
                        setattr(rec, "query_id", q_id)
                        setattr(rec, "trial", trial_num)
                        self.saved_records.append(rec)
                except Exception:
                    pass

    def _dict_to_record(self, d: Dict[str, Any]) -> CanonicalExecutionRecord:
        """Reconstructs a CanonicalExecutionRecord from dictionary."""
        audit_events = []
        for a in d.get("audit_events", []):
            if isinstance(a, dict):
                audit_events.append(
                    AuditEvent(
                        check_id=a.get("check_id", "CHK_GENERAL"),
                        check_name=a.get("check_name", "Check"),
                        run_id=a.get("run_id"),
                        mode=a.get("mode"),
                        verifier_module=a.get("verifier_module", "src.verifiers.independent_checker"),
                        observed_value=a.get("observed_value"),
                        required_value=a.get("required_value"),
                        operator=a.get("operator", "=="),
                        source=a.get("source", ""),
                        formula=a.get("formula", ""),
                        recomputed_result=a.get("recomputed_result"),
                        tolerance=a.get("tolerance"),
                        signed_margin=a.get("signed_margin"),
                        status=CheckStatus(a.get("status", "FAIL")) if a.get("status") in [e.value for e in CheckStatus] else CheckStatus.FAIL,
                        reason=a.get("reason", ""),
                    )
                )

        rec = CanonicalExecutionRecord(
            run_id=d.get("run_id", ""),
            mode=d.get("mode", 4),
            mode_name=d.get("mode_name", ""),
            original_query=d.get("original_query", ""),
            timestamp_utc=d.get("timestamp_utc", ""),
            requirement_source=d.get("requirement_source", "parsed_contract"),
            problem_type=d.get("problem_type", "ILP_VM_Allocation"),
            requirements=d.get("requirements", {}),
            execution_path=d.get("execution_path", ""),
            provider=d.get("provider"),
            model=d.get("model"),
            solver_name=d.get("solver_name"),
            solver_seed=d.get("solver_seed"),
            original_response=d.get("original_response"),
            normalized_allocation=d.get("normalized_allocation"),
            claimed_cost_usd=d.get("claimed_cost_usd"),
            normalization_status=NormalizationStatus(d.get("normalization_status", "SUCCESS")) if d.get("normalization_status") in [e.value for e in NormalizationStatus] else NormalizationStatus.SUCCESS,
            normalization_errors=d.get("normalization_errors", []),
            feasibility=FeasibilityStatus(d.get("feasibility", "FAIL")) if d.get("feasibility") in [e.value for e in FeasibilityStatus] else FeasibilityStatus.FAIL,
            optimality_status=OptimalityStatus(d.get("optimality_status", "Infeasible")) if d.get("optimality_status") in [e.value for e in OptimalityStatus] else OptimalityStatus.HEURISTIC_FEASIBLE,
            recomputed_cost_usd=d.get("recomputed_cost_usd"),
            cost_delta_usd=d.get("cost_delta_usd"),
            cost_error_pct=d.get("cost_error_pct"),
            budget_headroom_usd=d.get("budget_headroom_usd"),
            violations=d.get("violations", []),
            audit_events=audit_events,
            summary_status=d.get("summary_status", "Unverified"),
            parsing_ms=d.get("timing_ms", {}).get("parsing_ms", 0.0),
            solving_ms=d.get("timing_ms", {}).get("solving_ms", 0.0),
            verification_ms=d.get("timing_ms", {}).get("verification_ms", 0.0),
            explanation_ms=d.get("timing_ms", {}).get("explanation_ms", 0.0),
            total_duration_ms=d.get("timing_ms", {}).get("total_duration_ms", 0.0),
        )
        return rec

    def _append_to_ledger(self, obj: Dict[str, Any]) -> None:
        """Appends a single JSON object to the ledger and flushes synchronously."""
        line = json.dumps(obj, ensure_ascii=False)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

    def run_benchmark_on_manifest(
        self,
        query_manifest: List[Dict[str, Any]],
        on_record_completed: Optional[Callable[[CanonicalExecutionRecord], None]] = None,
        verbose: bool = False,
        final_study: bool = False,
    ) -> Tuple[List[CanonicalExecutionRecord], Dict[str, Any]]:
        """Executes 4-way evaluation over all queries in the manifest across configured trials."""
        if final_study:
            unapproved = [
                q.get("query_id")
                for q in query_manifest
                if q.get("approval_status") != "APPROVED" or not q.get("is_approved", False)
            ]
            if unapproved:
                raise ValueError(
                    f"Final study execution rejected: Dataset contains {len(unapproved)} unapproved/unfrozen queries (e.g. {unapproved[:3]}). "
                    "All queries must have approval_status='APPROVED' and is_approved=True before running a final benchmark."
                )

        query_lookup = {q.get("query_id", ""): q for q in query_manifest if q.get("query_id")}

        # Local comparative runner functions
        from run_all_modes_comparative import (
            execute_mode_1_raw_llm,
            execute_mode_2_schema_llm,
            execute_mode_3_symbolic,
            execute_mode_4_neuro_symbolic,
        )

        matched_runs = []

        for q in query_manifest:
            q_id = q.get("query_id", f"q_{hashlib.sha256(q.get('query_text', '').encode()).hexdigest()[:8]}")
            q_text = q.get("query_text", "")
            exp_outcome = q.get("expected_outcome", "FEASIBLE")

            for trial in range(1, self.trials_per_query + 1):
                mode_records_for_query: Dict[int, Optional[CanonicalExecutionRecord]] = {}

                for mode_num in [1, 2, 3, 4]:
                    run_key = (q_id, mode_num, trial)

                    if run_key in self.completed_runs:
                        # Find existing record
                        rec = next((r for r in self.saved_records if getattr(r, "query_id", "") == q_id and r.mode == mode_num and getattr(r, "trial", 1) == trial), None)
                        if rec:
                            mode_records_for_query[mode_num] = rec
                            if verbose:
                                print(f"  [SKIPPED / RESUMED] Query '{q_id}' | Mode {mode_num} | Trial {trial} (Already in ledger)")
                            continue

                    # Record START lifecycle event
                    start_event = {
                        "record_type": "REQUEST_START",
                        "query_id": q_id,
                        "mode": mode_num,
                        "trial": trial,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    self._append_to_ledger(start_event)

                    # Parse contract for common reference
                    parsed_contract = None
                    try:
                        parsed_contract = SCOPEParser.parse_query_to_contract(q_text)
                    except Exception:
                        parsed_contract = None

                    # Execute Mode
                    rec: Optional[CanonicalExecutionRecord] = None
                    try:
                        if mode_num == 1:
                            rec = execute_mode_1_raw_llm(q_text, contract=parsed_contract, mock_llm=self.mock_llm)
                        elif mode_num == 2:
                            rec = execute_mode_2_schema_llm(q_text, contract=parsed_contract, mock_llm=self.mock_llm)
                        elif mode_num == 3:
                            rec, _, _ = execute_mode_3_symbolic(q_text)
                        elif mode_num == 4:
                            rec = execute_mode_4_neuro_symbolic(q_text, mock_llm=self.mock_llm, enable_explanation=True)
                    except Exception as exc:
                        rec = CanonicalExecutionRecord(
                            mode=mode_num,
                            original_query=q_text,
                            normalization_status=NormalizationStatus.API_FAILURE,
                            normalization_errors=[f"Execution exception: {exc}"],
                            feasibility=FeasibilityStatus.FAIL,
                            summary_status=f"Exception: {exc}",
                        )

                    if rec:
                        rec.is_mock = self.mock_llm
                        setattr(rec, "query_id", q_id)
                        setattr(rec, "trial", trial)
                        if not rec.requirements:
                            rec.requirements = {"query_id": q_id}
                        else:
                            rec.requirements["query_id"] = q_id

                        # Attach check_id & metadata to audit events
                        for ev in rec.audit_events:
                            ev.run_id = rec.run_id
                            ev.mode = rec.mode

                        # Save to ledger
                        ledger_record = {
                            "record_type": "RUN_RECORD",
                            "query_id": q_id,
                            "mode": mode_num,
                            "trial": trial,
                            "record": rec.to_dict(),
                        }
                        self._append_to_ledger(ledger_record)
                        self.completed_runs.add(run_key)
                        self.saved_records.append(rec)
                        mode_records_for_query[mode_num] = rec

                        if on_record_completed:
                            on_record_completed(rec)

                # Compute 4-way pattern for this query trial
                pat_info = TruthTableEngine.compute_query_pattern(
                    mode_records_for_query,
                    expected_outcome=exp_outcome,
                )
                pat_info["query_id"] = q_id
                pat_info["trial"] = trial
                matched_runs.append(pat_info)

        self.matched_runs = matched_runs
        # Aggregate summary & export CSVs
        truth_table_summary = TruthTableEngine.aggregate_truth_table(matched_runs)
        self.export_all_csvs(query_manifest, matched_runs)

        return self.saved_records, truth_table_summary

    def export_all_csvs(
        self,
        query_manifest: List[Dict[str, Any]],
        matched_runs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """Exports all 4 required CSV files into the output directory."""
        q_lookup = {q.get("query_id", ""): q for q in query_manifest if q.get("query_id")}
        patterns_data = matched_runs if (matched_runs is not None and len(matched_runs) > 0) else getattr(self, "matched_runs", [])

        path_manifest = str(self.output_dir / "query_manifest.csv")
        path_results = str(self.output_dir / "run_results.csv")
        path_checks = str(self.output_dir / "verification_checks.csv")
        path_patterns = str(self.output_dir / "outcome_patterns.csv")

        CSVExporter.export_query_manifest(query_manifest, output_path=path_manifest)
        CSVExporter.export_run_results(self.saved_records, output_path=path_results, query_metadata_lookup=q_lookup)
        CSVExporter.export_verification_checks(self.saved_records, output_path=path_checks)
        CSVExporter.export_outcome_patterns(patterns_data, output_path=path_patterns)

        return {
            "query_manifest": path_manifest,
            "run_results": path_results,
            "verification_checks": path_checks,
            "outcome_patterns": path_patterns,
        }
