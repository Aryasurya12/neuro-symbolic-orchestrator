"""
Tests for Genuine NVIDIA-based Requirement Interpretation in Mode 4.
Validates:
1. VM, DR, and Scaling contracts reach the correct solver.
2. Solver arguments match validated neural requirements.
3. Mode 4 does NOT invoke local SCOPE requirement extraction or CARM (monkeypatched to raise).
4. Mode 3 does NOT invoke NVIDIA (monkeypatched to raise).
5. Changing a neural requirement changes the corresponding solver argument.
6. Invalid extraction and provider failures prevent solving (honest failure handling).
7. Separate explanation requests (Stage 1 vs Stage 6) with distinct provenance.
8. Controlled comparison integrity between Mode 3 and Mode 4.
9. ZERO live inference calls during test execution.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.semantic.schemas import CloudOptimizationContract, InterpretationOutcome
from src.semantic.nvidia_extractor import NVIDIAExtractor, NVIDIAExtractionResult
from src.proof.stage_trace import run_pipeline_trace, run_mode4_pipeline_trace, run_mode3_pipeline_trace
from src.verifiers.canonical_record import (
    CanonicalExecutionRecord,
    FeasibilityStatus,
    NormalizationStatus,
    ExplanationSource,
)
from src.verifiers.independent_checker import IndependentChecker
from src.semantic.scope_parser import SCOPEParser
from src.semantic.carm_matcher import CARMMatcher


# =============================================================================
# 1. VM, DR, and Scaling Routing Tests
# =============================================================================

def test_mode4_vm_contract_routes_to_ilp():
    """Mode 4 routes a VM contract directly to ILP solver with neural arguments."""
    query = "Deploy 8 vCPUs and 32 GB RAM on AWS under $300 monthly budget."
    rec = run_mode4_pipeline_trace(query, offline=True, silent=True)
    
    assert rec.problem_type == "ILP_VM_Allocation"
    assert "SciPy" in rec.solver_name or "ILP" in rec.solver_name or "HiGHS" in rec.solver_name
    assert "Groq" in rec.execution_path or "NVIDIA" in rec.execution_path or "OpenRouter" in rec.execution_path
    assert rec.requirements.get("required_vcpus") == 8
    assert rec.requirements.get("required_ram_gb") == 32.0
    assert rec.requirements.get("budget_max_usd") == 300.0
    assert rec.requirements.get("cloud_providers") == ["AWS"]
    assert rec.feasibility == FeasibilityStatus.PASS


def test_mode4_dr_contract_routes_to_z3():
    """Mode 4 routes a DR contract directly to Z3 SMT solver."""
    query = "Disaster recovery multi-region replication with 99.99% SLA and latency under 80ms under $1000 budget."
    rec = run_mode4_pipeline_trace(query, offline=True, silent=True)
    
    assert rec.problem_type == "Z3_Graph_Disaster_Recovery"
    assert "Z3" in rec.solver_name or "SMT" in rec.solver_name
    assert "Groq" in rec.execution_path or "NVIDIA" in rec.execution_path or "OpenRouter" in rec.execution_path
    assert rec.requirements.get("latency_max_ms") == 80.0
    assert rec.requirements.get("sla_availability_pct") == 99.99
    assert rec.feasibility == FeasibilityStatus.PASS


def test_mode4_scaling_contract_routes_to_pso():
    """Mode 4 routes a Scaling contract directly to PSO solver."""
    query = "Continuous dynamic scaling with target bandwidth 200 Mbps and 4 replicas under $1200."
    rec = run_mode4_pipeline_trace(query, offline=True, silent=True)
    
    assert rec.problem_type == "PSO_Continuous_Scaling"
    assert "PSO" in rec.solver_name or "Continuous" in rec.solver_name
    assert "Groq" in rec.execution_path or "NVIDIA" in rec.execution_path or "OpenRouter" in rec.execution_path
    assert rec.requirements.get("target_bandwidth_mbps") == 200.0 or rec.requirements.get("target_rps") is not None
    assert rec.feasibility == FeasibilityStatus.PASS


# =============================================================================
# 2. Strict Decoupling Tests (Monkeypatching SCOPE/CARM and NVIDIA)
# =============================================================================

def test_mode4_does_not_invoke_scope_or_carm(monkeypatch):
    """Mode 4 must NOT invoke SCOPEParser or CARMMatcher.
    Making those functions raise RuntimeError verifies strict neural extraction."""
    def forbidden_scope(*args, **kwargs):
        raise RuntimeError("FORBIDDEN: Mode 4 invoked SCOPEParser!")

    def forbidden_carm(*args, **kwargs):
        raise RuntimeError("FORBIDDEN: Mode 4 invoked CARMMatcher!")

    monkeypatch.setattr(SCOPEParser, "parse_query_to_contract", forbidden_scope)
    monkeypatch.setattr(SCOPEParser, "extract_constraints_from_text", forbidden_scope)
    monkeypatch.setattr(CARMMatcher, "compute_jaccard_score", forbidden_carm)

    query = "Deploy 4 vCPUs and 16 GB RAM on GCP under $200."
    # Should execute purely via NVIDIAExtractor without hitting forbidden functions
    rec = run_mode4_pipeline_trace(query, offline=True, silent=True)
    
    assert rec.mode == 4
    assert rec.requirements.get("required_vcpus") == 4
    assert rec.provider in ["Groq", "NVIDIA", "OpenRouter"]
    assert rec.feasibility == FeasibilityStatus.PASS


def test_mode3_does_not_invoke_nvidia(monkeypatch):
    """Mode 3 must NOT invoke NVIDIAExtractor.
    Making NVIDIAExtractor raise RuntimeError verifies pure rule-based Mode 3."""
    def forbidden_nvidia(*args, **kwargs):
        raise RuntimeError("FORBIDDEN: Mode 3 invoked NVIDIAExtractor!")

    monkeypatch.setattr(NVIDIAExtractor, "extract_contract_from_query", forbidden_nvidia)

    query = "Deploy 4 vCPUs and 16 GB RAM on GCP under $200."
    # Should execute purely via SCOPE / CARM without hitting NVIDIAExtractor
    rec = run_mode3_pipeline_trace(query, offline=True, silent=True)
    
    assert rec.mode == 3
    assert rec.requirements.get("required_vcpus") == 4
    assert rec.feasibility == FeasibilityStatus.PASS


# =============================================================================
# 3. Parameter Sensitivity & Contract Propagation Tests
# =============================================================================

def test_changing_neural_requirement_changes_solver_argument():
    """Changing neural requirements propagates directly to solver inputs."""
    query_small = "Deploy 2 vCPUs and 8 GB RAM on AWS under $100."
    rec_small = run_mode4_pipeline_trace(query_small, offline=True, silent=True)
    
    query_large = "Deploy 16 vCPUs and 64 GB RAM on AWS under $1000."
    rec_large = run_mode4_pipeline_trace(query_large, offline=True, silent=True)
    
    assert rec_small.requirements["required_vcpus"] == 2
    assert rec_large.requirements["required_vcpus"] == 16
    assert rec_small.requirements["required_ram_gb"] == 8.0
    assert rec_large.requirements["required_ram_gb"] == 64.0
    assert rec_large.claimed_cost_usd > rec_small.claimed_cost_usd


# =============================================================================
# 4. Honest Failure Handling (Clarification, Unsupported, Conflicting, Error)
# =============================================================================

def test_mode4_needs_clarification_halts_before_solving(monkeypatch):
    """When NVIDIA returns needs_clarification, pipeline halts at Stage 1 without solving."""
    def mock_extract(*args, **kwargs):
        return NVIDIAExtractionResult(
            status="NEEDS_CLARIFICATION",
            outcome="needs_clarification",
            clarification_questions=[
                "What is your target monthly budget ceiling?",
                "Which cloud provider do you prefer?"
            ],
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            elapsed_ms=45.0,
        )

    monkeypatch.setattr(NVIDIAExtractor, "extract_contract_from_query", mock_extract)

    query = "I need some servers somewhere."
    rec = run_mode4_pipeline_trace(query, offline=True, silent=True)
    
    assert rec.failed_stage == 1
    assert rec.normalization_status == NormalizationStatus.CLARIFICATION_REQUIRED
    assert rec.feasibility in [FeasibilityStatus.NOT_EVALUATED, FeasibilityStatus.FAIL]
    assert len(rec.violations) == 2
    assert "What is your target monthly budget ceiling?" in rec.violations


def test_mode4_unsupported_halts_before_solving(monkeypatch):
    """When NVIDIA returns unsupported, pipeline halts at Stage 1 without solving."""
    def mock_extract(*args, **kwargs):
        return NVIDIAExtractionResult(
            status="UNSUPPORTED",
            outcome="unsupported",
            unsupported_reasons=["Quantum annealing hardware scheduling is not supported in Neurasym."],
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            elapsed_ms=50.0,
        )

    monkeypatch.setattr(NVIDIAExtractor, "extract_contract_from_query", mock_extract)

    query = "Deploy a 50-qubit quantum circuit across D-Wave systems."
    rec = run_mode4_pipeline_trace(query, offline=True, silent=True)
    
    assert rec.failed_stage == 1
    assert rec.normalization_status == NormalizationStatus.TASK_INCOMPATIBLE
    assert rec.feasibility in [FeasibilityStatus.NOT_EVALUABLE, FeasibilityStatus.FAIL]
    assert any("Quantum annealing" in v for v in rec.violations)


def test_mode4_conflicting_halts_before_solving(monkeypatch):
    """When NVIDIA returns conflicting_requirements, pipeline halts at Stage 1."""
    def mock_extract(*args, **kwargs):
        return NVIDIAExtractionResult(
            status="CONFLICTING_REQUIREMENTS",
            outcome="conflicting_requirements",
            conflicting_reasons=["Physical latency across transatlantic regions cannot be 0.1ms."],
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            elapsed_ms=48.0,
        )

    monkeypatch.setattr(NVIDIAExtractor, "extract_contract_from_query", mock_extract)

    query = "Deploy DR replication between Tokyo and London with 0.1ms latency."
    rec = run_mode4_pipeline_trace(query, offline=True, silent=True)
    
    assert rec.failed_stage == 1
    assert rec.normalization_status == NormalizationStatus.NORMALIZATION_FAILURE
    assert rec.feasibility == FeasibilityStatus.FAIL
    assert any("transatlantic" in v for v in rec.violations)


def test_mode4_api_failure_halts_without_fallback(monkeypatch):
    """When NVIDIA extraction encounters an error/auth failure, it halts without silent fallback."""
    def mock_extract(*args, **kwargs):
        return NVIDIAExtractionResult(
            status="MISSING_CREDENTIALS",
            outcome="unsupported",
            error_message="HTTP 401: Invalid or missing NVIDIA API key",
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            elapsed_ms=10.0,
        )

    monkeypatch.setattr(NVIDIAExtractor, "extract_contract_from_query", mock_extract)

    query = "Deploy 4 vCPUs and 16 GB RAM."
    rec = run_mode4_pipeline_trace(query, offline=True, silent=True)
    
    assert rec.failed_stage == 1
    assert rec.normalization_status == NormalizationStatus.API_FAILURE
    assert rec.feasibility in [FeasibilityStatus.NOT_EVALUATED, FeasibilityStatus.FAIL]
    assert any("HTTP 401" in v for v in rec.violations)


# =============================================================================
# 5. Separation of Explanation Request Tests
# =============================================================================

def test_mode4_explanation_is_separate_request():
    """Stage 1 requirement extraction and Stage 6 explanation are distinct requests with separate metadata."""
    query = "Deploy 4 vCPUs and 16 GB RAM on AWS under $200."
    rec = run_mode4_pipeline_trace(query, offline=True, silent=True)
    
    # Stage 1 interpretation provenance
    assert rec.provider in ["Groq", "NVIDIA", "OpenRouter"]
    assert rec.model is not None and len(rec.model) > 0
    assert rec.parsing_ms > 0.0
    
    # Stage 6 explanation has distinct status and timing tracking
    assert rec.explanation_status == "SUCCESS"
    assert rec.explanation_source in [ExplanationSource.LOCAL_TEMPLATE, ExplanationSource.LIVE_PROVIDER]
    assert rec.explanation_ms >= 0.0


# =============================================================================
# 6. Controlled Comparison Integrity (Mode 3 vs Mode 4)
# =============================================================================

def test_modes_3_and_4_use_identical_solvers_and_checker():
    """For matching problem contracts, Mode 3 and Mode 4 invoke identical solvers and verifier."""
    query = "Deploy 4 vCPUs and 16 GB RAM on AWS under $500."
    
    rec_m3 = run_pipeline_trace(query, mode=3, offline=True, silent=True)
    rec_m4 = run_pipeline_trace(query, mode=4, offline=True, silent=True)
    
    # Both must use the same underlying solver for VM allocation
    assert rec_m3.solver_name == rec_m4.solver_name
    
    # Both solutions must pass the independent checker
    assert rec_m3.feasibility == FeasibilityStatus.PASS
    assert rec_m4.feasibility == FeasibilityStatus.PASS
    assert rec_m3.cost_delta_usd == 0.0
    assert rec_m4.cost_delta_usd == 0.0
    
    # Provenance correctly distinguishes local rule-based parsing vs Groq neural interpretation
    assert "SCOPE" in rec_m3.execution_path
    assert "Groq" in rec_m4.execution_path or "NVIDIA" in rec_m4.execution_path or "OpenRouter" in rec_m4.execution_path
