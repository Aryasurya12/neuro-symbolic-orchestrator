"""Live capability proof: demonstrates that the pure symbolic solver has no
natural-language processing path, by passing raw query text directly into
its numerically-typed parameters and capturing the genuine resulting
exception -- then running the identical solver after real SCOPE extraction
to show it succeeds. Both results are computed live on every call, never
cached or hardcoded.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.semantic.scope_parser import SCOPEParser
from templates.ILP_VM_Knapsack_Allocation import solve_ilp_vm_knapsack
from templates.Continuous_PSO_Dynamic_Scaling import solve_pso_continuous_scaling
from templates.Graph_SMT_Z3_MultiRegion_Placement import solve_z3_graph_disaster_recovery


@dataclass
class ProofResult:
    """Structured proof result recording live solver execution outcomes."""

    mode_name: str
    succeeded: bool
    raw_input_used: str
    solver_name: str = ""
    output: Optional[Dict[str, Any]] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    traceback_text: Optional[str] = None
    explanation: str = ""


# =============================================================================
# 1. ILP VM Knapsack Solver Capability Proof (Step 1)
# =============================================================================


def run_mode3_raw_text_proof(raw_query: str) -> ProofResult:
    """Passes raw text DIRECTLY into the ILP solver's numeric parameters."""
    try:
        result = solve_ilp_vm_knapsack(
            required_vcpus=raw_query,  # type: ignore[arg-type]
            required_ram_gb=raw_query,  # type: ignore[arg-type]
            budget_max_usd=raw_query,  # type: ignore[arg-type]
        )
        return ProofResult(
            mode_name="Mode 3: Pure Symbolic (ILP Solver)",
            succeeded=True,
            raw_input_used=raw_query,
            solver_name="solve_ilp_vm_knapsack",
            output=result,
            explanation=(
                "Unexpected: the solver accepted raw text without error. "
                "Investigate -- this should not be possible given the "
                "function's numeric type requirements."
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- intentional: capture whatever
        # real exception type actually occurs, do not anticipate one.
        return ProofResult(
            mode_name="Mode 3: Pure Symbolic (ILP Solver)",
            succeeded=False,
            raw_input_used=raw_query,
            solver_name="solve_ilp_vm_knapsack",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text=traceback.format_exc(),
            explanation=(
                "The pure symbolic solver has no natural-language processing "
                "step. Passing raw text directly into its numerically-typed "
                f"parameters causes an immediate, genuine runtime error "
                f"({type(exc).__name__}: {exc}). This is the concrete, reproducible "
                "basis for '0% NLU capability' -- there is no code path here "
                "that could succeed on raw text, by construction."
            ),
        )


def run_mode4_pipeline_proof(raw_query: str) -> ProofResult:
    """Runs the SAME raw query through real SCOPE extraction, then calls
    the IDENTICAL solver function used in Mode 3."""
    parser = SCOPEParser()
    try:
        contract, matched_template, jaccard_score = parser.parse_query_to_contract(raw_query)

        if contract.problem_type != "ILP_VM_Allocation":
            return ProofResult(
                mode_name="Mode 4: Full Neuro-Symbolic (ILP Solver)",
                succeeded=False,
                raw_input_used=raw_query,
                solver_name="solve_ilp_vm_knapsack",
                output={
                    "extracted_contract": contract.model_dump(),
                    "matched_template": matched_template,
                    "jaccard_score": jaccard_score,
                },
                explanation=(
                    f"This query matched '{contract.problem_type}', not "
                    f"'ILP_VM_Allocation' (Jaccard={jaccard_score:.3f}). "
                    "Use an ILP-type query to directly compare against Mode 3."
                ),
            )

        result = solve_ilp_vm_knapsack(
            required_vcpus=contract.required_vcpus,
            required_ram_gb=contract.required_ram_gb,
            budget_max_usd=contract.budget_max_usd,
            target_providers=contract.cloud_providers,
        )

        return ProofResult(
            mode_name="Mode 4: Full Neuro-Symbolic (ILP Solver)",
            succeeded=True,
            raw_input_used=raw_query,
            solver_name="solve_ilp_vm_knapsack",
            output={
                "extracted_contract": contract.model_dump(),
                "matched_template": matched_template,
                "jaccard_score": jaccard_score,
                "solver_result": result,
            },
            explanation=(
                f"SCOPE extracted {contract.required_vcpus} vCPUs, "
                f"{contract.required_ram_gb}GB RAM, ${contract.budget_max_usd:.2f} "
                f"budget from raw text (matched '{matched_template}', "
                f"Jaccard={jaccard_score:.3f}), then passed these as properly "
                "typed arguments into the EXACT SAME solve_ilp_vm_knapsack() "
                "function Mode 3 called directly. The solver is identical in "
                "both modes -- only the presence of a parsing layer differs."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return ProofResult(
            mode_name="Mode 4: Full Neuro-Symbolic (ILP Solver)",
            succeeded=False,
            raw_input_used=raw_query,
            solver_name="solve_ilp_vm_knapsack",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text=traceback.format_exc(),
            explanation="The full pipeline also failed on this query -- report honestly.",
        )


def generate_comparison_proof(raw_query: str) -> dict:
    """Runs both ILP proofs live and returns a structured result for display.
    Nothing here is cached, recorded, or substituted -- every field is
    computed fresh from this exact call."""
    mode3 = run_mode3_raw_text_proof(raw_query)
    mode4 = run_mode4_pipeline_proof(raw_query)
    return {
        "raw_query": raw_query,
        "solver_type": "ILP_VM_Knapsack",
        "solver_function": "solve_ilp_vm_knapsack",
        "mode3": mode3,
        "mode4": mode4,
        "summary": (
            f"Mode 3 {'succeeded' if mode3.succeeded else 'failed with ' + str(mode3.exception_type)}; "
            f"Mode 4 {'succeeded' if mode4.succeeded else 'failed'}. Both ran the "
            "identical solver function against the identical raw query -- the "
            "only variable is the SCOPE parsing layer."
        ),
    }


# =============================================================================
# 2. PSO Continuous Scaling Solver Capability Proof (Step 2)
# =============================================================================


def run_pso_raw_text_proof(raw_query: str) -> ProofResult:
    """Passes raw text DIRECTLY into the Continuous PSO solver's numeric parameters."""
    try:
        result = solve_pso_continuous_scaling(
            bandwidth_min_mbps=raw_query,  # type: ignore[arg-type]
            bandwidth_max_mbps=raw_query,  # type: ignore[arg-type]
            budget_max_usd=raw_query,  # type: ignore[arg-type]
        )
        return ProofResult(
            mode_name="Mode 3: Pure Symbolic (PSO Solver)",
            succeeded=True,
            raw_input_used=raw_query,
            solver_name="solve_pso_continuous_scaling",
            output=result,
            explanation=(
                "Unexpected: the PSO solver accepted raw text without error."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return ProofResult(
            mode_name="Mode 3: Pure Symbolic (PSO Solver)",
            succeeded=False,
            raw_input_used=raw_query,
            solver_name="solve_pso_continuous_scaling",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text=traceback.format_exc(),
            explanation=(
                "The pure continuous PSO solver has no natural-language processing "
                "layer. Passing raw text directly into NumPy boundary matrices "
                f"causes an immediate runtime error ({type(exc).__name__}: {exc}). "
                "This proves that continuous metaheuristics cannot ingest unstructured "
                "conversational text without semantic translation."
            ),
        )


def run_pso_pipeline_proof(raw_query: str) -> ProofResult:
    """Runs the raw query through SCOPE extraction, then calls the IDENTICAL
    solve_pso_continuous_scaling() function used in Mode 3."""
    parser = SCOPEParser()
    try:
        contract, matched_template, jaccard_score = parser.parse_query_to_contract(raw_query)

        if contract.problem_type != "PSO_Continuous_Scaling":
            return ProofResult(
                mode_name="Mode 4: Full Neuro-Symbolic (PSO Solver)",
                succeeded=False,
                raw_input_used=raw_query,
                solver_name="solve_pso_continuous_scaling",
                output={
                    "extracted_contract": contract.model_dump(),
                    "matched_template": matched_template,
                    "jaccard_score": jaccard_score,
                },
                explanation=(
                    f"This query matched '{contract.problem_type}', not "
                    f"'PSO_Continuous_Scaling' (Jaccard={jaccard_score:.3f}). "
                    "Use a continuous autoscale query to directly compare against Mode 3."
                ),
            )

        result = solve_pso_continuous_scaling(
            budget_max_usd=contract.budget_max_usd,
            target_providers=contract.cloud_providers,
            required_vcpus=contract.required_vcpus,
            required_ram_gb=contract.required_ram_gb,
        )

        return ProofResult(
            mode_name="Mode 4: Full Neuro-Symbolic (PSO Solver)",
            succeeded=True,
            raw_input_used=raw_query,
            solver_name="solve_pso_continuous_scaling",
            output={
                "extracted_contract": contract.model_dump(),
                "matched_template": matched_template,
                "jaccard_score": jaccard_score,
                "solver_result": result,
            },
            explanation=(
                f"SCOPE extracted continuous parameters and ${contract.budget_max_usd:.2f} "
                f"budget from raw text (matched '{matched_template}', "
                f"Jaccard={jaccard_score:.3f}), then executed the EXACT SAME "
                "solve_pso_continuous_scaling() function. The metaheuristic algorithm "
                "is identical in both modes."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return ProofResult(
            mode_name="Mode 4: Full Neuro-Symbolic (PSO Solver)",
            succeeded=False,
            raw_input_used=raw_query,
            solver_name="solve_pso_continuous_scaling",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text=traceback.format_exc(),
            explanation="The full pipeline failed on this PSO query -- report honestly.",
        )


def generate_pso_comparison_proof(raw_query: str) -> dict:
    """Runs both PSO proofs live and returns a structured result for display."""
    mode3 = run_pso_raw_text_proof(raw_query)
    mode4 = run_pso_pipeline_proof(raw_query)
    return {
        "raw_query": raw_query,
        "solver_type": "PSO_Continuous_Scaling",
        "solver_function": "solve_pso_continuous_scaling",
        "mode3": mode3,
        "mode4": mode4,
        "summary": (
            f"Mode 3 {'succeeded' if mode3.succeeded else 'failed with ' + str(mode3.exception_type)}; "
            f"Mode 4 {'succeeded' if mode4.succeeded else 'failed'}. Both ran the "
            "identical solve_pso_continuous_scaling() function."
        ),
    }


# =============================================================================
# 3. Z3 SMT Graph Disaster Recovery Solver Capability Proof (Step 2)
# =============================================================================


def run_z3_raw_text_proof(raw_query: str) -> ProofResult:
    """Passes raw text DIRECTLY into the Z3 SMT solver's numeric parameters."""
    try:
        result = solve_z3_graph_disaster_recovery(
            sla_pct=raw_query,  # type: ignore[arg-type]
            max_latency_ms=raw_query,  # type: ignore[arg-type]
            budget_max_usd=raw_query,  # type: ignore[arg-type]
        )
        return ProofResult(
            mode_name="Mode 3: Pure Symbolic (Z3 SMT Solver)",
            succeeded=True,
            raw_input_used=raw_query,
            solver_name="solve_z3_graph_disaster_recovery",
            output=result,
            explanation=(
                "Unexpected: the Z3 solver accepted raw text without error."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return ProofResult(
            mode_name="Mode 3: Pure Symbolic (Z3 SMT Solver)",
            succeeded=False,
            raw_input_used=raw_query,
            solver_name="solve_z3_graph_disaster_recovery",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text=traceback.format_exc(),
            explanation=(
                "The Z3 SMT graph solver enforces formal algebraic and graph constraints. "
                "Passing raw text directly into latency/SLA inequality checks "
                f"causes an immediate runtime error ({type(exc).__name__}: {exc}). "
                "This demonstrates that SMT logic engines require typed numerical ASTs."
            ),
        )


def run_z3_pipeline_proof(raw_query: str) -> ProofResult:
    """Runs the raw query through SCOPE extraction, then calls the IDENTICAL
    solve_z3_graph_disaster_recovery() function used in Mode 3."""
    parser = SCOPEParser()
    try:
        contract, matched_template, jaccard_score = parser.parse_query_to_contract(raw_query)

        if contract.problem_type != "Z3_Graph_Disaster_Recovery":
            return ProofResult(
                mode_name="Mode 4: Full Neuro-Symbolic (Z3 SMT Solver)",
                succeeded=False,
                raw_input_used=raw_query,
                solver_name="solve_z3_graph_disaster_recovery",
                output={
                    "extracted_contract": contract.model_dump(),
                    "matched_template": matched_template,
                    "jaccard_score": jaccard_score,
                },
                explanation=(
                    f"This query matched '{contract.problem_type}', not "
                    f"'Z3_Graph_Disaster_Recovery' (Jaccard={jaccard_score:.3f}). "
                    "Use a multi-region disaster recovery query to directly compare against Mode 3."
                ),
            )

        result = solve_z3_graph_disaster_recovery(
            sla_pct=contract.sla_availability_pct,
            max_latency_ms=contract.latency_max_ms,
            budget_max_usd=contract.budget_max_usd,
            target_providers=contract.cloud_providers,
            required_vcpus=contract.required_vcpus,
            required_ram_gb=contract.required_ram_gb,
        )

        return ProofResult(
            mode_name="Mode 4: Full Neuro-Symbolic (Z3 SMT Solver)",
            succeeded=True,
            raw_input_used=raw_query,
            solver_name="solve_z3_graph_disaster_recovery",
            output={
                "extracted_contract": contract.model_dump(),
                "matched_template": matched_template,
                "jaccard_score": jaccard_score,
                "solver_result": result,
            },
            explanation=(
                f"SCOPE extracted SLA ({contract.sla_availability_pct}%), "
                f"latency bound ({contract.latency_max_ms}ms), and ${contract.budget_max_usd:.2f} "
                f"budget from raw text (matched '{matched_template}', "
                f"Jaccard={jaccard_score:.3f}), then executed the EXACT SAME "
                "solve_z3_graph_disaster_recovery() function."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return ProofResult(
            mode_name="Mode 4: Full Neuro-Symbolic (Z3 SMT Solver)",
            succeeded=False,
            raw_input_used=raw_query,
            solver_name="solve_z3_graph_disaster_recovery",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text=traceback.format_exc(),
            explanation="The full pipeline failed on this Z3 query -- report honestly.",
        )


def generate_z3_comparison_proof(raw_query: str) -> dict:
    """Runs both Z3 proofs live and returns a structured result for display."""
    mode3 = run_z3_raw_text_proof(raw_query)
    mode4 = run_z3_pipeline_proof(raw_query)
    return {
        "raw_query": raw_query,
        "solver_type": "Z3_Graph_Disaster_Recovery",
        "solver_function": "solve_z3_graph_disaster_recovery",
        "mode3": mode3,
        "mode4": mode4,
        "summary": (
            f"Mode 3 {'succeeded' if mode3.succeeded else 'failed with ' + str(mode3.exception_type)}; "
            f"Mode 4 {'succeeded' if mode4.succeeded else 'failed'}. Both ran the "
            "identical solve_z3_graph_disaster_recovery() function."
        ),
    }


# =============================================================================
# 4. Multi-Solver Dispatcher & Comprehensive Proof Generator
# =============================================================================


def generate_solver_proof_by_type(raw_query: str, solver_type: str) -> dict:
    """Generates comparison proof for a specific solver archetype."""
    normalized = solver_type.strip().lower()
    if "pso" in normalized or "continuous" in normalized or "scaling" in normalized:
        return generate_pso_comparison_proof(raw_query)
    elif "z3" in normalized or "graph" in normalized or "disaster" in normalized or "smt" in normalized:
        return generate_z3_comparison_proof(raw_query)
    else:
        return generate_comparison_proof(raw_query)


def generate_comprehensive_capability_proof(raw_query: str) -> dict:
    """Analyzes the raw query with SCOPE to identify the matched archetype,
    and returns comparison proofs across all three solvers (ILP, PSO, Z3)."""
    parser = SCOPEParser()
    contract, matched_template, jaccard_score = parser.parse_query_to_contract(raw_query)

    ilp_proof = generate_comparison_proof(raw_query)
    pso_proof = generate_pso_comparison_proof(raw_query)
    z3_proof = generate_z3_comparison_proof(raw_query)

    matched_proof = ilp_proof
    if contract.problem_type == "PSO_Continuous_Scaling":
        matched_proof = pso_proof
    elif contract.problem_type == "Z3_Graph_Disaster_Recovery":
        matched_proof = z3_proof

    return {
        "raw_query": raw_query,
        "extracted_contract": contract.model_dump(),
        "matched_problem_type": contract.problem_type,
        "matched_template": matched_template,
        "jaccard_score": jaccard_score,
        "active_proof": matched_proof,
        "all_proofs": {
            "ILP_VM_Knapsack": ilp_proof,
            "PSO_Continuous_Scaling": pso_proof,
            "Z3_Graph_Disaster_Recovery": z3_proof,
        },
    }
