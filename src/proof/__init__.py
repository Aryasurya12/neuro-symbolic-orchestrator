"""Proof of capability package for neurasym."""

from src.proof.capability_demo import (
    ProofResult,
    run_mode3_raw_text_proof,
    run_mode4_pipeline_proof,
    generate_comparison_proof,
    run_pso_raw_text_proof,
    run_pso_pipeline_proof,
    generate_pso_comparison_proof,
    run_z3_raw_text_proof,
    run_z3_pipeline_proof,
    generate_z3_comparison_proof,
    generate_solver_proof_by_type,
    generate_comprehensive_capability_proof,
)

__all__ = [
    "ProofResult",
    "run_mode3_raw_text_proof",
    "run_mode4_pipeline_proof",
    "generate_comparison_proof",
    "run_pso_raw_text_proof",
    "run_pso_pipeline_proof",
    "generate_pso_comparison_proof",
    "run_z3_raw_text_proof",
    "run_z3_pipeline_proof",
    "generate_z3_comparison_proof",
    "generate_solver_proof_by_type",
    "generate_comprehensive_capability_proof",
]
