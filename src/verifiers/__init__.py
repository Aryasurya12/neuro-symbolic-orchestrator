"""Deterministic Mathematical Verification and Proof Engine for Neurasym.

Provides mathematical feasibility certificates, MIP duality gaps,
and catalog price delta verifications.
"""

from src.verifiers.proof_engine import (
    MathematicalProofEngine,
    compute_optimality_certificate,
    verify_feasibility,
)
from src.verifiers.independent_checker import IndependentChecker

__all__ = [
    "MathematicalProofEngine",
    "verify_feasibility",
    "compute_optimality_certificate",
    "IndependentChecker",
]

