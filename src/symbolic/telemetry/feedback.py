"""SYM-5: Feedback signal computation.

Determines which solver configuration label to assign to a telemetry record
based on OBSERVED execution outcomes. This is the feedback loop that converts
raw telemetry into training labels.

SAFETY NOTE:
    Feasibility is always determined by the symbolic constraint layer.
    This module does NOT change feasibility judgments — it only ranks
    solver configurations by performance on workloads where feasibility
    is already known.

Label priority (documented policy):
    1. Feasible > Infeasible      (correctness first)
    2. Lower runtime              (performance second)
    3. Lower cost                 (economy third)
    4. Deterministic tie-break    (stability)
"""

from typing import List, Optional
from .models import RoutingTelemetry


# Minimum required records before retraining is permitted
MIN_TELEMETRY_SAMPLES: int = 20

# Label definitions matching SYM-4 classifier
LABEL_GA_PSO = "GA_PSO"
LABEL_Z3 = "Z3"
LABEL_GA_PSO_Z3 = "GA_PSO_Z3"

CLASS_LABELS = [LABEL_GA_PSO, LABEL_Z3, LABEL_GA_PSO_Z3]


def derive_label(record: RoutingTelemetry) -> Optional[str]:
    """Derive the best solver configuration label from an observed telemetry record.

    Returns None if the record cannot be used for training (incomplete data,
    no feasibility information, etc.).

    Policy:
        - Records that resulted in a hard_constraint_violation are EXCLUDED
          (the constraint layer determined the result was invalid regardless
          of solver speed).
        - For feasible results: assign label based on which solver set was used.
        - For infeasible results from unconstrained problems: treat as data on
          solver performance in negative cases (useful for model learning).
    """
    # Exclude records with incomplete runtime data
    if record.total_runtime_ms <= 0:
        return None

    solvers = set(record.selected_solvers)

    # Derive label from solver set
    has_ga = "GA" in solvers
    has_pso = "PSO" in solvers
    has_z3 = "Z3" in solvers

    if has_ga and has_pso and has_z3:
        return LABEL_GA_PSO_Z3
    elif has_ga and has_pso and not has_z3:
        return LABEL_GA_PSO
    elif has_z3 and not has_ga and not has_pso:
        return LABEL_Z3
    elif has_z3:
        # Z3 + either GA or PSO → treat as GA_PSO_Z3
        return LABEL_GA_PSO_Z3
    else:
        # Only GA or only PSO — still GA_PSO family
        return LABEL_GA_PSO


def filter_valid_records(records: List[RoutingTelemetry]) -> List[RoutingTelemetry]:
    """Filter telemetry records that are suitable for training.

    Removes:
    - Records with hard_constraint_violation (solver produced invalid output)
    - Records with zero runtime (incomplete/corrupted)
    - Records with no selected solvers
    """
    valid = []
    for r in records:
        if r.hard_constraint_violation:
            continue
        if r.total_runtime_ms <= 0:
            continue
        if not r.selected_solvers:
            continue
        valid.append(r)
    return valid


def build_training_dataset(records: List[RoutingTelemetry]):
    """Convert valid telemetry records into (feature_matrix, labels) for training.

    Feature ordering matches SYM-4 exactly:
        [resource_scale, service_scale, budget_tightness, provider_count,
         has_latency_constraint, has_sla_constraint, constraint_density, is_highly_constrained]

    Returns:
        X (list of lists), y (list of int class indices), labels used

    Returns (None, None, None) if insufficient valid records.
    """
    valid = filter_valid_records(records)
    if len(valid) < MIN_TELEMETRY_SAMPLES:
        return None, None, None

    X = []
    y = []

    for r in valid:
        label_str = derive_label(r)
        if label_str is None:
            continue
        if label_str not in CLASS_LABELS:
            continue

        feature_vec = [
            r.resource_scale,
            float(r.service_scale),
            r.budget_tightness,
            float(r.provider_count),
            1.0 if r.has_latency_constraint else 0.0,
            1.0 if r.has_sla_constraint else 0.0,
            r.constraint_density,
            1.0 if r.is_highly_constrained else 0.0,
        ]
        X.append(feature_vec)
        y.append(CLASS_LABELS.index(label_str))

    if len(X) < MIN_TELEMETRY_SAMPLES:
        return None, None, None

    return X, y, CLASS_LABELS
