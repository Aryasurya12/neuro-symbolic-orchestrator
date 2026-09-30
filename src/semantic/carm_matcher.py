"""SEM-2: Context-Aware Retrieval Module (CARM) Pattern Matcher."""

from typing import Dict, Set, Tuple


class CARMMatcher:
    """Matches extracted query constraints to pre-compiled symbolic optimization templates

    using Jaccard similarity scoring.
    """

    # Template Constraint Knowledge Base & Archetype Index
    TEMPLATE_INDEX: Dict[str, Dict[str, object]] = {
        "ILP_VM_Allocation": {
            "constraints": {
                "Bounded_Integer_Variables",
                "Budget_Limit_Max",
                "Latency_Bound_Max",
                "Resource_Min_vCPU",
            },
            "template": "ilp_vm_allocation_template.py",
        },
        "PSO_Continuous_Scaling": {
            "constraints": {
                "Continuous_Bandwidth_Range",
                "CPU_Threshold_Max",
                "Cost_Minimization_Objective",
            },
            "template": "pso_continuous_scaling_template.py",
        },
        "Z3_Graph_Disaster_Recovery": {
            "constraints": {
                "Multi_Region_Disjoint",
                "SLA_Availability_Min",
                "Inter_Node_Latency_Max",
            },
            "template": "z3_graph_disaster_recovery_template.py",
        },
    }

    # Backward compatibility alias
    TEMPLATES: Dict[str, Set[str]] = {
        name: data["constraints"]  # type: ignore[misc]
        for name, data in TEMPLATE_INDEX.items()
    }

    @staticmethod
    def compute_jaccard_score(query_set: Set[str], template_set: Set[str]) -> float:
        """Computes Jaccard Similarity Coefficient: |Intersection| / |Union|."""
        if not query_set or not template_set:
            return 0.0

        intersection = query_set.intersection(template_set)
        union = query_set.union(template_set)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def match_template(
        self, extracted_constraints: Set[str]
    ) -> Tuple[str, str, float]:
        """Matches a set of extracted constraints against the template index.

        Returns:
            Tuple of (archetype_name, template_filename, jaccard_score)

        Raises:
            ValueError: If extracted_constraints is empty.
        """
        if not extracted_constraints:
            raise ValueError("Cannot match an empty constraint set.")

        best_archetype = "ILP_VM_Allocation"
        best_score = -1.0
        best_filename = "ilp_vm_allocation_template.py"

        for archetype, data in self.TEMPLATE_INDEX.items():
            template_constraints = data["constraints"]  # type: ignore[assignment]
            score = self.compute_jaccard_score(
                extracted_constraints, template_constraints
            )
            if score > best_score:
                best_score = score
                best_archetype = archetype
                best_filename = str(data["template"])

        return best_archetype, best_filename, round(best_score, 4)
