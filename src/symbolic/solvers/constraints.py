"""SYM-2: Reusable Z3 Hard Constraints Factory."""

import z3
from typing import List, Dict
from src.symbolic.models import SymbolicOptimizationRequest
from .graph_model import InfrastructureGraph

class Z3ConstraintFactory:
    """Constructs hard Z3 constraints for the Disaster Recovery placement domain."""

    @staticmethod
    def provider_constraint(
        selected_vars: Dict[str, z3.BoolRef],
        graph: InfrastructureGraph,
        allowed_providers: List[str]
    ) -> List[z3.BoolRef]:
        """Regions belonging to disallowed providers cannot be selected."""
        constraints = []
        for node in graph.get_all_nodes():
            if node.provider not in allowed_providers:
                constraints.append(z3.Not(selected_vars[node.id]))
        return constraints

    @staticmethod
    def exactly_two_regions_constraint(selected_vars: Dict[str, z3.BoolRef]) -> z3.BoolRef:
        """Enforces that exactly two distinct regions must be chosen for DR."""
        # Convert Bool to Int (1 if True, 0 if False)
        # Sum must equal 2
        sum_expr = z3.Sum([z3.If(var, 1, 0) for var in selected_vars.values()])
        return sum_expr == 2

    @staticmethod
    def budget_constraint(
        selected_vars: Dict[str, z3.BoolRef],
        graph: InfrastructureGraph,
        budget_max_usd: float
    ) -> z3.BoolRef:
        """Combined base cost of selected regions + inter-region latency penalty must be <= budget."""
        cost_exprs = []
        nodes = graph.get_all_nodes()
        
        # Base costs
        for node in nodes:
            cost_exprs.append(z3.If(selected_vars[node.id], node.base_cost_usd, 0.0))
            
        # Inter-region latency proxy cost (as modeled in Phase-0 prototype)
        # For simplicity in Z3 without Real variables multiplying booleans exponentially, 
        # we sum the pairwise cost penalties if both regions are selected.
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i < j:
                    lat = graph.get_latency(a.id, b.id)
                    lat_cost = lat * 0.25
                    both_selected = z3.And(selected_vars[a.id], selected_vars[b.id])
                    cost_exprs.append(z3.If(both_selected, lat_cost, 0.0))
                    
        return z3.Sum(cost_exprs) <= budget_max_usd

    @staticmethod
    def latency_constraint(
        selected_vars: Dict[str, z3.BoolRef],
        graph: InfrastructureGraph,
        latency_max_ms: float
    ) -> List[z3.BoolRef]:
        """If two regions are selected, their inter-region latency must be <= max_latency."""
        constraints = []
        nodes = graph.get_all_nodes()
        
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i < j:
                    lat = graph.get_latency(a.id, b.id)
                    both_selected = z3.And(selected_vars[a.id], selected_vars[b.id])
                    # If both selected, latency must be valid
                    constraints.append(z3.Implies(both_selected, lat <= latency_max_ms))
                    
        return constraints

    @staticmethod
    def sla_constraint(
        selected_vars: Dict[str, z3.BoolRef],
        graph: InfrastructureGraph,
        sla_pct: float
    ) -> List[z3.BoolRef]:
        """Composite availability must be >= target."""
        constraints = []
        nodes = graph.get_all_nodes()
        
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i < j:
                    unavail_a = 1.0 - (a.sla_pct / 100.0)
                    unavail_b = 1.0 - (b.sla_pct / 100.0)
                    composite_sla = (1.0 - (unavail_a * unavail_b)) * 100.0
                    
                    both_selected = z3.And(selected_vars[a.id], selected_vars[b.id])
                    constraints.append(z3.Implies(both_selected, composite_sla >= sla_pct))
                    
        return constraints
