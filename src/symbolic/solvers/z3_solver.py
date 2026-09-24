"""SYM-2: Graph-Steered Z3 SMT Solver Engine.

Combines the deterministic InfrastructureGraph, the soft GraphSteering preferences,
and the Z3 hard constraints to form a formal OptimizationEngine.
"""

import time
import z3
from typing import Dict, Any, Tuple, Optional

from src.symbolic.models import (
    SymbolicOptimizationRequest,
    OptimizationResult,
    OptimizationCandidate,
    OptimizationMetrics,
    ConstraintStatus
)
from src.symbolic.interfaces import OptimizationEngine
from .graph_model import InfrastructureGraph, RegionNode
from .graph_steering import GraphSteeringLayer
from .constraints import Z3ConstraintFactory

class GraphSteeredZ3Solver(OptimizationEngine):
    """Z3 SMT Solver directed by graph-based soft preferences."""

    def __init__(self, timeout_ms: int = 5000, steering_mode: str = "learned"):
        self.timeout_ms = timeout_ms
        self.steering_mode = steering_mode
        self.graph = InfrastructureGraph()
        
        # We handle initialization of the steering layer based on mode
        self.steering = None
        self.last_stats = {}
        if self.steering_mode != "none":
            self.steering = GraphSteeringLayer(self.graph)
            # If deterministic is requested, we actively suppress the learned model
            if self.steering_mode == "deterministic" and self.steering.model:
                self.steering.model = None

    def solve(self, request: SymbolicOptimizationRequest, progress_callback: Optional[callable] = None) -> OptimizationResult:
        start_time = time.perf_counter()

        if progress_callback:
            progress_callback({
                "event": "solver_started",
                "solver": "GraphSteeredZ3"
            })

        solver = z3.Optimize()
        solver.set("timeout", self.timeout_ms)

        nodes = self.graph.get_all_nodes()
        selected_vars: Dict[str, z3.BoolRef] = {
            node.id: z3.Bool(f"selected_{node.id}") for node in nodes
        }

        # 1. APPLY HARD CONSTRAINTS
        # Must select exactly 2 regions
        solver.add(Z3ConstraintFactory.exactly_two_regions_constraint(selected_vars))
        
        # Provider validity
        for c in Z3ConstraintFactory.provider_constraint(selected_vars, self.graph, request.cloud_providers):
            solver.add(c)
            
        # Budget
        solver.add(Z3ConstraintFactory.budget_constraint(selected_vars, self.graph, request.budget_max_usd))
        
        # Latency
        for c in Z3ConstraintFactory.latency_constraint(selected_vars, self.graph, request.latency_max_ms):
            solver.add(c)
            
        # SLA Availability
        for c in Z3ConstraintFactory.sla_constraint(selected_vars, self.graph, request.sla_availability_pct):
            solver.add(c)

        # 2. APPLY OBJECTIVE
        # Base cost objective
        total_cost_expr = self._build_cost_expr(selected_vars, nodes)
        
        # 3. APPLY GRAPH STEERING (Soft Preference)
        # Graph steering is integrated by adding its penalty score heavily discounted 
        # so it only breaks ties or biases choices WITHOUT violating budget limits.
        # It is strictly an objective term, never a hard assertion.
        if self.steering_mode != "none":
            graph_preference_expr = self._build_steering_expr(selected_vars, nodes)
            solver.minimize(total_cost_expr + (0.001 * graph_preference_expr))
        else:
            solver.minimize(total_cost_expr)

        # 4. SOLVE
        result = solver.check()
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Extract statistics for benchmarking
        try:
            stats = solver.statistics()
            self.last_stats = {
                "conflicts": stats.get_key_value("conflicts") if "conflicts" in stats.keys() else 0,
                "decisions": stats.get_key_value("decisions") if "decisions" in stats.keys() else 0,
                "propagations": stats.get_key_value("propagations") if "propagations" in stats.keys() else 0,
            }
        except:
            self.last_stats = {"conflicts": 0, "decisions": 0, "propagations": 0}

        if progress_callback:
            progress_callback({
                "event": "solver_completed",
                "solver": "GraphSteeredZ3"
            })

        if result == z3.sat:
            model = solver.model()
            candidate = self._extract_and_validate_candidate(model, selected_vars, request)
            if candidate is None or not candidate.is_feasible:
                # Post-validation failed
                return self._infeasible_result("Post-solve validation failed.", runtime_ms)
                
            return OptimizationResult(
                best_candidate=candidate,
                is_feasible=True,
                metrics=OptimizationMetrics(runtime_ms=runtime_ms),
                solver_name="GraphSteeredZ3"
            )
        elif result == z3.unsat:
            return self._infeasible_result("UNSAT: No configuration satisfies all hard constraints.", runtime_ms)
        else:
            return self._infeasible_result("UNKNOWN: Solver timed out or failed to resolve.", runtime_ms)

    def _build_cost_expr(self, selected_vars: Dict[str, z3.BoolRef], nodes: list[RegionNode]) -> z3.ArithRef:
        cost_exprs = []
        for node in nodes:
            cost_exprs.append(z3.If(selected_vars[node.id], node.base_cost_usd, 0.0))
            
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i < j:
                    lat = self.graph.get_latency(a.id, b.id)
                    lat_cost = lat * 0.25
                    both_selected = z3.And(selected_vars[a.id], selected_vars[b.id])
                    cost_exprs.append(z3.If(both_selected, lat_cost, 0.0))
        return z3.Sum(cost_exprs)

    def _build_steering_expr(self, selected_vars: Dict[str, z3.BoolRef], nodes: list[RegionNode]) -> z3.ArithRef:
        """Embeds graph steering penalty into Z3."""
        exprs = []
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i < j:
                    score = self.steering.score_pair(a.id, b.id)
                    both_selected = z3.And(selected_vars[a.id], selected_vars[b.id])
                    exprs.append(z3.If(both_selected, float(score), 0.0))
        # If no pairs are selected (should be impossible due to exact-2 constraint), returns 0
        if not exprs:
            return z3.RealVal(0.0)
        return z3.Sum(exprs)

    def _extract_and_validate_candidate(
        self, model: z3.ModelRef, selected_vars: Dict[str, z3.BoolRef], request: SymbolicOptimizationRequest
    ) -> OptimizationCandidate | None:
        """Extracts the solution and performs Python-side post-validation."""
        selected_nodes = []
        for node_id, var in selected_vars.items():
            if z3.is_true(model[var]):
                selected_nodes.append(self.graph.get_node(node_id))
                
        if len(selected_nodes) != 2:
            return None
            
        reg_a, reg_b = selected_nodes[0], selected_nodes[1]
        
        # Validate Provider
        if reg_a.provider not in request.cloud_providers or reg_b.provider not in request.cloud_providers:
            return None
            
        latency = self.graph.get_latency(reg_a.id, reg_b.id)
        
        # Validate Latency
        if latency > request.latency_max_ms:
            return None
            
        # Validate SLA
        unavail_a = 1.0 - (reg_a.sla_pct / 100.0)
        unavail_b = 1.0 - (reg_b.sla_pct / 100.0)
        composite_sla = (1.0 - (unavail_a * unavail_b)) * 100.0
        
        if composite_sla < request.sla_availability_pct:
            return None
            
        # Validate Budget
        cost = reg_a.base_cost_usd + reg_b.base_cost_usd + (latency * 0.25)
        if cost > request.budget_max_usd:
            return None
            
        status = ConstraintStatus(
            is_feasible=True,
            budget_ok=True,
            vcpu_ok=True,
            ram_ok=True,
            latency_ok=True,
            sla_ok=True,
            details=f"Z3 Verified. Latency={latency}ms, SLA={composite_sla:.4f}%"
        )
        
        return OptimizationCandidate(
            decision_variables={
                "primary_region": reg_a.id,
                "secondary_region": reg_b.id,
                "provider_a": reg_a.provider,
                "provider_b": reg_b.provider
            },
            objective_cost_usd=cost,
            is_feasible=True,
            constraint_status=status
        )

    def _infeasible_result(self, error_msg: str, runtime_ms: float) -> OptimizationResult:
        return OptimizationResult(
            best_candidate=None,
            is_feasible=False,
            metrics=OptimizationMetrics(runtime_ms=runtime_ms),
            solver_name="GraphSteeredZ3",
            error_message=error_msg
        )
