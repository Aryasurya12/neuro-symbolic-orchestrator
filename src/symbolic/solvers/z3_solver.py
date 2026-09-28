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
    ConstraintStatus,
    FeasibilityStatus,
)
from src.symbolic.interfaces import OptimizationEngine
from .graph_model import InfrastructureGraph, RegionNode
from .graph_steering import GraphSteeringLayer
from .constraints import Z3ConstraintFactory

class GraphSteeredZ3Solver(OptimizationEngine):
    """Z3 SMT Solver directed by graph-based soft preferences."""

    def __init__(
        self,
        timeout_ms: int = 5000,
        steering_mode: str = "learned",
        steering_layer: Optional[GraphSteeringLayer] = None,
        graph: Optional[InfrastructureGraph] = None,
    ):
        self.timeout_ms = timeout_ms
        self.steering_mode = steering_mode
        self.graph = graph or (steering_layer.graph if steering_layer else InfrastructureGraph())
        
        # We handle initialization of the steering layer based on mode or injected instance
        self.steering = steering_layer
        self.last_stats = {}
        if self.steering is None and self.steering_mode != "none":
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

        # Cross-provider disjointness (true multi-cloud DR when 2+ providers requested)
        for c in Z3ConstraintFactory.cross_provider_constraint(selected_vars, self.graph, request.cloud_providers):
            solver.add(c)
            
        # Budget
        solver.add(Z3ConstraintFactory.budget_constraint(selected_vars, self.graph, request.budget_max_usd))
        
        # Latency (only when constraint is active)
        if request.latency_max_ms > 0:
            for c in Z3ConstraintFactory.latency_constraint(selected_vars, self.graph, request.latency_max_ms):
                solver.add(c)
            
        # SLA Availability (only when constraint is active)
        if request.sla_availability_pct > 0:
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
                error_msg, cstatus = self._diagnose_infeasibility(request)
                return self._infeasible_result(error_msg, runtime_ms, cstatus)
                
            return OptimizationResult(
                best_candidate=candidate,
                is_feasible=True,
                status=FeasibilityStatus.FEASIBLE,
                metrics=OptimizationMetrics(runtime_ms=runtime_ms),
                solver_name="GraphSteeredZ3"
            )
        elif result == z3.unsat:
            error_msg, cstatus = self._diagnose_infeasibility(request)
            return self._infeasible_result(error_msg, runtime_ms, cstatus)
        else:
            return self._infeasible_result("UNKNOWN: Solver timed out or failed to resolve.", runtime_ms)

    def _diagnose_infeasibility(self, request: SymbolicOptimizationRequest) -> Tuple[str, ConstraintStatus]:
        """Diagnoses exactly which hard constraint(s) caused UNSAT across candidate graph pairs."""
        from config.settings import settings
        latency_cost_factor = getattr(settings, "LATENCY_COST_PER_MS", 0.25)

        nodes = self.graph.get_all_nodes()
        allowed_upper = {p.upper() for p in request.cloud_providers}
        is_multi_cloud = len(allowed_upper) >= 2

        valid_pairs = []
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i < j:
                    if a.provider.upper() not in allowed_upper or b.provider.upper() not in allowed_upper:
                        continue
                    if is_multi_cloud and a.provider.upper() == b.provider.upper():
                        continue
                    valid_pairs.append((a, b))

        if not valid_pairs:
            details = f"Provider disjointness violated: No valid region pairs for {request.cloud_providers}"
            return f"UNSAT: {details}", ConstraintStatus(
                is_feasible=False,
                budget_ok=True,
                vcpu_ok=True,
                ram_ok=True,
                latency_ok=True,
                sla_ok=True,
                details=details,
            )

        min_cost = float("inf")
        min_lat = float("inf")
        max_sla = 0.0

        for a, b in valid_pairs:
            lat = self.graph.get_latency(a.id, b.id)
            cost = a.base_cost_usd + b.base_cost_usd + (lat * latency_cost_factor)
            unavail_a = 1.0 - (a.sla_pct / 100.0)
            unavail_b = 1.0 - (b.sla_pct / 100.0)
            sla = (1.0 - (unavail_a * unavail_b)) * 100.0

            if cost < min_cost:
                min_cost = cost
            if lat < min_lat:
                min_lat = lat
            if sla > max_sla:
                max_sla = sla

        violations = []
        budget_ok = True
        latency_ok = True
        sla_ok = True

        if min_cost > request.budget_max_usd:
            budget_ok = False
            violations.append(f"Budget cap of ${request.budget_max_usd:.2f} violated (minimum cost is ${min_cost:.2f})")

        if request.latency_max_ms > 0 and min_lat > request.latency_max_ms:
            latency_ok = False
            violations.append(f"Latency threshold of {request.latency_max_ms:.1f}ms violated (minimum latency is {min_lat:.1f}ms)")

        if request.sla_availability_pct > 0 and max_sla < request.sla_availability_pct:
            sla_ok = False
            violations.append(f"SLA target of {request.sla_availability_pct:.3f}% is unreachable (maximum achievable is {max_sla:.5f}%)")

        if not violations:
            violations.append("Simultaneous satisfaction of budget, latency, and SLA constraints is impossible")

        msg = "UNSAT: " + "; ".join(violations)
        cstatus = ConstraintStatus(
            is_feasible=False,
            budget_ok=budget_ok,
            vcpu_ok=True,
            ram_ok=True,
            latency_ok=latency_ok,
            sla_ok=sla_ok,
            details=msg,
        )
        return msg, cstatus

    def _build_cost_expr(self, selected_vars: Dict[str, z3.BoolRef], nodes: list[RegionNode]) -> z3.ArithRef:
        from config.settings import settings
        latency_cost_factor = getattr(settings, "LATENCY_COST_PER_MS", 0.25)
        cost_exprs = []
        for node in nodes:
            cost_exprs.append(z3.If(selected_vars[node.id], node.base_cost_usd, 0.0))
            
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i < j:
                    lat = self.graph.get_latency(a.id, b.id)
                    lat_cost = lat * latency_cost_factor
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
        from config.settings import settings
        latency_cost_factor = getattr(settings, "LATENCY_COST_PER_MS", 0.25)
        selected_nodes = []
        for node_id, var in selected_vars.items():
            if z3.is_true(model[var]):
                selected_nodes.append(self.graph.get_node(node_id))
                
        if len(selected_nodes) != 2:
            return None
            
        reg_a, reg_b = selected_nodes[0], selected_nodes[1]
        
        # Validate Provider (case-insensitive)
        allowed_upper = {p.upper() for p in request.cloud_providers}
        if reg_a.provider.upper() not in allowed_upper or reg_b.provider.upper() not in allowed_upper:
            return None

        # Validate Cross-Provider Disjointness (defense-in-depth)
        if len(allowed_upper) >= 2:
            if reg_a.provider.upper() == reg_b.provider.upper():
                return None
            
        latency = self.graph.get_latency(reg_a.id, reg_b.id)
        
        # Validate Latency (only when constraint is active)
        if request.latency_max_ms > 0 and latency > request.latency_max_ms:
            return None
            
        # Validate SLA (only when constraint is active)
        unavail_a = 1.0 - (reg_a.sla_pct / 100.0)
        unavail_b = 1.0 - (reg_b.sla_pct / 100.0)
        composite_sla = (1.0 - (unavail_a * unavail_b)) * 100.0
        
        if request.sla_availability_pct > 0 and composite_sla < request.sla_availability_pct:
            return None
            
        # Validate Budget
        cost = reg_a.base_cost_usd + reg_b.base_cost_usd + (latency * latency_cost_factor)
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
                "provider_b": reg_b.provider,
                "latency_ms": latency,
                "achieved_sla": composite_sla,
            },
            objective_cost_usd=cost,
            is_feasible=True,
            constraint_status=status
        )

    def _infeasible_result(
        self, error_msg: str, runtime_ms: float, constraint_status: Optional[ConstraintStatus] = None
    ) -> OptimizationResult:
        return OptimizationResult(
            best_candidate=None,
            is_feasible=False,
            status=FeasibilityStatus.INFEASIBLE,
            metrics=OptimizationMetrics(runtime_ms=runtime_ms),
            solver_name="GraphSteeredZ3",
            error_message=error_msg,
        )
