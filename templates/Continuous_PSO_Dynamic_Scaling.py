"""SYM-1: Continuous Particle Swarm Optimization (PSO) Dynamic Scaling Solver."""

from typing import Any, Dict
import numpy as np


def solve_pso_continuous_scaling(
    bandwidth_min_mbps: float = 100.0,
    bandwidth_max_mbps: float = 1000.0,
    target_cpu_pct: float = 70.0,
    budget_max_usd: float = 500.0,
    num_particles: int = 30,
    max_iterations: int = 50,
) -> Dict[str, Any]:
    """Solves continuous dynamic autoscaling and bandwidth sizing using Particle Swarm Optimization (PSO).

    Decision variables:
      1. Bandwidth allocation (Mbps) in [bandwidth_min_mbps, bandwidth_max_mbps]
      2. Replicas count (continuous proxy) in [1.0, 16.0]
    """
    np.random.seed(42)

    # Search bounds: [bandwidth, replicas]
    lb = np.array([bandwidth_min_mbps, 1.0])
    ub = np.array([bandwidth_max_mbps, 16.0])

    # Particles initialization
    positions = np.random.uniform(lb, ub, (num_particles, 2))
    velocities = np.zeros((num_particles, 2))

    # Hyperparameters
    w = 0.729  # inertia weight
    c1 = 1.494  # cognitive parameter
    c2 = 1.494  # social parameter

    # Cost coefficients ($/Mbps-month: ~$0.08, $/Replica-month: ~$45.00)
    cost_per_mbps_month = 0.08
    cost_per_replica_month = 45.0

    def objective_fn(pos: np.ndarray) -> np.ndarray:
        # Bandwidth & Replicas
        bw = pos[:, 0]
        reps = pos[:, 1]

        # Calculate monthly infrastructure cost
        cost = (bw * cost_per_mbps_month) + (reps * cost_per_replica_month)

        # Performance penalty if CPU deviates from target_cpu_pct or budget exceeded
        # Simulated CPU utilization given bandwidth and replicas
        simulated_cpu = np.clip((bw / (reps * 75.0)) * 100.0, 10.0, 99.0)
        cpu_deviation_penalty = np.abs(simulated_cpu - target_cpu_pct) * 2.5
        budget_penalty = np.maximum(0.0, cost - budget_max_usd) * 50.0

        fitness = cost + cpu_deviation_penalty + budget_penalty
        return fitness

    # Fitness evaluation
    fitness = objective_fn(positions)
    pbest_positions = np.copy(positions)
    pbest_fitness = np.copy(fitness)

    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = np.copy(pbest_positions[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_curve = []

    # PSO Optimization Loop
    for _ in range(max_iterations):
        r1 = np.random.rand(num_particles, 2)
        r2 = np.random.rand(num_particles, 2)

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )

        positions = np.clip(positions + velocities, lb, ub)
        fitness = objective_fn(positions)

        # Update personal bests
        improved = fitness < pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = fitness[improved]

        # Update global best
        current_min_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_min_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[current_min_idx]
            gbest_position = np.copy(pbest_positions[current_min_idx])

        convergence_curve.append(float(gbest_fitness))

    optimal_bw = float(round(gbest_position[0], 2))
    optimal_replicas = int(max(1, round(gbest_position[1])))
    monthly_cost = round((optimal_bw * cost_per_mbps_month) + (optimal_replicas * cost_per_replica_month), 2)
    hourly_cost = round(monthly_cost / 730.0, 4)

    return {
        "status": "CONVERGED",
        "solver": "Continuous_Vectorized_PSO",
        "optimal_bandwidth_mbps": optimal_bw,
        "recommended_replicas": optimal_replicas,
        "target_cpu_utilization_pct": target_cpu_pct,
        "estimated_monthly_cost_usd": monthly_cost,
        "estimated_hourly_cost_usd": hourly_cost,
        "budget_max_usd": budget_max_usd,
        "budget_utilized_pct": round((monthly_cost / budget_max_usd) * 100, 2),
        "cost_savings_usd": round(max(0.0, budget_max_usd - monthly_cost), 2),
        "fitness_score": round(float(gbest_fitness), 4),
        "iterations_completed": max_iterations,
        "particles_count": num_particles,
    }
