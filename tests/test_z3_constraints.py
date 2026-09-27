import z3
import pytest
from src.symbolic.solvers.graph_model import InfrastructureGraph
from src.symbolic.solvers.constraints import Z3ConstraintFactory

def test_z3_provider_constraint():
    graph = InfrastructureGraph()
    vars = {n.id: z3.Bool(n.id) for n in graph.get_all_nodes()}
    
    # Only AWS allowed
    constraints = Z3ConstraintFactory.provider_constraint(vars, graph, ["AWS"])
    
    solver = z3.Solver()
    solver.add(constraints)
    
    # If we force an Azure region, it should be UNSAT
    solver.push()
    solver.add(vars["eastus"] == True)
    assert solver.check() == z3.unsat
    solver.pop()
    
    # If we force AWS, it should be SAT
    solver.add(vars["us-east-1"] == True)
    assert solver.check() == z3.sat

def test_exactly_two_regions_constraint():
    graph = InfrastructureGraph()
    vars = {n.id: z3.Bool(n.id) for n in graph.get_all_nodes()}
    
    solver = z3.Solver()
    solver.add(Z3ConstraintFactory.exactly_two_regions_constraint(vars))
    
    # Forcing 3 regions -> UNSAT
    solver.push()
    solver.add(vars["us-east-1"] == True)
    solver.add(vars["us-west-2"] == True)
    solver.add(vars["eu-west-1"] == True)
    assert solver.check() == z3.unsat
    solver.pop()
    
    # Forcing 2 regions -> SAT
    solver.add(vars["us-east-1"] == True)
    solver.add(vars["us-west-2"] == True)
    solver.add(vars["eastus"] == False)
    solver.add(vars["eu-west-1"] == False)
    solver.add(vars["us-central1"] == False)
    assert solver.check() == z3.sat
