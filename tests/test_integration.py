import pytest
from src.orchestrator.service import NeuroSymbolicOrchestrator

@pytest.fixture
def orchestrator():
    return NeuroSymbolicOrchestrator()

def test_integration_ilp_vm_allocation(orchestrator):
    query = "Deploy a web app that requires 8 vCPUs and 16GB RAM for under $300 a month on AWS."
    report = orchestrator.process_query(query)
    
    assert "Feasibility Status   : Feasible" in report
    assert "Problem Type         : ILP_VM_Allocation" in report
    assert "Optimization Engine  : Vectorized_GA" in report or "OptiHive" in report
    assert "Target Services Required : 1 service(s)" in report

def test_integration_pso_scaling(orchestrator):
    query = "Continuous dynamic scaling with target CPU 70% under $1500"
    report = orchestrator.process_query(query)

    assert "Feasibility Status   : Feasible" in report
    assert "Problem Type         : PSO_Continuous_Scaling" in report
    assert "Optimization Engine  : Vectorized_PSO" in report or "OptiHive" in report
    assert "CONTINUOUS AUTOSCALING PARAMETERS" in report

def test_integration_z3_dr(orchestrator):
    query = "We need disaster recovery setup across AWS and GCP with 99.99% SLA and max 50ms latency."
    report = orchestrator.process_query(query)

    assert "Feasibility Status   : Feasible" in report
    assert "Problem Type         : Z3_Graph_Disaster_Recovery" in report
    assert "Optimization Engine  : GraphSteeredZ3" in report or "OptiHive" in report
    assert "TOPOLOGICAL DISASTER RECOVERY PLACEMENT" in report

def test_integration_infeasible(orchestrator):
    query = "Deploy a service needing 64 vCPUs and 256GB RAM on Azure for under $15 a month."
    report = orchestrator.process_query(query)

    assert "Feasibility Status   : Infeasible" in report
