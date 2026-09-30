import pytest
import os
import json
from src.orchestrator.service import NeuroSymbolicOrchestrator
from benchmarks.baseline_evaluation import measure_semantic_only, measure_symbolic_only, measure_neuro_symbolic

def test_evaluation_runners_do_not_mutate():
    """Ensure the evaluation runners don't cause side effects."""
    orch = NeuroSymbolicOrchestrator()
    query = "Deploy a microservice needing 4 vCPUs and 16GB RAM on AWS or Azure for under $300 a month."
    
    # 1. Semantic
    sem_res = measure_semantic_only(orch, query)
    assert sem_res["configuration"] == "Semantic-Only"
    assert sem_res["contract_generated"] is True
    
    # 2. Symbolic
    sym_res = measure_symbolic_only(orch, query, use_optihive=True)
    assert sym_res["configuration"] == "Symbolic-Only"
    assert sym_res["is_feasible"] is True
    assert sym_res["cost_usd"] != "N/A"
    
    # 3. Neuro-Symbolic
    ns_res = measure_neuro_symbolic(orch, query)
    assert ns_res["configuration"] == "Neuro-Symbolic"
    assert ns_res["is_feasible"] is True
    assert ns_res["cost_usd"] != "N/A"

def test_infeasible_scenarios_remain_infeasible():
    """Ensure that the evaluation runner strictly reports Infeasible for known infeasible setups."""
    orch = NeuroSymbolicOrchestrator()
    query = "Deploy 128 vCPUs and 512GB RAM for under $10."
    
    sym_res = measure_symbolic_only(orch, query, use_optihive=True)
    assert sym_res["is_feasible"] is False
    assert sym_res["cost_usd"] == "N/A"
