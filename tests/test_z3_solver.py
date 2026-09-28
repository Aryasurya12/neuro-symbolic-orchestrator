import pytest
from src.symbolic.models import SymbolicOptimizationRequest, FeasibilityStatus
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver

def test_z3_solver_feasible():
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.99
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    assert res.solver_name == "GraphSteeredZ3"
    assert res.is_feasible
    assert res.best_candidate is not None
    assert "primary_region" in res.best_candidate.decision_variables

def test_z3_solver_unsat_budget():
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=10.0, # Impossible budget (regions cost 100+)
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.99
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    assert not res.is_feasible
    assert "UNSAT" in res.error_message

def test_z3_solver_unsat_latency():
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "Azure", "GCP"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=1.0, # Impossible latency
        sla_availability_pct=99.99
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    assert not res.is_feasible
    assert "UNSAT" in res.error_message

def test_hard_vs_soft_separation_security_test():
    """
    CRITICAL MANDATORY TEST.
    Candidate A has an amazing graph score (e.g. us-east-1 and eastus have 12ms latency, penalty 12).
    Candidate B has a worse graph score (e.g. us-east-1 and eu-west-1 have 85ms latency, diverse geo -> penalty 35).
    
    We set latency_max_ms to 10.0ms. 
    Even though (us-east-1, eastus) has a much better score in reality than others, 
    it still violates the HARD constraint of <= 10.0ms.
    Therefore, NO candidate should be feasible. The soft preference MUST NOT override the hard constraint.
    """
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=10.0, # Breaks all edges
        sla_availability_pct=99.0
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    # It must be explicitly rejected
    assert not res.is_feasible


def test_cross_provider_disjointness_multicloud_repetition():
    """Bug #1 Verification: Multi-cloud requests must never return same-provider regions across repeated solves."""
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "GCP"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.99
    )
    solver = GraphSteeredZ3Solver()
    
    # Run 25 repeated solves to verify across any non-deterministic tie breaks
    for _ in range(25):
        res = solver.solve(req)
        assert res.is_feasible, "Expected feasible multi-cloud placement"
        cand = res.best_candidate
        assert cand is not None
        provider_a = cand.decision_variables["provider_a"].upper()
        provider_b = cand.decision_variables["provider_b"].upper()
        
        # Must be different providers
        assert provider_a != provider_b, f"Disjointness violated: got {provider_a} and {provider_b}"
        assert {provider_a, provider_b} == {"AWS", "GCP"}


def test_cross_provider_disjointness_single_provider():
    """Bug #1 Verification: Single-provider requests must succeed and select two regions from that provider."""
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.99
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    assert res.is_feasible
    assert res.best_candidate is not None
    provider_a = res.best_candidate.decision_variables["provider_a"].upper()
    provider_b = res.best_candidate.decision_variables["provider_b"].upper()
    assert provider_a == "AWS"
    assert provider_b == "AWS"


def test_cross_provider_disjointness_infeasible_no_silent_fallback():
    """Bug #1 Verification: If one provider has no feasible region within latency cap, must return Infeasible (no fallback)."""
    # In graph: AWS <-> GCP latencies are 32ms (us-east-1), 42ms (us-west-2), 105ms (eu-west-1).
    # With latency_max_ms=20.0, NO AWS-GCP pair is valid.
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "GCP"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=20.0, # AWS-GCP lowest is 32ms
        sla_availability_pct=99.99
    )
    solver = GraphSteeredZ3Solver()
    res = solver.solve(req)
    
    assert not res.is_feasible
    assert res.status == FeasibilityStatus.INFEASIBLE
    assert res.best_candidate is None or not res.best_candidate.is_feasible


def test_latency_sla_report_data_integrity():
    """Bug #2 Verification: Report metrics must reflect real solver output and match graph model data."""
    from src.symbolic.solvers.graph_model import InfrastructureGraph
    from src.semantic.schemas import CloudOptimizationContract
    from src.symbolic.adapters import to_explainer_dict
    from src.semantic.explainer import FinOpsExplainer
    
    graph = InfrastructureGraph()
    solver = GraphSteeredZ3Solver()
    
    # 3 Distinct Scenarios:
    scenarios = [
        # Scenario 1: AWS + Azure, tight latency (forces us-east-1 <-> eastus, 12ms)
        {
            "contract": CloudOptimizationContract(
                problem_type="Z3_Graph_Disaster_Recovery",
                cloud_providers=["AWS", "Azure"],
                budget_max_usd=1000.0,
                service_count=1,
                required_vcpus=4,
                required_ram_gb=8.0,
                latency_max_ms=25.0,
                sla_availability_pct=99.99
            ),
        },
        # Scenario 2: AWS + GCP (forces us-east-1 <-> us-central1, 32ms)
        {
            "contract": CloudOptimizationContract(
                problem_type="Z3_Graph_Disaster_Recovery",
                cloud_providers=["AWS", "GCP"],
                budget_max_usd=1000.0,
                service_count=1,
                required_vcpus=4,
                required_ram_gb=8.0,
                latency_max_ms=50.0,
                sla_availability_pct=99.99
            ),
        },
        # Scenario 3: AWS only (forces us-east-1 <-> us-west-2, 65ms)
        {
            "contract": CloudOptimizationContract(
                problem_type="Z3_Graph_Disaster_Recovery",
                cloud_providers=["AWS"],
                budget_max_usd=1000.0,
                service_count=1,
                required_vcpus=4,
                required_ram_gb=8.0,
                latency_max_ms=80.0,
                sla_availability_pct=99.99
            ),
        },
    ]
    
    solved_latencies = []
    solved_slas = []
    
    for item in scenarios:
        contract = item["contract"]
        from src.symbolic.adapters import from_contract
        sym_req = from_contract(contract)
        res = solver.solve(sym_req)
        
        assert res.is_feasible
        cand = res.best_candidate
        reg_a = cand.decision_variables["primary_region"]
        reg_b = cand.decision_variables["secondary_region"]
        
        # Ground truth computation from graph model
        expected_latency = graph.get_latency(reg_a, reg_b)
        node_a = graph.get_node(reg_a)
        node_b = graph.get_node(reg_b)
        unavail_a = 1.0 - (node_a.sla_pct / 100.0)
        unavail_b = 1.0 - (node_b.sla_pct / 100.0)
        expected_sla = (1.0 - (unavail_a * unavail_b)) * 100.0
        
        # Verify candidate decision variables
        assert cand.decision_variables["latency_ms"] == pytest.approx(expected_latency, abs=1e-3)
        assert cand.decision_variables["achieved_sla"] == pytest.approx(expected_sla, abs=1e-5)
        
        # Verify Stage 6 report contents
        res_dict = to_explainer_dict(res, contract)
        report = FinOpsExplainer.generate_report(contract, res_dict, enable_llm_explainer=False)
        
        assert f"Cross-Region Sync Latency: {expected_latency:.2f} ms" in report
        assert f"Composite Availability   : {expected_sla:.5f}% SLA" in report
        
        # Regression: 0.00 ms and 99.99000% must never appear together unless actual ground truth
        assert not ("Cross-Region Sync Latency: 0.00 ms" in report and "Composite Availability   : 99.99000% SLA" in report)
        
        solved_latencies.append(expected_latency)
        solved_slas.append(expected_sla)
        
    # Verify the 3 runs produced distinct latencies
    assert len(set(solved_latencies)) == 3, f"Expected 3 distinct latencies, got {solved_latencies}"


def test_z3_solver_explicit_infeasibility_sla_and_budget():
    """Verify Z3 solver explicitly returns FeasibilityStatus.INFEASIBLE and logs violating constraint."""
    from src.symbolic.models import FeasibilityStatus
    solver = GraphSteeredZ3Solver()
    
    # 1. Unreachable SLA
    req_sla = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "Azure"],
        budget_max_usd=1000.0,
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.99999, # 7 nines impossible on 99.95% nodes
    )
    res_sla = solver.solve(req_sla)
    assert not res_sla.is_feasible
    assert res_sla.status == FeasibilityStatus.INFEASIBLE
    assert "SLA target" in res_sla.error_message
    assert res_sla.best_candidate is None

    # 2. Budget violation
    req_budget = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "GCP"],
        budget_max_usd=50.0, # Minimum pair is > $200
        service_count=1,
        required_vcpus=4,
        required_ram_gb=8.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.99,
    )
    res_budget = solver.solve(req_budget)
    assert not res_budget.is_feasible
    assert res_budget.status == FeasibilityStatus.INFEASIBLE
    assert "Budget cap" in res_budget.error_message
    assert res_budget.best_candidate is None


def test_template_disaster_recovery_solver():
    """Verify solve_z3_graph_disaster_recovery template handles disjointness, metrics, and infeasibility."""
    from templates.Graph_SMT_Z3_MultiRegion_Placement import solve_z3_graph_disaster_recovery

    # 1. Multi-provider: AWS + GCP
    res_multi = solve_z3_graph_disaster_recovery(
        target_providers=["AWS", "GCP"],
        budget_max_usd=500.0,
        max_latency_ms=100.0,
        sla_pct=99.99,
    )
    assert res_multi["is_feasible"]
    assert res_multi["status"] == "FEASIBLE"
    p_prov = res_multi["primary_region"].split(":")[0]
    s_prov = res_multi["secondary_region"].split(":")[0]
    assert p_prov != s_prov
    assert {p_prov, s_prov} == {"AWS", "GCP"}
    assert res_multi["inter_region_latency_ms"] > 0
    assert res_multi["achieved_sla_pct"] >= 99.99

    # 2. Single-provider: AWS only
    res_single = solve_z3_graph_disaster_recovery(
        target_providers=["AWS"],
        budget_max_usd=500.0,
        max_latency_ms=100.0,
        sla_pct=99.99,
    )
    assert res_single["is_feasible"]
    assert res_single["status"] == "FEASIBLE"
    p_prov = res_single["primary_region"].split(":")[0]
    s_prov = res_single["secondary_region"].split(":")[0]
    assert p_prov == "AWS"
    assert s_prov == "AWS"
    assert res_single["primary_region"] != res_single["secondary_region"]

    # 3. Explicit Infeasibility: Budget too low
    res_infeasible = solve_z3_graph_disaster_recovery(
        target_providers=["AWS", "GCP"],
        budget_max_usd=50.0,
        max_latency_ms=100.0,
        sla_pct=99.99,
    )
    assert not res_infeasible["is_feasible"]
    assert res_infeasible["status"] == "INFEASIBLE"
    assert "Budget cap" in res_infeasible["error_message"]
    assert not res_infeasible["constraint_status"]["budget_ok"]


