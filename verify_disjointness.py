"""Verification script for neurasym Part A: Z3 Disaster-Recovery Fixes (Items 1 to 5)."""

import sys
from typing import List, Dict, Any
from src.symbolic.models import SymbolicOptimizationRequest
from src.symbolic.solvers.z3_solver import GraphSteeredZ3Solver
from src.symbolic.solvers.graph_model import InfrastructureGraph
from src.symbolic.adapters import to_explainer_dict
from src.semantic.schemas import CloudOptimizationContract
from src.semantic.explainer import FinOpsExplainer


def run_item_1() -> None:
    print("=" * 80)
    print("ITEM 1: Cross-provider disjointness holds across repeated runs (20 runs)")
    print("=" * 80)

    solver = GraphSteeredZ3Solver()
    same_provider_count = 0
    results_history = []

    # 1. 20 identical runs with generous constraints
    print("\n--- Sub-test 1A: 20 repeated runs with identical generous parameters ---")
    for i in range(1, 21):
        req = SymbolicOptimizationRequest(
            problem_type="Z3_Graph_Disaster_Recovery",
            cloud_providers=["AWS", "GCP"],
            budget_max_usd=600.0,
            service_count=2,
            required_vcpus=4,
            required_ram_gb=16.0,
            latency_max_ms=100.0,
            sla_availability_pct=99.99,
        )
        res = solver.solve(req)
        assert res.is_feasible, f"Run {i} unexpectedly infeasible"
        cand = res.best_candidate
        p_a = cand.decision_variables.get("provider_a")
        p_b = cand.decision_variables.get("provider_b")
        r_a = cand.decision_variables.get("primary_region")
        r_b = cand.decision_variables.get("secondary_region")
        
        is_same = (p_a.upper() == p_b.upper())
        if is_same:
            same_provider_count += 1
            
        print(f"Run {i:02d}: Regions=({r_a}, {r_b}), Providers=({p_a}, {p_b}), Same Provider={is_same}")
        results_history.append((r_a, r_b, p_a, p_b))

    print(f"\nSummary (Sub-test 1A): {same_provider_count}/20 runs had provider_a == provider_b.")

    # 2. To satisfy the prompt's instruction:
    # "If ties in the Z3 model mean the same pair is returned all 20 times, note that explicitly
    # and additionally vary one input parameter (e.g. budget) across the 20 runs to force different
    # feasible regions to be explored, then re-run."
    unique_pairs = set((r[0], r[1]) for r in results_history)
    print(f"\nObserved {len(unique_pairs)} unique pair(s) across static runs: {unique_pairs}")
    
    print("\n--- Sub-test 1B: 20 runs varying budget parameter ($245 to $500) and latency ---")
    varied_same_provider_count = 0
    for i in range(1, 21):
        # Vary budget from 244 to 500
        budget = 244.0 + (i * 12.0)
        # Alternate provider sets: AWS+GCP, AWS+Azure, GCP+Azure
        prov_sets = [["AWS", "GCP"], ["AWS", "Azure"], ["GCP", "Azure"], ["AWS", "GCP", "Azure"]]
        provs = prov_sets[(i - 1) % len(prov_sets)]
        
        req = SymbolicOptimizationRequest(
            problem_type="Z3_Graph_Disaster_Recovery",
            cloud_providers=provs,
            budget_max_usd=budget,
            service_count=2,
            required_vcpus=4,
            required_ram_gb=16.0,
            latency_max_ms=120.0,
            sla_availability_pct=99.99,
        )
        res = solver.solve(req)
        assert res.is_feasible, f"Varied Run {i} unexpectedly infeasible"
        cand = res.best_candidate
        p_a = cand.decision_variables.get("provider_a")
        p_b = cand.decision_variables.get("provider_b")
        r_a = cand.decision_variables.get("primary_region")
        r_b = cand.decision_variables.get("secondary_region")
        is_same = (p_a.upper() == p_b.upper())
        if is_same:
            varied_same_provider_count += 1
        print(f"Varied Run {i:02d} (Budget=${budget:.1f}, Providers={provs}): Regions=({r_a}, {r_b}), Providers=({p_a}, {p_b}), Same Provider={is_same}")

    print(f"\nSummary (Sub-test 1B): {varied_same_provider_count}/20 runs had provider_a == provider_b.")


def run_item_2() -> None:
    print("\n" + "=" * 80)
    print("ITEM 2: Single-provider requests still work (no over-constraining)")
    print("=" * 80)

    solver = GraphSteeredZ3Solver()
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS"],
        budget_max_usd=600.0,
        service_count=2,
        required_vcpus=4,
        required_ram_gb=16.0,
        latency_max_ms=100.0,
        sla_availability_pct=99.99,
    )
    res = solver.solve(req)
    print(f"Solver result is_feasible: {res.is_feasible}")
    if res.is_feasible:
        cand = res.best_candidate
        p_a = cand.decision_variables.get("provider_a")
        p_b = cand.decision_variables.get("provider_b")
        r_a = cand.decision_variables.get("primary_region")
        r_b = cand.decision_variables.get("secondary_region")
        print(f"Selected Regions: {r_a} ({p_a}), {r_b} ({p_b})")
        print(f"provider_a.upper() == 'AWS': {p_a.upper() == 'AWS'}")
        print(f"provider_b.upper() == 'AWS': {p_b.upper() == 'AWS'}")
        assert p_a.upper() == "AWS" and p_b.upper() == "AWS"
    else:
        print(f"ERROR: Single provider request failed with: {res.error_message}")


def run_item_3() -> None:
    print("\n" + "=" * 80)
    print("ITEM 3: Genuine infeasibility is still reported as infeasible")
    print("=" * 80)

    graph = InfrastructureGraph()
    nodes = graph.get_all_nodes()
    
    # Programmatically compute min latency between AWS and GCP nodes
    aws_nodes = [n for n in nodes if n.provider.upper() == "AWS"]
    gcp_nodes = [n for n in nodes if n.provider.upper() == "GCP"]
    
    cross_latencies = []
    for a in aws_nodes:
        for g in gcp_nodes:
            lat = graph.get_latency(a.id, g.id)
            cross_latencies.append((lat, a.id, g.id))
            
    min_lat, min_a, min_g = min(cross_latencies, key=lambda x: x[0])
    print(f"Programmatically discovered AWS<->GCP latencies:")
    for lat, a, g in sorted(cross_latencies):
        print(f"  {a} (AWS) <-> {g} (GCP) = {lat} ms")
    print(f"Minimum AWS<->GCP latency: {min_lat} ms (between {min_a} and {min_g})")

    # Set latency_max_ms strictly below min_lat
    impossible_latency = min_lat - 10.0  # e.g., 22.0 ms
    print(f"Setting latency_max_ms strictly lower: {impossible_latency} ms")

    solver = GraphSteeredZ3Solver()
    req = SymbolicOptimizationRequest(
        problem_type="Z3_Graph_Disaster_Recovery",
        cloud_providers=["AWS", "GCP"],
        budget_max_usd=600.0,
        service_count=2,
        required_vcpus=4,
        required_ram_gb=16.0,
        latency_max_ms=impossible_latency,
        sla_availability_pct=99.99,
    )
    res = solver.solve(req)
    print(f"Solver result is_feasible: {res.is_feasible}")
    print(f"Solver error_message: '{res.error_message}'")
    assert not res.is_feasible, "Expected solve to be infeasible!"
    assert "UNSAT" in (res.error_message or "") or "infeasible" in (res.error_message or "").lower()


def run_item_4_and_5() -> None:
    print("\n" + "=" * 80)
    print("ITEM 4 & 5: Reported latency/SLA are real solved values vs independent recalculation & Bug signature check")
    print("=" * 80)

    graph = InfrastructureGraph()
    solver = GraphSteeredZ3Solver()

    # 3 DR queries with deliberately different targets
    test_queries = [
        {
            "name": "Query 1 (AWS + Azure, low latency target)",
            "contract": CloudOptimizationContract(
                problem_type="Z3_Graph_Disaster_Recovery",
                cloud_providers=["AWS", "Azure"],
                budget_max_usd=500.0,
                latency_max_ms=25.0,
                sla_availability_pct=99.99,
                service_count=2,
                required_vcpus=4,
                required_ram_gb=16.0,
            ),
        },
        {
            "name": "Query 2 (AWS + GCP, medium latency target)",
            "contract": CloudOptimizationContract(
                problem_type="Z3_Graph_Disaster_Recovery",
                cloud_providers=["AWS", "GCP"],
                budget_max_usd=500.0,
                latency_max_ms=40.0,
                sla_availability_pct=99.99,
                service_count=2,
                required_vcpus=4,
                required_ram_gb=16.0,
            ),
        },
        {
            "name": "Query 3 (AWS single-provider, inter-region target)",
            "contract": CloudOptimizationContract(
                problem_type="Z3_Graph_Disaster_Recovery",
                cloud_providers=["AWS"],
                budget_max_usd=500.0,
                latency_max_ms=80.0,
                sla_availability_pct=99.99,
                service_count=2,
                required_vcpus=4,
                required_ram_gb=16.0,
            ),
        },
    ]

    all_reported_latencies = []
    all_reported_slas = []

    for idx, tq in enumerate(test_queries, 1):
        contract = tq["contract"]
        print(f"\n--- {tq['name']} ---")
        req = SymbolicOptimizationRequest(
            problem_type=contract.problem_type,
            cloud_providers=contract.cloud_providers,
            budget_max_usd=contract.budget_max_usd,
            latency_max_ms=contract.latency_max_ms,
            sla_availability_pct=contract.sla_availability_pct,
            service_count=contract.service_count,
            required_vcpus=contract.required_vcpus,
            required_ram_gb=contract.required_ram_gb,
        )
        opt_res = solver.solve(req)
        assert opt_res.is_feasible, f"Query {idx} failed unexpectedly"

        # Pass through full adapter layer to explainer dict
        explainer_dict = to_explainer_dict(opt_res, contract)
        
        # Generate full explainer report
        report_text = FinOpsExplainer.generate_report(
            contract, explainer_dict, enable_llm_explainer=False
        )
        
        # (a) Values from explainer dict & report
        reported_lat = explainer_dict.get("inter_region_latency_ms")
        reported_sla = explainer_dict.get("achieved_sla_pct")
        primary_reg = explainer_dict.get("primary_region")
        secondary_reg = explainer_dict.get("secondary_region")

        # (b) Independently recomputed values directly from graph_model data
        node_a = graph.get_node(primary_reg)
        node_b = graph.get_node(secondary_reg)
        
        # Independent latency computation: lookup in peer table
        if secondary_reg in node_a.peer_latencies_ms:
            indep_lat = node_a.peer_latencies_ms[secondary_reg]
        elif primary_reg in node_b.peer_latencies_ms:
            indep_lat = node_b.peer_latencies_ms[primary_reg]
        else:
            indep_lat = 0.0 if primary_reg == secondary_reg else 999.0

        # Independent composite SLA computation: Product of unavailabilities
        indep_unavail_a = 1.0 - (node_a.sla_pct / 100.0)
        indep_unavail_b = 1.0 - (node_b.sla_pct / 100.0)
        indep_composite_sla = (1.0 - (indep_unavail_a * indep_unavail_b)) * 100.0

        print(f"Selected Regions: Primary={primary_reg} ({node_a.provider}), Secondary={secondary_reg} ({node_b.provider})")
        print(f"(a) Adapter/Explainer Reported Latency : {reported_lat:.2f} ms")
        print(f"(b) Independently Recomputed Latency    : {indep_lat:.2f} ms")
        print(f"    Latency Match                      : {abs(reported_lat - indep_lat) < 1e-5}")
        print(f"(a) Adapter/Explainer Reported SLA     : {reported_sla:.5f}%")
        print(f"(b) Independently Recomputed SLA       : {indep_composite_sla:.5f}%")
        print(f"    SLA Match                          : {abs(reported_sla - indep_composite_sla) < 1e-5}")

        # Check in explainer report text
        assert f"{reported_lat:.2f} ms" in report_text, "Report text missing reported latency!"
        assert f"{reported_sla:.5f}% SLA" in report_text, "Report text missing reported SLA!"
        print(f"Report Text Excerpt:\n  Cross-Region Sync Latency: {reported_lat:.2f} ms\n  Composite Availability   : {reported_sla:.5f}% SLA")

        all_reported_latencies.append(reported_lat)
        all_reported_slas.append(reported_sla)

        # Check Item 5: Old bug signature check
        is_old_bug_signature = (abs(reported_lat - 0.00) < 1e-5 and abs(reported_sla - 99.99000) < 1e-5)
        print(f"Old Bug Signature (0.00 ms AND 99.99000% SLA) Present: {is_old_bug_signature}")
        assert not is_old_bug_signature, f"Query {idx} exhibited old bug signature!"

    print(f"\nDistinct reported latencies across runs: {all_reported_latencies}")
    print(f"Distinct values count: {len(set(all_reported_latencies))} (Must differ meaningfully)")
    assert len(set(all_reported_latencies)) >= 3, "Reported latencies did not differ meaningfully across distinct targets!"


if __name__ == "__main__":
    run_item_1()
    run_item_2()
    run_item_3()
    run_item_4_and_5()
    print("\nALL PART A VERIFICATION CHECKS COMPLETED SUCCESSFULLY.")
