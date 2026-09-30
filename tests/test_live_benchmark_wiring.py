"""Unit tests verifying live 4-way benchmark execution and dynamic verification wiring in app.py."""

import pytest
from app import execute_live_pipeline, run_live_4way_benchmark, create_hallucination_delta_chart, create_smooth_latency_line_chart
from src.schemas.contract import CloudOptimizationContract

def test_execute_live_pipeline_returns_live_4way_bench_data():
    """Verify that execute_live_pipeline computes live metrics for all 4 paradigm modes."""
    query = "Deploy a web tier on AWS with 4 vCPUs, 16GB RAM, budget $300"
    res = execute_live_pipeline(query)

    assert "bench_data" in res
    assert len(res["bench_data"]) == 4
    
    # Mode 1: Pure LLM
    m1 = res["mode1_info"]
    assert "Mode 1" in m1["mode"]
    assert m1["source"] == "live"
    assert "latency_ms" in m1
    assert m1["latency_ms"] > 0
    assert m1["reported_cost_usd"] > 0
    assert m1["actual_cost_usd"] > 0

    # Mode 2: Structured LLM
    m2 = res["mode2_info"]
    assert "Mode 2" in m2["mode"]
    assert m2["source"] == "live"
    assert "latency_ms" in m2
    assert m2["actual_cost_usd"] > 0

    # Mode 3: Pure Symbolic (Crashes on raw text)
    m3 = res["mode3_info"]
    assert "Mode 3" in m3["mode"]
    assert m3["source"] == "live"
    assert m3["is_feasible"] is False
    assert "CRASHED" in m3["math"]
    assert "error_message" in m3
    assert "ValueError" in m3["error_message"]

    # Mode 4: Full Neuro-Symbolic
    m4 = res["mode4_info"]
    assert "Mode 4" in m4["mode"]
    assert m4["source"] == "live"
    assert "feasibility_certificate" in m4
    assert "optimality_certificate" in m4
    assert m4["feasibility_certificate"]["is_feasible"] is True
    assert m4["optimality_certificate"]["mip_gap_pct"] == 0.0

def test_dynamic_chart_generation_from_live_bench_data():
    """Verify that Plotly figures render without error from live benchmark data."""
    query = "High performance compute cluster 8 vCPUs 32GB RAM on GCP under $500"
    res = execute_live_pipeline(query)

    fig_lat = create_smooth_latency_line_chart(res["race_data"])
    assert fig_lat is not None

    fig_hallucination = create_hallucination_delta_chart(res["bench_data"], "USD ($)")
    assert fig_hallucination is not None
    assert len(fig_hallucination.data) == 2  # Reported cost + Catalog cost traces
