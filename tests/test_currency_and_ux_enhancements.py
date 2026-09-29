import pytest
from src.semantic.scope_parser import SCOPEParser
from src.semantic.explainer import FinOpsExplainer
import app


def test_centralized_offline_currency_converter():
    # 1. Static FX table verification
    assert app.FX_RATES == {
        "USD ($)": 1.0,
        "INR (₹)": 85.0,
        "EUR (€)": 0.92,
        "GBP (£)": 0.78,
        "JPY (¥)": 145.0,
    }

    # 2. format_currency Single Source of Truth
    assert app.format_currency(100.0, "USD ($)") == "$100.00"
    assert app.format_currency(100.0, "INR (₹)") == "₹8,500.00"
    assert app.format_currency(100.0, "EUR (€)") == "€92.00"
    assert app.format_currency(100.0, "GBP (£)") == "£78.00"
    assert app.format_currency(100.0, "JPY (¥)") == "¥14,500"

    # convert_currency float values
    assert app.convert_currency(100.0, "INR (₹)") == 8500.0
    assert app.convert_currency(100.0, "EUR (€)") == 92.0


def test_stage1_provenance_badges():
    parser = SCOPEParser()

    # Explicit query
    c_explicit, _, _ = parser.parse_query_to_contract(
        "Set up 16 vCPUs and 64GB RAM on GCP with a budget cap of $500 monthly"
    )
    prov = c_explicit.metadata["field_provenance"]
    assert prov["required_vcpus"] == "[EXPLICIT]"
    assert prov["required_ram_gb"] == "[EXPLICIT]"
    assert prov["budget_max_usd"] == "[EXPLICIT]"
    assert prov["cloud_providers"] == "[EXPLICIT]"

    # Inferred query
    c_inferred, _, _ = parser.parse_query_to_contract(
        "Run batch processing on GCP requiring high compute scaling under $250 monthly budget"
    )
    prov_inf = c_inferred.metadata["field_provenance"]
    assert "[INFERRED:" in prov_inf["required_vcpus"]
    assert "[INFERRED:" in prov_inf["required_ram_gb"]
    assert prov_inf["budget_max_usd"] == "[EXPLICIT]"

    # Default query
    c_default, _, _ = parser.parse_query_to_contract("optimize our workload")
    prov_def = c_default.metadata["field_provenance"]
    assert prov_def["required_vcpus"] == "[DEFAULT: Baseline Fallback]"
    assert prov_def["required_ram_gb"] == "[DEFAULT: Baseline Fallback]"


def test_percentage_error_and_overflow_benchmark_metrics():
    metrics = app.compute_calibrated_baseline_metrics(
        "Run batch processing on GCP requiring high compute scaling under $250 monthly budget",
        budget_max_usd=250.0,
        optimal_cost_usd=150.0,
    )

    m1 = metrics["mode1"]
    assert "error_usd" in m1
    assert "error_pct" in m1
    assert "overflow_usd" in m1
    assert m1["error_pct"] > 0
    assert m1["overflow_usd"] > 0

    m2 = metrics["mode2"]
    assert "error_usd" in m2
    assert "error_pct" in m2
    assert "overflow_usd" in m2
    assert m2["error_pct"] > 0
    assert m2["overflow_usd"] > 0


def test_itemized_sku_and_dynamic_recommendations():
    res = app.execute_live_pipeline(
        "Set up 16 vCPUs and 64GB RAM on GCP with a budget cap of $500 monthly"
    )

    # Itemized SKU breakdown
    assert len(res["allocations"]) >= 1
    for alloc in res["allocations"]:
        assert "provider" in alloc
        assert "sku" in alloc
        assert "qty" in alloc
        assert "vcpus" in alloc
        assert "ram" in alloc
        assert "monthly_cost" in alloc

    # Dynamic recommendations with multi-currency
    recs_inr = FinOpsExplainer.generate_dynamic_recommendations(
        contract=res["contract"],
        solver_result={
            "total_monthly_cost_usd": res["optimal_cost_usd"],
            "allocated_vms": res["allocations"],
        },
        currency_symbol="₹",
        currency_code="INR",
        currency_rate=85.0,
    )
    assert len(recs_inr) >= 3
    assert any("₹" in r or "INR" in r for r in recs_inr)
