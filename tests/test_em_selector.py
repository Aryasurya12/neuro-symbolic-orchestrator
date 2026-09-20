import pytest
import numpy as np
from src.symbolic.models import OptimizationCandidate
from src.symbolic.optihive.ilp_filter import FilteredCandidate
from src.symbolic.optihive.em_selector import EMSelector

def _mock_cand(cost, runtime):
    c = OptimizationCandidate(
        decision_variables={},
        objective_cost_usd=cost,
        is_feasible=True,
        metadata={"runtime_ms": runtime}
    )
    return FilteredCandidate(c, True, [], cost)

def test_em_selector_deterministic_behavior():
    cands = [
        _mock_cand(100.0, 50.0),
        _mock_cand(90.0, 60.0),
        _mock_cand(150.0, 10.0),
        _mock_cand(95.0, 55.0)
    ]
    
    selector_1 = EMSelector(seed=42)
    best_1, resp_1, ll_1 = selector_1.select(cands)
    
    selector_2 = EMSelector(seed=42)
    best_2, resp_2, ll_2 = selector_2.select(cands)
    
    # Assert completely deterministic
    assert best_1.candidate.objective_cost_usd == best_2.candidate.objective_cost_usd
    np.testing.assert_array_almost_equal(resp_1, resp_2)
    assert ll_1[0] == ll_2[0]

def test_em_selector_responsibilities_sum_to_one():
    cands = [
        _mock_cand(100.0, 50.0),
        _mock_cand(90.0, 60.0),
        _mock_cand(150.0, 10.0)
    ]
    
    selector = EMSelector()
    _, resp, _ = selector.select(cands)
    
    sums = np.sum(resp, axis=1)
    np.testing.assert_array_almost_equal(sums, np.ones(3))

def test_em_selector_picks_best():
    # Cand 0 and 1 are in a low-cost cluster. Cand 1 is slightly cheaper.
    # Cand 2 is a high-cost outlier.
    cands = [
        _mock_cand(100.0, 50.0),
        _mock_cand(90.0, 55.0),
        _mock_cand(250.0, 10.0)
    ]
    
    selector = EMSelector()
    best, _, _ = selector.select(cands)
    
    # Should pick 90.0
    assert best.candidate.objective_cost_usd == 90.0
    assert "latent_class" in best.candidate.metadata
    assert "latent_probability" in best.candidate.metadata
