"""Verification script for Item 11: Truncation-after-retry labeling."""

from unittest.mock import MagicMock
from benchmarks.run_4way_benchmark import FourWayBenchmarker

def test_forced_truncation():
    print("=" * 80)
    print("ITEM 11: Truncation-after-retry is labeled distinctly, not conflated")
    print("=" * 80)

    benchmarker = FourWayBenchmarker()

    # Mock OpenAI client that returns finish_reason="length" on all calls
    mock_choice = MagicMock()
    mock_choice.finish_reason = "length"
    mock_choice.message.content = "Here's a thinking process: 1. **Analyze"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    benchmarker.client = mock_client

    test_query = "AWS active-passive DR across us-east-1 and us-west-2 with 99.99% SLA and $850 budget cap."

    # Run Mode 1 with forced truncation
    print("\n--- Testing Mode 1 under forced truncation ---")
    res1 = benchmarker.run_mode_1_pure_llm(test_query)
    print(f"Mode 1 Math Feasibility String: '{res1.math_feasibility}'")
    print(f"Mode 1 Constraint Violations  : '{res1.constraint_violations}'")
    print(f"Mode 1 Total Cost Display     : '{res1.total_cost_display}'")
    print(f"Mode 1 Error Detail           : '{res1.details.get('error')}'")

    # Run Mode 2 with forced truncation
    print("\n--- Testing Mode 2 under forced truncation ---")
    res2 = benchmarker.run_mode_2_structured_llm(test_query)
    print(f"Mode 2 Math Feasibility String: '{res2.math_feasibility}'")
    print(f"Mode 2 Constraint Violations  : '{res2.constraint_violations}'")
    print(f"Mode 2 Total Cost Display     : '{res2.total_cost_display}'")
    print(f"Mode 2 Error Detail           : '{res2.details.get('error')}'")

    # Assertions
    expected_status = "Truncated (token budget exceeded)"
    assert res1.math_feasibility == expected_status, f"Mode 1 status was '{res1.math_feasibility}', expected '{expected_status}'"
    assert res2.math_feasibility == expected_status, f"Mode 2 status was '{res2.math_feasibility}', expected '{expected_status}'"
    assert res1.math_feasibility != "Unverified / Hallucinated"
    assert res2.math_feasibility != "Failed (Invalid JSON)"

    print("\n✓ SUCCESS: Both Mode 1 and Mode 2 return distinct status 'Truncated (token budget exceeded)'.")
    print("  Neither is conflated into 'Failed (Invalid JSON)' or 'Unverified / Hallucinated'.")

if __name__ == "__main__":
    test_forced_truncation()
