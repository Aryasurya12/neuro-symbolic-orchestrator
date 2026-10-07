"""tests/test_dashboard_llm.py
Offline Mock Unit Test Suite for Neurasym Dashboard LLM Request Handling.
Verifies all 10 required offline test scenarios without making any live network calls.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict

from config.settings import settings
from src.semantic.schemas import CloudOptimizationContract
from app import (
    execute_dashboard_llm_request,
    build_4way_comparison_data,
    execute_live_pipeline,
    format_currency,
    convert_currency,
    create_smooth_cost_trend_chart,
    create_hallucination_delta_chart,
)


class MockChoice:
    def __init__(self, content: str = "", finish_reason: str = "stop"):
        self.message = MagicMock()
        self.message.content = content
        self.finish_reason = finish_reason


class MockUsage:
    def __init__(self, prompt_tokens: int = 150, completion_tokens: int = 80):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockResponse:
    def __init__(self, content: str = "", finish_reason: str = "stop", resp_id: str = "gen-test-123"):
        self.id = resp_id
        self.choices = [MockChoice(content=content, finish_reason=finish_reason)]
        self.usage = MockUsage()


class TestDashboardLLMHandling(unittest.TestCase):
    def setUp(self):
        self.contract = CloudOptimizationContract(
            problem_type="ILP_VM_Allocation",
            cloud_providers=["AWS"],
            required_vcpus=4,
            required_ram_gb=16.0,
            budget_max_usd=500.0,
            service_count=2,
            sla_availability_pct=99.9,
            latency_max_ms=100.0,
        )
        self.query = "Deploy 2 web microservices on AWS requiring 4 vCPUs and 16GB RAM with $500 monthly budget."

    # -------------------------------------------------------------------------
    # 1. Successful plain-text response (Mode 1)
    # -------------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_mode1_successful_plaintext(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MockResponse(
            content="I recommend 2x AWS t3.xlarge instances. The total monthly cost is $245.50/month."
        )

        result = execute_dashboard_llm_request(
            mode_num=1,
            query=self.query,
            contract=self.contract,
            timeout_seconds=360.0,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["reported_cost_usd"], 245.50)
        self.assertIsNotNone(result["elapsed_ms"])
        self.assertIn("245.50", result["content"])
        self.assertIsInstance(result["error_pct"], (float, int))

        # Verify client configuration: timeout=360s, max_retries=0
        mock_openai_cls.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
            timeout=360.0,
            max_retries=0,
        )

    # -------------------------------------------------------------------------
    # 2. Successful structured response (Mode 2)
    # -------------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_mode2_successful_structured(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        json_payload = {
            "cloud_provider": "AWS",
            "instances": [{"sku": "t3.xlarge", "quantity": 2, "monthly_cost": 210.0}],
            "total_monthly_cost": 210.0,
            "total_vcpus": 4,
            "total_ram_gb": 16.0,
        }
        mock_client.chat.completions.create.return_value = MockResponse(
            content=json.dumps(json_payload)
        )

        result = execute_dashboard_llm_request(
            mode_num=2,
            query=self.query,
            contract=self.contract,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["reported_cost_usd"], 210.0)
        self.assertEqual(result["parsed_json"]["cloud_provider"], "AWS")
        self.assertIsNotNone(result["actual_cost_usd"])

    # -------------------------------------------------------------------------
    # 3. Invalid JSON (Mode 2)
    # -------------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_mode2_invalid_json(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MockResponse(
            content="Here is your plan: { cloud_provider: 'AWS', cost: broken_json_no_quotes "
        )

        result = execute_dashboard_llm_request(
            mode_num=2,
            query=self.query,
            contract=self.contract,
        )

        self.assertEqual(result["status"], "invalid_schema")
        self.assertIsNone(result["reported_cost_usd"])
        self.assertIn("error_message", result)
        self.assertNotIn("budget * 0.9", str(result))

    # -------------------------------------------------------------------------
    # 4. Empty final-answer content
    # -------------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_empty_final_answer_content(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MockResponse(content="")

        result = execute_dashboard_llm_request(
            mode_num=1,
            query=self.query,
            contract=self.contract,
        )

        self.assertEqual(result["status"], "empty_response")
        self.assertIsNone(result["reported_cost_usd"])
        self.assertIsNone(result["error_pct"])
        self.assertIn("empty content", result["error_message"].lower())

    # -------------------------------------------------------------------------
    # 5. Token-limit truncation
    # -------------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_token_limit_truncation(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MockResponse(
            content="Plan starts with 4x c5.xlarge instances at $180...",
            finish_reason="length",
        )

        result = execute_dashboard_llm_request(
            mode_num=1,
            query=self.query,
            contract=self.contract,
        )

        self.assertEqual(result["status"], "truncated")
        self.assertEqual(result["finish_reason"], "length")
        self.assertIn("truncated", result["error_message"].lower())
        self.assertIn("$180", result["content"])

    # -------------------------------------------------------------------------
    # 6. Daily-quota 429 (OpenRouter Free Tier Daily Limit)
    # -------------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_daily_quota_429(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        from openai import RateLimitError
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="429 Resource has been exhausted: free-models-per-day limit reached (limit_source=openrouter_free_tier_daily)",
            response=mock_resp,
            body={"error": {"message": "free-models-per-day limit reached"}},
        )

        result = execute_dashboard_llm_request(
            mode_num=1,
            query=self.query,
            contract=self.contract,
        )

        self.assertEqual(result["status"], "daily_quota_exhausted")
        self.assertIn("07 Oct 2026 at 05:30 IST", result["reset_str"])
        self.assertIsNone(result["reported_cost_usd"])
        self.assertIsNone(result["actual_cost_usd"])
        self.assertIn("daily free request limit", result["error_message"].lower())

    # -------------------------------------------------------------------------
    # 7. Temporary / provider-side 429
    # -------------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_temporary_provider_429(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        from openai import RateLimitError
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="429 Too Many Requests: Upstream server busy. Please slow down your requests.",
            response=mock_resp,
            body={"error": {"message": "Upstream server busy"}},
        )

        result = execute_dashboard_llm_request(
            mode_num=1,
            query=self.query,
            contract=self.contract,
        )

        self.assertEqual(result["status"], "rate_limited")
        self.assertNotIn("daily_quota_exhausted", result["status"])
        self.assertIsNone(result["reported_cost_usd"])

    # -------------------------------------------------------------------------
    # 8. Timeout handling
    # -------------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_timeout_handling(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        from openai import APITimeoutError
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )

        result = execute_dashboard_llm_request(
            mode_num=1,
            query=self.query,
            contract=self.contract,
            timeout_seconds=360.0,
        )

        self.assertEqual(result["status"], "timeout")
        self.assertIn("360s", result["error_message"])
        self.assertIsNone(result["reported_cost_usd"])
        self.assertIsNotNone(result["elapsed_ms"])

    # -------------------------------------------------------------------------
    # 9. A Streamlit rerun does not create a duplicate request
    # -------------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_no_duplicate_on_rerun(self, mock_openai_cls):
        # 1. First explicit run completes
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MockResponse(
            content="AWS t3.xlarge at $240.00/mo."
        )

        first_run = execute_dashboard_llm_request(
            mode_num=1,
            query=self.query,
            contract=self.contract,
        )
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

        # 2. Simulate subsequent Streamlit page reruns using build_4way_comparison_data
        # build_4way_comparison_data takes the cached run and constructs table without network calls
        for _ in range(5):
            res_table = build_4way_comparison_data(
                user_query=self.query,
                contract=self.contract,
                solver_res={"total_monthly_cost_usd": 180.0, "status": "feasible"},
                solve_latency_ms=8.5,
                total_latency_ms=12.0,
                m1_run=first_run,
                m2_run=None,
            )
            # Verify Mode 1 info reflects cached run
            self.assertEqual(res_table["mode1_info"]["source"], "live")
            self.assertEqual(res_table["mode1_info"]["reported_cost_usd"], 240.0)
            # Verify Mode 2 remains unexecuted
            self.assertEqual(res_table["mode2_info"]["source"], "not_run")

        # Confirm OpenAI create was NEVER called again during the 5 reruns
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    # -------------------------------------------------------------------------
    # 10. Local solver execution does not trigger optional LLM recommendations
    # -------------------------------------------------------------------------
    @patch("src.semantic.explainer.FinOpsExplainer.generate_llm_recommendations")
    @patch("openai.OpenAI")
    def test_local_solver_never_triggers_optional_llm_recs(
        self, mock_openai_cls, mock_gen_llm_recs
    ):
        # Running the local neuro-symbolic pipeline
        res = execute_live_pipeline(self.query)

        # 1. Pipeline succeeded locally
        self.assertIn("optimal_cost_usd", res)
        self.assertGreater(res["optimal_cost_usd"], 0.0)
        self.assertTrue(len(res["recommendations"]) > 0)

        # 2. Ensure NO LLM recommendations were called
        mock_gen_llm_recs.assert_not_called()
        mock_openai_cls.assert_not_called()

    # -------------------------------------------------------------------------
    # 11. Verification of Safe Formatters & Chart Handlers (No Fabricated Baselines)
    # -------------------------------------------------------------------------
    def test_safe_formatters_and_charts(self):
        # Test format_currency handles None without throwing TypeError
        self.assertEqual(format_currency(None), "—")
        self.assertEqual(format_currency(123.45), "$123.45")
        self.assertEqual(convert_currency(None), None)

        # Test create_smooth_cost_trend_chart with None costs (not run)
        fig_trend = create_smooth_cost_trend_chart(
            budget=500.0,
            optimal_cost=180.0,
            mode1_cost=None,
            mode2_cost=None,
        )
        self.assertIsNotNone(fig_trend)
        # Verify no fabricated multipliers (e.g. 500 * 1.25 = 625) in data
        x_data = list(fig_trend.data[0].x)
        self.assertNotIn("LLM Mode 1", x_data)
        self.assertIn("Stated Cap", x_data)
        self.assertTrue(any("Neurasym" in str(x) for x in x_data))

        # Test create_hallucination_delta_chart with not_run and error rows
        bench_data = [
            {"mode": "Mode 1: Pure LLM", "source": "not_run", "reported_cost_usd": None, "actual_cost_usd": None},
            {"mode": "Mode 2: Structured", "source": "live_error", "reported_cost_usd": None, "actual_cost_usd": None},
            {"mode": "Mode 3: Pure Symbolic", "source": "live", "reported_cost_usd": 0.0, "actual_cost_usd": 0.0},
            {"mode": "Mode 4: Neurasym", "source": "live", "reported_cost_usd": 180.0, "actual_cost_usd": 180.0},
        ]
        fig_delta = create_hallucination_delta_chart(bench_data)
        self.assertIsNotNone(fig_delta)
        self.assertEqual(fig_delta.data[0].text[0], "Awaiting Run")
        self.assertEqual(fig_delta.data[0].text[1], "Quota / Error")


if __name__ == "__main__":
    unittest.main()
