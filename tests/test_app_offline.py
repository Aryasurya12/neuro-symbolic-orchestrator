"""tests/test_app_offline.py
Offline Streamlit AppTest integration test verifying app.py runs and renders without live API calls.
"""

import unittest
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest


class TestStreamlitAppOffline(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_app_loads_and_runs_without_network_calls(self, mock_openai_cls):
        """Verify that launching app.py triggers 0 live API calls."""
        import os
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        self.assertFalse(at.exception)

        # Confirm OpenAI was never initialized during page load
        mock_openai_cls.assert_not_called()

    @patch("openai.OpenAI")
    def test_app_submits_query_and_executes_mode3_mode4_offline(self, mock_openai_cls):
        """Verify submitting a query executes Mode 3 & Mode 4 locally with zero API calls."""
        import os
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()

        # Switch to benchmark page and run a query
        at.session_state["current_page"] = "benchmark"
        at.session_state["active_query_text"] = "Deploy 4 vCPUs and 16GB RAM on AWS with a budget of $500/month"
        at.session_state["has_run"] = True
        at.run()

        self.assertFalse(at.exception)
        # Verify pipeline result was cached and generated locally
        self.assertIn("cached_pipeline_result", at.session_state)
        live_res = at.session_state["cached_pipeline_result"]
        self.assertEqual(live_res["problem_type"], "ILP_VM_Allocation")
        self.assertTrue(live_res["m4_check"]["feasible_against_contract"])
        self.assertTrue(live_res["m3_check"]["feasible_against_contract"])

        # OpenAI must not have been invoked
        mock_openai_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
