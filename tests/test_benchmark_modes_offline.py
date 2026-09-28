import json
import unittest
from benchmarks.run_4way_benchmark import strip_reasoning_preamble, extract_mode1_cost
from src.semantic.explainer import FinOpsExplainer


class TestBenchmarkModesOffline(unittest.TestCase):
    def test_strip_reasoning_xml_thinking_tags(self):
        sample = (
            "<thinking>\n"
            "Analyzing budget: $850.00\n"
            "Instances needed: 2x t3.medium\n"
            "</thinking>\n"
            "Recommendation:\n"
            "Provider: AWS\n"
            "Instances: 2x t3.medium\n"
            "Estimated Total Monthly Cost: $60.50"
        )
        cleaned = strip_reasoning_preamble(sample)
        self.assertNotIn("thinking", cleaned.lower())
        self.assertNotIn("850", cleaned)
        cost = extract_mode1_cost(sample)
        self.assertEqual(cost, 60.50)

    def test_strip_reasoning_unclosed_xml_tag(self):
        truncated_xml = (
            "<thinking>\n"
            "Analyzing budget: $850.00\n"
            "Step 1: check constraints..."
        )
        cleaned = strip_reasoning_preamble(truncated_xml)
        self.assertEqual(cleaned, "")
        cost = extract_mode1_cost(truncated_xml, finish_reason="length")
        self.assertIsNone(cost)

    def test_extract_mode1_cost_truncation_safety_returns_none(self):
        content = (
            "Here's a thinking process:\n"
            "1. **Analyze User Request:**\n"
            "   - Stated Budget: $850.00\n"
            "2. **Identify Instances:**\n"
            "   - Target c5.large ($62.00)"
        )
        # Even if finish_reason == 'length' and $850 or $62 is in the text, it must return None
        cost = extract_mode1_cost(content, finish_reason="length")
        self.assertIsNone(cost)

    def test_extract_mode1_cost_pure_thinking_returns_none(self):
        content = (
            "Here's a thinking process:\n"
            "1. **Analyze User Request:**\n"
            "   - Target budget cap: $850.00\n"
            "2. **Evaluate options:**"
        )
        cost = extract_mode1_cost(content)
        self.assertIsNone(cost)

    def test_explainer_clean_recommendations_strips_xml_tags(self):
        raw = (
            "<thinking>\n"
            "We should recommend Graviton instances and savings plans.\n"
            "</thinking>\n"
            "1. Commit to 1-Year AWS EC2 Instance Savings Plans to reduce steady-state cost.\n"
            "2. Consolidate requested microservices into containerized pods with strict CPU/memory limits.\n"
            "3. Apply mandatory Cost Allocation Tags for 100% cost attribution."
        )
        recs = FinOpsExplainer._clean_llm_recommendations(raw)
        self.assertNotIn("thinking", "".join(recs).lower())
        self.assertEqual(len(recs), 3)


if __name__ == "__main__":
    unittest.main()
