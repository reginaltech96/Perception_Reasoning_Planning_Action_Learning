import unittest
from unittest.mock import patch

from customer_service_agent import (
    create_action_plan,
    execute_action,
    learn_from_outcome,
    perceive_input,
    reason_about_intent,
)


class _BillingAPI:
    def get_account(self, user_id):
        return {"user_id": user_id, "balance": 42}


class _LLM:
    def generate_response(self, context, results):
        return {
            "message": f"Generated explanation for {context['user_id']}",
            "inputs": len(results),
        }


class CustomerServiceAgentTests(unittest.TestCase):
    def test_perceive_input_captures_expected_fields(self):
        context = {"user_id": "u1", "session": {"topic": "billing"}, "sentiment": "negative"}
        payload = perceive_input("I was overcharged", context)
        self.assertEqual(payload["message"], "I was overcharged")
        self.assertEqual(payload["user_id"], "u1")
        self.assertEqual(payload["session_state"], {"topic": "billing"})
        self.assertEqual(payload["sentiment"], "negative")
        self.assertIsNotNone(payload["timestamp"])

    def test_reason_about_intent_uses_sentiment_for_priority(self):
        perception_data = {
            "message": "My billing is wrong",
            "user_id": "u2",
            "sentiment": "negative",
            "session_state": {},
        }
        result = reason_about_intent(perception_data)
        self.assertEqual(result["intent"], "billing_issue")
        self.assertEqual(result["priority"], "urgent")
        self.assertEqual(result["context"], perception_data)

    def test_create_action_plan_for_billing_issue(self):
        plan = create_action_plan({"intent": "billing_issue", "priority": "normal"})
        self.assertEqual(
            plan,
            [
                "fetch_account_details",
                "analyze_billing_history",
                "generate_explanation",
                "offer_resolution",
            ],
        )

    def test_execute_action_runs_billing_flow(self):
        plan = [
            "fetch_account_details",
            "analyze_billing_history",
            "generate_explanation",
            "offer_resolution",
        ]
        context = {"user_id": "u3"}
        results = execute_action(plan, context, _BillingAPI(), _LLM())
        self.assertEqual(results[0]["user_id"], "u3")
        self.assertIn("analysis", results[1])
        self.assertIn("message", results[2])
        self.assertIn("resolution", results[3])

    def test_learn_from_outcome_flags_low_success(self):
        interaction_data = {"user_id": "u4"}
        with patch("customer_service_agent.update_user_preferences") as update_mock, patch(
            "customer_service_agent.flag_for_model_improvement"
        ) as flag_mock:
            score = learn_from_outcome(interaction_data, {"score": 0.5})
        update_mock.assert_called_once_with("u4", 0.5)
        flag_mock.assert_called_once_with(interaction_data)
        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
