"""Perception, reasoning, planning, action, and learning for a support agent."""

from datetime import datetime, timezone


def perceive_input(user_message, context):
    """Collect interaction input into a normalized perception payload."""
    return {
        "message": user_message,
        "timestamp": datetime.now(timezone.utc),
        "user_id": context.get("user_id"),
        "session_state": context.get("session"),
        "sentiment": context.get("sentiment", "neutral"),
    }


def classify_intent(message):
    """Lightweight intent classifier."""
    text = (message or "").lower()
    if any(token in text for token in ("bill", "billing", "charge", "invoice")):
        return "billing_issue"
    if any(token in text for token in ("refund", "cancel")):
        return "account_issue"
    return "general_inquiry"


def get_user_history(_user_id):
    """Stub for user history lookup."""
    return []


def determine_priority(intent, sentiment, user_history):
    """Set urgency based on intent, sentiment, and prior incidents."""
    recent_issues = len(user_history) if user_history else 0
    if intent == "billing_issue" and sentiment in {"negative", "angry"}:
        return "urgent"
    if recent_issues >= 3 and sentiment != "positive":
        return "urgent"
    return "normal"


def reason_about_intent(perception_data):
    """Infer intent and priority from perception data."""
    intent = classify_intent(perception_data["message"])
    priority = determine_priority(
        intent,
        perception_data.get("sentiment", "neutral"),
        user_history=get_user_history(perception_data.get("user_id")),
    )
    return {"intent": intent, "priority": priority, "context": perception_data}


def create_action_plan(reasoning_result):
    """Create a sequenced plan for the support workflow."""
    if reasoning_result["intent"] == "billing_issue":
        return [
            "fetch_account_details",
            "analyze_billing_history",
            "generate_explanation",
            "offer_resolution",
        ]
    if reasoning_result["priority"] == "urgent":
        return ["escalate_to_human", "log_urgent_case"]
    return ["respond_with_general_support"]


def execute_action(action_plan, context, billing_api, llm):
    """Execute each action in the plan and return collected outputs."""
    results = []
    for action in action_plan:
        if action == "fetch_account_details":
            result = billing_api.get_account(context["user_id"])
        elif action == "analyze_billing_history":
            account = results[-1] if results else {}
            result = {"analysis": f"reviewed {account}"}
        elif action == "generate_explanation":
            result = llm.generate_response(context, results)
        elif action == "offer_resolution":
            result = {"resolution": "offered billing adjustment options"}
        elif action == "escalate_to_human":
            result = {"escalated": True}
        elif action == "log_urgent_case":
            result = {"logged": True}
        else:
            result = {"status": f"skipped:{action}"}
        results.append(result)
    return results


def calculate_success(user_feedback):
    """Convert feedback to a bounded success score."""
    score = user_feedback.get("score", 0.0) if isinstance(user_feedback, dict) else 0.0
    return max(0.0, min(1.0, float(score)))


def update_user_preferences(_user_id, _success_score):
    """Stub for preference updates."""
    return None


def flag_for_model_improvement(_interaction_data):
    """Stub for model improvement tracking."""
    return None


def learn_from_outcome(interaction_data, user_feedback):
    """Adapt system state from interaction outcomes."""
    success_score = calculate_success(user_feedback)
    update_user_preferences(interaction_data["user_id"], success_score)
    if success_score < 0.7:
        flag_for_model_improvement(interaction_data)
    return success_score
