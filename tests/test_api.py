"""Phase 8A Integration and FastAPI Endpoint Tests for MOSAIC pipeline."""

import pytest
from fastapi.testclient import TestClient

from mosaic.executor import ExecutionStatus
from mosaic.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_simple_allowed_refund():
    """Test A: Simple allowed refund request executes through mock executor."""
    payload = {
        "customer_id": "cust_101",
        "raw_message": "Hi, please issue a refund for payment pay_101.",
        "initial_facts": {
            "payment_verified": True,
            "identity_verified": True,
        },
        "active_flags": [],
    }

    response = client.post("/api/v1/support/requests", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["customer_id"] == "cust_101"
    assert data["final_verdict"] == "ALLOW"
    assert data["execution_status"] == "EXECUTED"
    assert len(data["detected_intents"]) == 1
    assert data["detected_intents"][0]["intent_name"] == "refund_payment"
    assert len(data["action_execution_records"]) == 1
    assert data["action_execution_records"][0]["status"] == "EXECUTED"
    assert data["action_execution_records"][0]["action_type"] == "refund_payment"


def test_api_refund_missing_prerequisite_blocks():
    """Test B: Refund with missing prerequisite (unverified identity) -> BLOCK.

    Verifies that blocked actions NEVER reach mock executor (marked BLOCKED).
    """
    payload = {
        "customer_id": "cust_102",
        "raw_message": "Please process a refund for pay_102 right now!",
        "initial_facts": {
            "payment_verified": True,
            "identity_verified": False,
        },
        "active_flags": [],
    }

    response = client.post("/api/v1/support/requests", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["customer_id"] == "cust_102"
    assert data["final_verdict"] == "BLOCK"
    assert data["execution_status"] == "BLOCKED"
    assert len(data["primary_conflicts"]) > 0
    assert any(c["conflict_type"] == "PRECONDITION_UNSATISFIED" for c in data["primary_conflicts"])
    assert len(data["action_execution_records"]) == 1
    assert data["action_execution_records"][0]["status"] == "BLOCKED"
    assert "blocked by MOSAIC governance" in data["action_execution_records"][0]["message"]


def test_api_suspicious_account_recovery_escalates():
    """Test C: Suspicious account recovery request -> ESCALATE."""
    payload = {
        "customer_id": "cust_103",
        "raw_message": "Please restore login access for my account acc_707.",
        "initial_facts": {
            "account_status": "locked",
        },
        "active_flags": ["SUSPICIOUS_LOCATION_LOGIN"],
    }

    response = client.post("/api/v1/support/requests", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["customer_id"] == "cust_103"
    assert data["final_verdict"] == "ESCALATED"
    assert data["execution_status"] == "ESCALATED"
    assert len(data["primary_conflicts"]) > 0
    assert any(c["conflict_type"] == "AMBIGUOUS_EVIDENCE" for c in data["primary_conflicts"])
    assert len(data["action_execution_records"]) == 1
    assert data["action_execution_records"][0]["status"] == "ESCALATED"


def test_api_multi_intent_request():
    """Test D: Multi-intent request (cancel subscription + refund payment)."""
    payload = {
        "customer_id": "cust_104",
        "raw_message": "Cancel my active subscription sub_202 and also refund pay_101.",
        "initial_facts": {
            "subscription_active": True,
            "payment_verified": True,
            "identity_verified": True,
        },
        "active_flags": [],
    }

    response = client.post("/api/v1/support/requests", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["customer_id"] == "cust_104"
    assert data["final_verdict"] == "ALLOW"
    assert data["execution_status"] == "EXECUTED"
    assert len(data["detected_intents"]) == 2
    intent_names = {i["intent_name"] for i in data["detected_intents"]}
    assert intent_names == {"cancel_subscription", "refund_payment"}
    assert len(data["action_execution_records"]) == 2
    assert all(r["status"] == "EXECUTED" for r in data["action_execution_records"])


def test_api_invalid_empty_message_returns_400():
    """Test Error Handling: Empty raw message payload returns HTTP 400."""
    payload = {
        "customer_id": "cust_105",
        "raw_message": "   ",
    }

    response = client.post("/api/v1/support/requests", json=payload)
    assert response.status_code == 400
    assert "raw_message cannot be empty" in response.json()["detail"]
