"""Unit tests for MOSAIC Intent Decomposition Engine and Orchestrator."""

import pytest

from mosaic.domain.models import IntentCategory, ValidationVerdict
from mosaic.intake import IntentDecompositionEngine
from mosaic.orchestrator import MosaicOrchestrator


def test_intent_decomposition_engine_extraction():
    """Verify rule-based intent decomposition extracts multiple non-overlapping intents."""
    engine = IntentDecompositionEngine()
    message = "My account was hacked and I cannot log in. Please cancel my recurring subscription and refund my payment."

    spans = engine.decompose_message(message)

    assert len(spans) == 4
    categories = [s.category for s in spans]
    assert IntentCategory.SECURITY in categories
    assert IntentCategory.ACCESS_RESTORATION in categories
    assert IntentCategory.SUBSCRIPTION in categories
    assert IntentCategory.BILLING in categories


def test_orchestrator_end_to_end_single_valid_case():
    """Verify MosaicOrchestrator processes a single valid subscription cancel message to ALLOW."""
    orchestrator = MosaicOrchestrator()
    result = orchestrator.process_customer_case(
        customer_id="cust_1001",
        raw_message="Please cancel my recurring subscription plan.",
        initial_facts={"subscription_active": True}
    )

    assert result.verdict == ValidationVerdict.ALLOW
    assert len(result.allowed_action_ids) == 1
    assert len(result.conflicts) == 0


def test_orchestrator_end_to_end_unverified_refund_case():
    """Verify MosaicOrchestrator processes unverified refund request message to BLOCK."""
    orchestrator = MosaicOrchestrator()
    result = orchestrator.process_customer_case(
        customer_id="cust_1002",
        raw_message="Please refund my money back.",
        initial_facts={"payment_verified": False}
    )

    assert result.verdict == ValidationVerdict.BLOCK
    assert len(result.blocked_action_ids) == 1
    assert result.conflicts[0].affected_fact_key == "payment_verified"
