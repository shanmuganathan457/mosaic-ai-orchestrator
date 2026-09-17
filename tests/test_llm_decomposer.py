"""Unit and integration tests for LLMIntentDecomposer and Strategy Comparability."""

import json
import pytest

from mosaic.domain.models import IntentCategory, ValidationVerdict
from mosaic.intake import IntentDecompositionEngine, LLMIntentDecomposer
from mosaic.llm import LLMRequest, LLMResponse, LLMResponseError, MockLLMProvider
from mosaic.orchestrator import MosaicOrchestrator


def test_llm_intent_decomposer_single_intent():
    """Verify LLMIntentDecomposer correctly parses single billing intent from MockLLMProvider."""
    mock_payload = {
        "intents": [
            {
                "category": "BILLING",
                "intent_name": "refund_payment",
                "verbatim_text": "charged twice",
                "confidence": 0.98,
                "metadata": {"payment_id": "pay_101"}
            }
        ]
    }
    provider = MockLLMProvider(fixed_structured_output=mock_payload)
    decomposer = LLMIntentDecomposer(llm_provider=provider)

    raw_message = "I was charged twice for the same payment."
    spans = decomposer.decompose_message(raw_message)

    assert len(spans) == 1
    assert spans[0].category == IntentCategory.BILLING
    assert spans[0].intent_name == "refund_payment"
    assert spans[0].verbatim_text == "charged twice"
    assert spans[0].confidence == 0.98


def test_llm_intent_decomposer_multi_intent():
    """Verify LLMIntentDecomposer parses multi-intent billing + access restoration payload."""
    mock_payload = {
        "intents": [
            {
                "category": "BILLING",
                "intent_name": "refund_payment",
                "verbatim_text": "charged twice",
                "confidence": 0.95
            },
            {
                "category": "ACCESS_RESTORATION",
                "intent_name": "restore_login_access",
                "verbatim_text": "cannot log into my account",
                "confidence": 0.93
            }
        ]
    }
    provider = MockLLMProvider(fixed_structured_output=mock_payload)
    decomposer = LLMIntentDecomposer(llm_provider=provider)

    raw_message = "I was charged twice and now I cannot log into my account."
    spans = decomposer.decompose_message(raw_message)

    assert len(spans) == 2
    categories = [s.category for s in spans]
    assert IntentCategory.BILLING in categories
    assert IntentCategory.ACCESS_RESTORATION in categories


def test_llm_intent_decomposer_three_intents():
    """Verify LLMIntentDecomposer handles complex three-intent customer messages."""
    mock_payload = {
        "intents": [
            {
                "category": "SECURITY",
                "intent_name": "restrict_account",
                "verbatim_text": "suspicious activity",
                "confidence": 0.99
            },
            {
                "category": "SUBSCRIPTION",
                "intent_name": "cancel_subscription",
                "verbatim_text": "cancel subscription",
                "confidence": 0.96
            },
            {
                "category": "BILLING",
                "intent_name": "refund_payment",
                "verbatim_text": "refund payment",
                "confidence": 0.95
            }
        ]
    }
    provider = MockLLMProvider(fixed_structured_output=mock_payload)
    decomposer = LLMIntentDecomposer(llm_provider=provider)

    raw_message = "I noticed suspicious activity, so please cancel subscription and refund payment."
    spans = decomposer.decompose_message(raw_message)

    assert len(spans) == 3
    assert spans[0].category == IntentCategory.SECURITY
    assert spans[1].category == IntentCategory.SUBSCRIPTION
    assert spans[2].category == IntentCategory.BILLING


def test_llm_intent_decomposer_empty_message():
    """Verify LLMIntentDecomposer gracefully handles empty messages."""
    provider = MockLLMProvider()
    decomposer = LLMIntentDecomposer(llm_provider=provider)
    assert decomposer.decompose_message("") == []


def test_llm_intent_decomposer_malformed_json_handling():
    """Verify LLMIntentDecomposer raises LLMResponseError on malformed non-JSON output."""
    provider = MockLLMProvider(default_response_text="INVALID NON JSON STRING")
    decomposer = LLMIntentDecomposer(llm_provider=provider)

    with pytest.raises(LLMResponseError):
        decomposer.decompose_message("Test message")


def test_llm_intent_decomposer_schema_validation_error():
    """Verify LLMIntentDecomposer raises LLMResponseError when JSON fails Pydantic schema validation."""
    invalid_schema_payload = {"intents": [{"invalid_field": "test"}]}
    provider = MockLLMProvider(fixed_structured_output=invalid_schema_payload)
    decomposer = LLMIntentDecomposer(llm_provider=provider)

    with pytest.raises(LLMResponseError):
        decomposer.decompose_message("Test message")


def test_orchestrator_comparability_strategy_injection():
    """Verify Orchestrator accepts both Deterministic and LLM Decomposers yielding identical downstream pipeline flow."""
    mock_payload = {
        "intents": [
            {
                "category": "BILLING",
                "intent_name": "refund_payment",
                "verbatim_text": "refund",
                "confidence": 0.95,
                "metadata": {"payment_id": "pay_99"}
            }
        ]
    }
    llm_provider = MockLLMProvider(fixed_structured_output=mock_payload)
    llm_decomposer = LLMIntentDecomposer(llm_provider=llm_provider)

    # Strategy 1: Deterministic Intake
    det_orchestrator = MosaicOrchestrator(intake_engine=IntentDecompositionEngine())
    det_result = det_orchestrator.process_customer_case(
        customer_id="c1",
        raw_message="Please refund payment pay_99.",
        initial_facts={"payment_verified": True, "identity_verified": True}
    )

    # Strategy 2: LLM Intake
    llm_orchestrator = MosaicOrchestrator(intake_engine=llm_decomposer)
    llm_result = llm_orchestrator.process_customer_case(
        customer_id="c1",
        raw_message="Please refund payment pay_99.",
        initial_facts={"payment_verified": True, "identity_verified": True}
    )

    # Downstream Validation Verdict must be identical (ALLOW) across both strategy implementations
    assert det_result.verdict == ValidationVerdict.ALLOW
    assert llm_result.verdict == ValidationVerdict.ALLOW
