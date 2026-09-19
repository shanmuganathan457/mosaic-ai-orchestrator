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


# ==========================================
# QUALITY CORRECTION TESTS
# ==========================================

def test_llm_intent_decomposer_invalid_intent_category_rejection():
    """CORRECTION 1: Verify LLMIntentDecomposer raises LLMResponseError on unsupported IntentCategory."""
    invalid_category_payload = {
        "intents": [
            {
                "category": "UNSUPPORTED_UNKNOWN_CATEGORY",
                "intent_name": "refund_payment",
                "verbatim_text": "refund",
                "confidence": 0.9
            }
        ]
    }
    provider = MockLLMProvider(fixed_structured_output=invalid_category_payload)
    decomposer = LLMIntentDecomposer(llm_provider=provider)

    with pytest.raises(LLMResponseError):
        decomposer.decompose_message("Please refund my money.")


def test_llm_intent_decomposer_evidence_traceability_rejection():
    """CORRECTION 2: Verify LLMIntentDecomposer raises LLMResponseError on untraceable/hallucinated verbatim_text."""
    hallucinated_evidence_payload = {
        "intents": [
            {
                "category": "BILLING",
                "intent_name": "refund_payment",
                "verbatim_text": "text not present in message at all",
                "confidence": 0.95
            }
        ]
    }
    provider = MockLLMProvider(fixed_structured_output=hallucinated_evidence_payload)
    decomposer = LLMIntentDecomposer(llm_provider=provider)

    with pytest.raises(LLMResponseError, match="Evidence integrity error"):
        decomposer.decompose_message("Please refund my money.")


def test_llm_intent_decomposer_absent_metadata_non_fabrication():
    """CORRECTION 3: Verify LLMIntentDecomposer does not fabricate entity IDs if absent in customer text."""
    payload_without_ids = {
        "intents": [
            {
                "category": "BILLING",
                "intent_name": "refund_payment",
                "verbatim_text": "refund my money",
                "confidence": 0.95,
                "metadata": {}
            }
        ]
    }
    provider = MockLLMProvider(fixed_structured_output=payload_without_ids)
    decomposer = LLMIntentDecomposer(llm_provider=provider)

    raw_message = "Please refund my money."
    spans = decomposer.decompose_message(raw_message)

    assert len(spans) == 1
    assert "payment_id" not in spans[0].metadata
    assert "account_id" not in spans[0].metadata
    assert "subscription_id" not in spans[0].metadata


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


# ==========================================
# CANONICAL CONTRACT ENFORCEMENT TESTS
# These tests verify the strict LLM output contract:
# - category must be a domain IntentCategory value
# - intent_name must be a canonical agent-map key
# - verbatim_text must be traceable to the customer message
# ==========================================

class TestCanonicalContractEnforcement:
    """Verifies that ExtractedIntentItem and LLMIntentDecomposer enforce the canonical
    category/intent_name vocabulary and verbatim evidence traceability at the schema boundary."""

    # --- 1. Valid category + valid canonical intent accepted ---
    def test_valid_category_and_intent_accepted(self):
        """Valid BILLING + refund_payment must be accepted without error."""
        payload = {
            "intents": [{
                "category": "BILLING",
                "intent_name": "refund_payment",
                "verbatim_text": "charged twice",
                "confidence": 0.95,
            }]
        }
        provider = MockLLMProvider(fixed_structured_output=payload)
        decomposer = LLMIntentDecomposer(llm_provider=provider)
        spans = decomposer.decompose_message("I was charged twice.")
        assert len(spans) == 1
        assert spans[0].category == IntentCategory.BILLING
        assert spans[0].intent_name == "refund_payment"

    # --- 2. Invalid category (intent name used as category) must be rejected ---
    def test_invalid_category_restore_login_access_rejected(self):
        """RESTORE_LOGIN_ACCESS is an intent_name, not a category. Must be rejected by Pydantic."""
        payload = {
            "intents": [{
                "category": "RESTORE_LOGIN_ACCESS",  # invalid — this is an intent name
                "intent_name": "restore_login_access",
                "verbatim_text": "cannot log in",
                "confidence": 0.9,
            }]
        }
        provider = MockLLMProvider(fixed_structured_output=payload)
        decomposer = LLMIntentDecomposer(llm_provider=provider)
        with pytest.raises(LLMResponseError):
            decomposer.decompose_message("I cannot log in to my account.")

    # --- 3. Invalid intent name (free-form) must be rejected ---
    def test_invalid_intent_name_refund_request_rejected(self):
        """REFUND_REQUEST is a free-form name not in the canonical set. Must be rejected."""
        payload = {
            "intents": [{
                "category": "BILLING",
                "intent_name": "REFUND_REQUEST",  # invalid — not a canonical identifier
                "verbatim_text": "refund",
                "confidence": 0.9,
            }]
        }
        provider = MockLLMProvider(fixed_structured_output=payload)
        decomposer = LLMIntentDecomposer(llm_provider=provider)
        with pytest.raises(LLMResponseError):
            decomposer.decompose_message("Please process my refund.")

    # --- 4. ACCESS_RESTORATION + restore_login_access is valid ---
    def test_access_restoration_category_and_intent_accepted(self):
        """ACCESS_RESTORATION category + restore_login_access intent must be accepted."""
        payload = {
            "intents": [{
                "category": "ACCESS_RESTORATION",
                "intent_name": "restore_login_access",
                "verbatim_text": "cannot log in",
                "confidence": 0.93,
            }]
        }
        provider = MockLLMProvider(fixed_structured_output=payload)
        decomposer = LLMIntentDecomposer(llm_provider=provider)
        spans = decomposer.decompose_message("I cannot log in to my account.")
        assert len(spans) == 1
        assert spans[0].category == IntentCategory.ACCESS_RESTORATION
        assert spans[0].intent_name == "restore_login_access"

    # --- 5. Verbatim text copied exactly from customer message is accepted ---
    def test_exact_verbatim_from_message_accepted(self):
        """Verbatim text that is an exact substring of the customer message must be accepted."""
        raw_message = "Please cancel my active subscription sub_202."
        payload = {
            "intents": [{
                "category": "SUBSCRIPTION",
                "intent_name": "cancel_subscription",
                "verbatim_text": "cancel my active subscription",
                "confidence": 0.97,
            }]
        }
        provider = MockLLMProvider(fixed_structured_output=payload)
        decomposer = LLMIntentDecomposer(llm_provider=provider)
        spans = decomposer.decompose_message(raw_message)
        assert len(spans) == 1
        assert spans[0].verbatim_text == "cancel my active subscription"
        assert spans[0].intent_name == "cancel_subscription"

    # --- 6. Paraphrased verbatim text is rejected ---
    def test_paraphrased_verbatim_text_rejected(self):
        """Paraphrased text not present in the customer message must be rejected."""
        raw_message = "I want to stop my plan."
        payload = {
            "intents": [{
                "category": "SUBSCRIPTION",
                "intent_name": "cancel_subscription",
                "verbatim_text": "cancel my subscription",  # paraphrase — not in raw_message
                "confidence": 0.9,
            }]
        }
        provider = MockLLMProvider(fixed_structured_output=payload)
        decomposer = LLMIntentDecomposer(llm_provider=provider)
        with pytest.raises(LLMResponseError, match="Evidence integrity error"):
            decomposer.decompose_message(raw_message)

    # --- 7. System-prompt description text used as verbatim is rejected ---
    def test_system_prompt_description_as_verbatim_rejected(self):
        """Text from the system prompt (not the customer message) used as verbatim must be rejected."""
        raw_message = "Please cancel my subscription."
        # This text came from the old system prompt examples, not the customer message
        payload = {
            "intents": [{
                "category": "SUBSCRIPTION",
                "intent_name": "cancel_subscription",
                "verbatim_text": "cancel/terminate/stop subscription, plan, or membership",
                "confidence": 0.9,
            }]
        }
        provider = MockLLMProvider(fixed_structured_output=payload)
        decomposer = LLMIntentDecomposer(llm_provider=provider)
        with pytest.raises(LLMResponseError, match="Evidence integrity error"):
            decomposer.decompose_message(raw_message)

    # --- 8. Existing metadata/evidence integrity check: hallucinated verbatim rejected ---
    def test_hallucinated_verbatim_completely_absent_rejected(self):
        """Completely invented verbatim text not present anywhere in customer message is rejected."""
        payload = {
            "intents": [{
                "category": "BILLING",
                "intent_name": "refund_payment",
                "verbatim_text": "refund payment",  # not in the message below
                "confidence": 0.9,
            }]
        }
        provider = MockLLMProvider(fixed_structured_output=payload)
        decomposer = LLMIntentDecomposer(llm_provider=provider)
        with pytest.raises(LLMResponseError, match="Evidence integrity error"):
            decomposer.decompose_message("I noticed a charge on my account and I want my money back.")

    # --- 9. Existing mock-provider tests: unsupported unknown category still rejected ---
    def test_completely_unknown_category_rejected(self):
        """A completely unknown/unsupported category string must be rejected."""
        payload = {
            "intents": [{
                "category": "UNSUPPORTED_UNKNOWN_CATEGORY",
                "intent_name": "refund_payment",
                "verbatim_text": "refund",
                "confidence": 0.9,
            }]
        }
        provider = MockLLMProvider(fixed_structured_output=payload)
        decomposer = LLMIntentDecomposer(llm_provider=provider)
        with pytest.raises(LLMResponseError):
            decomposer.decompose_message("Please refund my money.")

    # --- JSON schema verification ---
    def test_json_schema_exposes_category_enum(self):
        """The generated JSON schema must expose category as an enum with all six IntentCategory values."""
        from mosaic.intake.llm_decomposer import ExtractedIntentItem
        schema = ExtractedIntentItem.model_json_schema()
        cat_schema = schema["properties"]["category"]
        assert "enum" in cat_schema, "category must have enum constraint in JSON schema"
        enum_values = set(cat_schema["enum"])
        expected = {"SECURITY", "BILLING", "SUBSCRIPTION", "ACCESS_RESTORATION", "TECHNICAL_SUPPORT", "GENERAL_INQUIRY"}
        assert enum_values == expected, f"Unexpected category enum values: {enum_values}"

    def test_json_schema_exposes_intent_name_enum(self):
        """The generated JSON schema must expose intent_name as an enum with all four canonical identifiers."""
        from mosaic.intake.llm_decomposer import ExtractedIntentItem
        schema = ExtractedIntentItem.model_json_schema()
        intent_schema = schema["properties"]["intent_name"]
        assert "enum" in intent_schema, "intent_name must have enum constraint in JSON schema"
        enum_values = set(intent_schema["enum"])
        expected = {"restrict_account", "refund_payment", "cancel_subscription", "restore_login_access"}
        assert enum_values == expected, f"Unexpected intent_name enum values: {enum_values}"
