"""Unit tests for Phase 5A Provider-Agnostic LLM Abstraction Layer."""

import pytest

from mosaic.config import settings
from mosaic.llm import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    MockLLMProvider,
)


def test_llm_request_response_instantiation():
    """Verify strongly-typed Pydantic LLMRequest and LLMResponse models."""
    req = LLMRequest(
        system_prompt="You are a support classifier.",
        user_prompt="I want to cancel my subscription.",
        structured_schema={"type": "object"},
        temperature=0.2,
    )
    assert req.system_prompt == "You are a support classifier."
    assert req.temperature == 0.2

    resp = LLMResponse(
        content="Response text",
        structured_output={"intent": "cancel"},
        provider_name="mock",
        model_name="test-model",
    )
    assert resp.provider_name == "mock"
    assert resp.structured_output["intent"] == "cancel"


def test_mock_llm_provider_generation():
    """Verify MockLLMProvider operates deterministically without external dependencies."""
    provider: BaseLLMProvider = MockLLMProvider(
        model_name="mock-v1",
        default_response_text="Determined mock response",
        fixed_structured_output={"intent": "refund_payment", "confidence": 0.99},
    )

    assert provider.provider_name == "mock"
    assert provider.model_name == "mock-v1"

    req = LLMRequest(system_prompt="Test", user_prompt="Test prompt")
    resp = provider.generate(req)

    assert resp.provider_name == "mock"
    assert resp.model_name == "mock-v1"
    assert resp.content == "Determined mock response"
    assert resp.structured_output == {"intent": "refund_payment", "confidence": 0.99}
    assert resp.finish_reason == "stop"


def test_mock_llm_provider_repeatability():
    """Verify repeated invocations on MockLLMProvider yield identical deterministic output."""
    provider = MockLLMProvider()
    req = LLMRequest(system_prompt="Sys", user_prompt="User")

    resp1 = provider.generate(req)
    resp2 = provider.generate(req)

    assert resp1.content == resp2.content
    assert resp1.usage_info == resp2.usage_info


def test_llm_configuration_defaults():
    """Verify default LLM settings point to offline mock configuration."""
    assert settings.LLM_PROVIDER == "mock"
    assert settings.LLM_MODEL == "mock-deterministic-v1"
    assert settings.LLM_ENABLED is False
