"""Tests for Google Gemini LLM Provider & Factory Integration."""

import os
from unittest.mock import MagicMock, patch
import pytest

from google.genai.errors import APIError

from mosaic.llm.factory import LLMProviderFactory
from mosaic.llm.gemini import GeminiProvider
from mosaic.llm.mock import MockLLMProvider
from mosaic.llm.models import (
    LLMConfigurationError,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
)
from mosaic.llm.ollama import OllamaProvider


def test_gemini_provider_init_requires_api_key(monkeypatch):
    """Verify GeminiProvider raises LLMConfigurationError if API key is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError, match="Gemini API key is missing"):
        GeminiProvider(api_key="")


def test_gemini_provider_init_with_key():
    """Verify GeminiProvider initializes successfully when API key is provided."""
    with patch("google.genai.Client"):
        provider = GeminiProvider(api_key="TEST_FAKE_KEY", model_name="gemini-2.5-flash")
        assert provider.provider_name == "gemini"
        assert provider.model_name == "gemini-2.5-flash"


def test_gemini_provider_generate_basic_request():
    """Verify standard text request conversion into LLMResponse."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello world from Gemini"
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client, model_name="gemini-2.5-flash")
    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    res = provider.generate(req)

    assert isinstance(res, LLMResponse)
    assert res.content == "Hello world from Gemini"
    assert res.provider_name == "gemini"
    assert res.model_name == "gemini-2.5-flash"
    assert res.usage_info["prompt_tokens"] == 10
    assert res.usage_info["completion_tokens"] == 20
    assert res.usage_info["total_tokens"] == 30


def test_gemini_provider_generate_structured_json_request():
    """Verify structured schema request forces json mime type and parses JSON output."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"intents": [{"category": "BILLING"}]}'
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client)
    req = LLMRequest(
        system_prompt="sys",
        user_prompt="usr",
        structured_schema={"type": "object"}
    )
    res = provider.generate(req)

    assert res.structured_output == {"intents": [{"category": "BILLING"}]}
    mock_client.models.generate_content.assert_called_once()
    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"].response_mime_type == "application/json"


def test_gemini_provider_malformed_json_response():
    """Verify LLMResponseError is raised when structured output cannot be parsed."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "INVALID NON-JSON TEXT"
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client)
    req = LLMRequest(system_prompt="sys", user_prompt="usr", structured_schema={"type": "object"})

    with pytest.raises(LLMResponseError, match="Failed to parse structured JSON output"):
        provider.generate(req)


def test_gemini_provider_api_error_handling():
    """Verify APIError is wrapped in LLMError without leaking secrets."""
    mock_client = MagicMock()
    mock_client._api_key = "SECRET_API_KEY_123"
    mock_client.models.generate_content.side_effect = APIError(
        code=400, response_json={"error": {"message": "Invalid request with key SECRET_API_KEY_123"}}
    )

    provider = GeminiProvider(client=mock_client)
    req = LLMRequest(system_prompt="sys", user_prompt="usr")

    with pytest.raises(LLMError) as exc_info:
        provider.generate(req)

    assert "SECRET_API_KEY_123" not in str(exc_info.value)


def test_factory_gemini_resolution(monkeypatch):
    """Verify LLMProviderFactory resolves GeminiProvider when requested."""
    monkeypatch.setenv("GEMINI_API_KEY", "FAKE_KEY_FACTORY")
    with patch("google.genai.Client"):
        provider = LLMProviderFactory.get_provider("gemini", model_name="gemini-2.5-flash")
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"
        assert provider.model_name == "gemini-2.5-flash"


def test_factory_gemini_missing_key(monkeypatch):
    """Verify LLMProviderFactory raises LLMConfigurationError if key is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError):
        LLMProviderFactory.get_provider("gemini", api_key="")


def test_existing_providers_unbroken():
    """Verify Mock and Ollama providers continue resolving normally."""
    mock_p = LLMProviderFactory.get_provider("mock")
    assert isinstance(mock_p, MockLLMProvider)

    ollama_p = LLMProviderFactory.get_provider("ollama")
    assert isinstance(ollama_p, OllamaProvider)


@pytest.mark.skipif(
    os.environ.get("GEMINI_LIVE_TEST") != "1",
    reason="Live Gemini API test requires explicit opt-in via GEMINI_LIVE_TEST=1 and GEMINI_API_KEY"
)
def test_gemini_live_api_integration():  # pragma: no cover
    """Optional live integration test executed only when GEMINI_LIVE_TEST=1 is set."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")

    provider = GeminiProvider(api_key=key, model_name="gemini-2.5-flash")
    req = LLMRequest(system_prompt="You are a helpful assistant.", user_prompt="Say hello in one word.")
    res = provider.generate(req)
    assert len(res.content) > 0
