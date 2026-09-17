"""Tests for Ollama Local LLM Provider & Factory."""

import json
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from mosaic.llm.factory import LLMProviderFactory
from mosaic.llm.mock import MockLLMProvider
from mosaic.llm.models import (
    LLMConfigurationError,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
)
from mosaic.llm.ollama import OllamaProvider


def test_ollama_provider_configuration():
    """Verify initialization and property access of OllamaProvider."""
    provider = OllamaProvider(base_url="http://localhost:11434/", model_name="llama3.2", timeout=30.0)
    assert provider.provider_name == "ollama"
    assert provider.model_name == "llama3.2"
    assert provider.base_url == "http://localhost:11434"

    with pytest.raises(LLMConfigurationError):
        OllamaProvider(base_url="")

    with pytest.raises(LLMConfigurationError):
        OllamaProvider(model_name="")


@patch("urllib.request.urlopen")
def test_ollama_provider_generate_success(mock_urlopen):
    """Verify successful generation and JSON parsing from Ollama server response."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "model": "llama3.2",
        "response": "Hello world from Ollama",
        "done": True,
        "prompt_eval_count": 15,
        "eval_count": 25,
        "total_duration": 1234567,
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    provider = OllamaProvider()
    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    res = provider.generate(req)

    assert isinstance(res, LLMResponse)
    assert res.content == "Hello world from Ollama"
    assert res.provider_name == "ollama"
    assert res.model_name == "llama3.2"
    assert res.usage_info["prompt_tokens"] == 15
    assert res.usage_info["completion_tokens"] == 25
    assert res.usage_info["total_tokens"] == 40


@patch("urllib.request.urlopen")
def test_ollama_provider_generate_structured_success(mock_urlopen):
    """Verify structured output parsing when structured_schema is provided."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "model": "llama3.2",
        "response": json.dumps({"intents": [{"category": "BILLING"}]}),
        "done": True,
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    provider = OllamaProvider()
    req = LLMRequest(system_prompt="sys", user_prompt="usr", structured_schema={"type": "object"})
    res = provider.generate(req)

    assert res.structured_output == {"intents": [{"category": "BILLING"}]}


@patch("urllib.request.urlopen")
def test_ollama_provider_connection_failure(mock_urlopen):
    """Verify LLMError is raised on connection failure."""
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    provider = OllamaProvider()
    req = LLMRequest(system_prompt="sys", user_prompt="usr")

    with pytest.raises(LLMError, match="Failed to connect to Ollama server"):
        provider.generate(req)


@patch("urllib.request.urlopen")
def test_ollama_provider_timeout(mock_urlopen):
    """Verify LLMError is raised on request timeout."""
    mock_urlopen.side_effect = TimeoutError("Request timed out")

    provider = OllamaProvider(timeout=5.0)
    req = LLMRequest(system_prompt="sys", user_prompt="usr")

    with pytest.raises(LLMError, match="timed out"):
        provider.generate(req)


@patch("urllib.request.urlopen")
def test_ollama_provider_malformed_json_response(mock_urlopen):
    """Verify LLMResponseError is raised when Ollama returns non-JSON content."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"BAD NON-JSON RESPONSE"
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    provider = OllamaProvider()
    req = LLMRequest(system_prompt="sys", user_prompt="usr")

    with pytest.raises(LLMResponseError, match="Failed to parse JSON response"):
        provider.generate(req)


@patch("urllib.request.urlopen")
def test_ollama_provider_malformed_structured_output(mock_urlopen):
    """Verify LLMResponseError is raised when structured output cannot be parsed as JSON."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "model": "llama3.2",
        "response": "NOT VALID JSON STRUCTURED OUTPUT",
        "done": True,
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    provider = OllamaProvider()
    req = LLMRequest(system_prompt="sys", user_prompt="usr", structured_schema={"type": "object"})

    with pytest.raises(LLMResponseError, match="Failed to parse structured JSON output"):
        provider.generate(req)


def test_provider_factory_resolution():
    """Verify LLMProviderFactory instantiates mock and ollama providers correctly."""
    mock_p = LLMProviderFactory.get_provider("mock")
    assert isinstance(mock_p, MockLLMProvider)
    assert mock_p.provider_name == "mock"

    ollama_p = LLMProviderFactory.get_provider("ollama", model_name="qwen2.5")
    assert isinstance(ollama_p, OllamaProvider)
    assert ollama_p.provider_name == "ollama"
    assert ollama_p.model_name == "qwen2.5"

    with pytest.raises(LLMConfigurationError):
        LLMProviderFactory.get_provider("invalid_provider")


@pytest.mark.skipif(
    True,
    reason="Optional integration test — requires a live running local Ollama server at http://localhost:11434"
)
def test_ollama_live_server_integration():  # pragma: no cover
    """Optional live integration test executed only when a local Ollama server is running."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            if resp.status != 200:
                pytest.skip("Ollama server not running locally")
    except Exception:
        pytest.skip("Ollama server unreachable locally")

    provider = OllamaProvider(model_name="llama3.2")
    req = LLMRequest(system_prompt="You are a helpful assistant.", user_prompt="Say hello in one word.")
    res = provider.generate(req)
    assert len(res.content) > 0
