"""Unit Tests for EvaluationRateLimitedProvider Pacing and Rate Limiting Logic."""

from unittest.mock import MagicMock
import pytest

from mosaic.evaluation.runner import EvaluationRateLimitedProvider
from mosaic.llm.base import BaseLLMProvider
from mosaic.llm.models import LLMError, LLMRequest, LLMResponse


def test_rate_limiter_first_request_immediate():
    """Verify first request executes immediately without sleep."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.provider_name = "mock_provider"
    mock_provider.model_name = "mock_model"
    mock_response = LLMResponse(content="ok", provider_name="mock_provider", model_name="mock_model", structured_output={"status": "ok"})
    mock_provider.generate.return_value = mock_response

    mock_time = MagicMock(side_effect=[100.0, 100.0])
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    res = limiter.generate(req)

    assert res == mock_response
    mock_sleep.assert_not_called()
    mock_provider.generate.assert_called_once_with(req)


def test_rate_limiter_second_request_sleeps_remaining_interval():
    """Verify second request sleeps for remaining interval when requested too quickly."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_response = LLMResponse(content="ok", provider_name="mock_provider", model_name="mock_model")
    mock_provider.generate.return_value = mock_response

    # Call 1: time_func returns 100.0 before, 100.0 after
    # Call 2: time_func returns 105.0 before, 118.0 after
    mock_time = MagicMock(side_effect=[100.0, 100.0, 105.0, 118.0])
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    req = LLMRequest(system_prompt="sys", user_prompt="usr")

    # Request 1 at t=100.0
    limiter.generate(req)
    mock_sleep.assert_not_called()

    # Request 2 at t=105.0 (elapsed = 5.0s < 13.0s) -> should sleep 8.0s
    limiter.generate(req)
    mock_sleep.assert_called_once_with(8.0)


def test_rate_limiter_no_sleep_if_enough_time_elapsed():
    """Verify no sleep occurs if elapsed time exceeds minimum interval."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_response = LLMResponse(content="ok", provider_name="mock_provider", model_name="mock_model")
    mock_provider.generate.return_value = mock_response

    # Call 1: time_func returns 100.0 before, 100.0 after
    # Call 2: time_func returns 115.0 before (elapsed 15s > 13s)
    mock_time = MagicMock(side_effect=[100.0, 100.0, 115.0, 115.0])
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    limiter.generate(req)
    limiter.generate(req)

    mock_sleep.assert_not_called()
    assert mock_provider.generate.call_count == 2


def test_rate_limiter_delegates_properties_and_exceptions():
    """Verify provider properties and exceptions propagate unchanged without modifications to LLMRequest."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.provider_name = "gemini"
    mock_provider.model_name = "gemini-3.6-flash"
    mock_provider.generate.side_effect = LLMError("API failure")

    mock_time = MagicMock(return_value=100.0)
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    assert limiter.provider_name == "gemini"
    assert limiter.model_name == "gemini-3.6-flash"

    req = LLMRequest(system_prompt="sys", user_prompt="usr", temperature=0.0)
    with pytest.raises(LLMError, match="API failure"):
        limiter.generate(req)

    # Confirm LLMRequest was passed unchanged
    assert req.system_prompt == "sys"
    assert req.user_prompt == "usr"
    assert req.temperature == 0.0
