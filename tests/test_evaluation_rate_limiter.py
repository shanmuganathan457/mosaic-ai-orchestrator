"""Unit Tests for EvaluationRateLimitedProvider Pacing and Retry Logic."""

from unittest.mock import MagicMock
import pytest

from mosaic.evaluation.runner import EvaluationRateLimitedProvider
from mosaic.llm.base import BaseLLMProvider
from mosaic.llm.models import LLMError, LLMRequest, LLMResponse, LLMResponseError


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
    """Verify provider properties and non-retryable exceptions propagate unchanged without modifying LLMRequest."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.provider_name = "gemini"
    mock_provider.model_name = "gemini-3.6-flash"
    mock_provider.generate.side_effect = LLMError("General failure")

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
    with pytest.raises(LLMError, match="General failure"):
        limiter.generate(req)

    # Confirm LLMRequest was passed unchanged
    assert req.system_prompt == "sys"
    assert req.user_prompt == "usr"
    assert req.temperature == 0.0


# --- TRANSIENT 503 RETRY POLICY TESTS ---

def test_retry_503_succeeds_on_first_retry():
    """Verify 503 error succeeds on first retry (attempt 1)."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_response = LLMResponse(content="ok", provider_name="gemini", model_name="gemini-3.6-flash")
    mock_provider.generate.side_effect = [LLMError("Gemini API request failed: 503 UNAVAILABLE"), mock_response]

    # time_func called 2 times per attempt loop (now before sleep check, _last_request_time recording)
    mock_time = MagicMock(side_effect=[100.0, 100.0, 105.0, 105.0])
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        max_retries=3,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    res = limiter.generate(req)

    assert res == mock_response
    assert mock_provider.generate.call_count == 2
    mock_sleep.assert_any_call(2.0)  # Backoff delay 2s


def test_retry_503_succeeds_on_second_retry():
    """Verify 503 error succeeds on second retry (attempt 2)."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_response = LLMResponse(content="ok", provider_name="gemini", model_name="gemini-3.6-flash")
    mock_provider.generate.side_effect = [
        LLMError("503 Server Error"),
        LLMError("503 UNAVAILABLE"),
        mock_response,
    ]

    mock_time = MagicMock(side_effect=[100.0, 100.0, 105.0, 105.0, 110.0, 110.0])
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        max_retries=3,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    res = limiter.generate(req)

    assert res == mock_response
    assert mock_provider.generate.call_count == 3
    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(4.0)


def test_retry_503_succeeds_on_third_retry():
    """Verify 503 error succeeds on third retry (attempt 3)."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_response = LLMResponse(content="ok", provider_name="gemini", model_name="gemini-3.6-flash")
    mock_provider.generate.side_effect = [
        LLMError("503 Server Error"),
        LLMError("503 UNAVAILABLE"),
        LLMError("503 UNAVAILABLE"),
        mock_response,
    ]

    mock_time = MagicMock(side_effect=[100.0, 100.0, 105.0, 105.0, 110.0, 110.0, 115.0, 115.0])
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        max_retries=3,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    res = limiter.generate(req)

    assert res == mock_response
    assert mock_provider.generate.call_count == 4
    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(4.0)
    mock_sleep.assert_any_call(8.0)


def test_retry_503_fails_after_max_retries():
    """Verify 503 error raises exception after max retries (3 retries = 4 attempts total)."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.generate.side_effect = LLMError("Gemini API request failed: 503 UNAVAILABLE")

    mock_time = MagicMock(side_effect=[100.0, 100.0, 105.0, 105.0, 110.0, 110.0, 115.0, 115.0])
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        max_retries=3,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    with pytest.raises(LLMError, match="503 UNAVAILABLE"):
        limiter.generate(req)

    assert mock_provider.generate.call_count == 4


def test_retry_policy_ignores_429_quota_error():
    """Verify 429 RESOURCE_EXHAUSTED error is NOT retried (hard stop)."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.generate.side_effect = LLMError("Gemini API request failed: 429 RESOURCE_EXHAUSTED")

    mock_time = MagicMock(return_value=100.0)
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        max_retries=3,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    with pytest.raises(LLMError, match="429 RESOURCE_EXHAUSTED"):
        limiter.generate(req)

    assert mock_provider.generate.call_count == 1
    mock_sleep.assert_not_called()


def test_retry_policy_ignores_client_errors():
    """Verify 400/401/403/404 client errors are NOT retried."""
    for status_code in ["400 Bad Request", "401 Unauthorized", "403 Forbidden", "404 Not Found"]:
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.generate.side_effect = LLMError(f"Gemini API request failed: {status_code}")

        mock_time = MagicMock(return_value=100.0)
        mock_sleep = MagicMock()

        limiter = EvaluationRateLimitedProvider(
            provider=mock_provider,
            min_interval_seconds=13.0,
            max_retries=3,
            time_func=mock_time,
            sleep_func=mock_sleep,
        )

        req = LLMRequest(system_prompt="sys", user_prompt="usr")
        with pytest.raises(LLMError, match=status_code):
            limiter.generate(req)

        assert mock_provider.generate.call_count == 1
        mock_sleep.assert_not_called()


def test_retry_policy_ignores_malformed_json_and_validation_errors():
    """Verify malformed JSON and Pydantic validation errors are NOT retried."""
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.generate.side_effect = LLMResponseError("LLM response malformed JSON")

    mock_time = MagicMock(return_value=100.0)
    mock_sleep = MagicMock()

    limiter = EvaluationRateLimitedProvider(
        provider=mock_provider,
        min_interval_seconds=13.0,
        max_retries=3,
        time_func=mock_time,
        sleep_func=mock_sleep,
    )

    req = LLMRequest(system_prompt="sys", user_prompt="usr")
    with pytest.raises(LLMResponseError, match="LLM response malformed JSON"):
        limiter.generate(req)

    assert mock_provider.generate.call_count == 1
    mock_sleep.assert_not_called()
