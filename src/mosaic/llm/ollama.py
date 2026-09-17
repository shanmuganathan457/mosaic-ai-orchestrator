"""MOSAIC Ollama Local LLM Provider Implementation.

Provides an HTTP-based local Ollama provider conforming to BaseLLMProvider.
Uses standard library urllib.request for zero external framework overhead.
"""

import json
import logging
from typing import Any, Dict, Optional
import urllib.error
import urllib.request

from mosaic.llm.base import BaseLLMProvider
from mosaic.llm.models import (
    LLMConfigurationError,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
)

logger = logging.getLogger("mosaic.llm.ollama")


class OllamaProvider(BaseLLMProvider):
    """Local LLM Provider communicating with Ollama server via HTTP REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2",
        timeout: float = 60.0,
    ) -> None:
        if not base_url:
            raise LLMConfigurationError("Ollama base_url cannot be empty.")
        if not model_name:
            raise LLMConfigurationError("Ollama model_name cannot be empty.")

        self.base_url = base_url.rstrip("/")
        self._model_name = model_name
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Executes LLM generation synchronously via Ollama /api/generate endpoint.

        Args:
            request: Standardized LLMRequest container.

        Returns:
            Standardized LLMResponse container.

        Raises:
            LLMError: On connection failure, timeout, or malformed/invalid HTTP response.
        """
        endpoint = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "system": request.system_prompt,
            "prompt": request.user_prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens

        if request.structured_schema is not None:
            payload["format"] = "json"

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status != 200:
                    raise LLMResponseError(
                        f"Ollama server returned non-200 HTTP status code: {response.status}"
                    )
                resp_bytes = response.read()
        except urllib.error.HTTPError as err:
            raise LLMResponseError(
                f"Ollama HTTP error {err.code}: {err.reason}"
            ) from err
        except urllib.error.URLError as err:
            raise LLMError(
                f"Failed to connect to Ollama server at {endpoint}: {err.reason}"
            ) from err
        except TimeoutError as err:
            raise LLMError(
                f"Ollama request timed out after {self.timeout} seconds"
            ) from err
        except Exception as err:
            raise LLMError(f"Unexpected error communicating with Ollama: {err}") from err

        try:
            resp_json = json.loads(resp_bytes.decode("utf-8"))
        except Exception as err:
            raise LLMResponseError(
                f"Failed to parse JSON response from Ollama: {err}"
            ) from err

        raw_content = resp_json.get("response", "")
        if not raw_content and not isinstance(raw_content, str):
            raise LLMResponseError("Ollama response missing text content.")

        structured_output: Optional[Dict[str, Any]] = None
        if request.structured_schema is not None:
            try:
                structured_output = json.loads(raw_content)
            except Exception as err:
                raise LLMResponseError(
                    f"Failed to parse structured JSON output from Ollama content: {err}"
                ) from err

        done_reason = "stop" if resp_json.get("done", True) else "length"
        eval_count = resp_json.get("eval_count", 0)
        prompt_eval_count = resp_json.get("prompt_eval_count", 0)

        usage_info = {
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count,
            "total_duration": resp_json.get("total_duration"),
        }
        self.record_usage(usage_info)

        return LLMResponse(
            content=raw_content,
            structured_output=structured_output,
            provider_name=self.provider_name,
            model_name=self.model_name,
            usage_info=usage_info,
            finish_reason=done_reason,
        )
