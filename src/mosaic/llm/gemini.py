"""MOSAIC Google Gemini LLM Provider Implementation.

Provides a Google Gemini provider conforming to BaseLLMProvider.
Uses the official google-genai SDK (`from google import genai`).
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

from mosaic.llm.base import BaseLLMProvider
from mosaic.llm.models import (
    LLMConfigurationError,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
)

logger = logging.getLogger("mosaic.llm.gemini")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider implementing BaseLLMProvider via google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        client: Optional[genai.Client] = None,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key and client is None:
            raise LLMConfigurationError("Gemini API key is missing. Set GEMINI_API_KEY in environment or settings.")

        self._model_name = model_name or "gemini-2.5-flash"
        
        if client is not None:
            self.client = client
        else:
            self.client = genai.Client(api_key=key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Executes LLM generation synchronously via Google GenAI SDK.

        Args:
            request: Standardized LLMRequest container.

        Returns:
            Standardized LLMResponse object.

        Raises:
            LLMError: On API call failure.
            LLMResponseError: On malformed or invalid JSON/structured response.
        """
        # Combine system prompt and user prompt via GenerateContentConfig
        config_kwargs: Dict[str, Any] = {
            "temperature": request.temperature,
            "system_instruction": request.system_prompt,
        }

        if request.max_tokens is not None:
            config_kwargs["max_output_tokens"] = request.max_tokens

        # Configure structured JSON output if schema requested
        if request.structured_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=request.user_prompt,
                config=config,
            )
        except APIError as e:
            # Clean error string to prevent any potential API key leakage
            err_msg = str(e).replace(self.client._api_key if hasattr(self.client, "_api_key") and self.client._api_key else "", "[REDACTED]")
            logger.error("Gemini API error during generation: %s", err_msg)
            raise LLMError(f"Gemini API request failed: {err_msg}") from e
        except Exception as e:
            err_msg = str(e)
            logger.error("Unexpected error communicating with Gemini API: %s", err_msg)
            raise LLMError(f"Unexpected error communicating with Gemini API: {err_msg}") from e

        if not response or not hasattr(response, "text") or response.text is None:
            raise LLMResponseError("Gemini API returned empty response text.")

        raw_content = response.text.strip()
        structured_output: Optional[Dict[str, Any]] = None

        if request.structured_schema is not None:
            try:
                structured_output = json.loads(raw_content)
            except Exception as err:
                raise LLMResponseError(
                    f"Failed to parse structured JSON output from Gemini response: {err}"
                ) from err

        usage_info: Dict[str, Any] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            candidates_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": candidates_tokens,
                "total_tokens": prompt_tokens + candidates_tokens,
            }

        finish_reason = "stop"

        return LLMResponse(
            content=raw_content,
            structured_output=structured_output,
            provider_name=self.provider_name,
            model_name=self.model_name,
            usage_info=usage_info,
            finish_reason=finish_reason,
        )
