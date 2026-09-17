"""MOSAIC Deterministic Mock LLM Provider Implementation.

Provides a 100% offline, deterministic mock implementation of BaseLLMProvider
for unit testing and pipeline validation without network or GPU requirements.
"""

import json
import logging
from typing import Any, Dict, Optional

from mosaic.llm.base import BaseLLMProvider
from mosaic.llm.models import LLMRequest, LLMResponse

logger = logging.getLogger("mosaic.llm.mock")


class MockLLMProvider(BaseLLMProvider):
    """Deterministic Mock LLM Provider for offline testing."""

    def __init__(
        self,
        model_name: str = "mock-deterministic-v1",
        default_response_text: str = "Mock LLM response generated successfully.",
        fixed_structured_output: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model_name = model_name
        self.default_response_text = default_response_text
        self.fixed_structured_output = fixed_structured_output

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generates a deterministic LLMResponse based on request payload."""
        logger.debug("MockLLMProvider processing request with user prompt snippet: '%s'", request.user_prompt[:50])

        structured_output = self.fixed_structured_output

        # If a structured schema was requested and no fixed mock output was set, provide a simple schema-conforming dict
        if request.structured_schema and structured_output is None:
            structured_output = {"intents": []}

        content = self.default_response_text
        if structured_output is not None and self.default_response_text == "Mock LLM response generated successfully.":
            content = json.dumps(structured_output)

        usage_info = {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        self.record_usage(usage_info)

        return LLMResponse(
            content=content,
            structured_output=structured_output,
            provider_name=self.provider_name,
            model_name=self.model_name,
            usage_info=usage_info,
            finish_reason="stop",
        )
