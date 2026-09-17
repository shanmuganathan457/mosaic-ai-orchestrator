"""MOSAIC Abstract LLM Provider Interface.

Defines the abstract BaseLLMProvider class establishing a strict contract
for provider implementations (Mock, Ollama, Gemini, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
from mosaic.llm.models import LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    """Abstract Base Class for all MOSAIC LLM provider implementations."""

    def __init__(self) -> None:
        self._usage_history: list[dict] = []

    def record_usage(self, usage_info: dict) -> None:
        """Records token usage info from a generation call."""
        if not hasattr(self, "_usage_history"):
            self._usage_history = []
        if usage_info:
            self._usage_history.append(usage_info)

    def pop_recorded_usage(self) -> dict:
        """Consolidates and returns accumulated token usage since last pop, then clears history."""
        if not hasattr(self, "_usage_history") or not self._usage_history:
            return {}

        history = self._usage_history
        self._usage_history = []

        valid_usages = [u for u in history if isinstance(u, dict) and "total_tokens" in u]
        if not valid_usages:
            return {}

        prompt_tokens = sum(u.get("prompt_tokens", 0) for u in valid_usages)
        completion_tokens = sum(u.get("completion_tokens", 0) for u in valid_usages)
        total_tokens = sum(u.get("total_tokens", 0) for u in valid_usages)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier of the LLM provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the active model backing this provider instance."""
        pass

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Executes LLM generation synchronously and returns a standardized LLMResponse.

        Args:
            request: Standardized LLMRequest container.

        Returns:
            Standardized LLMResponse object.

        Raises:
            LLMError: On execution failure.
        """
        pass

