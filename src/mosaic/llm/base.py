"""MOSAIC Abstract LLM Provider Interface.

Defines the abstract BaseLLMProvider class establishing a strict contract
for provider implementations (Mock, Ollama, Gemini, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
from mosaic.llm.models import LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    """Abstract Base Class for all MOSAIC LLM provider implementations."""

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
