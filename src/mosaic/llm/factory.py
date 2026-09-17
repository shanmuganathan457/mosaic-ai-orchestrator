"""MOSAIC LLM Provider Registry and Factory.

Provides a centralized factory to register and resolve BaseLLMProvider instances.
"""

from typing import Callable, Dict, Type
from mosaic.config.settings import settings
from mosaic.llm.base import BaseLLMProvider
from mosaic.llm.mock import MockLLMProvider
from mosaic.llm.models import LLMConfigurationError
from mosaic.llm.ollama import OllamaProvider


class LLMProviderFactory:
    """Registry and Factory for MOSAIC LLM Providers."""

    _registry: Dict[str, Callable[..., BaseLLMProvider]] = {}

    @classmethod
    def register_provider(cls, name: str, factory_fn: Callable[..., BaseLLMProvider]) -> None:
        """Registers a provider factory function under a name."""
        cls._registry[name.lower()] = factory_fn

    @classmethod
    def get_provider(
        cls,
        provider_name: str | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Instantiates and returns an LLM provider based on name or environment settings.

        Args:
            provider_name: Optional provider key ('mock', 'ollama'). Defaults to settings.LLM_PROVIDER.
            model_name: Optional model override. Defaults to settings.LLM_MODEL.
            **kwargs: Additional provider-specific keyword arguments.

        Returns:
            An initialized BaseLLMProvider subclass instance.

        Raises:
            LLMConfigurationError: If requested provider is unregistered.
        """
        p_name = (provider_name or settings.LLM_PROVIDER).lower()
        m_name = model_name or settings.LLM_MODEL

        if p_name == "mock":
            return MockLLMProvider(model_name=m_name, **kwargs)
        elif p_name == "ollama":
            base_url = kwargs.pop("base_url", settings.OLLAMA_BASE_URL)
            timeout = kwargs.pop("timeout", settings.LLM_TIMEOUT)
            # Default model name for Ollama if setting is mock default
            if m_name == "mock-deterministic-v1":
                m_name = "llama3.2"
            return OllamaProvider(base_url=base_url, model_name=m_name, timeout=timeout)
        elif p_name in cls._registry:
            return cls._registry[p_name](model_name=m_name, **kwargs)
        else:
            supported = ["mock", "ollama"] + list(cls._registry.keys())
            raise LLMConfigurationError(
                f"Unsupported LLM provider '{p_name}'. Supported providers: {supported}"
            )
