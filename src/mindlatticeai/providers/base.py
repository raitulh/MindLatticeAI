"""Provider contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mindlatticeai.schemas import AIRequest, ModelProfile, ModelResponse


class ModelProvider(ABC):
    """Base class for model providers."""

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    @abstractmethod
    async def complete(self, prompt: str, request: AIRequest) -> ModelResponse:
        """Generate a completion for a rendered prompt."""


class ProviderNotConfiguredError(RuntimeError):
    """Raised when a provider needs credentials or an SDK that is missing."""

