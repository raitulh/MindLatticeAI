"""Optional provider stubs.

These classes define the extension points for network-backed providers without
forcing heavyweight SDK dependencies into the core package.
"""

from __future__ import annotations

import os

from mindlatticeai.providers.base import ModelProvider, ProviderNotConfiguredError
from mindlatticeai.schemas import AIRequest, ModelProfile, ModelResponse, TaskType


class _EnvironmentProvider(ModelProvider):
    env_key_name: str = ""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        capabilities: list[TaskType] | None = None,
        quality_score: float = 0.80,
        avg_latency_ms: int = 1200,
    ) -> None:
        super().__init__(
            ModelProfile(
                provider=provider,
                model=model,
                capabilities=capabilities
                or [TaskType.CHAT, TaskType.REASONING, TaskType.RAG, TaskType.AGENT],
                avg_latency_ms=avg_latency_ms,
                quality_score=quality_score,
                availability=1.0 if os.getenv(self.env_key_name) else 0.0,
            )
        )

    async def complete(self, prompt: str, request: AIRequest) -> ModelResponse:
        raise ProviderNotConfiguredError(
            f"{self.profile.id} is a provider contract. Install the provider SDK "
            f"and set {self.env_key_name} before enabling network completions."
        )


class OpenAIProvider(_EnvironmentProvider):
    env_key_name = "OPENAI_API_KEY"

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        super().__init__(provider="openai", model=model, quality_score=0.86)


class GeminiProvider(_EnvironmentProvider):
    env_key_name = "GEMINI_API_KEY"

    def __init__(self, model: str = "gemini-1.5-pro") -> None:
        super().__init__(provider="gemini", model=model, quality_score=0.84)


class ClaudeProvider(_EnvironmentProvider):
    env_key_name = "ANTHROPIC_API_KEY"

    def __init__(self, model: str = "claude-3-5-sonnet") -> None:
        super().__init__(provider="anthropic", model=model, quality_score=0.87)


class OllamaProvider(_EnvironmentProvider):
    env_key_name = "OLLAMA_HOST"

    def __init__(self, model: str = "llama3.1") -> None:
        super().__init__(provider="ollama", model=model, quality_score=0.70)


class DeepSeekProvider(_EnvironmentProvider):
    env_key_name = "DEEPSEEK_API_KEY"

    def __init__(self, model: str = "deepseek-chat") -> None:
        super().__init__(provider="deepseek", model=model, quality_score=0.82)

