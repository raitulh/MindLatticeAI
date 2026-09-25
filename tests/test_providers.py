import pytest

from mindlatticeai.providers import (
    ClaudeProvider,
    DeepSeekProvider,
    GeminiProvider,
    LocalHeuristicProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderNotConfiguredError,
)
from mindlatticeai.schemas import AIRequest


@pytest.mark.asyncio
async def test_local_provider_completes() -> None:
    provider = LocalHeuristicProvider()
    response = await provider.complete("Test prompt", AIRequest(input="What is MindLattice?"))
    assert response.text is not None
    assert response.latency_ms >= 0
    assert response.model_profile.provider == "local"


@pytest.mark.asyncio
async def test_network_providers_raise_not_configured_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    openai = OpenAIProvider()
    gemini = GeminiProvider()
    claude = ClaudeProvider()
    deepseek = DeepSeekProvider()

    req = AIRequest(input="Test")
    with pytest.raises(ProviderNotConfiguredError):
        await openai.complete("Prompt", req)

    with pytest.raises(ProviderNotConfiguredError):
        await gemini.complete("Prompt", req)

    with pytest.raises(ProviderNotConfiguredError):
        await claude.complete("Prompt", req)

    with pytest.raises(ProviderNotConfiguredError):
        await deepseek.complete("Prompt", req)
