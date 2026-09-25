"""Network-backed model provider implementations.

These classes provide ready-to-use API integrations for major LLM providers
(OpenAI, Google Gemini, Anthropic Claude, Ollama, DeepSeek) using standard
library HTTP utilities, avoiding mandatory heavyweight external SDKs.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

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

    def _get_api_key(self) -> str:
        key = os.getenv(self.env_key_name, "").strip()
        if not key:
            raise ProviderNotConfiguredError(
                f"Missing environment variable {self.env_key_name} for {self.profile.id}. "
                f"Please set {self.env_key_name} to enable completions."
            )
        return key

    def _http_post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Provider {self.profile.id} HTTP {exc.code} error: {err_body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Provider {self.profile.id} request failed: {exc}") from exc


class OpenAIProvider(_EnvironmentProvider):
    env_key_name = "OPENAI_API_KEY"

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        super().__init__(provider="openai", model=model, quality_score=0.86)

    async def complete(self, prompt: str, request: AIRequest) -> ModelResponse:
        api_key = self._get_api_key()
        start = time.perf_counter()

        def _call() -> dict[str, Any]:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.profile.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            return self._http_post_json(url, headers, payload)

        res = await asyncio.to_thread(_call)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        choice = res.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = res.get("usage", {})

        return ModelResponse(
            text=text,
            model_profile=self.profile,
            input_tokens=usage.get("prompt_tokens", len(prompt.split())),
            output_tokens=usage.get("completion_tokens", len(text.split())),
            latency_ms=elapsed_ms,
        )


class GeminiProvider(_EnvironmentProvider):
    env_key_name = "GEMINI_API_KEY"

    def __init__(self, model: str = "gemini-1.5-pro") -> None:
        super().__init__(provider="gemini", model=model, quality_score=0.84)

    async def complete(self, prompt: str, request: AIRequest) -> ModelResponse:
        api_key = self._get_api_key()
        start = time.perf_counter()

        def _call() -> dict[str, Any]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.profile.model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ]
            }
            return self._http_post_json(url, headers, payload)

        res = await asyncio.to_thread(_call)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        candidates = res.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", [{}]) if candidates else [{}]
        text = parts[0].get("text", "")
        usage = res.get("usageMetadata", {})

        return ModelResponse(
            text=text,
            model_profile=self.profile,
            input_tokens=usage.get("promptTokenCount", len(prompt.split())),
            output_tokens=usage.get("candidatesTokenCount", len(text.split())),
            latency_ms=elapsed_ms,
        )


class ClaudeProvider(_EnvironmentProvider):
    env_key_name = "ANTHROPIC_API_KEY"

    def __init__(self, model: str = "claude-3-5-sonnet") -> None:
        super().__init__(provider="anthropic", model=model, quality_score=0.87)

    async def complete(self, prompt: str, request: AIRequest) -> ModelResponse:
        api_key = self._get_api_key()
        start = time.perf_counter()

        def _call() -> dict[str, Any]:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self.profile.model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
            return self._http_post_json(url, headers, payload)

        res = await asyncio.to_thread(_call)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        content = res.get("content", [{}])
        text = content[0].get("text", "") if content else ""
        usage = res.get("usage", {})

        return ModelResponse(
            text=text,
            model_profile=self.profile,
            input_tokens=usage.get("input_tokens", len(prompt.split())),
            output_tokens=usage.get("output_tokens", len(text.split())),
            latency_ms=elapsed_ms,
        )


class OllamaProvider(_EnvironmentProvider):
    env_key_name = "OLLAMA_HOST"

    def __init__(self, model: str = "llama3.1") -> None:
        super().__init__(provider="ollama", model=model, quality_score=0.70)
        # Ollama can default to localhost if not explicitly specified
        if not os.getenv("OLLAMA_HOST"):
            self.profile.availability = 1.0

    async def complete(self, prompt: str, request: AIRequest) -> ModelResponse:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        start = time.perf_counter()

        def _call() -> dict[str, Any]:
            url = f"{host}/api/generate"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.profile.model,
                "prompt": prompt,
                "stream": False,
            }
            return self._http_post_json(url, headers, payload)

        res = await asyncio.to_thread(_call)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        text = res.get("response", "")

        return ModelResponse(
            text=text,
            model_profile=self.profile,
            input_tokens=res.get("prompt_eval_count", len(prompt.split())),
            output_tokens=res.get("eval_count", len(text.split())),
            latency_ms=elapsed_ms,
        )


class DeepSeekProvider(_EnvironmentProvider):
    env_key_name = "DEEPSEEK_API_KEY"

    def __init__(self, model: str = "deepseek-chat") -> None:
        super().__init__(provider="deepseek", model=model, quality_score=0.82)

    async def complete(self, prompt: str, request: AIRequest) -> ModelResponse:
        api_key = self._get_api_key()
        start = time.perf_counter()

        def _call() -> dict[str, Any]:
            url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.profile.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            return self._http_post_json(url, headers, payload)

        res = await asyncio.to_thread(_call)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        choice = res.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = res.get("usage", {})

        return ModelResponse(
            text=text,
            model_profile=self.profile,
            input_tokens=usage.get("prompt_tokens", len(prompt.split())),
            output_tokens=usage.get("completion_tokens", len(text.split())),
            latency_ms=elapsed_ms,
        )
