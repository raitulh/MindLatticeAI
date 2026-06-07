"""Deterministic local provider used for tests and offline development."""

from __future__ import annotations

import re
import time

from mindlatticeai.providers.base import ModelProvider
from mindlatticeai.schemas import AIRequest, ModelProfile, ModelResponse, TaskType


class LocalHeuristicProvider(ModelProvider):
    """A no-network provider that makes the framework runnable out of the box."""

    def __init__(self) -> None:
        super().__init__(
            ModelProfile(
                provider="local",
                model="heuristic-v1",
                capabilities=[
                    TaskType.CHAT,
                    TaskType.REASONING,
                    TaskType.RAG,
                    TaskType.TOOL,
                    TaskType.AGENT,
                    TaskType.BENCHMARK,
                ],
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                avg_latency_ms=40,
                quality_score=0.55,
                availability=1.0,
            )
        )

    async def complete(self, prompt: str, request: AIRequest) -> ModelResponse:
        started = time.perf_counter()
        observations = self._extract_observations(prompt)
        answer = self._compose_answer(request.input, observations)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ModelResponse(
            text=answer,
            model_profile=self.profile,
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(answer.split())),
            latency_ms=elapsed_ms,
        )

    def _extract_observations(self, prompt: str) -> str | None:
        match = re.search(r"Tool observations:\s*(.+?)\n\nOutput format:", prompt, re.S)
        if not match:
            return None
        return match.group(1).strip()

    def _compose_answer(self, user_input: str, observations: str | None) -> str:
        lines = [
            "MindLatticeAI local runtime response",
            f"Interpreted request: {user_input}",
        ]
        if observations and observations != "{}":
            lines.append(f"Tool-backed observation: {observations}")
        lines.append(
            "Answer: This response was produced by the offline provider after "
            "planning, policy checks, DAG execution, reflection, evaluation, "
            "learning, and memory update."
        )
        return "\n".join(lines)

