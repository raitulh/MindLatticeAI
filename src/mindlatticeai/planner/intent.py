"""Intent parsing for incoming requests."""

from __future__ import annotations

import re

from mindlatticeai.schemas import AIRequest, Intent, TaskType


class IntentParser:
    """Lightweight parser that can later be replaced by a learned classifier."""

    CALCULATOR_PATTERN = re.compile(r"^[\s\d\.\+\-\*\/\(\)\^]+$")

    def parse(self, request: AIRequest) -> Intent:
        text = request.input.strip()
        lowered = text.lower()
        tools: list[str] = []
        task_type = request.task_type or TaskType.CHAT

        normalized_math = text.replace("^", "**")
        if (
            "calculate" in lowered
            or "sum" in lowered
            or "multiply" in lowered
            or self.CALCULATOR_PATTERN.match(normalized_math)
        ):
            tools.append("calculator")
            task_type = TaskType.TOOL

        if "time" in lowered or "date" in lowered:
            tools.append("utc_time")
            task_type = TaskType.TOOL

        if "search" in lowered or "retrieve" in lowered or "rag" in lowered:
            tools.append("local_search")
            task_type = TaskType.RAG

        if any(word in lowered for word in ["prove", "reason", "derive", "analyze"]):
            task_type = TaskType.REASONING

        risk_level = "elevated" if any(word in lowered for word in ["secret", "password", "key"]) else "normal"
        return Intent(
            task_type=task_type,
            goals=[text],
            required_tools=tools,
            risk_level=risk_level,
            constraints=request.constraints,
            confidence=0.80 if tools else 0.65,
        )

