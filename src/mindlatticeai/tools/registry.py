"""Tool registry and execution layer."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mindlatticeai.schemas import ToolResult

ToolHandler = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    handler: ToolHandler
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Named tool catalog with async execution support."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    async def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        started = time.perf_counter()
        try:
            value = tool.handler(arguments)
            if inspect.isawaitable(value):
                value = await value
            return ToolResult(
                tool_name=name,
                output=value,
                latency_ms=int((time.perf_counter() - started) * 1000),
                cost_usd=tool.cost_usd,
                metadata=tool.metadata,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=name,
                output=None,
                success=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                cost_usd=tool.cost_usd,
                error=str(exc),
                metadata=tool.metadata,
            )

