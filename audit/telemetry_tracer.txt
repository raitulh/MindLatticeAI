"""Telemetry and trace capture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mindlatticeai.schemas import ExecutionGraph, ExecutionTrace, TraceEvent, now_utc


class ExecutionTracer:
    """Collects execution traces and can export JSONL."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.traces: dict[str, ExecutionTrace] = {}

    def start(self, request_id: str) -> ExecutionTrace:
        trace = ExecutionTrace(request_id=request_id)
        self.traces[trace.id] = trace
        self.event(trace.id, "trace.started", {"request_id": request_id})
        return trace

    def event(self, trace_id: str, name: str, payload: dict[str, Any] | None = None) -> None:
        trace = self.traces[trace_id]
        trace.events.append(TraceEvent(name=name, payload=payload or {}))

    def finish(self, trace_id: str) -> ExecutionTrace:
        trace = self.traces[trace_id]
        trace.ended_at = now_utc()
        self.event(trace_id, "trace.finished", {})
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace.model_dump(mode="json")) + "\n")
        return trace

    def graph_to_dot(self, graph: ExecutionGraph) -> str:
        lines = [f'digraph "{graph.id}" {{']
        for node in graph.nodes:
            lines.append(f'  "{node.id}" [label="{node.name}\\n{node.state.value}"];')
        for edge in graph.edges:
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{edge.label}"];')
        lines.append("}")
        return "\n".join(lines)

