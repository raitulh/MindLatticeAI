"""Replay capture and debugging helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mindlatticeai.schemas import AIRequest, AIResponse, ExecutionGraph, ExecutionTrace, new_id


class ReplayEngine:
    """Stores execution snapshots for replay and debugging."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._items: dict[str, dict[str, Any]] = {}

    def record(
        self,
        *,
        request: AIRequest,
        response: AIResponse,
        graph: ExecutionGraph,
        trace: ExecutionTrace,
    ) -> str:
        replay_id = new_id("replay")
        payload = {
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json", exclude={"graph"}),
            "graph": graph.model_dump(mode="json"),
            "trace": trace.model_dump(mode="json"),
        }
        self._items[replay_id] = payload
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / f"{replay_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return replay_id

    def get(self, replay_id: str) -> dict[str, Any]:
        if replay_id in self._items:
            return self._items[replay_id]
        if self.path:
            path = self.path / f"{replay_id}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        raise KeyError(f"Unknown replay id: {replay_id}")

    def summarize(self, replay_id: str) -> dict[str, Any]:
        payload = self.get(replay_id)
        graph = payload["graph"]
        return {
            "request_id": payload["request"]["id"],
            "graph_id": graph["id"],
            "nodes": [
                {"id": node["id"], "state": node["state"], "attempts": node["attempts"]}
                for node in graph["nodes"]
            ],
            "quality_score": payload["response"]["metrics"]["quality_score"],
        }

