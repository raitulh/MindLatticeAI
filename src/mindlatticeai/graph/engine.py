"""Async DAG execution engine with retries and checkpointing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from mindlatticeai.schemas import ExecutionGraph, ExecutionState, GraphNode, NodeType, now_utc

NodeHandler = Callable[[GraphNode, dict[str, Any]], Awaitable[Any]]


class GraphExecutionError(RuntimeError):
    """Raised when the graph cannot finish successfully."""


class CheckpointStore:
    """In-memory checkpoint store.

    Production deployments should replace this with durable storage keyed by
    request fingerprint, graph id, node id, and strategy version.
    """

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def key(self, graph_id: str, node_id: str) -> str:
        return f"{graph_id}:{node_id}"

    def get(self, graph_id: str, node_id: str) -> Any | None:
        return self._items.get(self.key(graph_id, node_id))

    def save(self, graph_id: str, node_id: str, value: Any) -> str:
        key = self.key(graph_id, node_id)
        self._items[key] = value
        return key


class ExecutionGraphEngine:
    """Executes a DAG and records node state transitions."""

    def __init__(self, checkpoints: CheckpointStore | None = None, *, enabled: bool = True) -> None:
        self.checkpoints = checkpoints or CheckpointStore()
        self.enabled = enabled

    async def execute(
        self,
        graph: ExecutionGraph,
        handlers: dict[NodeType, NodeHandler],
        context: dict[str, Any],
    ) -> ExecutionGraph:
        node_map = graph.node_map()
        self._validate_dependencies(graph, node_map)

        while True:
            pending = [node for node in graph.nodes if node.state in {ExecutionState.PENDING, ExecutionState.RETRYING}]
            if not pending:
                break

            ready = [
                node
                for node in pending
                if all(node_map[dep].state == ExecutionState.SUCCEEDED for dep in node.dependencies)
            ]
            if not ready:
                failed = [node for node in graph.nodes if node.state == ExecutionState.FAILED]
                if failed:
                    raise GraphExecutionError(f"Graph failed at node {failed[0].id}: {failed[0].error}")
                waiting = ", ".join(node.id for node in pending)
                raise GraphExecutionError(f"Graph has a cycle or unsatisfied dependency among: {waiting}")

            await asyncio.gather(*(self._run_node(graph, node, handlers, context) for node in ready))

        failed_nodes = [node for node in graph.nodes if node.state == ExecutionState.FAILED]
        if failed_nodes:
            raise GraphExecutionError(f"Graph failed at node {failed_nodes[0].id}: {failed_nodes[0].error}")
        return graph

    async def _run_node(
        self,
        graph: ExecutionGraph,
        node: GraphNode,
        handlers: dict[NodeType, NodeHandler],
        context: dict[str, Any],
    ) -> None:
        checkpoint = self.checkpoints.get(graph.id, node.id) if self.enabled else None
        if checkpoint is not None:
            node.state = ExecutionState.SUCCEEDED
            node.result = checkpoint
            node.checkpoint_key = self.checkpoints.key(graph.id, node.id)
            context.setdefault("node_results", {})[node.id] = checkpoint
            return

        handler = handlers.get(node.node_type)
        if handler is None:
            raise GraphExecutionError(f"No handler registered for node type {node.node_type}")

        max_attempts = max(1, node.max_retries + 1)
        for attempt in range(max_attempts):
            node.attempts += 1
            node.started_at = now_utc()
            node.state = ExecutionState.RUNNING if attempt == 0 else ExecutionState.RETRYING
            try:
                result = await handler(node, context)
                node.result = result
                node.error = None
                node.ended_at = now_utc()
                node.state = ExecutionState.SUCCEEDED
                node.checkpoint_key = self.checkpoints.save(graph.id, node.id, result) if self.enabled else None
                context.setdefault("node_results", {})[node.id] = result
                return
            except Exception as exc:
                node.error = str(exc)
                node.ended_at = now_utc()
                if attempt + 1 >= max_attempts:
                    node.state = ExecutionState.FAILED
                    context.setdefault("failures", []).append({"node_id": node.id, "error": str(exc)})
                    return

    def _validate_dependencies(self, graph: ExecutionGraph, node_map: dict[str, GraphNode]) -> None:
        for node in graph.nodes:
            missing = [dependency for dependency in node.dependencies if dependency not in node_map]
            if missing:
                raise GraphExecutionError(f"Node {node.id} has missing dependencies: {missing}")

