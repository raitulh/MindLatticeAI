import asyncio

from mindlatticeai.graph import CheckpointStore, ExecutionGraphEngine
from mindlatticeai.schemas import ExecutionGraph, ExecutionState, GraphNode, NodeType


def test_graph_retries_failed_node_then_succeeds() -> None:
    calls = {"count": 0}
    node = GraphNode(id="model.retry", node_type=NodeType.MODEL_CALL, name="retry", max_retries=1)
    graph = ExecutionGraph(request_id="req.test", nodes=[node])

    async def handler(graph_node, context):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient")
        return {"ok": True}

    engine = ExecutionGraphEngine(CheckpointStore())
    asyncio.run(engine.execute(graph, {NodeType.MODEL_CALL: handler}, {}))

    assert calls["count"] == 2
    assert graph.nodes[0].attempts == 2
    assert graph.nodes[0].state == ExecutionState.SUCCEEDED

