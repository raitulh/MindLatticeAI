"""Compiles strategies into execution DAGs."""

from __future__ import annotations

from mindlatticeai.schemas import ExecutionGraph, GraphEdge, GraphNode, PolicyDecision, Strategy


class ExecutionGraphBuilder:
    """Turns a policy-approved strategy into a checkpointable DAG."""

    def build(self, *, request_id: str, strategy: Strategy, policy: PolicyDecision) -> ExecutionGraph:
        nodes = [
            GraphNode(
                id=step.id,
                node_type=step.node_type,
                name=step.name,
                payload=step.payload,
                dependencies=step.dependencies,
                max_retries=min(step.max_retries, policy.budgets.max_retries),
            )
            for step in strategy.steps
        ]
        edges = [
            GraphEdge(source=dependency, target=step.id)
            for step in strategy.steps
            for dependency in step.dependencies
        ]
        return ExecutionGraph(
            request_id=request_id,
            nodes=nodes,
            edges=edges,
            metadata={"strategy_id": strategy.id, "policy_reasons": policy.reasons},
        )

