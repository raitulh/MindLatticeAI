from mindlatticeai import Agent
from mindlatticeai.schemas import ExecutionState


def test_agent_runs_full_tool_loop() -> None:
    agent = Agent()
    response = agent.run("Calculate 2 + 2")

    assert "4.0" in response.text
    assert response.metrics.success is True
    assert response.metrics.model_id == "local:heuristic-v1"
    assert response.replay_id is not None
    assert response.graph is not None
    assert all(node.state == ExecutionState.SUCCEEDED for node in response.graph.nodes)
    assert "digraph" in response.artifacts["graph_dot"]


def test_learning_updates_model_performance() -> None:
    agent = Agent()
    response = agent.run("Explain adaptive routing briefly.")

    assert response.metrics.success is True
    assert "local:heuristic-v1" in agent.learning.model_performance

