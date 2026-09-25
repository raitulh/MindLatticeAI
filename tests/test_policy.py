from mindlatticeai.policy import PolicyEngine
from mindlatticeai.schemas import AIRequest, Intent, NodeType, PlanStep, RuntimeBudget, Strategy, TaskType


def test_policy_engine_allows_valid_request() -> None:
    engine = PolicyEngine()
    request = AIRequest(input="What is 2+2?")
    intent = Intent(task_type=TaskType.CHAT)
    strategy = Strategy(
        name="test_strat",
        prompt_genome_id="genome.default.cognitive",
        steps=[PlanStep(id="step.1", node_type=NodeType.MODEL_CALL, name="step1", description="step1")],
    )

    decision = engine.evaluate(request, intent, strategy)
    assert decision.allowed is True
    assert "Policy checks passed." in decision.reasons


def test_policy_engine_blocks_disabled_tools() -> None:
    engine = PolicyEngine(disabled_tools={"calculator", "bash"})
    request = AIRequest(input="Calculate 10 * 10")
    intent = Intent(
        task_type=TaskType.TOOL,
        required_tools=["calculator"],
    )
    strategy = Strategy(
        name="test_tool_strat",
        prompt_genome_id="genome.default.cognitive",
        steps=[PlanStep(id="step.1", node_type=NodeType.TOOL_CALL, name="step1", description="step1")],
    )

    decision = engine.evaluate(request, intent, strategy)
    assert decision.allowed is False
    assert any("Disabled tools requested" in r for r in decision.reasons)


def test_policy_engine_flags_negative_budget() -> None:
    engine = PolicyEngine()
    request = AIRequest(input="Hello", budget=RuntimeBudget(max_cost_usd=-0.5))
    intent = Intent(task_type=TaskType.CHAT)
    strategy = Strategy(
        name="test_negative_budget",
        prompt_genome_id="genome.default.cognitive",
        steps=[PlanStep(id="step.1", node_type=NodeType.MODEL_CALL, name="step1", description="step1")],
    )

    decision = engine.evaluate(request, intent, strategy)
    assert decision.allowed is False
    assert any("Negative cost budget" in r for r in decision.reasons)
