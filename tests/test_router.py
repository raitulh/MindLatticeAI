import pytest

from mindlatticeai.router import ModelRouter
from mindlatticeai.schemas import Intent, ModelProfile, PolicyDecision, RuntimeBudget, TaskType


def test_model_router_selects_best_matching_model() -> None:
    fast_profile = ModelProfile(
        provider="test",
        model="fast-v1",
        capabilities=[TaskType.CHAT],
        quality_score=0.75,
        avg_latency_ms=200,
        availability=1.0,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
    )
    smart_profile = ModelProfile(
        provider="test",
        model="smart-v1",
        capabilities=[TaskType.CHAT, TaskType.REASONING],
        quality_score=0.95,
        avg_latency_ms=1500,
        availability=1.0,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
    )

    router = ModelRouter([fast_profile, smart_profile])
    intent = Intent(
        task_type=TaskType.REASONING,
        confidence=0.9,
    )
    policy = PolicyDecision(
        allowed=True,
        reasons=["ok"],
        budgets=RuntimeBudget(max_cost_usd=1.0, latency_budget_ms=3000),
        disabled_tools=[],
        routing_preferences={},
        metadata={},
    )

    selected = router.select(intent, policy)
    assert selected.model == "smart-v1"


def test_model_router_raises_when_no_model_supports_capability() -> None:
    profile = ModelProfile(
        provider="test",
        model="chat-only",
        capabilities=[TaskType.CHAT],
        availability=1.0,
    )
    router = ModelRouter([profile])
    intent = Intent(task_type=TaskType.BENCHMARK)
    policy = PolicyDecision(
        allowed=True,
        reasons=["ok"],
        budgets=RuntimeBudget(),
        disabled_tools=[],
        routing_preferences={},
        metadata={},
    )

    with pytest.raises(RuntimeError, match="No available model supports task type"):
        router.select(intent, policy)


def test_model_router_respects_low_latency_preference() -> None:
    fast_profile = ModelProfile(
        provider="test",
        model="fast-v1",
        capabilities=[TaskType.CHAT],
        quality_score=0.78,
        avg_latency_ms=100,
        availability=1.0,
    )
    slow_profile = ModelProfile(
        provider="test",
        model="slow-v1",
        capabilities=[TaskType.CHAT],
        quality_score=0.82,
        avg_latency_ms=4000,
        availability=1.0,
    )
    router = ModelRouter([fast_profile, slow_profile])
    intent = Intent(task_type=TaskType.CHAT)
    policy = PolicyDecision(
        allowed=True,
        reasons=["ok"],
        budgets=RuntimeBudget(latency_budget_ms=1000),
        disabled_tools=[],
        routing_preferences={"prefer_low_latency": True},
        metadata={},
    )

    selected = router.select(intent, policy)
    assert selected.model == "fast-v1"
