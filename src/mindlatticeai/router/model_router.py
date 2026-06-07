"""Adaptive model routing."""

from __future__ import annotations

from mindlatticeai.schemas import Intent, ModelProfile, PolicyDecision


class ModelRouter:
    """Selects a model using policy constraints and historical performance."""

    def __init__(self, profiles: list[ModelProfile], performance: dict[str, float] | None = None) -> None:
        self.profiles = profiles
        self.performance = performance if performance is not None else {}

    def select(self, intent: Intent, policy: PolicyDecision) -> ModelProfile:
        candidates = [
            profile
            for profile in self.profiles
            if intent.task_type in profile.capabilities and profile.availability > 0
        ]
        if not candidates:
            raise RuntimeError(f"No available model supports task type: {intent.task_type.value}")

        return max(candidates, key=lambda profile: self._score(profile, policy))

    def _score(self, profile: ModelProfile, policy: PolicyDecision) -> float:
        historical = self.performance.get(profile.id, profile.quality_score)
        quality = 0.40 * profile.quality_score + 0.25 * historical
        availability = 0.15 * profile.availability
        latency_penalty = min(profile.avg_latency_ms / max(policy.budgets.latency_budget_ms, 1), 2.0) * 0.10
        estimated_cost = profile.cost_per_1k_input + profile.cost_per_1k_output
        cost_penalty = min(estimated_cost / max(policy.budgets.max_cost_usd, 0.001), 2.0) * 0.10

        if policy.routing_preferences.get("prefer_low_latency"):
            latency_penalty *= 1.8
        if policy.routing_preferences.get("prefer_low_cost"):
            cost_penalty *= 1.8
        return quality + availability - latency_penalty - cost_penalty

