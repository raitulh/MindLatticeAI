"""Feedback-driven strategy optimization."""

from __future__ import annotations

from dataclasses import dataclass

from mindlatticeai.schemas import LearningSignal


@dataclass
class AdaptiveStatistic:
    score: float = 0.50
    count: int = 0

    def update(self, value: float, alpha: float = 0.20) -> None:
        self.score = (1 - alpha) * self.score + alpha * value
        self.count += 1


class LearningEngine:
    """Records execution feedback and exposes updated priorities."""

    def __init__(self) -> None:
        self.strategy_stats: dict[str, AdaptiveStatistic] = {}
        self.prompt_stats: dict[str, AdaptiveStatistic] = {}
        self.model_stats: dict[str, AdaptiveStatistic] = {}
        self.tool_stats: dict[str, AdaptiveStatistic] = {}
        self.memory_importance_bias: dict[str, float] = {}
        self.history: list[LearningSignal] = []

    @property
    def model_performance(self) -> dict[str, float]:
        return {key: stat.score for key, stat in self.model_stats.items()}

    def observe(self, signal: LearningSignal) -> None:
        self.history.append(signal)
        scalar = self._score_signal(signal)
        self._stat(self.strategy_stats, signal.strategy_id).update(scalar)
        self._stat(self.prompt_stats, signal.prompt_genome_id).update(scalar)
        if signal.model_id:
            self._stat(self.model_stats, signal.model_id).update(scalar)
        for tool_name, score in signal.tool_scores.items():
            self._stat(self.tool_stats, tool_name).update(score)
        self.memory_importance_bias[signal.strategy_id] = max(
            0.05,
            min(1.0, signal.quality_score * (1.0 - signal.hallucination_probability)),
        )

    def tool_priority(self, tool_name: str) -> float:
        return self.tool_stats.get(tool_name, AdaptiveStatistic()).score

    def _score_signal(self, signal: LearningSignal) -> float:
        latency_component = max(0.0, 1.0 - min(signal.latency_ms / 15_000, 1.0))
        cost_component = max(0.0, 1.0 - min(signal.cost_usd / 1.0, 1.0))
        feedback = signal.user_feedback if signal.user_feedback is not None else signal.quality_score
        score = (
            0.42 * signal.quality_score
            + 0.18 * latency_component
            + 0.14 * cost_component
            + 0.16 * feedback
            + 0.10 * (1.0 - signal.hallucination_probability)
        )
        if not signal.success:
            score *= 0.70
        return max(0.0, min(1.0, score))

    def _stat(self, store: dict[str, AdaptiveStatistic], key: str) -> AdaptiveStatistic:
        if key not in store:
            store[key] = AdaptiveStatistic()
        return store[key]

