"""Execution scoring."""

from __future__ import annotations

from mindlatticeai.schemas import EvaluationResult, ReflectionResult, ToolResult


class Evaluator:
    """Scores an execution so the learner can update runtime strategy."""

    def evaluate(
        self,
        *,
        reflection: ReflectionResult,
        tool_results: dict[str, ToolResult],
        latency_ms: int,
        cost_usd: float,
        quality_floor: float,
    ) -> EvaluationResult:
        success = bool(reflection.revised_text) and all(result.success for result in tool_results.values())
        tool_efficiency = self._tool_efficiency(tool_results)
        latency_score = max(0.0, 1.0 - min(latency_ms / 15_000, 1.0))
        cost_score = max(0.0, 1.0 - min(cost_usd / 1.0, 1.0))
        issue_penalty = min(0.35, 0.07 * len(reflection.issues))
        quality = (
            0.45 * reflection.confidence
            + 0.20 * tool_efficiency
            + 0.20 * latency_score
            + 0.15 * cost_score
            - issue_penalty
        )
        quality = max(0.0, min(1.0, quality))
        reasons = [
            f"confidence={reflection.confidence:.2f}",
            f"tool_efficiency={tool_efficiency:.2f}",
            f"latency_ms={latency_ms}",
            f"cost_usd={cost_usd:.4f}",
        ]
        if quality < quality_floor:
            reasons.append("quality below requested floor")
        return EvaluationResult(
            success=success and quality >= quality_floor,
            quality_score=quality,
            hallucination_probability=reflection.hallucination_probability,
            tool_efficiency_score=tool_efficiency,
            reasons=reasons,
        )

    def _tool_efficiency(self, tool_results: dict[str, ToolResult]) -> float:
        if not tool_results:
            return 0.70
        successful = sum(1 for result in tool_results.values() if result.success)
        latency_penalty = sum(min(result.latency_ms / 5000, 1.0) for result in tool_results.values())
        raw = successful / len(tool_results)
        return max(0.0, min(1.0, raw - latency_penalty * 0.05))

