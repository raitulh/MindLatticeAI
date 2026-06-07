"""Reflection and self-check loop."""

from __future__ import annotations

from mindlatticeai.schemas import AIRequest, ReflectionResult, ToolResult


class ReflectionEngine:
    """Performs inexpensive deterministic self-evaluation."""

    def reflect(
        self,
        *,
        request: AIRequest,
        draft_text: str,
        tool_results: dict[str, ToolResult],
    ) -> ReflectionResult:
        issues: list[str] = []
        revised = draft_text.strip()

        if not revised:
            issues.append("Draft response is empty.")
            revised = "I could not produce a reliable answer from the current strategy."
        if tool_results and "Tool-backed observation" not in revised:
            issues.append("Draft did not explicitly use available tool observations.")
            revised = f"{revised}\n\nTool observations were available and should be considered."
        if len(revised.split()) < 12:
            issues.append("Draft response may be under-specified.")

        hallucination = 0.15 if tool_results else 0.28
        if issues:
            hallucination = min(0.70, hallucination + 0.08 * len(issues))
        confidence = max(0.10, 1.0 - hallucination)
        return ReflectionResult(
            original_text=draft_text,
            revised_text=revised,
            issues=issues,
            hallucination_probability=hallucination,
            confidence=confidence,
        )

