"""Minimal evaluation framework."""

from __future__ import annotations

from pydantic import BaseModel, Field

from mindlatticeai.schemas import AIRequest


class BenchmarkCase(BaseModel):
    name: str
    input: str
    expected_keywords: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    total_cases: int
    success_rate: float
    mean_quality_score: float
    mean_latency_ms: float
    case_results: list[dict]


class BenchmarkSuite:
    def __init__(self, cases: list[BenchmarkCase]) -> None:
        self.cases = cases

    async def arun(self, agent) -> BenchmarkReport:
        results = []
        for case in self.cases:
            response = await agent.arun(AIRequest(input=case.input, session_id=f"bench.{case.name}"))
            keyword_success = all(keyword.lower() in response.text.lower() for keyword in case.expected_keywords)
            results.append(
                {
                    "name": case.name,
                    "success": response.metrics.success and keyword_success,
                    "quality_score": response.metrics.quality_score,
                    "latency_ms": response.metrics.latency_ms,
                    "replay_id": response.replay_id,
                }
            )
        total = len(results) or 1
        return BenchmarkReport(
            total_cases=len(results),
            success_rate=sum(1 for item in results if item["success"]) / total,
            mean_quality_score=sum(item["quality_score"] for item in results) / total,
            mean_latency_ms=sum(item["latency_ms"] for item in results) / total,
            case_results=results,
        )

