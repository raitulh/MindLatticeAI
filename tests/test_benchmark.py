import pytest

from mindlatticeai import Agent
from mindlatticeai.benchmarks import BenchmarkCase, BenchmarkSuite


@pytest.mark.asyncio
async def test_benchmark_suite_runs_and_aggregates_metrics() -> None:
    agent = Agent()
    suite = BenchmarkSuite(
        [
            BenchmarkCase(name="math", input="Calculate 10 + 20", expected_keywords=["30.0"]),
            BenchmarkCase(name="greeting", input="Hello world", expected_keywords=["response"]),
        ]
    )

    report = await suite.arun(agent)
    assert report.total_cases == 2
    assert report.success_rate == 1.0
    assert report.mean_quality_score > 0.0
    assert len(report.case_results) == 2
