import asyncio

from mindlatticeai import Agent
from mindlatticeai.benchmarks import BenchmarkCase, BenchmarkSuite


async def main() -> None:
    suite = BenchmarkSuite(
        [
            BenchmarkCase(name="math", input="Calculate 10 + 5", expected_keywords=["15"]),
            BenchmarkCase(name="runtime", input="Explain adaptive model routing."),
        ]
    )
    report = await suite.arun(Agent())
    print(report.model_dump())


if __name__ == "__main__":
    asyncio.run(main())

