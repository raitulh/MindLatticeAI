"""Async execution helpers."""

from __future__ import annotations

import asyncio

from mindlatticeai.schemas import AIRequest, AIResponse


async def run_many(agent, requests: list[AIRequest], *, concurrency: int = 4) -> list[AIResponse]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(request: AIRequest) -> AIResponse:
        async with semaphore:
            return await agent.arun(request)

    return await asyncio.gather(*(guarded(request) for request in requests))

