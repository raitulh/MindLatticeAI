"""Runtime configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from mindlatticeai.schemas import RuntimeBudget


class RuntimeConfig(BaseModel):
    """Top-level runtime configuration for an Agent."""

    default_budget: RuntimeBudget = Field(default_factory=RuntimeBudget)
    telemetry_path: Path | None = None
    replay_path: Path | None = None
    memory_path: Path | None = None
    enable_response_cache: bool = True
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = 0.86
    checkpointing_enabled: bool = True
    local_provider_name: str = "local"

