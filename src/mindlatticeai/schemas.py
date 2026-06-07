"""Shared Pydantic models for the MindLatticeAI runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class TaskType(str, Enum):
    CHAT = "chat"
    REASONING = "reasoning"
    RAG = "rag"
    TOOL = "tool"
    AGENT = "agent"
    BENCHMARK = "benchmark"


class NodeType(str, Enum):
    INTENT = "intent"
    PLAN = "plan"
    POLICY = "policy"
    TOOL_CALL = "tool_call"
    MODEL_CALL = "model_call"
    REFLECTION = "reflection"
    EVALUATION = "evaluation"
    MEMORY_WRITE = "memory_write"
    LEARNING = "learning"
    FINALIZE = "finalize"


class ExecutionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    CHECKPOINTED = "checkpointed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class MemoryLayer(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class RuntimeBudget(BaseModel):
    max_cost_usd: float = 0.25
    latency_budget_ms: int = 15_000
    quality_floor: float = 0.60
    max_retries: int = 2


class AIRequest(BaseModel):
    id: str = Field(default_factory=lambda: new_id("req"))
    input: str
    session_id: str = "default"
    user_id: str | None = None
    task_type: TaskType | None = None
    budget: RuntimeBudget = Field(default_factory=RuntimeBudget)
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now_utc)


class Intent(BaseModel):
    task_type: TaskType
    goals: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    risk_level: str = "normal"
    constraints: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.75


class PlanStep(BaseModel):
    id: str
    node_type: NodeType
    name: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 1


class Strategy(BaseModel):
    id: str = Field(default_factory=lambda: new_id("strategy"))
    name: str
    prompt_genome_id: str
    steps: list[PlanStep]
    mutation_parent_id: str | None = None
    score: float = 0.50
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    budgets: RuntimeBudget
    disabled_tools: list[str] = Field(default_factory=list)
    routing_preferences: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    node_type: NodeType
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    state: ExecutionState = ExecutionState.PENDING
    attempts: int = 0
    max_retries: int = 1
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None
    result: Any = None
    checkpoint_key: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str = "depends_on"


class ExecutionGraph(BaseModel):
    id: str = Field(default_factory=lambda: new_id("graph"))
    request_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def node_map(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}


class ModelProfile(BaseModel):
    provider: str
    model: str
    capabilities: list[TaskType]
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    avg_latency_ms: int = 250
    quality_score: float = 0.65
    availability: float = 1.0
    context_window: int = 16_000
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"


class ModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str
    model_profile: ModelProfile
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptGenome(BaseModel):
    id: str = Field(default_factory=lambda: new_id("genome"))
    role: str = "cognitive execution assistant"
    instruction: str
    context: str = ""
    constraints: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    output_format: str = "clear markdown"
    safety_rules: list[str] = Field(default_factory=list)
    self_evaluation_rules: list[str] = Field(default_factory=list)
    generation: int = 0
    score: float = 0.50
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render(
        self,
        *,
        user_input: str,
        memory_context: list[str] | None = None,
        tool_observations: dict[str, Any] | None = None,
    ) -> str:
        parts = [
            f"Role: {self.role}",
            f"Instruction: {self.instruction}",
        ]
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.constraints:
            parts.append("Constraints:\n" + "\n".join(f"- {item}" for item in self.constraints))
        if self.examples:
            parts.append("Examples:\n" + "\n".join(f"- {item}" for item in self.examples))
        if self.safety_rules:
            parts.append("Safety rules:\n" + "\n".join(f"- {item}" for item in self.safety_rules))
        if self.self_evaluation_rules:
            parts.append(
                "Self-evaluation rules:\n"
                + "\n".join(f"- {item}" for item in self.self_evaluation_rules)
            )
        if memory_context:
            parts.append("Memory context:\n" + "\n".join(f"- {item}" for item in memory_context))
        if tool_observations:
            parts.append(f"Tool observations: {tool_observations}")
        parts.append(f"Output format: {self.output_format}")
        parts.append(f"User input: {user_input}")
        return "\n\n".join(parts)


class ToolResult(BaseModel):
    tool_name: str
    output: Any
    success: bool = True
    latency_ms: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReflectionResult(BaseModel):
    original_text: str
    revised_text: str
    issues: list[str] = Field(default_factory=list)
    hallucination_probability: float = 0.20
    confidence: float = 0.70
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    success: bool
    quality_score: float
    hallucination_probability: float
    tool_efficiency_score: float
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    latency_ms: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    quality_score: float = 0.0
    success: bool = False
    hallucination_probability: float = 1.0
    tool_efficiency_score: float = 0.0
    retries: int = 0
    model_id: str | None = None


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("mem"))
    layer: MemoryLayer
    content: str
    importance: float = 0.50
    embedding: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    relationships: list[tuple[str, str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class LearningSignal(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    strategy_id: str
    prompt_genome_id: str
    model_id: str | None
    tool_scores: dict[str, float] = Field(default_factory=dict)
    latency_ms: int
    cost_usd: float
    quality_score: float
    success: bool
    hallucination_probability: float
    user_feedback: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    timestamp: datetime = Field(default_factory=now_utc)
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    id: str = Field(default_factory=lambda: new_id("trace"))
    request_id: str
    events: list[TraceEvent] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=now_utc)
    ended_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    request_id: str
    trace_id: str
    graph_id: str
    strategy_id: str
    prompt_genome_id: str
    replay_id: str | None
    metrics: ExecutionMetrics
    evaluation: EvaluationResult | None = None
    graph: ExecutionGraph | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
