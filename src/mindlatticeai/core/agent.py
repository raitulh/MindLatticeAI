"""High-level MindLatticeAI agent runtime."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from mindlatticeai.cache import ResponseCache, SemanticCache
from mindlatticeai.core.config import RuntimeConfig
from mindlatticeai.graph import CheckpointStore, ExecutionGraphBuilder, ExecutionGraphEngine
from mindlatticeai.learner import LearningEngine
from mindlatticeai.memory import MemorySystem
from mindlatticeai.planner import AdaptivePlanner, IntentParser
from mindlatticeai.plugins import PluginManager
from mindlatticeai.policy import PolicyEngine
from mindlatticeai.prompts import PromptGenomeRegistry
from mindlatticeai.providers import LocalHeuristicProvider, ModelProvider
from mindlatticeai.reflection import Evaluator, ReflectionEngine
from mindlatticeai.replay import ReplayEngine
from mindlatticeai.router import ModelRouter
from mindlatticeai.schemas import (
    AIRequest,
    AIResponse,
    EvaluationResult,
    ExecutionGraph,
    ExecutionMetrics,
    GraphNode,
    Intent,
    LearningSignal,
    MemoryLayer,
    ModelResponse,
    NodeType,
    PolicyDecision,
    ReflectionResult,
    Strategy,
    ToolResult,
)
from mindlatticeai.security import SandboxPolicy
from mindlatticeai.telemetry import ExecutionTracer
from mindlatticeai.tools import ToolRegistry, register_builtin_tools


class Agent:
    """Adaptive AI runtime facade.

    The starter implementation is intentionally local-first. It can execute the
    full experiment loop without network access, while provider, memory, cache,
    policy, and plugin interfaces are ready for production replacement.
    """

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        providers: list[ModelProvider] | None = None,
        tools: ToolRegistry | None = None,
        plugins: PluginManager | None = None,
        policy_engine: PolicyEngine | None = None,
        sandbox_policy: SandboxPolicy | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.prompt_genomes = PromptGenomeRegistry()
        self.intent_parser = IntentParser()
        self.planner = AdaptivePlanner(self.prompt_genomes)
        self.policy_engine = policy_engine or PolicyEngine()
        self.graph_builder = ExecutionGraphBuilder()
        self.checkpoints = CheckpointStore()
        self.graph_engine = ExecutionGraphEngine(
            self.checkpoints,
            enabled=self.config.checkpointing_enabled,
        )
        self.learning = LearningEngine()
        self.memory = MemorySystem(self.config.memory_path)
        self.reflection = ReflectionEngine()
        self.evaluator = Evaluator()
        self.telemetry = ExecutionTracer(self.config.telemetry_path)
        self.replay = ReplayEngine(self.config.replay_path)
        self.plugins = plugins or PluginManager()
        self.sandbox_policy = sandbox_policy or SandboxPolicy()

        self.tools = tools or ToolRegistry()
        if tools is None:
            register_builtin_tools(self.tools)

        self.providers = providers or [LocalHeuristicProvider()]
        self.provider_by_id = {provider.profile.id: provider for provider in self.providers}
        self.router = ModelRouter([provider.profile for provider in self.providers], self.learning.model_performance)
        self.response_cache = ResponseCache()
        self.semantic_cache = SemanticCache(self.config.semantic_cache_threshold)

    def run(self, request: str | AIRequest, **overrides: Any) -> AIResponse:
        """Synchronous wrapper around ``arun``."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(request, **overrides))
        raise RuntimeError("Agent.run() cannot run inside an active event loop. Use await agent.arun(...).")

    async def arun(self, request: str | AIRequest, **overrides: Any) -> AIResponse:
        """Run a request through the full adaptive experiment loop."""

        ai_request = self._coerce_request(request, overrides)
        trace = self.telemetry.start(ai_request.id)
        context: dict[str, Any] = {
            "request": ai_request,
            "trace_id": trace.id,
            "started_perf": time.perf_counter(),
            "tool_results": {},
            "metrics": ExecutionMetrics(),
        }

        try:
            intent = self.intent_parser.parse(ai_request)
            context["intent"] = intent
            self.telemetry.event(trace.id, "intent.parsed", intent.model_dump(mode="json"))

            await self.plugins.emit("before_plan", request=ai_request, intent=intent)
            strategy = self.planner.create_strategy(ai_request, intent)
            context["strategy"] = strategy
            await self.plugins.emit("after_plan", request=ai_request, intent=intent, strategy=strategy)
            self.telemetry.event(trace.id, "strategy.created", strategy.model_dump(mode="json"))

            policy = self.policy_engine.evaluate(ai_request, intent, strategy)
            context["policy"] = policy
            self.telemetry.event(trace.id, "policy.evaluated", policy.model_dump(mode="json"))
            if not policy.allowed:
                raise PermissionError("; ".join(policy.reasons))

            graph = self.graph_builder.build(request_id=ai_request.id, strategy=strategy, policy=policy)
            context["graph"] = graph
            self.telemetry.event(trace.id, "graph.built", {"graph_id": graph.id, "nodes": len(graph.nodes)})

            await self.plugins.emit("before_execute", request=ai_request, graph=graph, context=context)
            graph = await self.graph_engine.execute(graph, self._handlers(), context)
            await self.plugins.emit("after_execute", request=ai_request, graph=graph, context=context)

            trace = self.telemetry.finish(trace.id)
            metrics: ExecutionMetrics = context["metrics"]
            metrics.latency_ms = int((time.perf_counter() - context["started_perf"]) * 1000)
            metrics.retries = sum(max(0, node.attempts - 1) for node in graph.nodes)

            evaluation: EvaluationResult | None = context.get("evaluation")
            response = AIResponse(
                text=context.get("final_text", context.get("reflected_response", "")),
                request_id=ai_request.id,
                trace_id=trace.id,
                graph_id=graph.id,
                strategy_id=strategy.id,
                prompt_genome_id=strategy.prompt_genome_id,
                replay_id=None,
                metrics=metrics,
                evaluation=evaluation,
                graph=graph,
                artifacts={
                    "policy_reasons": policy.reasons,
                    "graph_dot": self.telemetry.graph_to_dot(graph),
                    "trace_events": [event.name for event in trace.events],
                },
            )
            replay_id = self.replay.record(request=ai_request, response=response, graph=graph, trace=trace)
            response.replay_id = replay_id
            return response
        except Exception:
            self.telemetry.finish(trace.id)
            raise

    def _coerce_request(self, request: str | AIRequest, overrides: dict[str, Any]) -> AIRequest:
        if isinstance(request, AIRequest):
            if overrides:
                return request.model_copy(update=overrides)
            return request
        payload = {"input": request}
        payload.update(overrides)
        if "budget" not in payload:
            payload["budget"] = self.config.default_budget
        return AIRequest(**payload)

    def _handlers(self) -> dict[NodeType, Any]:
        return {
            NodeType.TOOL_CALL: self._handle_tool,
            NodeType.MODEL_CALL: self._handle_model,
            NodeType.REFLECTION: self._handle_reflection,
            NodeType.EVALUATION: self._handle_evaluation,
            NodeType.MEMORY_WRITE: self._handle_memory_write,
            NodeType.LEARNING: self._handle_learning,
            NodeType.FINALIZE: self._handle_finalize,
        }

    async def _handle_tool(self, node: GraphNode, context: dict[str, Any]) -> dict[str, Any]:
        trace_id = context["trace_id"]
        tool_name = node.payload["tool_name"]
        self.sandbox_policy.validate_tool(tool_name)
        self.telemetry.event(trace_id, "tool.started", {"node_id": node.id, "tool_name": tool_name})
        result = await self.tools.run(tool_name, node.payload.get("arguments", {}))
        context["tool_results"][tool_name] = result
        metrics: ExecutionMetrics = context["metrics"]
        metrics.cost_usd += result.cost_usd
        if not result.success:
            self.telemetry.event(trace_id, "tool.failed", result.model_dump(mode="json"))
            raise RuntimeError(result.error or f"Tool failed: {tool_name}")
        self.telemetry.event(trace_id, "tool.finished", result.model_dump(mode="json"))
        return result.model_dump(mode="json")

    async def _handle_model(self, node: GraphNode, context: dict[str, Any]) -> dict[str, Any]:
        request: AIRequest = context["request"]
        intent: Intent = context["intent"]
        policy: PolicyDecision = context["policy"]
        strategy: Strategy = context["strategy"]
        trace_id = context["trace_id"]

        self.router.performance = self.learning.model_performance
        profile = self.router.select(intent, policy)
        provider = self.provider_by_id[profile.id]
        genome = self.prompt_genomes.get(node.payload.get("prompt_genome_id", strategy.prompt_genome_id))
        memory_records = self.memory.recall(request.input, session_id=request.session_id)
        tool_results: dict[str, ToolResult] = context["tool_results"]
        prompt = genome.render(
            user_input=request.input,
            memory_context=[record.content for record in memory_records],
            tool_observations={name: result.output for name, result in tool_results.items()},
        )
        self.telemetry.event(
            trace_id,
            "model.routed",
            {"node_id": node.id, "model_id": profile.id, "genome_id": genome.id},
        )

        response = None
        cache_metadata = {"model_id": profile.id, "genome_id": genome.id}
        if self.config.enable_response_cache:
            response = self.response_cache.get(prompt, cache_metadata)
        if response is None and self.config.enable_semantic_cache:
            response = self.semantic_cache.get(request.input)

        if response is not None:
            model_response = ModelResponse(**response)
            self.telemetry.event(trace_id, "model.cache_hit", {"model_id": model_response.model_profile.id})
        else:
            model_response = await provider.complete(prompt, request)
            model_response.cost_usd = model_response.cost_usd or self._estimate_cost(model_response)
            serialized = model_response.model_dump(mode="json")
            if self.config.enable_response_cache:
                self.response_cache.set(prompt, serialized, cache_metadata)
            if self.config.enable_semantic_cache:
                self.semantic_cache.set(request.input, serialized)

        metrics: ExecutionMetrics = context["metrics"]
        metrics.cost_usd += model_response.cost_usd
        metrics.input_tokens += model_response.input_tokens
        metrics.output_tokens += model_response.output_tokens
        metrics.model_id = model_response.model_profile.id
        context["draft_response"] = model_response.text
        context["model_response"] = model_response
        self.telemetry.event(
            trace_id,
            "model.finished",
            {
                "model_id": model_response.model_profile.id,
                "input_tokens": model_response.input_tokens,
                "output_tokens": model_response.output_tokens,
                "cost_usd": model_response.cost_usd,
            },
        )
        return model_response.model_dump(mode="json")

    async def _handle_reflection(self, node: GraphNode, context: dict[str, Any]) -> dict[str, Any]:
        request: AIRequest = context["request"]
        trace_id = context["trace_id"]
        await self.plugins.emit("before_reflect", request=request, context=context)
        result = self.reflection.reflect(
            request=request,
            draft_text=context.get("draft_response", ""),
            tool_results=context["tool_results"],
        )
        await self.plugins.emit("after_reflect", request=request, reflection=result, context=context)
        context["reflection"] = result
        context["reflected_response"] = result.revised_text
        self.telemetry.event(trace_id, "reflection.finished", result.model_dump(mode="json"))
        return result.model_dump(mode="json")

    async def _handle_evaluation(self, node: GraphNode, context: dict[str, Any]) -> dict[str, Any]:
        request: AIRequest = context["request"]
        reflection: ReflectionResult = context["reflection"]
        metrics: ExecutionMetrics = context["metrics"]
        latency_ms = int((time.perf_counter() - context["started_perf"]) * 1000)
        evaluation = self.evaluator.evaluate(
            reflection=reflection,
            tool_results=context["tool_results"],
            latency_ms=latency_ms,
            cost_usd=metrics.cost_usd,
            quality_floor=request.budget.quality_floor,
        )
        metrics.quality_score = evaluation.quality_score
        metrics.success = evaluation.success
        metrics.hallucination_probability = evaluation.hallucination_probability
        metrics.tool_efficiency_score = evaluation.tool_efficiency_score
        context["evaluation"] = evaluation
        self.telemetry.event(context["trace_id"], "evaluation.finished", evaluation.model_dump(mode="json"))
        return evaluation.model_dump(mode="json")

    async def _handle_memory_write(self, node: GraphNode, context: dict[str, Any]) -> dict[str, Any]:
        request: AIRequest = context["request"]
        evaluation: EvaluationResult = context["evaluation"]
        response_text = context.get("reflected_response", "")
        importance = max(0.05, evaluation.quality_score * (1.0 - evaluation.hallucination_probability))
        content = f"Request: {request.input}\nResponse: {response_text}"
        short = self.memory.remember(
            content=content,
            layer=MemoryLayer.SHORT_TERM,
            session_id=request.session_id,
            importance=importance,
            tags=["session", request.task_type.value if request.task_type else "auto"],
        )
        episodic = self.memory.remember(
            content=f"Executed {request.id} with quality={evaluation.quality_score:.3f}",
            layer=MemoryLayer.EPISODIC,
            session_id=request.session_id,
            importance=importance,
            tags=["execution"],
        )
        semantic = self.memory.remember(
            content=content,
            layer=MemoryLayer.SEMANTIC,
            session_id=request.session_id,
            importance=importance,
            tags=["semantic"],
        )
        if evaluation.success and evaluation.quality_score >= 0.80:
            self.memory.remember(
                content=content,
                layer=MemoryLayer.LONG_TERM,
                session_id=request.session_id,
                importance=importance,
                tags=["successful"],
            )
        self.memory.compress(session_id=request.session_id)
        result = {"records": [short.id, episodic.id, semantic.id], "importance": importance}
        self.telemetry.event(context["trace_id"], "memory.updated", result)
        return result

    async def _handle_learning(self, node: GraphNode, context: dict[str, Any]) -> dict[str, Any]:
        request: AIRequest = context["request"]
        strategy: Strategy = context["strategy"]
        evaluation: EvaluationResult = context["evaluation"]
        metrics: ExecutionMetrics = context["metrics"]
        tool_results: dict[str, ToolResult] = context["tool_results"]

        await self.plugins.emit("before_learn", request=request, context=context)
        signal = LearningSignal(
            strategy_id=strategy.id,
            prompt_genome_id=strategy.prompt_genome_id,
            model_id=metrics.model_id,
            tool_scores={name: 1.0 if result.success else 0.0 for name, result in tool_results.items()},
            latency_ms=metrics.latency_ms,
            cost_usd=metrics.cost_usd,
            quality_score=evaluation.quality_score,
            success=evaluation.success,
            hallucination_probability=evaluation.hallucination_probability,
            user_feedback=request.metadata.get("user_feedback"),
        )
        self.learning.observe(signal)
        self.prompt_genomes.update_score(strategy.prompt_genome_id, evaluation.quality_score)
        if evaluation.quality_score < request.budget.quality_floor:
            child = self.prompt_genomes.mutate(strategy.prompt_genome_id, reason="quality_below_floor")
            context["mutated_prompt_genome_id"] = child.id
        await self.plugins.emit("after_learn", request=request, signal=signal, context=context)
        self.telemetry.event(context["trace_id"], "learning.updated", signal.model_dump(mode="json"))
        return signal.model_dump(mode="json")

    async def _handle_finalize(self, node: GraphNode, context: dict[str, Any]) -> dict[str, str]:
        final_text = context.get("reflected_response") or context.get("draft_response") or ""
        context["final_text"] = final_text
        self.telemetry.event(context["trace_id"], "response.finalized", {"characters": len(final_text)})
        return {"text": final_text}

    def _estimate_cost(self, response: ModelResponse) -> float:
        profile = response.model_profile
        input_cost = (response.input_tokens / 1000) * profile.cost_per_1k_input
        output_cost = (response.output_tokens / 1000) * profile.cost_per_1k_output
        return input_cost + output_cost

