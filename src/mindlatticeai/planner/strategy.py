"""Adaptive planning and strategy generation."""

from __future__ import annotations

import re

from mindlatticeai.prompts.genome import PromptGenomeRegistry
from mindlatticeai.schemas import AIRequest, Intent, NodeType, PlanStep, Strategy


class AdaptivePlanner:
    """Builds a strategy that the graph builder can compile into a DAG."""

    def __init__(self, prompt_genomes: PromptGenomeRegistry) -> None:
        self.prompt_genomes = prompt_genomes

    def create_strategy(self, request: AIRequest, intent: Intent) -> Strategy:
        genome = self.prompt_genomes.select(intent.task_type.value)
        steps: list[PlanStep] = []
        previous_tool_ids: list[str] = []

        for tool_name in intent.required_tools:
            step_id = f"tool.{tool_name}"
            payload = {"tool_name": tool_name, "arguments": self._tool_arguments(tool_name, request.input)}
            steps.append(
                PlanStep(
                    id=step_id,
                    node_type=NodeType.TOOL_CALL,
                    name=f"Run {tool_name}",
                    description=f"Execute required tool: {tool_name}",
                    payload=payload,
                    max_retries=request.budget.max_retries,
                )
            )
            previous_tool_ids.append(step_id)

        steps.append(
            PlanStep(
                id="model.main",
                node_type=NodeType.MODEL_CALL,
                name="Adaptive model call",
                description="Route to the best available model and produce a draft response.",
                dependencies=previous_tool_ids,
                payload={"prompt_genome_id": genome.id},
                max_retries=request.budget.max_retries,
            )
        )
        steps.extend(
            [
                PlanStep(
                    id="reflect.self_check",
                    node_type=NodeType.REFLECTION,
                    name="Reflect on draft",
                    description="Run self-checks and revise the draft response.",
                    dependencies=["model.main"],
                ),
                PlanStep(
                    id="evaluate.score",
                    node_type=NodeType.EVALUATION,
                    name="Evaluate execution",
                    description="Score latency, cost, quality, hallucination risk, and tool efficiency.",
                    dependencies=["reflect.self_check"],
                ),
                PlanStep(
                    id="memory.write",
                    node_type=NodeType.MEMORY_WRITE,
                    name="Update memory",
                    description="Persist useful short-term, episodic, semantic, and long-term memories.",
                    dependencies=["evaluate.score"],
                ),
                PlanStep(
                    id="learn.update",
                    node_type=NodeType.LEARNING,
                    name="Update adaptive policies",
                    description="Update strategy, prompt, model, tool, and memory scores.",
                    dependencies=["evaluate.score"],
                ),
                PlanStep(
                    id="final.response",
                    node_type=NodeType.FINALIZE,
                    name="Finalize response",
                    description="Return the best response with explainable execution metadata.",
                    dependencies=["memory.write", "learn.update"],
                ),
            ]
        )
        return Strategy(
            name=f"{intent.task_type.value}.adaptive.default",
            prompt_genome_id=genome.id,
            steps=steps,
            metadata={"intent_confidence": intent.confidence},
        )

    def _tool_arguments(self, tool_name: str, user_input: str) -> dict[str, str]:
        if tool_name == "calculator":
            expression = user_input.lower().replace("calculate", "").replace("^", "**").strip()
            match = re.search(r"[-+*/().\d\s\*]+", expression)
            return {"expression": match.group(0).strip() if match else expression}
        return {"query": user_input}

