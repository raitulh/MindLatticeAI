"""Prompt genome registry and mutation logic."""

from __future__ import annotations

from copy import deepcopy

from mindlatticeai.schemas import PromptGenome, new_id


class PromptGenomeRegistry:
    """Stores, scores, mutates, and selects prompt genomes."""

    def __init__(self) -> None:
        self._genomes: dict[str, PromptGenome] = {}
        self.add(self._default_genome())

    def add(self, genome: PromptGenome) -> None:
        self._genomes[genome.id] = genome

    def get(self, genome_id: str) -> PromptGenome:
        return self._genomes[genome_id]

    def all(self) -> list[PromptGenome]:
        return list(self._genomes.values())

    def select(self, task_hint: str | None = None) -> PromptGenome:
        candidates = self.all()
        if task_hint:
            tagged = [
                genome
                for genome in candidates
                if task_hint in genome.metadata.get("task_hints", [])
            ]
            if tagged:
                candidates = tagged
        return max(candidates, key=lambda item: item.score)

    def mutate(self, genome_id: str, *, reason: str) -> PromptGenome:
        parent = self.get(genome_id)
        child = deepcopy(parent)
        child.id = new_id("genome")
        child.generation = parent.generation + 1
        child.score = max(0.05, parent.score * 0.98)
        child.metadata = {**parent.metadata, "mutation_parent_id": parent.id, "reason": reason}
        child.self_evaluation_rules = [
            *parent.self_evaluation_rules,
            "Estimate uncertainty explicitly when evidence is weak.",
        ]
        self.add(child)
        return child

    def update_score(self, genome_id: str, score: float, alpha: float = 0.20) -> None:
        genome = self.get(genome_id)
        genome.score = (1 - alpha) * genome.score + alpha * score

    def _default_genome(self) -> PromptGenome:
        return PromptGenome(
            id="genome.default.cognitive",
            role="policy-aware cognitive execution assistant",
            instruction=(
                "Solve the user's task using available tool observations, memory, "
                "and runtime constraints. Be concise, grounded, and explicit about "
                "uncertainty."
            ),
            constraints=[
                "Use tool observations when they are available.",
                "Do not invent external facts when the runtime has not retrieved evidence.",
                "Prefer a useful answer over framework narration.",
            ],
            output_format="markdown with clear final answer",
            safety_rules=[
                "Respect policy constraints and avoid unsafe operational guidance.",
                "Ask for clarification when required data is missing.",
            ],
            self_evaluation_rules=[
                "Check whether the answer follows the request.",
                "Check whether any claim lacks evidence.",
                "Check whether a cheaper or faster strategy would have sufficed.",
            ],
            metadata={"task_hints": ["chat", "reasoning", "tool"]},
        )

