from mindlatticeai.prompts import PromptGenomeRegistry
from mindlatticeai.schemas import PromptGenome


def test_prompt_genome_registry_defaults_and_select() -> None:
    registry = PromptGenomeRegistry()
    genomes = registry.all()
    assert len(genomes) >= 1

    default = registry.select("chat")
    assert "cognitive" in default.id
    assert default.generation == 0


def test_prompt_genome_mutation() -> None:
    registry = PromptGenomeRegistry()
    parent = registry.select()

    child = registry.mutate(parent.id, reason="low quality score on reasoning")
    assert child.generation == parent.generation + 1
    assert child.metadata["mutation_parent_id"] == parent.id
    assert len(child.self_evaluation_rules) > len(parent.self_evaluation_rules)
    assert child.id in [g.id for g in registry.all()]


def test_prompt_genome_update_score() -> None:
    registry = PromptGenomeRegistry()
    genome = registry.select()
    initial_score = genome.score

    registry.update_score(genome.id, 1.0, alpha=0.5)
    assert registry.get(genome.id).score > initial_score
