from mindlatticeai.memory import MemorySystem
from mindlatticeai.schemas import MemoryLayer


def test_memory_multi_layer_storage_and_recall() -> None:
    mem = MemorySystem()

    mem.remember(
        content="MindLatticeAI uses execution graphs",
        layer=MemoryLayer.SHORT_TERM,
        session_id="test_sess",
        importance=0.8,
    )
    mem.remember(
        content="DAG nodes can be checkpointed and retried",
        layer=MemoryLayer.LONG_TERM,
        importance=0.9,
    )
    mem.remember(
        content="Reinforcement learning and bandit routing",
        layer=MemoryLayer.SEMANTIC,
        importance=0.7,
    )

    recalled = mem.recall("execution graphs DAG", session_id="test_sess", limit=2)
    assert len(recalled) == 2
    assert any("execution graphs" in r.content for r in recalled)


def test_memory_knowledge_graph_triples() -> None:
    mem = MemorySystem()
    mem.remember(
        content="Euler is a mathematician",
        layer=MemoryLayer.KNOWLEDGE_GRAPH,
        relationships=[("Euler", "is_a", "mathematician"), ("Euler", "born_in", "Basel")],
    )

    assert ("Euler", "is_a", "mathematician") in mem.knowledge_graph
    assert ("Euler", "born_in", "Basel") in mem.knowledge_graph


def test_memory_compression() -> None:
    mem = MemorySystem()
    for i in range(15):
        mem.remember(
            content=f"Message {i}",
            layer=MemoryLayer.SHORT_TERM,
            session_id="sess_compress",
            importance=0.1 * (i % 5),
        )

    mem.compress(max_short_term=5, session_id="sess_compress")
    assert len(mem.short_term["sess_compress"]) == 5
    assert len(mem.long_term) >= 1
    assert "Compressed session memory" in mem.long_term[0].content
