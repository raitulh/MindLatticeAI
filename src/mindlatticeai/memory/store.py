"""Multi-layer memory implementation."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from mindlatticeai.schemas import MemoryLayer, MemoryRecord


class MemorySystem:
    """Short, long, semantic, episodic, and graph memory in one facade."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.short_term: dict[str, list[MemoryRecord]] = {}
        self.long_term: list[MemoryRecord] = []
        self.semantic: list[MemoryRecord] = []
        self.episodic: list[MemoryRecord] = []
        self.knowledge_graph: set[tuple[str, str, str]] = set()

    def remember(
        self,
        *,
        content: str,
        layer: MemoryLayer,
        session_id: str = "default",
        importance: float = 0.50,
        tags: list[str] | None = None,
        relationships: list[tuple[str, str, str]] | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            layer=layer,
            content=content,
            importance=importance,
            embedding=self._embed(content),
            tags=tags or [],
            relationships=relationships or [],
            metadata=metadata or {},
        )
        if layer == MemoryLayer.SHORT_TERM:
            self.short_term.setdefault(session_id, []).append(record)
        elif layer == MemoryLayer.LONG_TERM:
            self.long_term.append(record)
        elif layer == MemoryLayer.SEMANTIC:
            self.semantic.append(record)
        elif layer == MemoryLayer.EPISODIC:
            self.episodic.append(record)
        elif layer == MemoryLayer.KNOWLEDGE_GRAPH:
            self.knowledge_graph.update(record.relationships)

        for triple in record.relationships:
            self.knowledge_graph.add(triple)
        self.deduplicate()
        return record

    def recall(self, query: str, *, session_id: str = "default", limit: int = 5) -> list[MemoryRecord]:
        query_embedding = self._embed(query)
        candidates = [
            *self.short_term.get(session_id, []),
            *self.long_term,
            *self.semantic,
            *self.episodic[-20:],
        ]
        scored = [
            (self._similarity(query_embedding, record.embedding) * 0.70 + record.importance * 0.30, record)
            for record in candidates
        ]
        return [record for _, record in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]

    def decay(self, *, half_life_hours: float = 72.0) -> None:
        now = datetime.now(timezone.utc)
        for record in self._all_records():
            age_hours = max(0.0, (now - record.updated_at).total_seconds() / 3600)
            record.importance *= math.pow(0.5, age_hours / half_life_hours)

    def compress(self, *, max_short_term: int = 20, session_id: str = "default") -> None:
        records = self.short_term.get(session_id, [])
        if len(records) <= max_short_term:
            return
        records.sort(key=lambda item: item.importance, reverse=True)
        archived = records[max_short_term:]
        self.short_term[session_id] = records[:max_short_term]
        summary = "\n".join(record.content for record in archived[:5])
        self.remember(
            content=f"Compressed session memory:\n{summary}",
            layer=MemoryLayer.LONG_TERM,
            session_id=session_id,
            importance=max((record.importance for record in archived), default=0.4),
            tags=["compressed"],
        )

    def deduplicate(self) -> None:
        for collection in [self.long_term, self.semantic, self.episodic]:
            seen: dict[str, MemoryRecord] = {}
            for record in collection:
                existing = seen.get(record.content)
                if existing is None or existing.importance < record.importance:
                    seen[record.content] = record
            collection[:] = list(seen.values())

    def resolve_conflicts(self, records: list[MemoryRecord]) -> MemoryRecord | None:
        if not records:
            return None
        return max(records, key=lambda item: (item.importance, item.updated_at))

    def _all_records(self) -> list[MemoryRecord]:
        short = [record for records in self.short_term.values() for record in records]
        return [*short, *self.long_term, *self.semantic, *self.episodic]

    def _embed(self, text: str) -> dict[str, float]:
        tokens = [token.lower() for token in text.split() if token.strip()]
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {token: count / total for token, count in counts.items()}

    def _similarity(self, left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        shared = set(left).intersection(right)
        dot = sum(left[token] * right[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

