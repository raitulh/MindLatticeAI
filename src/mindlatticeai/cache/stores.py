"""Response and semantic caches."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any


class ResponseCache:
    """Exact cache keyed by normalized request text and constraints."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def key(self, text: str, metadata: dict | None = None) -> str:
        payload = f"{text.strip().lower()}::{sorted((metadata or {}).items())}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, text: str, metadata: dict | None = None) -> Any | None:
        return self._items.get(self.key(text, metadata))

    def set(self, text: str, value: Any, metadata: dict | None = None) -> None:
        self._items[self.key(text, metadata)] = value


class SemanticCache:
    """Small lexical semantic cache for dependency-free development."""

    def __init__(self, threshold: float = 0.86) -> None:
        self.threshold = threshold
        self._items: list[tuple[str, dict[str, float], Any]] = []

    def get(self, text: str) -> Any | None:
        vector = self._embed(text)
        best_score = 0.0
        best_value = None
        for _, cached_vector, value in self._items:
            score = self._similarity(vector, cached_vector)
            if score > best_score:
                best_score = score
                best_value = value
        return best_value if best_score >= self.threshold else None

    def set(self, text: str, value: Any) -> None:
        self._items.append((text, self._embed(text), value))

    def _embed(self, text: str) -> dict[str, float]:
        counts = Counter(token.lower() for token in text.split())
        total = sum(counts.values()) or 1
        return {token: count / total for token, count in counts.items()}

    def _similarity(self, left: dict[str, float], right: dict[str, float]) -> float:
        shared = set(left).intersection(right)
        dot = sum(left[token] * right[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

