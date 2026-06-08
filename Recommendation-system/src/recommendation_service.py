from __future__ import annotations

import re
import unicodedata
from typing import Any

from .config import DATA_PATH, DEFAULT_SCORE_THRESHOLD, TOP_K
from .generator import Generator
from .pipeline_runtime import LegalRecommendationPipeline
from .retriever import Retriever
from .schemas import RecommendationResult
from .truth_lookup import TruthLookup

FOLLOW_UP_MARKERS = {
    "ca",
    "cela",
    "ce cas",
    "cette",
    "celle-ci",
    "celui-ci",
    "et pour ca",
    "et pour cela",
    "et pour ce cas",
}

VAGUE_FOLLOW_UP_TOKENS = {
    "aussi",
    "ca",
    "cas",
    "ce",
    "cela",
    "celle-ci",
    "celui-ci",
    "cette",
    "et",
    "idem",
    "meme",
    "non",
    "oui",
    "pareil",
    "pour",
}


def _normalize_follow_up_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower().strip()
    value = re.sub(r"[^\w\s'-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _needs_previous_user_context(query: str) -> bool:
    normalized = _normalize_follow_up_text(query)
    if normalized in FOLLOW_UP_MARKERS:
        return True

    tokens = set(normalized.split())
    return bool(tokens) and len(tokens) <= 4 and tokens <= VAGUE_FOLLOW_UP_TOKENS


class RecommendationService:
    def __init__(
        self,
        data_path=DATA_PATH,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
        retriever: Retriever | None = None,
        lookup: TruthLookup | None = None,
        generator: Generator | None = None,
    ):
        self.pipeline = LegalRecommendationPipeline(
            data_path=data_path,
            score_threshold=score_threshold,
            retriever=retriever,
            lookup=lookup,
            generator=generator,
        )
        self.initialized = False
        self.generator_warmed = False

    def initialize(self) -> None:
        self.pipeline.retriever.health()
        self.generator_warmed = self.pipeline.generator.warmup()
        self.initialized = True

    def health(self) -> dict[str, Any]:
        retriever_health = self.pipeline.retriever.health()
        return {
            "mode": "recommendation",
            "dataset_path": str(DATA_PATH),
            "score_threshold": self.pipeline.score_threshold,
            "initialized": self.initialized,
            "generator_backend": self.pipeline.generator.backend,
            "generator_warmed": self.generator_warmed,
            "healthy": True,
            "retriever": retriever_health,
        }

    def _build_effective_query(
        self,
        query: str,
        conversation_history: list[dict[str, Any]] | None,
    ) -> str:
        if not conversation_history or not _needs_previous_user_context(query):
            return query

        for item in reversed(conversation_history):
            if item.get("role") != "user":
                continue
            previous_content = str(item.get("content", "")).strip()
            if previous_content and previous_content != query:
                return f"{previous_content}\n{query}"

        return query

    def recommend(
        self,
        query: str,
        *,
        with_generation: bool = True,
        top_k: int = TOP_K,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> RecommendationResult:
        effective_query = self._build_effective_query(query, conversation_history)
        result = self.pipeline.run(
            query=effective_query,
            top_k=top_k,
            with_generation=with_generation,
        )
        result.query = query
        return result
