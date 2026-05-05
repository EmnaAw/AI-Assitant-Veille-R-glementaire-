from __future__ import annotations

import sys
import logging
from pathlib import Path
from collections import OrderedDict
from threading import RLock
from time import monotonic
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RECOMMENDATION_ROOT = PROJECT_ROOT / "Recommendation-system"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

if str(RECOMMENDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(RECOMMENDATION_ROOT))

from rag_service import RAGService  # noqa: E402
from src.recommendation_service import RecommendationService  # noqa: E402


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str = Field(min_length=1)


class AIRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    top_k: int = 3


class OrchestratedAIRequest(AIRequest):
    mode: Literal["rag", "recommendation"]


app = FastAPI(
    title="AI Engine Service API",
    description="Thin API layer exposing RAG and Recommendation services to the main backend.",
    version="1.0.0",
)
logger = logging.getLogger(__name__)
CACHE_TTL_SECONDS = 12 * 60 * 60
MAX_CACHE_ENTRIES = 250
response_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
response_cache_lock = RLock()
INSTANT_REPLIES = {
    "bonjour": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "bon jour": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "salut": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "hello": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "hi": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "coucou": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "bonsoir": "Bonsoir ! Comment puis-je vous aider ce soir ?",
    "bonne soiree": "Bonsoir ! Comment puis-je vous aider ce soir ?",
    "bye": "Au revoir ! N'hesitez pas a revenir si vous avez une autre question.",
    "goodbye": "Au revoir ! N'hesitez pas a revenir si vous avez une autre question.",
    "au revoir": "Au revoir ! N'hesitez pas a revenir si vous avez une autre question.",
    "comment allez vous": "Je vais bien, merci. Comment puis-je vous aider ?",
    "comment vas tu": "Je vais bien, merci. Comment puis-je vous aider ?",
    "ca va": "Je vais bien, merci. Comment puis-je vous aider ?",
    "comment ca va": "Je vais bien, merci. Comment puis-je vous aider ?",
    "vous allez bien": "Je vais bien, merci. Comment puis-je vous aider ?",
    "tu vas bien": "Je vais bien, merci. Comment puis-je vous aider ?",
    "merci": "Avec plaisir. Je reste disponible si vous avez une autre question.",
    "merci beaucoup": "Avec plaisir. Je reste disponible si vous avez une autre question.",
}
PROMPT_STOP_WORDS = {
    "the", "and", "for", "are", "you", "please", "tell", "about", "what", "which", "how",
    "est", "sont", "une", "des", "les", "aux", "avec", "pour", "dans", "sur", "par",
    "quel", "quelle", "quels", "quelles", "quoi", "comment", "pouvez", "peux", "vous",
    "merci", "svp", "donne", "donner", "dire", "explique", "expliquer", "concernant",
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
        },
    )

rag_service = RAGService()
recommendation_service = RecommendationService()
rag_startup_error: str | None = None


def _history_as_dicts(history: list[ConversationMessage]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in history]


def _normalize_message(value: str) -> str:
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFD", value or "")
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    normalized = normalized.lower()
    normalized = re.sub(r"[^\w\s']", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _instant_reply(message: str, mode: str) -> dict[str, Any] | None:
    normalized = _normalize_message(message)
    answer = INSTANT_REPLIES.get(normalized) or _find_fuzzy_instant_reply(normalized)
    if not answer:
        return None
    return {
        "mode": mode,
        "answer": answer,
        "decision_reason": "instant_greeting",
    }


def _find_fuzzy_instant_reply(normalized: str) -> str | None:
    if not _is_short_communication_phrase(normalized):
        return None

    for phrase, answer in INSTANT_REPLIES.items():
        allowed_distance = 1 if len(phrase) <= 8 else 2
        if abs(len(normalized) - len(phrase)) > allowed_distance:
            continue
        if _levenshtein_distance_at_most(normalized, phrase, allowed_distance):
            return answer
    return None


def _is_short_communication_phrase(normalized: str) -> bool:
    return len(normalized) <= 24 and len(normalized.split()) <= 4


def _levenshtein_distance_at_most(left: str, right: str, max_distance: int) -> bool:
    previous = list(range(len(right) + 1))
    current = [0] * (len(right) + 1)

    for left_index, left_char in enumerate(left, start=1):
        current[0] = left_index
        row_minimum = current[0]

        for right_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current[right_index] = min(
                current[right_index - 1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + substitution_cost,
            )
            row_minimum = min(row_minimum, current[right_index])

        if row_minimum > max_distance:
            return False

        previous, current = current, previous

    return previous[len(right)] <= max_distance


def _cache_key(request: OrchestratedAIRequest) -> str:
    history_key = "|".join(
        f"{item.role}:{_normalize_message(item.content)}"
        for item in request.conversation_history[-6:]
    )
    return "|".join(
        [
            request.mode,
            str(max(1, request.top_k)),
            _normalize_message(request.message),
            history_key,
        ]
    )


def _get_cached_response(key: str) -> dict[str, Any] | None:
    with response_cache_lock:
        cached = response_cache.get(key)
        if not cached:
            return None

        cached_at, payload = cached
        if monotonic() - cached_at > CACHE_TTL_SECONDS:
            response_cache.pop(key, None)
            return None

        response_cache.move_to_end(key)
        return dict(payload)


def _get_similar_cached_response(request: OrchestratedAIRequest) -> dict[str, Any] | None:
    candidate_mode = request.mode
    candidate_top_k = str(max(1, request.top_k))
    candidate_prompt = _normalize_message(request.message)

    with response_cache_lock:
        for cache_key, (cached_at, payload) in list(response_cache.items()):
            if monotonic() - cached_at > CACHE_TTL_SECONDS:
                response_cache.pop(cache_key, None)
                continue

            parts = cache_key.split("|", 3)
            if len(parts) < 3:
                continue

            cached_mode, cached_top_k, cached_prompt = parts[:3]
            if (
                cached_mode == candidate_mode
                and cached_top_k == candidate_top_k
                and _are_similar_prompts(cached_prompt, candidate_prompt)
            ):
                response_cache.move_to_end(cache_key)
                return dict(payload)

    return None


def _remember_response(key: str, payload: dict[str, Any]) -> None:
    with response_cache_lock:
        response_cache[key] = (monotonic(), dict(payload))
        response_cache.move_to_end(key)
        while len(response_cache) > MAX_CACHE_ENTRIES:
            response_cache.popitem(last=False)


def _are_similar_prompts(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True

    max_distance = max(2, min(12, max(len(left), len(right)) // 10))
    if abs(len(left) - len(right)) <= max_distance and _levenshtein_distance_at_most(
        left, right, max_distance
    ):
        return True

    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return False

    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = intersection / union if union else 0
    overlap = intersection / min(len(left_tokens), len(right_tokens))

    return jaccard >= 0.72 or (overlap >= 0.82 and intersection >= 4)


def _meaningful_tokens(normalized: str) -> set[str]:
    return {
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in PROMPT_STOP_WORDS
    }


@app.on_event("startup")
def startup_event() -> None:
    global rag_startup_error
    try:
        rag_service.initialize()
        rag_startup_error = None
    except Exception as exc:
        rag_startup_error = str(exc)
        logger.exception("Failed to initialize RAG service during startup")


@app.get("/ai/health")
def health() -> dict[str, Any]:
    recommendation_health: dict[str, Any]
    try:
        recommendation_health = recommendation_service.health()
    except Exception as exc:
        recommendation_health = {
            "mode": "recommendation",
            "healthy": False,
            "error": str(exc),
        }

    rag_health = rag_service.health()
    if rag_startup_error:
        rag_health["healthy"] = False
        rag_health["startup_error"] = rag_startup_error

    rag_healthy = bool(rag_health.get("healthy", False))
    recommendation_healthy = bool(recommendation_health.get("healthy", False))
    overall_status = "ok" if rag_healthy and recommendation_healthy else "degraded"
    return {
        "status": overall_status,
        "services": {
            "rag": rag_health,
            "recommendation": recommendation_health,
        },
    }


@app.post("/ai/rag/respond")
def rag_respond(request: AIRequest) -> dict[str, Any]:
    try:
        result = rag_service.answer(
            question=request.message,
            conversation_history=_history_as_dicts(request.conversation_history),
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG service failed: {exc}") from exc

    return result.model_dump()


@app.post("/ai/recommendation/respond")
def recommendation_respond(request: AIRequest) -> dict[str, Any]:
    try:
        result = recommendation_service.recommend(
            query=request.message,
            conversation_history=_history_as_dicts(request.conversation_history),
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation service failed: {exc}") from exc

    payload = result.model_dump()
    payload["mode"] = "recommendation"
    return payload


@app.post("/ai/respond")
def respond(request: OrchestratedAIRequest) -> dict[str, Any]:
    instant_response = _instant_reply(request.message, request.mode)
    if instant_response:
        return instant_response

    cache_key = _cache_key(request)
    cached_response = _get_cached_response(cache_key)
    if cached_response:
        cached_response["cache_hit"] = True
        return cached_response

    similar_cached_response = _get_similar_cached_response(request)
    if similar_cached_response:
        similar_cached_response["cache_hit"] = True
        similar_cached_response["cache_match"] = "similar_prompt"
        return similar_cached_response

    if request.mode == "rag":
        payload = rag_respond(request)
        _remember_response(cache_key, payload)
        return payload
    if request.mode == "recommendation":
        payload = recommendation_respond(request)
        _remember_response(cache_key, payload)
        return payload
    raise HTTPException(status_code=400, detail=f"Unsupported mode '{request.mode}'")
