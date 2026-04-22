from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .text_utils import clean_client_text


RecommendationMode = Literal["verified", "ambiguous", "advisory", "no_match"]


class CandidateMatch(BaseModel):
    ncid: str
    nc: str
    distance: float


class CandidateEvidence(BaseModel):
    ncid: str
    nc: str
    distance: float
    official_plan: str | None = None
    gap_type: str | None = None
    semantic_score: float | None = None
    fuzzy_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    gap_type_alignment: float | None = None
    group_key: str | None = None
    group_size: int | None = None
    unique_plan_count: int | None = None
    exact_query_match: bool = False
    deterministic_score: float | None = None


class RecommendationResult(BaseModel):
    mode: RecommendationMode
    query: str
    normalized_query: str
    decision_reason: str
    query_gap_type: str | None = None
    matched_ncid: str | None = None
    matched_nc: str | None = None
    official_plan: str | None = None
    display_plan_fr: str | None = None
    explanation_fr: str | None = None
    matched_gap_type: str | None = None
    advisory_disclaimer: str | None = None
    confidence_distance: float | None = None
    rerank_score: float | None = None
    top_candidates: list[CandidateEvidence] = Field(default_factory=list)
    ambiguous_matches: list[CandidateEvidence] = Field(default_factory=list)

    @field_validator("display_plan_fr", "explanation_fr", "advisory_disclaimer", mode="before")
    @classmethod
    def _clean_client_facing_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return clean_client_text(value)
