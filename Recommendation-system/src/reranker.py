import re
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_set(text: str) -> set[str]:
    return set(normalize_text(text).split())


def jaccard_score(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def sequence_score(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def semantic_score_from_distance(distance: float) -> float:
    return max(0.0, 1.0 - float(distance))


STATUS_PATTERNS = {
    "missing": (
        r"\babsence\b",
        r"\babsent(?:e|s)?\b",
        r"\bsans\b",
        r"\bmanque\b",
    ),
    "not_done": (
        r"\bnest pas realis\w*\b",
        r"\bpas realis\w*\b",
        r"\bpas fait\b",
        r"\bnon suivi\b",
        r"\bnon realise\b",
        r"\bnon effect\w*\b",
    ),
    "non_compliant": (
        r"\bnon conforme\b",
        r"\bpas conforme\b",
    ),
    "insufficient": (
        r"\binsuffisant\w*\b",
        r"\bnatteint pas\b",
        r"\bnombre insuffisant\b",
    ),
    "damaged": (
        r"\bendommage\w*\b",
        r"\babim\w*\b",
        r"\bdefect\w*\b",
    ),
    "administrative": (
        r"\bautorisation\b",
        r"\bdossier\b",
        r"\bbureau detude\b",
        r"\bdeclaration\b",
    ),
}


def status_features(text: str) -> set[str]:
    normalized = normalize_text(text)
    features = set()
    for label, patterns in STATUS_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            features.add(label)
    return features


def specificity_bonus(query: str, candidate: str) -> float:
    query_tokens = token_set(query)
    candidate_tokens = token_set(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    extra_detail = len(candidate_tokens) - len(query_tokens)
    return max(0.0, min(0.05, extra_detail * 0.01))


def hybrid_rerank(query: str, candidates: list[dict], query_gap_type: str | None = None) -> list[dict]:
    reranked = []
    query_tokens = token_set(query)
    query_status = status_features(query)

    for c in candidates:
        semantic = semantic_score_from_distance(c["distance"])
        fuzzy = sequence_score(query, c["nc"])
        lexical = jaccard_score(query, c["nc"])
        candidate_tokens = token_set(c["nc"])
        candidate_status = status_features(c["nc"])
        candidate_gap_type = c.get("gap_type")
        containment = 1.0 if query_tokens and query_tokens <= candidate_tokens else 0.0
        exact_bonus = 1.0 if normalize_text(query) == normalize_text(c["nc"]) else 0.0
        ambiguity_penalty = 0.08 if c.get("unique_plan_count", 1) > 1 else 0.0

        status_alignment = 0.0
        if query_status:
            overlap = query_status & candidate_status
            if overlap:
                status_alignment = 1.0
            elif candidate_status:
                status_alignment = -0.35

        gap_type_alignment = 0.0
        if query_gap_type and query_gap_type != "autre":
            if candidate_gap_type == query_gap_type:
                gap_type_alignment = 1.0
            elif candidate_gap_type and candidate_gap_type != "autre":
                gap_type_alignment = -0.2

        final_score = (
            0.40 * semantic
            + 0.18 * fuzzy
            + 0.18 * lexical
            + 0.08 * containment
            + 0.05 * exact_bonus
            + 0.10 * status_alignment
            + 0.08 * gap_type_alignment
            + specificity_bonus(query, c["nc"])
            - ambiguity_penalty
        )

        reranked.append(
            {
                **c,
                "semantic_score": round(semantic, 4),
                "fuzzy_score": round(fuzzy, 4),
                "lexical_score": round(lexical, 4),
                "status_alignment": round(status_alignment, 4),
                "gap_type_alignment": round(gap_type_alignment, 4),
                "rerank_score": round(max(0.0, final_score), 4),
            }
        )

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked
