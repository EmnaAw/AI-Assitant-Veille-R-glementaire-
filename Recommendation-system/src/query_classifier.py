from dataclasses import dataclass
import re
import unicodedata


GAP_TYPE_AUTRE = "autre"
AMBIGUOUS_GAP_CONFIDENCE_THRESHOLD = 0.8
AMBIGUOUS_GAP_MARGIN = 0.06


@dataclass(frozen=True)
class GapTypeResult:
    gap_type: str
    confidence: float
    matched_rules: tuple[str, ...] = ()


GAP_TYPE_RULES: dict[str, tuple[tuple[str, float], ...]] = {
    "nomination_responsable_securite": (
        (r"\bfiche nominative\b.*\bresponsable securite\b", 0.99),
        (r"\bfiche nomination\b.*\bresponsable securite\b", 0.99),
        (r"\bnomination\b.*\bresponsable securite\b", 0.98),
        (r"\bresponsable securite\b.*\bnomm\w*\b", 0.94),
        (r"\bresponsable sst\b.*\bnomm\w*\b", 0.94),
    ),
    "equipe_role_securite": (
        (r"\bequipe de securite\b", 0.98),
        (r"\bchef d equipe\b", 0.70),
        (r"\bresponsable securite\b", 0.74),
        (r"\bdesignation\b", 0.62),
        (r"\bdesigner\b", 0.62),
    ),
    "registre_securite": (
        (r"\bregistre\b.*\bsecurite\b", 0.98),
        (r"\bmain courante\b", 0.58),
    ),
    "formation": (
        (r"\bformation\b", 0.92),
        (r"\bgestes et postures\b", 0.96),
        (r"\bmanutention manuelle\b", 0.70),
        (r"\bsensibilisation\b", 0.58),
    ),
    "autorisation_administratif": (
        (r"\bautorisation\b", 0.95),
        (r"\bdeclaration\b", 0.82),
        (r"\bdossier\b", 0.72),
        (r"\bbureau detude\b", 0.82),
        (r"\bexploitation\b", 0.65),
    ),
    "dechets_sanitaires": (
        (r"\bdechets? sanitaires?\b", 0.98),
        (r"\bdechets? de soins\b", 0.95),
        (r"\bconteneur specifique\b", 0.88),
        (r"\bconteneur\b", 0.52),
    ),
    "analyse_controle": (
        (r"\banalyse\b", 0.66),
        (r"\bcontrole\b", 0.62),
        (r"\bmesure\b", 0.58),
        (r"\bair ambiant\b", 0.96),
        (r"\bverification\b", 0.50),
    ),
    "equipement_securite": (
        (r"\beclairage de securite\b", 0.97),
        (r"\beclairage de secours\b", 0.95),
        (r"\bbloc autonome\b", 0.92),
        (r"\bextincteur\b", 0.78),
        (r"\balarme\b", 0.62),
    ),
    "personnes_handicapees": (
        (r"\bpersonnes? handicapees?\b", 0.97),
        (r"\bhandicapees?\b", 0.90),
    ),
}


def _strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def _normalize_for_classification(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = _strip_accents(value).lower().strip()
    value = re.sub(r"\b([cdjlmnst])'\s*(\w+)", r"\1\2", value, flags=re.IGNORECASE)
    value = re.sub(r"[^\w\s'-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def classify_gap_type(text: str) -> GapTypeResult:
    normalized = _normalize_for_classification(text)
    if not normalized:
        return GapTypeResult(gap_type=GAP_TYPE_AUTRE, confidence=0.0, matched_rules=())

    scored_gap_types: list[tuple[str, float, list[str]]] = []

    for gap_type, rules in GAP_TYPE_RULES.items():
        current_matches: list[str] = []
        current_score = 0.0
        for pattern, weight in rules:
            if re.search(pattern, normalized):
                current_score = max(current_score, weight)
                current_matches.append(pattern)

        if current_score > 0.0:
            scored_gap_types.append((gap_type, current_score, current_matches))

    if not scored_gap_types:
        return GapTypeResult(gap_type=GAP_TYPE_AUTRE, confidence=0.0, matched_rules=())

    scored_gap_types.sort(key=lambda item: item[1], reverse=True)
    best_gap_type, best_confidence, matched_rules = scored_gap_types[0]
    second_best = scored_gap_types[1] if len(scored_gap_types) > 1 else None

    if (
        second_best is not None
        and best_confidence >= AMBIGUOUS_GAP_CONFIDENCE_THRESHOLD
        and second_best[1] >= AMBIGUOUS_GAP_CONFIDENCE_THRESHOLD
        and (best_confidence - second_best[1]) <= AMBIGUOUS_GAP_MARGIN
    ):
        combined_rules = tuple(dict.fromkeys(matched_rules + second_best[2]))
        return GapTypeResult(
            gap_type=GAP_TYPE_AUTRE,
            confidence=0.0,
            matched_rules=combined_rules,
        )

    return GapTypeResult(
        gap_type=best_gap_type,
        confidence=round(best_confidence, 4),
        matched_rules=tuple(matched_rules),
    )


def infer_gap_type_from_record(nc: str, plan: str | None = None) -> GapTypeResult:
    combined = f"{nc or ''} {plan or ''}".strip()
    return classify_gap_type(combined)
