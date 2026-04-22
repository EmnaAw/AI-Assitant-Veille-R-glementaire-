from .config import (
    AMBIGUITY_RERANK_GAP_THRESHOLD,
    DATA_PATH,
    DEFAULT_SCORE_THRESHOLD,
    EXACT_MATCH_DISTANCE_THRESHOLD,
    EXACT_MATCH_RERANK_THRESHOLD,
    SHORTLIST_ACCEPT_MARGIN_THRESHOLD,
    SHORTLIST_ACCEPT_SCORE_THRESHOLD,
    TOP1_ACCEPT_DISTANCE_THRESHOLD,
    TOP1_ACCEPT_RERANK_THRESHOLD,
    TOP_K,
)
from .generator import Generator
from .query_classifier import classify_gap_type
from .query_normalizer import normalize_query_text
from .reranker import hybrid_rerank
from .retriever import Retriever
from .schemas import CandidateEvidence, RecommendationResult
from .text_utils import clean_client_text
from .truth_lookup import TruthLookup

DISAMBIGUATION_STOPWORDS = {
    "absence",
    "cas",
    "conformite",
    "de",
    "des",
    "du",
    "dune",
    "et",
    "la",
    "le",
    "les",
    "non",
    "pour",
    "sur",
    "une",
}

NEGATION_CUES = {
    "absence",
    "absent",
    "absente",
    "non",
    "pas",
    "na",
    "nest",
    "insuffisant",
    "insuffisante",
    "nonconforme",
    "conforme",
    "realise",
    "realisee",
    "realisee",
    "endommages",
}


class LegalRecommendationPipeline:
    def __init__(
        self,
        data_path=DATA_PATH,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
        retriever: Retriever | None = None,
        lookup: TruthLookup | None = None,
        generator: Generator | None = None,
    ):
        self.lookup = lookup or TruthLookup(data_path)
        self.retriever = retriever or Retriever(data_path=data_path, lookup=self.lookup)
        self.generator = generator or Generator()
        self.score_threshold = score_threshold

    def _normalize_group_text(self, text: str) -> str:
        return self.lookup.normalize_nc(text)

    def _enrich_candidate(self, ncid: str, nc: str, distance: float) -> dict:
        truth = self.lookup.safe_get_by_id(ncid)
        return {
            "ncid": ncid,
            "nc": nc,
            "distance": distance,
            "official_plan": truth["Plan"] if truth else None,
            "gap_type": truth.get("gap_type") if truth else None,
        }

    def _as_candidate_models(self, candidates: list[dict]) -> list[CandidateEvidence]:
        return [CandidateEvidence(**candidate) for candidate in candidates]

    def _salient_tokens(self, text: str) -> set[str]:
        tokens = set(self.lookup.normalize_nc(text).split())
        return {
            token
            for token in tokens
            if len(token) > 2 and token not in DISAMBIGUATION_STOPWORDS and not token.isdigit()
        }

    def _negation_tokens(self, text: str) -> set[str]:
        tokens = set(self.lookup.normalize_nc(text).split())
        return {token for token in tokens if token in NEGATION_CUES}

    def _deterministic_candidate_score(self, normalized_query: str, candidate: dict) -> float:
        query_tokens = self._salient_tokens(normalized_query)
        candidate_tokens = self._salient_tokens(candidate["nc"])
        query_negations = self._negation_tokens(normalized_query)
        candidate_negations = self._negation_tokens(candidate["nc"])

        if not query_tokens or not candidate_tokens:
            return round(candidate["rerank_score"], 4)

        overlap = query_tokens & candidate_tokens
        total_weight = sum(self.lookup.token_weight(token) for token in query_tokens)
        overlap_weight = sum(self.lookup.token_weight(token) for token in overlap)
        weighted_coverage = overlap_weight / total_weight if total_weight else 0.0

        containment_bonus = 1.0 if query_tokens <= candidate_tokens else 0.0
        exact_bonus = 1.0 if self.lookup.normalize_nc(candidate["nc"]) == normalized_query else 0.0
        missing_penalty = max(0.0, 1.0 - weighted_coverage)
        negation_bonus = 0.0
        if query_negations:
            negation_bonus = 1.0 if query_negations & candidate_negations else -0.4

        score = (
            0.55 * weighted_coverage
            + 0.15 * candidate["fuzzy_score"]
            + 0.10 * candidate["lexical_score"]
            + 0.10 * candidate["semantic_score"]
            + 0.05 * containment_bonus
            + 0.05 * exact_bonus
            + 0.08 * negation_bonus
            - 0.10 * missing_penalty
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _apply_shortlist_disambiguation(
        self,
        normalized_query: str,
        reranked: list[dict],
    ) -> tuple[dict | None, list[dict]]:
        shortlist = []
        for candidate in reranked[:3]:
            enriched = {
                **candidate,
                "deterministic_score": self._deterministic_candidate_score(normalized_query, candidate),
            }
            shortlist.append(enriched)

        shortlist.sort(
            key=lambda item: (item["deterministic_score"], item["rerank_score"]),
            reverse=True,
        )

        best = shortlist[0] if shortlist else None
        second = shortlist[1] if len(shortlist) > 1 else None
        if best is None:
            return None, reranked

        margin = best["deterministic_score"] - (second["deterministic_score"] if second else 0.0)
        if (
            best["deterministic_score"] >= SHORTLIST_ACCEPT_SCORE_THRESHOLD
            and margin >= SHORTLIST_ACCEPT_MARGIN_THRESHOLD
        ):
            reranked = [{**candidate} for candidate in reranked]
            reranked_by_id = {candidate["ncid"]: candidate for candidate in reranked}
            if best["ncid"] in reranked_by_id:
                reranked_by_id[best["ncid"]]["deterministic_score"] = best["deterministic_score"]
            return best, list(reranked_by_id.values())

        reranked_by_id = {candidate["ncid"]: {**candidate} for candidate in reranked}
        for candidate in shortlist:
            if candidate["ncid"] in reranked_by_id:
                reranked_by_id[candidate["ncid"]]["deterministic_score"] = candidate["deterministic_score"]
        return None, list(reranked_by_id.values())

    def _prefer_specific_shortlist_candidate(
        self,
        normalized_query: str,
        reranked: list[dict],
    ) -> tuple[dict | None, list[dict]]:
        shortlist = []
        for candidate in reranked[:3]:
            shortlist.append(
                {
                    **candidate,
                    "deterministic_score": self._deterministic_candidate_score(normalized_query, candidate),
                }
            )

        shortlist.sort(
            key=lambda item: (item["deterministic_score"], item["rerank_score"]),
            reverse=True,
        )

        best = shortlist[0] if shortlist else None
        current_best = reranked[0] if reranked else None
        if best is None or current_best is None:
            return None, reranked

        reranked_by_id = {candidate["ncid"]: {**candidate} for candidate in reranked}
        for candidate in shortlist:
            if candidate["ncid"] in reranked_by_id:
                reranked_by_id[candidate["ncid"]]["deterministic_score"] = candidate["deterministic_score"]

        det_gap = best["deterministic_score"] - float(
            reranked_by_id.get(current_best["ncid"], {}).get("deterministic_score", current_best["rerank_score"])
        )
        if best["ncid"] != current_best["ncid"] and best["deterministic_score"] >= 0.48 and det_gap >= 0.08:
            return best, list(reranked_by_id.values())

        return None, list(reranked_by_id.values())

    def _effective_distance_threshold(self, normalized_query: str) -> float:
        word_count = len(normalized_query.split())
        if word_count <= 4:
            return 0.45
        if word_count <= 7:
            return 0.42
        return self.score_threshold

    def _build_verified_result(
        self,
        *,
        query: str,
        normalized_query: str,
        query_gap_type: str | None,
        truth: dict,
        top_candidates: list[dict] | None,
        best: dict | None,
        decision_reason: str,
        with_generation: bool,
    ) -> RecommendationResult:
        explanation = None
        display_plan = clean_client_text(truth["Plan"])
        if with_generation:
            display_plan = self.generator.generate_french_action(truth["Plan"])
            explanation = self.generator.generate_french_explanation(
                nc=truth["NC"],
                plan=truth["Plan"],
                gap_type=truth.get("gap_type", query_gap_type or "autre"),
            )

        return RecommendationResult(
            mode="verified",
            query=query,
            normalized_query=normalized_query,
            decision_reason=decision_reason,
            query_gap_type=query_gap_type,
            matched_ncid=truth["NCid"],
            matched_nc=truth["NC"],
            official_plan=truth["Plan"],
            display_plan_fr=display_plan,
            explanation_fr=explanation,
            matched_gap_type=truth.get("gap_type"),
            confidence_distance=best["distance"] if best else 0.0,
            rerank_score=best["rerank_score"] if best else 1.0,
            top_candidates=self._as_candidate_models(top_candidates or []),
        )

    def _build_no_match_result(
        self,
        query: str,
        normalized_query: str,
        query_gap_type: str | None,
    ) -> RecommendationResult:
        return RecommendationResult(
            mode="no_match",
            query=query,
            normalized_query=normalized_query,
            decision_reason="no_candidates",
            query_gap_type=query_gap_type,
            explanation_fr=(
                "Aucune correspondance suffisamment proche n'a été trouvée dans la base. "
                "Merci de reformuler la non-conformité avec davantage de précision."
            ),
            advisory_disclaimer=(
                "Aucune action officielle n'a été validée pour cette requête."
            ),
        )

    def _build_advisory_result(
        self,
        query: str,
        normalized_query: str,
        query_gap_type: str | None,
        top_candidates: list[dict],
        decision_reason: str,
        best: dict | None = None,
    ) -> RecommendationResult:
        explanation = self.generator.generate_french_advisory(
            query=query,
            normalized_query=normalized_query,
            top_candidates=top_candidates[:3],
        )

        return RecommendationResult(
            mode="advisory",
            query=query,
            normalized_query=normalized_query,
            decision_reason=decision_reason,
            query_gap_type=query_gap_type,
            explanation_fr=explanation,
            advisory_disclaimer=(
                "Cette réponse est une suggestion non confirmée et ne constitue pas "
                "une recommandation réglementaire validée."
            ),
            confidence_distance=best["distance"] if best else None,
            rerank_score=best["rerank_score"] if best else None,
            top_candidates=self._as_candidate_models(top_candidates),
        )

    def _build_ambiguous_result(
        self,
        query: str,
        normalized_query: str,
        query_gap_type: str | None,
        top_candidates: list[dict],
        conflicting_matches: list[dict],
        best: dict,
    ) -> RecommendationResult:
        explanation = self.generator.generate_french_ambiguity(
            query=query,
            conflicting_matches=conflicting_matches,
        )

        return RecommendationResult(
            mode="ambiguous",
            query=query,
            normalized_query=normalized_query,
            decision_reason="conflicting_official_matches",
            query_gap_type=query_gap_type,
            explanation_fr=explanation,
            advisory_disclaimer=(
                "Plusieurs actions officielles restent possibles. "
                "Un contexte supplémentaire ou une validation humaine est nécessaire."
            ),
            confidence_distance=best["distance"],
            rerank_score=best["rerank_score"],
            top_candidates=self._as_candidate_models(top_candidates),
            ambiguous_matches=self._as_candidate_models(conflicting_matches),
        )

    def run(
        self,
        query: str,
        top_k: int = TOP_K,
        with_generation: bool = True,
    ) -> RecommendationResult:
        normalized_query = normalize_query_text(query)
        query_gap = classify_gap_type(normalized_query or query)
        alias_truth = self.lookup.resolve_query_alias(normalized_query)
        if alias_truth is not None:
            alias_candidates = [
                {
                    "ncid": alias_truth["NCid"],
                    "nc": alias_truth["NC"],
                    "official_plan": alias_truth["Plan"],
                    "gap_type": alias_truth.get("gap_type"),
                    "distance": 0.0,
                    "semantic_score": 1.0,
                    "fuzzy_score": 1.0,
                    "lexical_score": 1.0,
                    "rerank_score": 1.0,
                    "deterministic_score": 1.0,
                    "exact_query_match": True,
                }
            ]
            return self._build_verified_result(
                query=query,
                normalized_query=normalized_query,
                query_gap_type=query_gap.gap_type,
                truth=alias_truth,
                top_candidates=alias_candidates,
                best=alias_candidates[0],
                decision_reason="verified_query_alias",
                with_generation=with_generation,
            )

        direct_truth, direct_truths = self.lookup.resolve_records_by_normalized_nc(normalized_query)

        if direct_truth is not None:
            direct_candidates = [
                {
                    "ncid": record["NCid"],
                    "nc": record["NC"],
                    "official_plan": record["Plan"],
                    "gap_type": record.get("gap_type"),
                    "distance": 0.0,
                    "semantic_score": 1.0,
                    "fuzzy_score": 1.0,
                    "lexical_score": 1.0,
                    "rerank_score": 1.0,
                    "group_key": self._normalize_group_text(record["NC"]),
                    "group_size": len(direct_truths),
                    "unique_plan_count": 1,
                    "exact_query_match": True,
                }
                for record in direct_truths
            ]
            return self._build_verified_result(
                query=query,
                normalized_query=normalized_query,
                query_gap_type=query_gap.gap_type,
                truth=direct_truth,
                top_candidates=direct_candidates,
                best=direct_candidates[0],
                decision_reason="verified_exact_lookup",
                with_generation=with_generation,
            )

        if direct_truths:
            conflicting_matches = [
                {
                    "ncid": record["NCid"],
                    "nc": record["NC"],
                    "official_plan": record["Plan"],
                    "gap_type": record.get("gap_type"),
                    "distance": 0.0,
                    "rerank_score": 1.0,
                    "group_key": self._normalize_group_text(record["NC"]),
                    "group_size": len(direct_truths),
                    "unique_plan_count": len(
                        {self.lookup.normalize_plan(resolved["Plan"]) for resolved in direct_truths}
                    ),
                    "exact_query_match": True,
                }
                for record in direct_truths
            ]
            return self._build_ambiguous_result(
                query=query,
                normalized_query=normalized_query,
                query_gap_type=query_gap.gap_type,
                top_candidates=conflicting_matches,
                conflicting_matches=conflicting_matches,
                best=conflicting_matches[0],
            )

        candidates = self.retriever.search(
            query=normalized_query,
            top_k=top_k,
            query_gap=query_gap,
        )

        if not candidates:
            return self._build_no_match_result(query, normalized_query, query_gap.gap_type)

        candidate_dicts = [
            self._enrich_candidate(candidate.ncid, candidate.nc, candidate.distance)
            for candidate in candidates
        ]

        reranked = hybrid_rerank(
            normalized_query,
            candidate_dicts,
            query_gap_type=query_gap.gap_type,
        )
        preferred_truth, reranked = self._prefer_specific_shortlist_candidate(normalized_query, reranked)
        if preferred_truth is not None:
            truth = self.lookup.safe_get_by_id(preferred_truth["ncid"])
            if truth is not None:
                return self._build_verified_result(
                    query=query,
                    normalized_query=normalized_query,
                    query_gap_type=query_gap.gap_type,
                    truth=truth,
                    top_candidates=reranked,
                    best=preferred_truth,
                    decision_reason="verified_specific_shortlist_match",
                    with_generation=with_generation,
                )

        best = reranked[0]
        second = reranked[1] if len(reranked) > 1 else None

        is_high_confidence_exact_match = (
            self._normalize_group_text(best["nc"]) == normalized_query
            and best["distance"] <= EXACT_MATCH_DISTANCE_THRESHOLD
            and best["rerank_score"] >= EXACT_MATCH_RERANK_THRESHOLD
        )

        passes_relaxed_top1_gate = (
            best["distance"] <= TOP1_ACCEPT_DISTANCE_THRESHOLD
            and best["rerank_score"] >= TOP1_ACCEPT_RERANK_THRESHOLD
        )

        if (
            best["distance"] > self._effective_distance_threshold(normalized_query)
            and not is_high_confidence_exact_match
            and not passes_relaxed_top1_gate
        ):
            shortlist_truth, reranked = self._apply_shortlist_disambiguation(normalized_query, reranked)
            if shortlist_truth is not None:
                truth = self.lookup.safe_get_by_id(shortlist_truth["ncid"])
                if truth is not None:
                    return self._build_verified_result(
                        query=query,
                        normalized_query=normalized_query,
                        query_gap_type=query_gap.gap_type,
                        truth=truth,
                        top_candidates=reranked,
                        best=shortlist_truth,
                        decision_reason="verified_shortlist_disambiguation",
                        with_generation=with_generation,
                    )
            return self._build_advisory_result(
                query=query,
                normalized_query=normalized_query,
                query_gap_type=query_gap.gap_type,
                top_candidates=reranked,
                decision_reason="weak_retrieval",
                best=best,
            )

        if second is not None:
            score_gap = best["rerank_score"] - second["rerank_score"]
            best_det = best.get("deterministic_score")
            second_det = second.get("deterministic_score")
            deterministic_gap = None
            if best_det is not None and second_det is not None:
                deterministic_gap = best_det - second_det

            if (
                score_gap < AMBIGUITY_RERANK_GAP_THRESHOLD
                and not is_high_confidence_exact_match
                and not (
                    best_det is not None
                    and best_det >= 0.48
                    and deterministic_gap is not None
                    and deterministic_gap >= 0.08
                )
            ):
                return self._build_advisory_result(
                    query=query,
                    normalized_query=normalized_query,
                    query_gap_type=query_gap.gap_type,
                    top_candidates=reranked,
                    decision_reason="near_tie_between_candidates",
                    best=best,
                )

        truth = self.lookup.safe_get_by_id(best["ncid"])
        if truth is None:
            return self._build_advisory_result(
                query=query,
                normalized_query=normalized_query,
                query_gap_type=query_gap.gap_type,
                top_candidates=reranked,
                decision_reason="missing_truth_record",
                best=best,
            )

        return self._build_verified_result(
            query=query,
            normalized_query=normalized_query,
            query_gap_type=query_gap.gap_type,
            truth=truth,
            top_candidates=reranked,
            best=best,
            decision_reason=(
                "verified_relaxed_top1_gate"
                if passes_relaxed_top1_gate and not is_high_confidence_exact_match
                else "verified_unique_match"
            ),
            with_generation=with_generation,
        )
