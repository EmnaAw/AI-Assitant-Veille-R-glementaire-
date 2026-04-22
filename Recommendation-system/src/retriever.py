from difflib import SequenceMatcher

from sentence_transformers import util

from .config import DATA_PATH
from .embedder import Embedder
from .query_classifier import GAP_TYPE_AUTRE, GapTypeResult
from .schemas import CandidateMatch
from .truth_lookup import TruthLookup
from .vector_store import VectorStore


class Retriever:
    def __init__(self, data_path=DATA_PATH, lookup: TruthLookup | None = None):
        self.lookup = lookup or TruthLookup(data_path)
        self.embedder = None
        self.store = None
        self.store_error = None
        self.store_health = None
        self.embedding_backend_available = False
        self.store_in_sync = False
        self.semantic_mode = "disabled"
        self.records = self._build_records()
        self._record_embeddings = None

        try:
            self.store = VectorStore()
            self.store_health = self.store.health()
            self.store_in_sync = self._is_store_in_sync()
            if self.store_in_sync:
                self.embedding_backend_available = True
                self.semantic_mode = "chroma"
        except Exception:
            self.embedder = None
            self.store = None
            self.store_health = None
            self.store_error = "persistent_store_unavailable"
            self.embedding_backend_available = False
            self.store_in_sync = False
            self.semantic_mode = "disabled"

        if not self.embedding_backend_available:
            self._enable_in_memory_semantic_fallback()

        if self.store is not None and not self.store_in_sync:
            self.store_error = "persistent_store_out_of_sync"

    def _build_records(self) -> list[dict]:
        records = []
        for idx, record in enumerate(self.lookup.by_ncid.values()):
            normalized_nc = self.lookup.normalize_nc(record["NC"])
            records.append(
                {
                    "idx": idx,
                    "ncid": str(record["NCid"]),
                    "nc": record["NC"],
                    "normalized_nc": normalized_nc,
                    "tokens": set(normalized_nc.split()),
                    "gap_type": record.get("gap_type", GAP_TYPE_AUTRE),
                }
            )
        return records

    def _is_store_in_sync(self) -> bool:
        if self.store is None:
            return False

        try:
            store_ids = set(self.store.get_ids())
        except Exception:
            return False

        dataset_ids = set(self.lookup.by_ncid.keys())
        return store_ids == dataset_ids

    def _enable_in_memory_semantic_fallback(self) -> None:
        try:
            self.embedder = Embedder()
            self._record_embeddings = self.embedder.encode(
                [record["nc"] for record in self.records]
            )
            self.embedding_backend_available = True
            self.semantic_mode = "in_memory"
        except Exception:
            self.embedder = None
            self._record_embeddings = None
            self.embedding_backend_available = False
            self.semantic_mode = "disabled"

    def health(self) -> dict:
        dataset_ids = set(self.lookup.by_ncid.keys())
        store_ids: set[str] = set()
        if self.store is not None:
            try:
                store_ids = set(self.store.get_ids())
            except Exception:
                store_ids = set()

        return {
            "semantic_mode": self.semantic_mode,
            "embedding_backend_available": self.embedding_backend_available,
            "store_in_sync": self.store_in_sync,
            "store_error": self.store_error,
            "record_count": len(self.records),
            "dataset_id_count": len(dataset_ids),
            "store_id_count": len(store_ids),
            "missing_store_ids": sorted(dataset_ids - store_ids)[:10],
            "extra_store_ids": sorted(store_ids - dataset_ids)[:10],
            "store": self.store_health,
        }

    def _candidate_pool(self, query_gap: GapTypeResult | None, top_k: int) -> list[dict]:
        if (
            query_gap is None
            or query_gap.gap_type == GAP_TYPE_AUTRE
            or query_gap.confidence < 0.78
        ):
            return self.records

        matching = [record for record in self.records if record["gap_type"] == query_gap.gap_type]
        if len(matching) >= max(top_k * 3, 8):
            return matching
        return self.records

    def _gap_type_bonus(self, candidate_gap_type: str, query_gap: GapTypeResult | None) -> float:
        if query_gap is None or query_gap.gap_type == GAP_TYPE_AUTRE:
            return 0.0
        if candidate_gap_type == query_gap.gap_type:
            return min(0.10, 0.04 + (0.05 * query_gap.confidence))
        if candidate_gap_type != GAP_TYPE_AUTRE:
            return -0.03
        return 0.0

    def _lexical_candidates(
        self,
        query: str,
        top_k: int,
        query_gap: GapTypeResult | None,
    ) -> list[tuple[str, str, float, str]]:
        query_norm = self.lookup.normalize_nc(query)
        query_tokens = set(query_norm.split())
        scored = []
        candidate_pool = self._candidate_pool(query_gap, top_k)

        for record in candidate_pool:
            candidate_nc = record["nc"]
            candidate_norm = record["normalized_nc"]
            candidate_tokens = record["tokens"]
            sequence = SequenceMatcher(None, query_norm, candidate_norm).ratio()
            lexical = 0.0
            if query_tokens and candidate_tokens:
                lexical = len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)
            exact = 1.0 if query_norm == candidate_norm else 0.0
            containment = 1.0 if query_tokens and query_tokens <= candidate_tokens else 0.0
            score = (
                0.45 * sequence
                + 0.25 * lexical
                + 0.20 * exact
                + 0.10 * containment
                + self._gap_type_bonus(record["gap_type"], query_gap)
            )
            scored.append((record["ncid"], candidate_nc, score, record["gap_type"]))

        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[: max(top_k * 4, 10)]

    def _semantic_candidates(
        self,
        query: str,
        top_k: int,
        query_gap: GapTypeResult | None,
    ) -> list[tuple[str, str, float, str]]:
        if not self.embedding_backend_available:
            return []

        if self.embedder is None:
            try:
                self.embedder = Embedder()
            except Exception:
                self.embedding_backend_available = False
                self.semantic_mode = "disabled"
                return []

        query_embedding = self.embedder.encode([query])
        if self.store is not None and self.store_in_sync:
            where = None
            if (
                query_gap is not None
                and query_gap.gap_type != GAP_TYPE_AUTRE
                and query_gap.confidence >= 0.88
            ):
                where = {"gap_type": query_gap.gap_type}

            try:
                raw = self.store.query(
                    query_embedding=query_embedding,
                    top_k=max(top_k * 4, 10),
                    where=where,
                )
            except Exception:
                raw = self.store.query(
                    query_embedding=query_embedding,
                    top_k=max(top_k * 4, 10),
                )

            ids = raw.get("ids", [[]])[0]
            docs = raw.get("documents", [[]])[0]
            distances = raw.get("distances", [[]])[0]

            if not ids and where is not None:
                raw = self.store.query(
                    query_embedding=query_embedding,
                    top_k=max(top_k * 4, 10),
                )
                ids = raw.get("ids", [[]])[0]
                docs = raw.get("documents", [[]])[0]
                distances = raw.get("distances", [[]])[0]

            semantic_results = []
            for ncid, nc, distance in zip(ids, docs, distances):
                truth = self.lookup.safe_get_by_id(str(ncid))
                semantic_results.append(
                    (
                        str(ncid),
                        nc,
                        max(0.0, 1.0 - float(distance)),
                        truth.get("gap_type", GAP_TYPE_AUTRE) if truth else GAP_TYPE_AUTRE,
                    )
                )
            return semantic_results

        candidate_pool = self._candidate_pool(query_gap, top_k)
        if self._record_embeddings is None:
            self._record_embeddings = self.embedder.encode(
                [record["nc"] for record in self.records]
            )

        candidate_indexes = [record["idx"] for record in candidate_pool]
        candidate_embeddings = self._record_embeddings[candidate_indexes]
        similarities = util.cos_sim(query_embedding, candidate_embeddings)[0].tolist()
        scored = [
            (record["ncid"], record["nc"], float(score), record["gap_type"])
            for record, score in zip(candidate_pool, similarities)
        ]
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[: max(top_k * 4, 10)]

    def search(self, query: str, top_k: int = 3, query_gap: GapTypeResult | None = None):
        semantic_candidates = self._semantic_candidates(query, top_k, query_gap)
        lexical_candidates = self._lexical_candidates(query, top_k, query_gap)
        merged: dict[str, dict] = {}

        for ncid, nc, score, gap_type in lexical_candidates:
            merged.setdefault(
                ncid,
                {
                    "ncid": ncid,
                    "nc": nc,
                    "gap_type": gap_type,
                    "semantic": 0.0,
                    "lexical": 0.0,
                },
            )
            merged[ncid]["lexical"] = max(merged[ncid]["lexical"], score)

        for ncid, nc, score, gap_type in semantic_candidates:
            merged.setdefault(
                ncid,
                {
                    "ncid": ncid,
                    "nc": nc,
                    "gap_type": gap_type,
                    "semantic": 0.0,
                    "lexical": 0.0,
                },
            )
            merged[ncid]["semantic"] = max(merged[ncid]["semantic"], score)

        ranked = []
        for candidate in merged.values():
            combined_score = (
                0.65 * candidate["semantic"] + 0.35 * candidate["lexical"]
                if self.embedding_backend_available
                else candidate["lexical"]
            )
            combined_score += self._gap_type_bonus(candidate["gap_type"], query_gap)
            ranked.append(
                CandidateMatch(
                    ncid=candidate["ncid"],
                    nc=candidate["nc"],
                    distance=max(0.0, 1.0 - combined_score),
                )
            )

        ranked.sort(key=lambda item: item.distance)
        return ranked[:top_k]
