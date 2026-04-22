import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

FRENCH_LEGAL_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "à", "au", "aux",
    "est", "sont", "dans", "par", "sur", "ou", "que", "qui", "ce", "se", "sa", "ses",
    "il", "ils", "elle", "elles", "pour", "avec", "plus", "ledit", "ladite",
    "lesdits", "lesdites", "susvisé", "susmentionné", "nonobstant",
    "conformément", "notamment", "toutefois", "néanmoins", "alinéa",
    "paragraphe", "ci-dessus", "ci-après", "ci-dessous", "présent", "présente",
}

TUNISIAN_ABBREVIATIONS = {
    r"\bCSP\b": "Code du Statut Personnel",
    r"\bCOT\b": "Code des Obligations et des Contrats",
    r"\bCNSS\b": "Caisse Nationale de Sécurité Sociale",
    r"\bJORT\b": "Journal Officiel de la République Tunisienne",
    r"\bINNORPI\b": "Institut National de la Normalisation et de la Propriété Industrielle",
    r"\bANM\b": "Agence Nationale de Métrologie",
    r"\bERP\b": "Etablissements Recevant du Public",
    r"\bPOI\b": "Plan d'Opération Interne Plan d'intervention interne",
    r"\bPII\b": "Plan d'intervention interne",
    r"\bFDS\b": "Fiche de Données de Sécurité",
    r"\bTVA\b": "Taxe sur la Valeur Ajoutée",
    r"\bSARL\b": "Société à Responsabilité Limitée",
}

AUTHORITY_BOOST = {
    "primary": 0.35,
    "secondary": 0.15,
    "fallback": -0.10,
}

STATUT_BOOST = {
    "en vigueur": 0.20,
    "modifié": 0.05,
    "suspendu": -0.05,
    "abrogé": -0.60,
}


def expand_query(query: str) -> list[str]:
    variants = [query]
    expanded = query
    for pattern, replacement in TUNISIAN_ABBREVIATIONS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    if expanded != query:
        variants.append(expanded)
    return variants


def reciprocal_rank_fusion(results_lists: list[list], k: int = 60) -> list:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content.strip()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def build_bm25_index(vector_db) -> BM25Retriever:
    data = vector_db.get()
    all_docs = [Document(page_content=d, metadata=m) for d, m in zip(data["documents"], data["metadatas"])]
    retriever = BM25Retriever.from_documents(
        all_docs,
        preprocess_func=lambda text: [
            w for w in text.lower().split() if w not in FRENCH_LEGAL_STOPWORDS and len(w) > 2
        ],
    )
    return retriever


def _normalize_statut(value: str | None) -> str:
    return (value or "").strip().lower()


def _legal_priority_adjustment(doc: Document) -> float:
    meta = doc.metadata or {}
    authority = (meta.get("authority") or "fallback").strip().lower()
    statut = _normalize_statut(meta.get("statut"))

    adjustment = AUTHORITY_BOOST.get(authority, 0.0)
    adjustment += STATUT_BOOST.get(statut, 0.0)

    try:
        rank = int(meta.get("legal_rank", 5))
    except (TypeError, ValueError):
        rank = 5
    adjustment += max(0, 5 - rank) * 0.03

    return adjustment


def _matches_metadata_filter(doc: Document, metadata_filter: dict | None) -> bool:
    if not metadata_filter:
        return True

    # Support logical filters built by main_rag.py
    if "$and" in metadata_filter:
        clauses = metadata_filter["$and"] or []
        return all(_matches_metadata_filter(doc, clause) for clause in clauses)

    if "$or" in metadata_filter:
        clauses = metadata_filter["$or"] or []
        return any(_matches_metadata_filter(doc, clause) for clause in clauses)

    for key, expected in metadata_filter.items():
        actual = doc.metadata.get(key)

        if isinstance(expected, dict):
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$nin" in expected and actual in expected["$nin"]:
                return False
        else:
            if actual != expected:
                return False

    return True


def hybrid_search(
    query: str,
    vector_db,
    bm25_retriever: BM25Retriever,
    k: int = 6,
    candidate_k: int = 20,
    metadata_filter: dict = None,
) -> list[Document]:
    query_variants = expand_query(query)

    all_semantic = []
    search_kwargs = {"k": candidate_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    for q in query_variants:
        docs = vector_db.similarity_search(q, **search_kwargs)
        all_semantic.extend(docs)

    seen = set()
    semantic_docs = []
    for doc in all_semantic:
        key = doc.page_content.strip()
        if key not in seen:
            semantic_docs.append(doc)
            seen.add(key)
    semantic_docs = semantic_docs[:candidate_k]

    bm25_retriever.k = candidate_k
    keyword_docs = bm25_retriever.invoke(query)
    if metadata_filter:
        keyword_docs = [doc for doc in keyword_docs if _matches_metadata_filter(doc, metadata_filter)]

    fused = reciprocal_rank_fusion([semantic_docs, keyword_docs])
    if not fused:
        return []

    pairs = [(query, doc.page_content) for doc in fused]
    scores = RERANKER.predict(pairs, show_progress_bar=False)

    ranked = []
    for score, doc in zip(scores, fused):
        final_score = float(score) + _legal_priority_adjustment(doc)
        ranked.append((final_score, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]