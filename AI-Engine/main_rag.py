import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import unicodedata
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from database import hybrid_search, build_bm25_index

DB_DIR = os.getenv("RAG_DB_DIR", "./db_vigogne_bge_m3")
EMB_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", os.getenv("OLLAMA_MODEL", "vig3:latest"))
MAX_GENERATION_CHUNK_CHARS = int(os.getenv("MAX_GENERATION_CHUNK_CHARS", "1400"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "384"))
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "0"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_LOCAL_FILES_ONLY = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "1").lower() in {
    "1",
    "true",
    "yes",
}

APP_ID_MAP = {
    "QUALITÉ": "1",
    "SÉCURITÉ": "2",
    "ENVIRONNEMENT": "3",
    "ENERGIE": "4",
    "Autres": "5",
    "SOCIALE": "7",
    "Sécurité alimentaire": "8",
    "Qualité alimentaire": "9",
    "QUALITE ALIMENTAIRE": "10",
}

CLASSIFY_THRESHOLD = 0.30
VALID_STATUTS = {"en vigueur", "abrogé", "modifié", "suspendu"}
HISTORICAL_KEYWORDS = {
    "abrogé", "abroge", "historique", "ancienne", "ancien", "avant", "version antérieure",
}

EXACT_OBJECT_PATTERNS = {
    "fiche_entreprise": [
        "fiche d'entreprise",
        "fiche d’entreprise",
        "fiche de l'entreprise",
        "fiche de l’entreprise",
        "fiche de lentreprise",
    ],
    "etude_dangers": [
        "étude de dangers",
        "etude de dangers",
    ],
    "registre_securite": [
        "registre de sécurité",
        "registre de securite",
    ],
    "plan_interieur_intervention": [
        "plan intérieur d'intervention",
        "plan interieur d'intervention",
    ],
    "liste_nominative": [
        "liste nominative",
    ],
    "ria": [
        "robinets d'incendie armés",
        "robinets d’incendie armés",
        "ria",
    ],
}

_app_label_embeddings: dict[str, np.ndarray] | None = None


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = strip_accents(text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_apostrophe_variants(query: str) -> str:
    q = normalize_text(query)

    q = q.replace("de l'entreprise", "d'entreprise")
    q = q.replace("de l’entreprise", "d'entreprise")
    q = q.replace("de l entreprise", "d'entreprise")
    q = q.replace("de lentreprise", "d'entreprise")

    return q


def detect_exact_object_keys(query: str) -> list[str]:
    q = normalize_apostrophe_variants(query)
    found = []

    for key, phrases in EXACT_OBJECT_PATTERNS.items():
        for phrase in phrases:
            if normalize_text(phrase) in q:
                found.append(key)
                break

    return found


def _get_app_embeddings(emb_model: HuggingFaceEmbeddings) -> dict[str, np.ndarray]:
    global _app_label_embeddings
    if _app_label_embeddings is None:
        labels = list(APP_ID_MAP.keys())
        vectors = emb_model.embed_documents(labels)
        _app_label_embeddings = {label: np.array(vec) for label, vec in zip(labels, vectors)}
    return _app_label_embeddings


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def classify_query(query: str, emb_model: HuggingFaceEmbeddings) -> dict | None:
    app_embeddings = _get_app_embeddings(emb_model)
    retrieval_query = normalize_apostrophe_variants(query)
    query_vec = np.array(emb_model.embed_query(retrieval_query))

    best_label, best_score = None, -1.0
    for label, vec in app_embeddings.items():
        score = _cosine(query_vec, vec)
        if score > best_score:
            best_score, best_label = score, label

    if best_score >= CLASSIFY_THRESHOLD and best_label:
        app_id = APP_ID_MAP[best_label]
        print(f"  📊 Domaine détecté: {best_label} (score={best_score:.2f})")
        return {"app_id": app_id}

    print(f"  📊 Aucun domaine détecté avec confiance (meilleur score={best_score:.2f})")
    return None


def is_historical_query(query: str) -> bool:
    q = normalize_text(query)
    return any(normalize_text(keyword) in q for keyword in HISTORICAL_KEYWORDS)


def has_authoritative_source(docs: list) -> bool:
    for doc in docs:
        authority = (doc.metadata.get("authority") or "").lower()
        statut = (doc.metadata.get("statut") or "").lower()
        if authority in {"primary", "secondary"} and statut != "abrogé":
            return True
    return False


def _clean_year(date_str: str | None) -> str | None:
    if not date_str:
        return None
    cleaned = date_str.strip()
    if cleaned.lower() in ("inconnue", "inconnu", "?", "", "inco"):
        return None
    year = cleaned[:4]
    if year.isdigit():
        return year
    return None


def _clean_statut(statut: str | None) -> str | None:
    if not statut:
        return None
    s = statut.strip().lower()
    if s in VALID_STATUTS:
        return statut.strip().capitalize()
    return None


def build_source_header(meta: dict, index: int) -> str:
    parts = [f"[Source {index + 1}"]

    if meta.get("type_texte"):
        parts.append(meta["type_texte"].capitalize())

    if meta.get("numero"):
        parts.append(f"n°{meta['numero']}")

    year = _clean_year(meta.get("date"))
    if year:
        parts.append(year)

    if meta.get("titre"):
        parts.append(f"— {meta['titre'][:60]}")

    if meta.get("article"):
        parts.append(f"| {meta['article']}")

    statut = _clean_statut(meta.get("statut"))
    if statut:
        parts.append(f"| {statut}")

    parts.append("]")
    return " ".join(parts)


def build_source_label(meta: dict) -> str | None:
    type_texte = meta.get("type_texte", "").strip()
    numero = meta.get("numero", "").strip()
    if not type_texte or not numero:
        return None

    year = _clean_year(meta.get("date"))
    year_part = f" ({year})" if year else ""
    return f"{type_texte.capitalize()} n°{numero}{year_part}"


def doc_search_blob(doc) -> str:
    parts = [
        doc.page_content or "",
        doc.metadata.get("titre", "") or "",
        doc.metadata.get("article", "") or "",
        doc.metadata.get("type_texte", "") or "",
        doc.metadata.get("numero", "") or "",
    ]
    return normalize_text(" ".join(parts))


def doc_matches_exact_object(user_query: str, doc) -> bool:
    object_keys = detect_exact_object_keys(user_query)
    if not object_keys:
        return True

    blob = doc_search_blob(doc)

    for key in object_keys:
        phrases = EXACT_OBJECT_PATTERNS[key]
        if any(normalize_text(phrase) in blob for phrase in phrases):
            return True

    return False


def score_doc_relevance(user_query: str, doc) -> int:
    q = normalize_text(user_query)
    blob = doc_search_blob(doc)
    score = 0

    for key in detect_exact_object_keys(user_query):
        phrases = EXACT_OBJECT_PATTERNS[key]
        if any(normalize_text(phrase) in blob for phrase in phrases):
            score += 20

    if q.startswith("qui "):
        for marker in ["est tenu", "est charge", "doit", "est responsable", "est etabli", "est etabli par"]:
            if marker in blob:
                score += 3

    authority = normalize_text(doc.metadata.get("authority", ""))
    if authority == "primary":
        score += 5
    elif authority == "secondary":
        score += 3

    statut = normalize_text(doc.metadata.get("statut", ""))
    if "abroge" in statut and not is_historical_query(user_query):
        score -= 8

    return score


def filter_and_rank_docs(user_query: str, docs: list, top_k: int = 3) -> list:
    if not docs:
        return []

    exact_docs = [doc for doc in docs if doc_matches_exact_object(user_query, doc)]
    if exact_docs:
        docs = exact_docs

    ranked = sorted(docs, key=lambda d: score_doc_relevance(user_query, d), reverse=True)
    return ranked[:top_k]


def repair_answer_format(answer: str) -> str:
    text = str(answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return text

    match = re.search(r"(?im)^\s*(?:#{1,6}\s*)?(?:r[ée]ponse|reponse)\s*:", text)
    if match:
        text = text[match.start():].strip()

    text = re.sub(r"(?im)^\s*(?:#{1,6}\s*)?(?:r[ée]ponse|reponse)\s*:?", "Réponse:", text, count=1)
    text = re.sub(r"(?im)^\s*(?:#{1,6}\s*)?base\s+l[ée]gale\s*:?", "Base légale:", text)

    lines = text.split("\n")
    cleaned = []
    in_base = False
    saw_base_bullet = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"(?i)^base\s+l[ée]gale\s*:", stripped):
            in_base = True
            cleaned.append("Base légale:")
            continue
        if in_base:
            if stripped.startswith("-"):
                saw_base_bullet = True
                cleaned.append(line.rstrip())
                continue
            if saw_base_bullet:
                break
        cleaned.append(line.rstrip())

    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


SYSTEM_PROMPT = """### Instruction:
Tu es un assistant juridique spécialisé en réglementation tunisienne (HSE, qualité, environnement, sécurité au travail).
Tu travailles pour une entreprise qui fait de la veille réglementaire.

Le contexte ci-dessous contient des extraits de textes juridiques tunisiens récupérés automatiquement.
Chaque extrait commence par un en-tête [Source N ...] indiquant le type de texte, son numéro, son année et son statut.

═══════════════════════════════════
RÈGLES ABSOLUES (ne jamais enfreindre)
═══════════════════════════════════
1. Tu réponds UNIQUEMENT à partir du contexte fourni. Aucune connaissance externe.
2. Tu ne dois JAMAIS inventer ou compléter un numéro de texte, un numéro d'article, une date, un acteur, une obligation ou un chiffre.
3. Si l'information demandée n'apparaît pas explicitement dans le contexte, réponds UNIQUEMENT :
"Je n'ai pas l'information dans le contexte fourni."
4. Si le contexte contient des textes proches du sujet mais ne répond pas exactement à la question posée, réponds UNIQUEMENT :
"Je n'ai pas l'information dans le contexte fourni."
5. Un texte avec le statut "Abrogé" ne fait plus force de loi : mentionne-le explicitement avec ⚠️ ABROGÉ.
6. Ne réponds jamais par analogie, déduction large, approximation ou rapprochement thématique.

═══════════════════════════════════
RÈGLES DE PERTINENCE (TRÈS IMPORTANT)
═══════════════════════════════════
- La réponse doit viser EXACTEMENT l'objet juridique demandé dans la question.
- Si la question porte sur "fiche d'entreprise", ne réponds pas avec des règles sur registre, liste nominative, médecin du travail en général, employeur en général, ou toute autre obligation voisine, sauf si le passage mentionne explicitement la fiche d'entreprise.
- Si la question demande "qui", la réponse doit identifier explicitement l'acteur désigné dans le contexte.
- Si la question demande "quelle réglementation concerne ...", privilégie les textes qui mentionnent explicitement cet objet, cette obligation ou ce document.
- Ignore tout extrait seulement partiellement lié ou simplement du même domaine.

═══════════════════════════════════
RÈGLES DE CITATION
═══════════════════════════════════
- Cite toujours : Type + n° + Année (ex : "Décret n°2000-1989 de 2000")
- Si un article précis est mentionné dans la source, cite-le : "Article 3 du Décret n°2000-1989"
- Ne cite jamais un article qui ne figure pas explicitement dans le contexte
- Ne cite que les sources directement utiles à la réponse

═══════════════════════════════════
RÈGLES SPÉCIFIQUES À LA BASE LÉGALE
═══════════════════════════════════
- La section "Base légale" doit être PLUS COURTE que la section "Réponse".
- Chaque puce de la base légale doit contenir UNE seule phrase.
- Ne fais jamais de sous-puces, ni de liste imbriquée.
- Ne répète pas en détail ce qui est déjà dit dans la réponse.
- Garde seulement les textes les plus directement utiles pour justifier la réponse.
- Limite-toi à 2 ou 3 sources maximum.
- Si plusieurs sources disent la même chose, n'en garde qu'une seule dans la base légale.

═══════════════════════════════════
STYLE ATTENDU
═══════════════════════════════════
- Réponse courte, directe, juridique, sans commentaire inutile.
- Commence obligatoirement par "Réponse:" sur une ligne seule.
- Ensuite donne une réponse directe en 1 ou 2 phrases maximum.
- Puis écris obligatoirement "Base légale:" sur une ligne seule.
- N'ajoute aucun texte avant "Réponse:" ni après la dernière puce de "Base légale:".
- N'ajoute jamais de note, remarque, commentaire, précision complémentaire ou avertissement hors de la section "Base légale:".
- Quand la question commence par "Qui", réponds d'abord par l'acteur exact, puis complète en une phrase si nécessaire.
- Si une source mentionne seulement un modèle, un formulaire ou une annexe sans répondre directement à la question, elle peut être citée dans la base légale mais ne doit pas faire l'objet d'un commentaire séparé.
═══════════════════════════════════
EXEMPLES
═══════════════════════════════════

Question: Qui remplit la fiche d'entreprise ?
Réponse:
Le service autonome de médecine du travail est tenu d'établir et de mettre à jour une fiche d'entreprise.

Base légale:
- [Décret n°2000-1985 (2000)] : Le service autonome de médecine du travail est tenu d'établir et de mettre à jour une fiche d'entreprise.
- [Arrêté n°2009-1060 (2009)] : Cet arrêté fixe le modèle de la fiche d'entreprise.

Question: Qui valide l'étude de l'impacte sur l'environnement ?
Réponse:
L'étude d'impact sur l'environnement doit être élaborée par des bureaux d'études ou des experts spécialisés dans le domaine et validée par l'agence nationale de protection de l'environnement, qui peut demander l'avis du gestionnaire de zones bénéficiant d'une protection juridique en cas d'effets prévisibles sur ces zones.       

Base légale:
- Décret n°2005-1991 2005 : L'étude d'impact sur l'environnement doit être élaborée par des bureaux d'études ou des experts spécialisés dans le domaine.
- Communiqué n°26-11-2025 2025 ⚠️ ABROGÉ : Le ministère de l’Industrie appelle les établissements classés à dééposer leur étude de dépollution avant fin 2025.

═══════════════════════════════════
FORMAT DE RÉPONSE (respecter strictement)
═══════════════════════════════════
Réponse:
<Réponse directe et concise en prose. Utilise des tirets (-) si la réponse est une liste d'obligations ou de membres.>

Base légale:
- [Type n°Numero (Année)] : <résumé très bref de la règle pertinente en une seule phrase>
- [Type n°Numero (Année)] ⚠️ ABROGÉ : <résumé très bref en une seule phrase> (si applicable)
"""


SYSTEM_PROMPT += """

REGLES COMPLEMENTAIRES POUR LES SEUILS ET CONDITIONS:
- Si une meme source contient plusieurs cas, seuils, conditions ou tranches d'effectif qui repondent a la question, garde tous les cas utiles.
- Pour une question sur l'effectif, le nombre de travailleurs, une categorie ou une condition introduite par "Si", ne t'arrete jamais au premier cas trouve.
- Dans la section "Base legale", tu peux citer plusieurs puces avec la meme source si chaque puce justifie une condition differente.
- Ne fusionne pas deux seuils differents en une seule regle.

Exemple attendu:
Question: Qui doit exercer la fonction du responsable securite selon l'effectif ?
Reponse:
Si l'effectif est de 500 travailleurs et plus, un ingenieur en plein temps exerce la fonction du responsable securite. Si l'effectif est de 40 employes ou plus et inferieur ou egal a 500, un ingenieur ou un technicien superieur l'exerce a plein temps en sus de son travail personnel.

Base legale:
- [Decret n°2000-1989 (2000)] : Si l'effectif est de 500 travailleurs et plus, un ingenieur en plein temps exerce la fonction du responsable securite.
- [Decret n°2000-1989 (2000)] : Si l'effectif est de 40 employes ou plus et inferieur ou egal a 500, un ingenieur ou un technicien superieur l'exerce a plein temps en sus de son travail personnel.
"""


SYSTEM_PROMPT = """Tu es un assistant juridique specialise en reglementation tunisienne HSE.
Tu reponds uniquement a partir du CONTEXTE JURIDIQUE fourni.

Regles absolues:
- N'invente jamais un acteur, une obligation, un seuil, un chiffre, un numero de texte, un article ou une date.
- Si l'information n'est pas explicitement dans le contexte, reponds: Je n'ai pas l'information dans le contexte juridique fourni.
- Ignore les extraits seulement proches du sujet mais qui ne repondent pas exactement a la question.
- Si un texte est abroge, mentionne-le avec ABROGE.

Pertinence:
- La reponse doit viser exactement l'objet juridique demande.
- Si la question demande "qui", identifie l'acteur exact designe dans le contexte.
- Si la reponse depend d'un effectif, d'un seuil, d'une categorie ou d'une condition introduite par "Si", reprends toutes les conditions explicitement presentes dans le contexte qui repondent a la question.
- Ne t'arrete pas au premier cas lorsqu'une meme source contient plusieurs seuils ou tranches d'effectif.

Format obligatoire:
Reponse:
<reponse directe en 1 ou 2 phrases. Utilise plusieurs phrases si plusieurs conditions sont necessaires.>

Base legale:
- [Type n°Numero (Annee)] : <justification courte en une phrase>

Regles de base legale:
- Cite seulement les sources directement utiles.
- Tu peux citer plusieurs puces avec la meme source si chaque puce justifie une condition differente.
- Chaque puce doit contenir une seule condition ou regle.
- N'ajoute aucun texte avant "Reponse:" ni apres la derniere puce de "Base legale:".

Exemple:
Question: Qui doit exercer la fonction du responsable securite selon l'effectif ?
Reponse:
Si l'effectif est de 500 travailleurs et plus, un ingenieur en plein temps exerce la fonction du responsable securite. Si l'effectif est de 40 employes ou plus et inferieur ou egal a 500, un ingenieur ou un technicien superieur l'exerce a plein temps en sus de son travail personnel.

Base legale:
- [Decret n°2000-1989 (2000)] : Si l'effectif est de 500 travailleurs et plus, un ingenieur en plein temps exerce la fonction du responsable securite.
- [Decret n°2000-1989 (2000)] : Si l'effectif est de 40 employes ou plus et inferieur ou egal a 500, un ingenieur ou un technicien superieur l'exerce a plein temps en sus de son travail personnel.
"""


def run_rag():
    print(f"--- Initializing: {LLM_MODEL} + BGE-M3 ---")
    emb = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": "cpu", "local_files_only": EMBEDDING_LOCAL_FILES_ONLY},
        encode_kwargs={"normalize_embeddings": True},
    )

    if not os.path.exists(DB_DIR):
        print("❌ Database not found. Run ingest.py first.")
        return

    db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=OLLAMA_NUM_PREDICT,
        num_gpu=OLLAMA_NUM_GPU,
    )

    print("⚙ Building BM25 index (one-time)...")
    bm25 = build_bm25_index(db)
    print("✅ Assistant Prêt. (Tapez 'exit' pour quitter)\n")

    while True:
        user_query = input("[VOUS]: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        if not user_query:
            continue

        clauses = []

        domain_filter = classify_query(user_query, emb)
        if domain_filter:
            clauses.append(domain_filter)

        if not is_historical_query(user_query):
            clauses.append({"statut": {"$ne": "abrogé"}})

        if len(clauses) == 1:
            meta_filter = clauses[0]
        elif len(clauses) > 1:
            meta_filter = {"$and": clauses}
        else:
            meta_filter = {}

        if meta_filter:
            print(f"  🏷  Filtre appliqué: {meta_filter}")

        retrieval_query = normalize_apostrophe_variants(user_query)
        if retrieval_query != normalize_text(user_query):
            print(f"  ✍️ Requête normalisée: {retrieval_query}")

        print("🔍 Recherche hybride...")
        docs = hybrid_search(
            query=retrieval_query,
            vector_db=db,
            bm25_retriever=bm25,
            k=8,
            candidate_k=24,
            metadata_filter=meta_filter,
        )

        docs = filter_and_rank_docs(user_query, docs, top_k=3)

        exact_keys = detect_exact_object_keys(user_query)
        has_exact_match = any(doc_matches_exact_object(user_query, doc) for doc in docs) if docs else False

        # Fallback ciblé : on enlève seulement app_id si l'objet exact n'a pas été trouvé
        if exact_keys and not has_exact_match:
            fallback_filter = {"statut": {"$ne": "abrogé"}} if not is_historical_query(user_query) else {}
            print("  ↪ Fallback ciblé sans filtre app_id...")

            fallback_docs = hybrid_search(
                query=retrieval_query,
                vector_db=db,
                bm25_retriever=bm25,
                k=8,
                candidate_k=24,
                metadata_filter=fallback_filter,
            )

            fallback_docs = filter_and_rank_docs(user_query, fallback_docs, top_k=3)

            if fallback_docs and any(doc_matches_exact_object(user_query, doc) for doc in fallback_docs):
                docs = fallback_docs

        if not docs:
            print("Je n'ai pas l'information dans le contexte juridique fourni.\n")
            continue

        if exact_keys and not any(doc_matches_exact_object(user_query, doc) for doc in docs):
            print("Je n'ai pas l'information dans le contexte juridique fourni.\n")
            continue

        context_parts = []
        sources_seen = {}

        for i, doc in enumerate(docs):
            header = build_source_header(doc.metadata, i)
            clipped_content = doc.page_content.strip()
            if len(clipped_content) > MAX_GENERATION_CHUNK_CHARS:
                clipped_content = clipped_content[:MAX_GENERATION_CHUNK_CHARS].rsplit(" ", 1)[0].strip()
            context_parts.append(f"{header}\n{clipped_content}")

            label = build_source_label(doc.metadata)
            if label and label not in sources_seen:
                sources_seen[label] = True

        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""{SYSTEM_PROMPT}
CONTEXTE JURIDIQUE ({len(docs)} extraits recuperes) :
{context_text}

QUESTION : {user_query}

### Reponse:"""

        print("💬 Génération...\n")
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            print(f"❌ LLM error: {e}\n")
            if "runner process has terminated" in str(e).lower():
                print(
                    "Ollama a probablement manque de VRAM. "
                    "Essayez de redemarrer Ollama, ou baissez OLLAMA_NUM_CTX/OLLAMA_NUM_GPU.\n"
                )
            continue

        print("-" * 60)
        print(repair_answer_format(str(response)))
        print("-" * 60)

        if sources_seen:
            print(f"📎 Sources: {' | '.join(sources_seen.keys())}\n")
        else:
            print()


if __name__ == "__main__":
    run_rag()
