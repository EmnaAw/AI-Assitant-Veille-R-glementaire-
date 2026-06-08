import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env", PROJECT_ROOT.parent.parent / ".env"):
    load_dotenv(env_path)

DATA_DIR = PROJECT_ROOT / "data"
DATASET_FILENAME = "final_dedup_by_actionplan_recovered.xlsx"
DATA_PATH = DATA_DIR / DATASET_FILENAME
NC_RESOLUTION_OVERRIDES_PATH = DATA_DIR / "nc_resolution_overrides.csv"
QUERY_ALIAS_OVERRIDES_PATH = DATA_DIR / "query_alias_overrides.csv"
DEFAULT_CHROMA_DIR = Path(os.getenv("LOCALAPPDATA", str(PROJECT_ROOT))) / "RecommendationSystem" / "chroma_store"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DEFAULT_CHROMA_DIR)))

DEFAULT_EMBEDDING_MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--BAAI--bge-m3"
    / "snapshots"
    / "5617a9f61b028005a4858fdac845db406aefb181"
)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_MODEL_PATH = Path(
    os.getenv("EMBEDDING_MODEL_PATH", str(DEFAULT_EMBEDDING_MODEL_PATH))
)
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR")
EMBEDDING_LOCAL_FILES_ONLY = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "1").lower() in {
    "1",
    "true",
    "yes",
}
COLLECTION_NAME = "legal_recommendation_nc_index"
REQUIRED_COLUMNS = ["NCid", "NC", "Plan"]

TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.35
EXACT_MATCH_DISTANCE_THRESHOLD = 0.28
EXACT_MATCH_RERANK_THRESHOLD = 0.82
TOP1_ACCEPT_DISTANCE_THRESHOLD = 0.55
TOP1_ACCEPT_RERANK_THRESHOLD = 0.35
SHORTLIST_ACCEPT_SCORE_THRESHOLD = 0.67
SHORTLIST_ACCEPT_MARGIN_THRESHOLD = 0.08
AMBIGUITY_RERANK_GAP_THRESHOLD = 0.01

GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "template")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "vigogne-llama-3:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "0"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")

SYSTEM_PROMPT_FR = """Tu es un assistant IA expert en droit tunisien.

Ta mission est de reformuler une action corrective officielle et d'en expliquer le sens de manière claire et professionnelle.

Contraintes STRICTES :
- Utiliser uniquement le contenu fourni
- Ne pas ajouter d'information externe
- Ne pas modifier le sens juridique
- Employer un français naturel avec des accents corrects
- L'explication doit clarifier le sens pratique de l'action sans recopier simplement la phrase source

Interdictions :
- Pas de \"Résumé court\"
- Pas de \"Point de vigilance\"
- Pas de listes
"""
