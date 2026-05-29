import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_ENGINE_ROOT = Path(__file__).resolve().parent

load_dotenv(AI_ENGINE_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env", override=True)

MISTRAL_API_KEY = os.getenv("HF_MISTRAL")
EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"
DATA_DIR = "./data"
DB_DIR = "./db"

_PLACEHOLDER_OLLAMA_API_KEYS = {
    "your-secure-token-if-needed",
    "replace-with-the-token-printed-by-colab",
    "replace-with-your-reverse-proxy-token",
    "replace-with-your-ollama-cloud-api-key",
}


def _clean_ollama_base_url(value: str | None) -> str:
    cleaned = (value or "").strip().rstrip("/")
    return cleaned or "http://localhost:11434"


def _clean_ollama_api_key(value: str | None) -> str:
    cleaned = (value or "").strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if not cleaned:
        return ""
    if lowered in _PLACEHOLDER_OLLAMA_API_KEYS or lowered.startswith("replace-with-"):
        return ""
    return cleaned


OLLAMA_BASE_URL = _clean_ollama_base_url(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "vigogne-llama-3:latest")
RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", OLLAMA_MODEL)
OLLAMA_API_KEY = _clean_ollama_api_key(os.getenv("OLLAMA_API_KEY"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))


def ollama_headers() -> dict[str, str]:
    if not OLLAMA_API_KEY:
        return {}
    return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}


def ollama_client_kwargs(timeout: float | int | None = None) -> dict:
    kwargs: dict = {}
    headers = ollama_headers()
    if headers:
        kwargs["headers"] = headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs
