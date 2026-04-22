import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")

from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_CACHE_DIR,
    EMBEDDING_LOCAL_FILES_ONLY,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_MODEL_NAME,
)


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        model_source = str(EMBEDDING_MODEL_PATH) if EMBEDDING_MODEL_PATH.exists() else model_name
        try:
            self.model = SentenceTransformer(
                model_source,
                cache_folder=EMBEDDING_CACHE_DIR,
                local_files_only=EMBEDDING_LOCAL_FILES_ONLY,
            )
        except Exception as exc:
            details = [
                f"Failed to load embedding model '{model_source}'.",
                "The recommendation retriever cannot run until the embedding model is available.",
            ]
            if EMBEDDING_LOCAL_FILES_ONLY:
                details.append(
                    "Offline-only mode is enabled via EMBEDDING_LOCAL_FILES_ONLY, so the model must already exist in the local cache."
                )
            else:
                details.append(
                    "If this environment has no network access, pre-download the model or set EMBEDDING_LOCAL_FILES_ONLY=1 once the cache is populated."
                )
            if EMBEDDING_MODEL_PATH:
                details.append(f"Configured model path: {EMBEDDING_MODEL_PATH}")
            if EMBEDDING_CACHE_DIR:
                details.append(f"Configured cache directory: {EMBEDDING_CACHE_DIR}")
            raise RuntimeError(" ".join(details)) from exc

    def encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
