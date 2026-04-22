from datetime import datetime
from pathlib import Path
import shutil

import chromadb
from chromadb.config import Settings

from .config import CHROMA_DIR, COLLECTION_NAME


class VectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME, persist_dir=None):
        self.persist_dir = Path(persist_dir or CHROMA_DIR)
        self.collection_name = collection_name
        self.recovery_backup_dir: Path | None = None
        self.initialization_error: str | None = None
        self.client = self._create_client_with_recovery()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _create_client(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def _backup_corrupt_store(self) -> Path | None:
        if not self.persist_dir.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.persist_dir.parent / f"{self.persist_dir.name}_corrupt_backup_{timestamp}"
        shutil.move(str(self.persist_dir), str(backup_dir))
        self.recovery_backup_dir = backup_dir
        return backup_dir

    def _create_client_with_recovery(self):
        try:
            return self._create_client()
        except Exception as exc:
            self.initialization_error = str(exc)
            self._backup_corrupt_store()
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            try:
                return self._create_client()
            except Exception as recovery_exc:
                raise RuntimeError(
                    "Failed to initialize the Chroma persistent store even after recovery. "
                    f"Store path: {self.persist_dir}. "
                    f"Initial error: {exc}. "
                    f"Recovery error: {recovery_exc}."
                ) from recovery_exc

    def reset(self):
        name = self.collection.name
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=name)

    def add_documents(self, ids, documents, embeddings, metadatas):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()

    def get_ids(self) -> list[str]:
        return [str(item) for item in self.collection.get(include=[])["ids"]]

    def query(self, query_embedding, top_k=3, where=None):
        kwargs = {
            "query_embeddings": query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding,
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(
            **kwargs,
        )

    def health(self) -> dict:
        return {
            "persist_dir": str(self.persist_dir),
            "collection_name": self.collection_name,
            "count": self.count(),
            "recovery_backup_dir": str(self.recovery_backup_dir) if self.recovery_backup_dir else None,
            "initialization_error": self.initialization_error,
        }
