from .data_loader import load_dataset
from .embedder import Embedder
from .query_classifier import infer_gap_type_from_record
from .vector_store import VectorStore


def build_index(data_path, reset=True):
    df = load_dataset(data_path)
    embedder = Embedder()
    store = VectorStore()

    if reset:
        store.reset()

    embeddings = embedder.encode(df["NC"].tolist())
    metadatas = [
        {
            "NCid": str(row["NCid"]).strip(),
            "gap_type": infer_gap_type_from_record(
                str(row["NC"]).strip(),
                str(row["Plan"]).strip(),
            ).gap_type,
        }
        for _, row in df.iterrows()
    ]

    store.add_documents(
        ids=df["NCid"].tolist(),
        documents=df["NC"].tolist(),
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(df)
