import json

import requests
import typer
from rich import print

from .config import DATA_PATH, OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K
from .indexer import build_index
from .pipeline_runtime import LegalRecommendationPipeline
from .retriever import Retriever

app = typer.Typer(help="Legal Recommendation Engine CLI")


@app.command()
def index():
    count = build_index(DATA_PATH, reset=True)
    print(f"[green]Index built successfully.[/green] {count} records indexed.")


@app.command()
def search(query: str, top_k: int = TOP_K):
    retriever = Retriever()
    results = retriever.search(query=query, top_k=top_k)
    print(json.dumps([result.model_dump() for result in results], ensure_ascii=False, indent=2))


@app.command()
def health():
    retriever = Retriever()
    retriever_health = retriever.health()

    ollama_health = {
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "reachable": False,
        "models": [],
    }
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
        models = [item.get("name") for item in payload.get("models", []) if item.get("name")]
        ollama_health["reachable"] = True
        ollama_health["models"] = models
    except requests.RequestException as exc:
        ollama_health["error"] = str(exc)

    print(
        json.dumps(
            {
                "retriever": retriever_health,
                "ollama": ollama_health,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def query(query: str):
    pipeline = LegalRecommendationPipeline()
    result = pipeline.run(query=query, with_generation=True)

    print("\n=== Recommandation ===\n")

    if result.mode == "verified":
        print("Action recommandée :\n")
        print(result.display_plan_fr or result.official_plan or "")
        print("\nExplication :\n")
        print(result.explanation_fr or "")
        return

    if result.mode == "ambiguous":
        print(result.explanation_fr or "")
    elif result.mode == "no_match":
        print(result.explanation_fr or "")
    else:
        print(result.explanation_fr or "")


__all__ = ["app"]
