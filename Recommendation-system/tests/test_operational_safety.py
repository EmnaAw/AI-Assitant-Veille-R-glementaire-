from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import _validate_duplicate_ncid_rows
from src.query_classifier import classify_gap_type
from src.retriever import Retriever
from src.schemas import RecommendationResult


def test_load_dataset_rejects_conflicting_duplicate_ncid_rows():
    df = pd.DataFrame(
        [
            {"NCid": "NC-1", "NC": "Absence d'autorisation", "Plan": "Regulariser l'autorisation"},
            {"NCid": "NC-1", "NC": "Absence d'autorisation", "Plan": "Suspendre l'activite"},
        ]
    )

    with pytest.raises(ValueError, match="conflicting duplicate NCid rows"):
        _validate_duplicate_ncid_rows(df)


def test_query_classifier_returns_autre_for_conflicting_high_confidence_gap_signals():
    result = classify_gap_type(
        "absence autorisation exploitation pour eclairage de securite et bloc autonome"
    )

    assert result.gap_type == "autre"
    assert result.confidence == 0.0
    assert result.matched_rules


def test_recommendation_result_repairs_client_facing_mojibake():
    result = RecommendationResult(
        mode="no_match",
        query="test",
        normalized_query="test",
        decision_reason="no_candidates",
        explanation_fr="Cette rÃ©ponse n'a pas Ã©tÃ© validÃ©e.",
        advisory_disclaimer="Aucune action officielle n'a Ã©tÃ© validÃ©e pour cette requÃªte.",
    )

    assert result.explanation_fr == "Cette réponse n'a pas été validée."
    assert result.advisory_disclaimer == "Aucune action officielle n'a été validée pour cette requête."


def test_retriever_health_reports_dataset_and_store_drift():
    retriever = Retriever.__new__(Retriever)
    retriever.lookup = type("LookupStub", (), {"by_ncid": {"1": {}, "2": {}, "3": {}}})()
    retriever.store = type("StoreStub", (), {"get_ids": lambda self: ["2", "4"]})()
    retriever.semantic_mode = "chroma"
    retriever.embedding_backend_available = True
    retriever.store_in_sync = False
    retriever.store_error = "persistent_store_out_of_sync"
    retriever.records = [object(), object(), object()]
    retriever.store_health = {"persist_dir": "dummy"}

    health = Retriever.health(retriever)

    assert health["dataset_id_count"] == 3
    assert health["store_id_count"] == 2
    assert health["missing_store_ids"] == ["1", "3"]
    assert health["extra_store_ids"] == ["4"]
