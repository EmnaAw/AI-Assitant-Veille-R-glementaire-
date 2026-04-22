from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reranker import hybrid_rerank


def test_reranker_prefers_candidate_with_matching_gap_type():
    candidates = [
        {
            "ncid": "A",
            "nc": "Absence d'une equipe de securite",
            "distance": 0.30,
            "gap_type": "equipe_role_securite",
        },
        {
            "ncid": "B",
            "nc": "Absence d'un registre de securite",
            "distance": 0.28,
            "gap_type": "registre_securite",
        },
    ]

    reranked = hybrid_rerank(
        "absence d'une equipe de securite",
        candidates,
        query_gap_type="equipe_role_securite",
    )

    assert reranked[0]["ncid"] == "A"
    assert reranked[0]["gap_type_alignment"] > reranked[1]["gap_type_alignment"]
