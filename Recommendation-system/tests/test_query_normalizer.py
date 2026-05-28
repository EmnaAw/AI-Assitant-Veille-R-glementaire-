from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_normalizer import normalize_query_text


def test_normalizer_handles_security_team_training_request_without_demo_shortcut():
    normalized = normalize_query_text(
        "Notre equipe de securite n'est pas formee, que devons nous faire ?"
    )

    assert normalized == "equipe de securite insuffisamment formee"
