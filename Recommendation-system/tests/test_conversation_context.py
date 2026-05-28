from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation_service import _needs_previous_user_context


def test_short_specific_query_does_not_inherit_previous_context():
    assert not _needs_previous_user_context("nomination responsable securite")


def test_vague_follow_up_inherits_previous_context():
    assert _needs_previous_user_context("et pour ca")
