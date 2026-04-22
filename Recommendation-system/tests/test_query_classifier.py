from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_classifier import classify_gap_type, infer_gap_type_from_record


def test_query_classifier_detects_security_team_gap_type():
    result = classify_gap_type("absence d'une equipe de securite")

    assert result.gap_type == "equipe_role_securite"
    assert result.confidence >= 0.9


def test_record_gap_type_uses_nc_and_plan_context():
    result = infer_gap_type_from_record(
        nc="Absence d'autorisation d'exploitation",
        plan="Faire avancer le dossier avec le bureau d'etude jusqu'a regularisation",
    )

    assert result.gap_type == "autorisation_administratif"
