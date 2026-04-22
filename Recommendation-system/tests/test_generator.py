from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import Generator


def test_template_explanation_uses_nc_plan_and_gap_type_for_security_team():
    generator = Generator(backend="template")

    explanation = generator.generate_french_explanation(
        nc="Absence d'une equipe de securite",
        plan="Mettre en place une equipe de securite conformement aux dispositions de l article 161",
        gap_type="equipe_role_securite",
    )

    lowered = explanation.lower()
    assert "équipe de sécurité" in lowered
    assert "absence" in lowered or "insuffisance" in lowered
    assert "fonctionnement conforme et tracable" not in lowered
    assert "équipe de sécurité" in explanation
    assert "définissant" in explanation


def test_ollama_generic_explanation_falls_back_to_gap_aware_template():
    generator = Generator(backend="ollama")
    generator._generate_with_ollama = lambda prompt: (
        "Cette action precise la mesure concrete a mettre en oeuvre pour corriger "
        "la non conformite et assurer un fonctionnement conforme et tracable."
    )

    explanation = generator.generate_french_explanation(
        nc="Absence d'une equipe de securite",
        plan="Mettre en place une equipe de securite conformement aux dispositions de l article 161",
        gap_type="equipe_role_securite",
    )

    lowered = explanation.lower()
    assert "équipe de sécurité" in lowered
    assert "absence" in lowered or "insuffisance" in lowered
    assert "fonctionnement conforme et tracable" not in lowered
    assert "équipe de sécurité" in explanation
    assert "définissant" in explanation


def test_template_action_keeps_french_accents_for_security_team():
    generator = Generator(backend="template")

    action = generator.generate_french_action(
        "Mettre en place une equipe de securite conformement aux dispositions de l article 161"
    )

    assert action == "Constituer une équipe de sécurité, désigner ses membres et assurer les formations obligatoires."
