from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import Generator


def test_template_action_and_explanation_for_responsable_securite_nomination():
    generator = Generator(backend="template")
    plan = (
        "1 preparer une fiche nominative du responsable securite avec linsertion de ses missions "
        "conformement a larticle 1545 du code de travail 2 assurer lapprobation de la fiche "
        "par linspection de medecine de travail"
    )

    action = generator.generate_french_action(plan)
    explanation = generator.generate_french_explanation(
        nc="nomination responsable securite",
        plan=plan,
        gap_type="nomination_responsable_securite",
    )

    assert "fiche nominative du responsable securite" in generator._normalize_for_analysis(action)
    assert "inspection de medecine du travail" in generator._normalize_for_analysis(action)
    assert "formaliser la nomination du responsable securite" in generator._normalize_for_analysis(explanation)
