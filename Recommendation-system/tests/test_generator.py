from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import Generator
import src.generator as generator_module
from src.config import _clean_ollama_api_key, _clean_ollama_base_url


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


def test_ollama_generation_sends_configured_headers(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200
        text = '{"response": "ok"}'

        def json(self):
            return {"response": "ok"}

    class DummySession:
        def post(self, url, json, timeout, headers):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            captured["headers"] = headers
            return DummyResponse()

    monkeypatch.setattr(
        generator_module,
        "ollama_headers",
        lambda: {"Authorization": "Bearer test-token"},
    )

    generator = Generator(backend="ollama")
    generator.session = DummySession()

    assert generator._generate_with_ollama("prompt") == "ok"
    assert captured["headers"] == {"Authorization": "Bearer test-token"}


def test_placeholder_ollama_api_keys_are_ignored():
    assert _clean_ollama_api_key("your-secure-token-if-needed") == ""
    assert _clean_ollama_api_key("replace-with-the-token-printed-by-colab") == ""
    assert _clean_ollama_api_key("real-token") == "real-token"


def test_ollama_base_url_is_normalized():
    assert _clean_ollama_base_url("https://example.trycloudflare.com/") == "https://example.trycloudflare.com"
    assert _clean_ollama_base_url("") == "http://localhost:11434"


def test_template_action_keeps_french_accents_for_security_team():
    generator = Generator(backend="template")

    action = generator.generate_french_action(
        "Mettre en place une equipe de securite conformement aux dispositions de l article 161"
    )

    assert action == "Constituer une équipe de sécurité, désigner ses membres et assurer les formations obligatoires."
