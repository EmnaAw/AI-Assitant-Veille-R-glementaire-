from .config import SYSTEM_PROMPT_FR
from .text_utils import clean_client_text


def build_french_action_prompt(plan: str) -> str:
    return f"""
{SYSTEM_PROMPT_FR}

Contexte :
Action corrective officielle source : {plan}

Instructions STRICTES :
- Reformuler l'action pour un client final dans un français clair et naturel
- Corriger la forme, les accents, les apostrophes et la ponctuation
- Conserver le sens principal sans recopier mot à mot la phrase source
- Produire une seule phrase courte et propre
- Ne rien ajouter avant ou après la phrase
"""


def build_french_explanation_prompt(nc: str, plan: str, gap_type: str) -> str:
    return f"""
{SYSTEM_PROMPT_FR}

Contexte :
Non-conformité : {nc}
Action corrective officielle : {plan}
Type d'écart : {gap_type}

Instructions STRICTES :
- Utiliser uniquement le contenu fourni
- Expliquer pourquoi cette action répond à cette non-conformité précise
- Expliquer concrètement ce qu'il faut faire dans l'entreprise pour corriger la non-conformité
- Reformuler l'action corrective avec des mots simples et opérationnels
- Ne pas recopier simplement l'action corrective
- Ne pas ajouter de référence juridique ou de détail externe
- Répondre avec un seul paragraphe court en français naturel
- Ne pas utiliser de titre
- Ne pas produire de phrase vague ou passe-partout
- Interdiction d'écrire une formule du type :
  \"Cette action précise la mesure concrète à mettre en oeuvre pour corriger la non-conformité et assurer un fonctionnement conforme et traçable.\"

Exemple attendu :
- Action corrective officielle : Mettre en place un registre de sécurité
- Bonne explication : Cette action consiste à créer un registre de sécurité, à y consigner les informations utiles et à le tenir à jour pour assurer un suivi clair.
"""


def build_french_ambiguity_message(query: str, conflicting_matches: list[dict]) -> str:
    lines = [
        "Plusieurs actions officielles plausibles correspondent à cette non-conformité.",
        "",
        "Cas officiels à confirmer :",
    ]

    for idx, candidate in enumerate(conflicting_matches, start=1):
        nc = clean_client_text(candidate.get("nc", ""))
        plan = clean_client_text(candidate.get("official_plan", ""))
        lines.append(f"{idx}. Non-conformité : {nc}")
        lines.append(f"   Action officielle : {plan}")

    lines.extend(
        [
            "",
            "Merci de préciser davantage le contexte afin de sélectionner l'action corrective officielle appropriée.",
        ]
    )

    return "\n".join(lines)


def build_french_advisory_prompt(
    query: str,
    normalized_query: str,
    top_candidates: list[dict],
) -> str:
    formatted_candidates = (
        "\n".join([f"- Cas proche : {c.get('nc')}" for c in top_candidates])
        if top_candidates
        else "- Aucun cas proche disponible"
    )

    return f"""
Tu es un assistant d'aide à l'analyse de non-conformités.

Contexte important :
- La non-conformité demandée n'a pas été retrouvée de façon fiable dans la base de données.
- Tu ne dois pas présenter la réponse comme officielle ou validée.
- Tu peux proposer une hypothèse prudente basée sur des cas proches.
- Tu ne dois jamais mentionner de score, de NCid, de classement ou de détail technique interne.
- La réponse sera affichée à un client final.
- Tu dois t'adresser directement au client en utilisant \"vous\".

Requête utilisateur :
{query}

Requête normalisée :
{normalized_query}

Cas proches pour raisonnement interne uniquement :
{formatted_candidates}

Instructions STRICTES :
- Répondre en français
- Commencer par : \"Cette non-conformité n'existe pas actuellement dans notre base.\"
- Ensuite écrire \"Suggestion :\" puis un court texte adressé au client
- Terminer par une phrase indiquant qu'il s'agit d'une suggestion non confirmée
- Ne jamais afficher les détails internes du système
"""
