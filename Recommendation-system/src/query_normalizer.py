import re
import unicodedata


PREFIX_PATTERNS = [
    r"^bonsoir\s*,*\s*",
    r"^bonjour\s*,*\s*",
    r"^salut\s*,*\s*",
    r"^nous avons constate le point suivant\s*:\s*",
    r"^nous avons constate le probleme suivant\s*:\s*",
    r"^nous avons le probleme suivant\s*:\s*",
    r"^audit a signale\s*",
    r"^on a remarque\s*",
    r"^probleme\s*:\s*",
    r"^probleme\s+",
    r"^non[- ]?conformite\s*",
    r"^cas de\s*",
    r"^besoin d[' ]une solution\s*",
    r"^quelle action corrective recommandez[- ]vous pour\s*:\s*",
    r"^quelle action officielle correspond a\s*:\s*",
    r"^quelle recommandation faut[- ]il appliquer pour\s*:\s*",
    r"^quel est le traitement recommande pour\s*:\s*",
    r"^quelle mesure faut[- ]il prendre pour le cas suivant\s*:\s*",
    r"^que doit faire l'entreprise en cas de\s*",
    r"^comment corriger la situation suivante\s*:\s*",
    r"^comment traiter la non[- ]conformite suivante\s*:\s*",
    r"^comment regler\s*",
    r"^que faire concretement\s*",
]

SUFFIX_PATTERNS = [
    r"\s*que recommandez[- ]vous\s*\??$",
    r"\s*que recommendez[- ]vous\s*\??$",
    r"\s*que faut[- ]il faire\s*\??$",
    r"\s*que faut il faire\s*\??$",
    r"\s*comment corriger\s*\??$",
    r"\s*comment regler ca\s*\??$",
    r"\s*comment regler ca urgent\s*\??$",
    r"\s*quelles actions\s*\??$",
    r"\s*que proposez[- ]vous\s*\??$",
    r"\s*besoin d[' ]une solution\s*\??$",
    r"\s*urgent\s*\??$",
]

STOPWORD_TOKENS = {
    "action",
    "actions",
    "besoin",
    "bonsoir",
    "bonjour",
    "cas",
    "chez",
    "comment",
    "concretement",
    "corriger",
    "faire",
    "faut",
    "il",
    "le",
    "les",
    "mesure",
    "mesures",
    "non",
    "notre",
    "on",
    "probleme",
    "que",
    "quel",
    "quelle",
    "quelles",
    "recommandation",
    "recommandations",
    "recommandez",
    "recommendez",
    "regler",
    "signale",
    "situation",
    "sont",
    "solution",
    "traitement",
    "toujours",
    "urgent",
    "vous",
    "votre",
    "bien",
    "jamais",
}

PHRASE_REPLACEMENTS = (
    (r"\bclaration\b", "declaration"),
    (r"\bxploitation\b", "exploitation"),
    (r"\bdonnes\b", "donnees"),
    (r"\bmecin\b", "medecin"),
    (r"\bmedcin\b", "medecin"),
    (r"\bproces\s+s\b", "process"),
    (r"\bdexploitation\b", "declaration dexploitation"),
    (r"\bdune etude securite incendie\b", "absence dune etude de securite incendie"),
    (r"\beclairage de securite\b", "absence declairage de securite"),
    (r"\beclairage de secours\b", "eclairage de securite"),
    (r"\bblocs? d eclairage de secours\b", "eclairage de securite"),
    (r"\bblocs? d eclairage de securite\b", "eclairage de securite"),
    (r"\bne marchent pas bien\b", "non conforme"),
    (r"\babimes\b", "endommages"),
    (r"\bendommages\b", "non conforme"),
    (r"\bmanque\s+s?\b", "manque de"),
    (r"\bporter les charges correctement\b", "gestes et postures"),
    (r"\bporter des charges\b", "gestes et postures"),
    (r"\bcharges correctement\b", "gestes et postures"),
    (r"\bmanutention manuelle\b", "gestes et postures"),
    (r"\bna aucune personne[s]?\s+handicapee?s?\b", "absence des personnes handicapees"),
    (r"\bn a aucune personne[s]?\s+handicapee?s?\b", "absence des personnes handicapees"),
    (r"\bsans personne[s]?\s+handicapee?s?\b", "absence des personnes handicapees"),
    (r"\baucune personne[s]?\s+handicapee?s?\b", "absence des personnes handicapees"),
    (r"\bpersonne[s]?\s+handicapee?s?\b", "personnes handicapees"),
    (r"\bdans notre societe\b", "a lentreprise"),
    (r"\bdans notre entreprise\b", "a lentreprise"),
    (r"\bau sein de notre societe\b", "a lentreprise"),
    (r"\bau sein de notre entreprise\b", "a lentreprise"),
    (r"\bsociete\b", "entreprise"),
    (r"\bbureau d etude\b", "bureau detude"),
    (r"\bbureau detude\b", "bureau detude"),
    (r"\bautorisation traine\b", "autorisation dossier en cours"),
    (r"\bdossier .*bureau detude\b", "dossier bureau detude autorisation"),
    (r"\bdossier .*autorisation.*bureau\b", "absence dautorisation dexploitation dossier bureau detude"),
    (r"\bautorisation .*bureau detude\b", "absence dautorisation dexploitation"),
    (r"\bdechets? de soins\b", "dechets sanitaires"),
    (r"\bcontenant dedie\b", "conteneur specifique"),
    (r"\bsans contenant dedie\b", "absence de conteneur specifique"),
    (r"\bon jette .*dechets sanitaires\b", "absence de conteneur specifique pour dechets sanitaires"),
    (r"\bna pas realis\w* une analyse de lair ambiant\b", "analyse de lair ambiant nest pas realise"),
    (r"\bn a pas realis\w* une analyse de lair ambiant\b", "analyse de lair ambiant nest pas realise"),
    (r"\bpas realis\w* .*analyse de lair ambiant\b", "analyse de lair ambiant nest pas realise"),
    (r"\banalyse de lair ambiant .*pas realis\w*\b", "analyse de lair ambiant nest pas realise"),
)

TOKEN_REPLACEMENTS = {
    "aucun": "absence",
    "aucune": "absence",
    "biometrique": "biometriques",
    "biometriques": "biometriques",
    "claration": "declaration",
    "dexploitation": "declaration dexploitation",
    "donnes": "donnees",
    "handicape": "handicapees",
    "handicapee": "handicapees",
    "handicapes": "handicapees",
    "mecin": "medecin",
    "medcin": "medecin",
    "societe": "entreprise",
    "abimes": "endommages",
    "abime": "endommages",
    "soins": "sanitaires",
    "xploitation": "exploitation",
}


def _strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return _strip_accents(text)


def _normalize_french_elisions(text: str) -> str:
    return re.sub(r"\b([cdjlmnst])'\s*(\w+)", r"\1\2", text, flags=re.IGNORECASE)


def _repair_common_phrases(text: str) -> str:
    repaired = text
    for pattern, replacement in PHRASE_REPLACEMENTS:
        repaired = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)
    return repaired


def _repair_common_tokens(text: str) -> str:
    repaired_tokens = []
    for token in text.split():
        repaired_tokens.append(TOKEN_REPLACEMENTS.get(token, token))
    return " ".join(repaired_tokens)


def _drop_request_fluff(text: str) -> str:
    kept_tokens = [token for token in text.split() if token not in STOPWORD_TOKENS]
    return " ".join(kept_tokens)


def normalize_query_text(query: str) -> str:
    q = _normalize_unicode(query).strip().lower()
    q = _normalize_french_elisions(q)

    for pattern in PREFIX_PATTERNS:
        q = re.sub(pattern, "", q, flags=re.IGNORECASE)

    for pattern in SUFFIX_PATTERNS:
        q = re.sub(pattern, "", q, flags=re.IGNORECASE)

    q = re.sub(r"[^\w\s'-]", " ", q)
    q = re.sub(r"\s+", " ", q)
    q = _repair_common_phrases(q)
    q = _repair_common_tokens(q)
    q = _drop_request_fluff(q)
    q = re.sub(r"\s+", " ", q)
    q = q.strip(" .?!:;-")

    return q
