import re


FRENCH_ACCENT_REPLACEMENTS = (
    (r"\ba caractere\b", "à caractère"),
    (r"\ba l'article\b", "à l'article"),
    (r"\ba jour\b", "à jour"),
    (r"\ba l article\b", "à l'article"),
    (r"\ba propos les\b", "à propos des"),
    (r"\baupres\b", "auprès"),
    (r"\bcameras\b", "caméras"),
    (r"\bconformement\b", "conformément"),
    (r"\bdeclaration\b", "déclaration"),
    (r"\bdeposer\b", "déposer"),
    (r"\bdonnees\b", "données"),
    (r"\betablir\b", "établir"),
    (r"\bmedecine\b", "médecine"),
    (r"\bnecessaire\b", "nécessaire"),
    (r"\bnecessaires\b", "nécessaires"),
    (r"\bpreparer\b", "préparer"),
    (r"\bprevention\b", "prévention"),
    (r"\bsecurite\b", "sécurité"),
    (r"\bbiometrique\b", "biométrique"),
    (r"\bbiometriques\b", "biométriques"),
    (r"\binstallee\b", "installée"),
    (r"\binstallees\b", "installées"),
)


def repair_text_encoding(text: str) -> str:
    value = str(text or "")

    for _ in range(2):
        if any(marker in value for marker in ("Ã", "â", "Â")):
            try:
                value = value.encode("latin-1").decode("utf-8")
                continue
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        break

    replacements = {
        "\u00a0": " ",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)

    return value


def clean_client_text(text: str) -> str:
    value = repair_text_encoding(text)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def sentence_case(text: str) -> str:
    value = clean_client_text(text)
    if not value:
        return value
    return value[0].upper() + value[1:]


def polish_french_text(text: str) -> str:
    value = sentence_case(text)
    if not value:
        return value

    value = re.sub(r"\bd\s+exploitation\b", "d'exploitation", value, flags=re.IGNORECASE)
    value = re.sub(r"\bl\s+article\b", "l'article", value, flags=re.IGNORECASE)
    value = re.sub(r"\bl\s+instance\b", "l'instance", value, flags=re.IGNORECASE)
    value = re.sub(r"\bl\s+entreprise\b", "l'entreprise", value, flags=re.IGNORECASE)

    for pattern, replacement in FRENCH_ACCENT_REPLACEMENTS:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)

    step_verbs = (
        "assurer|deposer|etablir|finaliser|former|mettre|nommer|planifier|"
        "preparer|realiser|regulariser|verifier"
    )
    value = re.sub(rf"^(\d+)\s+(?=({step_verbs})\b)", r"\1. ", value, flags=re.IGNORECASE)
    value = re.sub(
        rf"(?<=[.;])\s+(\d+)\s+(?=({step_verbs})\b)",
        r" \1. ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        rf"(?<=\d)\s+(\d+)\s+(?=({step_verbs})\b)",
        r". \1. ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip()

    if value and value[-1] not in ".!?":
        value += "."
    return value
