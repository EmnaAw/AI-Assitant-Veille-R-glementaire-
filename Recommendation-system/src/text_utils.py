import re


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
