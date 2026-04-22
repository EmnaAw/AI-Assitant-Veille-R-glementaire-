import os
import re
import traceback

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DATA_DIR = "./data"
PDF_DIR = "./data/pdfs"
DB_DIR = "./db_vigogne_bge_m3"
EMB_MODEL = "BAAI/bge-m3"

# Set this to your Poppler bin folder on Windows.
# Example: r"C:\poppler\Library\bin"
POPPLER_PATH = r"C:\Release-25.12.0-0\poppler-25.12.0\Library\bin"

LEGAL_SEPARATORS = [
    r"\n(?=Article\s+\d+)",
    r"\n(?=Art\.\s*\d+)",
    r"\n(?=Chapitre\s+[IVXLC\d]+)",
    r"\n(?=Section\s+\d+)",
    r"\n(?=Titre\s+[IVXLC\d]+)",
    "\n\n",
    "\n",
]

TXT_FILES = ["Lois.txt", "Articles.txt", "TermesJuridiques.txt"]

KNOWN_FIELDS = [
    "ID", "Application", "App_ID", "Theme", "Theme_ID", "Type",
    "Numero", "Date", "Journal", "Titre", "Statut", "PDF",
    "Resume", "Complement",
]
FIELD_PATTERN = re.compile(
    r"^(" + "|".join(KNOWN_FIELDS) + r"):\s*", re.MULTILINE
)

LEGAL_RANKS = {
    "loi": 1,
    "code": 1,
    "décret": 2,
    "decret": 2,
    "arrêté": 3,
    "arrete": 3,
    "décision": 4,
    "decision": 4,
    "circulaire": 4,
    "inconnu": 5,
}


def normalize_pdf_name(name: str) -> str:
    stem = name.lower().replace(".pdf", "")
    stem = re.sub(r"[_\-\s]+", "-", stem)
    return stem


def normalize_statut(value: str | None) -> str:
    if not value:
        return ""
    s = value.strip().lower()
    replacements = {
        "abroge": "abrogé",
        "envigueur": "en vigueur",
        "en_vigueur": "en vigueur",
    }
    return replacements.get(s, s)


def get_legal_rank(type_texte: str | None) -> int:
    if not type_texte:
        return 5
    return LEGAL_RANKS.get(type_texte.strip().lower(), 5)


def build_pdf_disk_index(pdf_dir: str) -> dict[str, str]:
    index = {}
    for root, _, files in os.walk(pdf_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                full_path = os.path.join(root, f)
                index[f.lower()] = full_path
                index[normalize_pdf_name(f)] = full_path
    file_count = sum(1 for k in index if k.endswith(".pdf"))
    print(f"  📂 PDF disk index: {file_count} files ({len(index)} keys) under {pdf_dir}")
    return index


def find_pdf_on_disk(pdf_ref: str, pdf_disk_index: dict[str, str]) -> str | None:
    if not pdf_ref:
        return None

    exact = pdf_ref.lower()
    if exact in pdf_disk_index:
        return pdf_disk_index[exact]

    normalized = normalize_pdf_name(pdf_ref)
    if normalized in pdf_disk_index:
        return pdf_disk_index[normalized]

    return None


def parse_record_block(block: str) -> dict:
    fields = {}
    matches = list(FIELD_PATTERN.finditer(block))

    for i, match in enumerate(matches):
        field_name = match.group(1)
        value_start = match.end()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        value = block[value_start:value_end].strip()
        value = " ".join(value.split())
        fields[field_name] = value

    return fields


def parse_unified_txt(filepath: str) -> list[Document]:
    with open(filepath, encoding="utf-8") as f:
        raw = f.read()

    source_name = os.path.basename(filepath)

    lines = raw.splitlines()
    body_start = next(
        (i for i, l in enumerate(lines) if l.strip() and not l.startswith("#")),
        0,
    )
    body = "\n".join(lines[body_start:])

    records = re.split(r"─{10,}", body)
    documents = []

    for block in records:
        block = block.strip()
        if not block:
            continue

        fields = parse_record_block(block)

        resume = fields.get("Resume", "").strip()
        titre = fields.get("Titre", "").strip()
        if not resume and not titre:
            continue
        if resume.upper() == "N/A" and not titre:
            continue

        content_parts = []
        if titre:
            content_parts.append(f"Titre: {titre}")
        numero = fields.get("Numero", "")
        if numero and numero.upper() != "N/A":
            content_parts.append(f"Numéro: {numero}")
        if fields.get("Theme"):
            content_parts.append(f"Thème: {fields['Theme']}")
        if fields.get("Type"):
            content_parts.append(f"Type: {fields['Type']}")
        if fields.get("Date") and fields["Date"].upper() != "N/A":
            content_parts.append(f"Date: {fields['Date']}")
        if fields.get("Journal"):
            content_parts.append(f"Journal: {fields['Journal']}")
        if resume and resume.upper() != "N/A":
            content_parts.append(f"Résumé: {resume}")
        complement = fields.get("Complement", "").strip()
        if complement and complement.upper() not in ("N/A", ""):
            content_parts.append(f"Complément: {complement}")

        page_content = "\n".join(content_parts)
        type_texte = fields.get("Type", "").lower()
        statut = normalize_statut(fields.get("Statut", ""))

        metadata = {
            "record_id": fields.get("ID", ""),
            "source": source_name,
            "source_kind": "txt",
            "authority": "primary",
            "app_id": fields.get("App_ID", ""),
            "theme_id": fields.get("Theme_ID", ""),
            "application": fields.get("Application", ""),
            "theme": fields.get("Theme", ""),
            "type_texte": type_texte,
            "legal_rank": get_legal_rank(type_texte),
            "numero": numero,
            "date": fields.get("Date", "inconnue") if fields.get("Date", "").upper() != "N/A" else "inconnue",
            "journal": fields.get("Journal", ""),
            "titre": titre,
            "statut": statut,
            "pdf_file": fields.get("PDF", "").strip(),
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"  ✅ Parsed {len(documents)} records from {source_name}")
    return documents


def infer_metadata_from_path(filename: str, subfolder: str) -> dict:
    meta = {
        "application": "",
        "app_id": "",
        "theme": subfolder,
        "type_texte": "inconnu",
        "authority": "fallback",
        "source_kind": "pdf",
        "statut": "",
        "numero": "",
        "date": "inconnue",
    }

    name = filename.lower().replace(".pdf", "")

    if name.startswith("arrete") or name.startswith("arrêté"):
        meta["type_texte"] = "arrêté"
    elif name.startswith("decret") or name.startswith("décret"):
        meta["type_texte"] = "décret"
    elif name.startswith("loi"):
        meta["type_texte"] = "loi"
    elif name.startswith("code"):
        meta["type_texte"] = "code"

    m = re.search(r"(\d{2,4}[-_]\d{1,5})", name)
    if m:
        meta["numero"] = m.group(1).replace("_", "-")

    m_year = re.search(r"\b(19|20)\d{2}\b", name)
    if m_year:
        meta["date"] = m_year.group(0)

    securite_keywords = [
        "accident", "amiante", "bruit", "établissement dangereux",
        "hydrocarbures", "matières dangereuses", "incendie", "travail",
        "sécurité", "santé",
    ]
    sub_lower = subfolder.lower()
    if any(k in sub_lower for k in securite_keywords):
        meta["application"] = "SÉCURITÉ"
        meta["app_id"] = "2"
    else:
        meta["application"] = ""
        meta["app_id"] = ""

    meta["legal_rank"] = get_legal_rank(meta.get("type_texte"))
    return meta


def sanitize_loaded_pages(pages) -> list[Document]:
    cleaned_pages = []

    if not pages:
        return cleaned_pages

    for page in pages:
        page_content = getattr(page, "page_content", "") or ""
        metadata = getattr(page, "metadata", {}) or {}

        if not isinstance(metadata, dict):
            metadata = {}

        if not page_content.strip():
            continue

        cleaned_pages.append(
            Document(
                page_content=page_content,
                metadata=dict(metadata),
            )
        )

    return cleaned_pages


def ocr_pdf_with_pytesseract(full_path: str) -> list[Document]:
    try:
        images = convert_from_path(full_path, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"  ⚠ OCR image conversion failed for {os.path.basename(full_path)}: {type(e).__name__}: {e}")
        return []

    pages = []
    for page_num, image in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(image, lang="fra")
        except Exception as e:
            print(f"  ⚠ Tesseract failed on page {page_num} of {os.path.basename(full_path)}: {type(e).__name__}: {e}")
            text = ""

        text = (text or "").strip()
        if not text:
            continue

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "ocr_page": page_num,
                    "ocr_engine": "pytesseract",
                },
            )
        )

    return pages


def load_pdfs(
    pdf_dir: str,
    txt_records: list[Document],
    pdf_disk_index: dict[str, str],
) -> list[Document]:
    pdf_to_records: dict[str, list[dict]] = {}
    for doc in txt_records:
        pdf_ref = doc.metadata.get("pdf_file", "").strip()
        if pdf_ref:
            keys = {pdf_ref.lower(), normalize_pdf_name(pdf_ref)}

            resolved_path = find_pdf_on_disk(pdf_ref, pdf_disk_index)
            if resolved_path:
                resolved_name = os.path.basename(resolved_path)
                keys.add(resolved_name.lower())
                keys.add(normalize_pdf_name(resolved_name))

            for key in keys:
                pdf_to_records.setdefault(key, [])
                if doc.metadata not in pdf_to_records[key]:
                    pdf_to_records[key].append(doc.metadata)

    linked_count = 0
    unlinked_count = 0
    ocr_count = 0

    legal_splitter = RecursiveCharacterTextSplitter(
        separators=LEGAL_SEPARATORS,
        is_separator_regex=True,
        chunk_size=800,
        chunk_overlap=150,
        keep_separator=True,
    )

    documents = []

    for filename_lower, full_path in pdf_disk_index.items():
        if not filename_lower.endswith(".pdf"):
            continue

        filename = os.path.basename(full_path)
        subfolder = os.path.basename(os.path.dirname(full_path))
        norm_key = normalize_pdf_name(filename)

        try:
            loader = PyPDFLoader(full_path)
            pages = loader.load()

            if not pages:
                print(f"  ⚠ Empty PDF: {filename}")
                continue

            cleaned_pages = sanitize_loaded_pages(pages)

            if not cleaned_pages:
                print(f"  ↪ OCR fallback (pdf2image + pytesseract): {filename}")
                cleaned_pages = ocr_pdf_with_pytesseract(full_path)

                if not cleaned_pages:
                    print(f"  ⚠ No readable text pages: {filename}")
                    continue

                ocr_count += 1

            inherited = {}
            matched_key = None
            for try_key in (filename_lower, norm_key):
                if try_key in pdf_to_records:
                    matched_key = try_key
                    break

            if matched_key:
                matched_records = pdf_to_records.get(matched_key) or []
                ref = matched_records[0] if matched_records else {}

                inherited = {
                    "application": ref.get("application", ""),
                    "theme": ref.get("theme", ""),
                    "type_texte": ref.get("type_texte", ""),
                    "legal_rank": ref.get("legal_rank", get_legal_rank(ref.get("type_texte"))),
                    "numero": ref.get("numero", ""),
                    "date": ref.get("date", "inconnue"),
                    "journal": ref.get("journal", ""),
                    "titre": ref.get("titre", ""),
                    "statut": normalize_statut(ref.get("statut", "")),
                    "app_id": ref.get("app_id", ""),
                    "theme_id": ref.get("theme_id", ""),
                    "authority": "secondary",
                    "source_kind": "pdf",
                    "linked_record_ids": ",".join(
                        (r or {}).get("record_id", "") for r in matched_records if r
                    ),
                }
                linked_count += 1
            else:
                inherited = infer_metadata_from_path(filename, subfolder)
                unlinked_count += 1

            chunks = legal_splitter.split_documents(cleaned_pages)

            for chunk in chunks:
                if chunk.metadata is None:
                    chunk.metadata = {}

                chunk.metadata["source"] = filename
                chunk.metadata["pdf_file"] = filename
                chunk.metadata["subfolder"] = subfolder
                chunk.metadata.update(inherited)

                m = re.match(
                    r"(Article\s+\d+[^\n]*|Art\.\s*\d+[^\n]*)",
                    (chunk.page_content or "").strip(),
                )
                if m:
                    chunk.metadata["article"] = m.group(0)[:80]

            documents.extend(chunks)

        except Exception as e:
            print(f"  ⚠ Could not load {filename}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n  PDF loading complete:")
    print(f"    Linked to TXT record : {linked_count} PDFs")
    print(f"    Unlinked (inferred)  : {unlinked_count} PDFs")
    print(f"    OCR fallback used    : {ocr_count} PDFs")
    print(f"    Total chunks         : {len(documents)}")
    return documents


def ingest_data():
    all_documents: list[Document] = []

    print("\n📄 Parsing TXT files...")
    txt_records: list[Document] = []
    for fname in TXT_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            docs = parse_unified_txt(fpath)
            txt_records.extend(docs)
        else:
            print(f"  ⚠ Not found: {fpath}")

    all_documents.extend(txt_records)
    print(f"\n  Total TXT records: {len(txt_records)}")

    print("\n📑 Scanning PDF directory recursively...")
    pdf_disk_index = build_pdf_disk_index(PDF_DIR)

    print("\n📑 Loading and chunking PDFs...")
    pdf_docs = load_pdfs(PDF_DIR, txt_records, pdf_disk_index)
    all_documents.extend(pdf_docs)

    print(f"\n{'=' * 50}")
    print("INGESTION SUMMARY")
    print(f"{'=' * 50}")
    print(
        f"  TXT records (Lois)             : {sum(1 for d in txt_records if 'Lois' in d.metadata.get('source', ''))}"
    )
    print(
        f"  TXT records (Articles)         : {sum(1 for d in txt_records if 'Articles' in d.metadata.get('source', ''))}"
    )
    print(
        f"  TXT records (TermesJuridiques) : {sum(1 for d in txt_records if 'Termes' in d.metadata.get('source', ''))}"
    )
    print(f"  PDF chunks total               : {len(pdf_docs)}")
    print(f"  GRAND TOTAL documents          : {len(all_documents)}")
    print(f"{'=' * 50}")

    print(f"\n🔢 Embedding with {EMB_MODEL}...")
    print(f"   (This will take several minutes for {len(all_documents)} documents)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    BATCH_SIZE = 1000
    print(f"   Ingesting in batches of {BATCH_SIZE}...")

    vector_db = None
    for i in range(0, len(all_documents), BATCH_SIZE):
        batch = all_documents[i : i + BATCH_SIZE]
        print(f"   Batch {i // BATCH_SIZE + 1}: docs {i}–{i + len(batch) - 1}")

        if vector_db is None:
            vector_db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_DIR,
            )
        else:
            vector_db.add_documents(batch)

    print(f"\n✅ Database saved → {DB_DIR}")
    print(f"   Total vectors stored: {vector_db._collection.count()}")


if __name__ == "__main__":
    ingest_data()