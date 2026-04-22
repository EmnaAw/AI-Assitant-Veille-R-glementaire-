import csv
import math
import re
import unicodedata
from pathlib import Path

from .config import NC_RESOLUTION_OVERRIDES_PATH, QUERY_ALIAS_OVERRIDES_PATH
from .data_loader import load_dataset
from .query_classifier import infer_gap_type_from_record


class TruthLookup:
    def __init__(self, data_path):
        self.df = load_dataset(data_path)
        self.by_ncid = {
            str(row["NCid"]).strip(): {
                "NCid": str(row["NCid"]).strip(),
                "NC": str(row["NC"]).strip(),
                "Plan": str(row["Plan"]).strip(),
                "gap_type": infer_gap_type_from_record(
                    str(row["NC"]).strip(),
                    str(row["Plan"]).strip(),
                ).gap_type,
            }
            for _, row in self.df.iterrows()
        }
        self.by_normalized_nc: dict[str, list[dict]] = {}

        for record in self.by_ncid.values():
            key = self.normalize_nc(record["NC"])
            self.by_normalized_nc.setdefault(key, []).append(record)
        self.overrides_by_normalized_nc = self._load_resolution_overrides()
        self.query_aliases = self._load_query_alias_overrides()
        self.token_document_frequency = self._build_token_document_frequency()

    def _load_resolution_overrides(self) -> dict[str, dict]:
        path = Path(NC_RESOLUTION_OVERRIDES_PATH)
        if not path.exists():
            return {}

        overrides = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized_nc = self.normalize_nc(row.get("normalized_nc", ""))
                if not normalized_nc:
                    continue
                overrides[normalized_nc] = {
                    "chosen_ncid": str(row.get("chosen_ncid", "")).strip(),
                    "chosen_plan": self.normalize_plan(row.get("chosen_plan", "")),
                    "note": str(row.get("note", "")).strip(),
                }
        return overrides

    def _load_query_alias_overrides(self) -> dict[str, dict]:
        path = Path(QUERY_ALIAS_OVERRIDES_PATH)
        if not path.exists():
            return {}

        aliases = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized_query = self.normalize_nc(row.get("normalized_query", ""))
                chosen_ncid = str(row.get("chosen_ncid", "")).strip()
                if not normalized_query or not chosen_ncid:
                    continue
                aliases[normalized_query] = {
                    "chosen_ncid": chosen_ncid,
                    "note": str(row.get("note", "")).strip(),
                }
        return aliases

    def _build_token_document_frequency(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for normalized_nc in self.by_normalized_nc.keys():
            for token in set(normalized_nc.split()):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    @staticmethod
    def _strip_accents(text: str) -> str:
        return "".join(
            char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
        )

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text))
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        return TruthLookup._strip_accents(text)

    @staticmethod
    def _normalize_french_elisions(text: str) -> str:
        return re.sub(r"\b([cdjlmnst])'\s*(\w+)", r"\1\2", text, flags=re.IGNORECASE)

    @staticmethod
    def normalize_nc(text: str) -> str:
        text = TruthLookup._normalize_unicode(text).lower().strip()
        text = TruthLookup._normalize_french_elisions(text)
        text = re.sub(r"[^\w\s'-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def normalize_plan(text: str) -> str:
        text = TruthLookup._normalize_unicode(text).lower().strip()
        text = TruthLookup._normalize_french_elisions(text)
        text = re.sub(r"[\-*]+", " ", text)
        text = re.sub(r"[^\w\s'/]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def token_weight(self, token: str) -> float:
        token = self.normalize_nc(token)
        if not token:
            return 0.0
        document_frequency = self.token_document_frequency.get(token, 0)
        total_documents = max(len(self.by_normalized_nc), 1)
        return 1.0 + math.log((total_documents + 1) / (document_frequency + 1))

    def resolve_query_alias(self, normalized_query: str) -> dict | None:
        alias = self.query_aliases.get(self.normalize_nc(normalized_query))
        if not alias:
            return None
        return self.safe_get_by_id(alias["chosen_ncid"])

    def get_by_id(self, ncid: str):
        record = self.by_ncid.get(str(ncid).strip())
        if record is None:
            raise KeyError(f"No row found for NCid={ncid}")
        return record

    def safe_get_by_id(self, ncid: str):
        return self.by_ncid.get(str(ncid).strip())

    def get_records_by_normalized_nc(self, nc: str) -> list[dict]:
        return list(self.by_normalized_nc.get(self.normalize_nc(nc), []))

    def apply_resolution_override(self, records: list[dict]) -> tuple[dict | None, list[dict]]:
        if not records:
            return None, []

        normalized_nc = self.normalize_nc(records[0]["NC"])
        override = self.overrides_by_normalized_nc.get(normalized_nc)
        if not override:
            return None, records

        chosen_ncid = override["chosen_ncid"]
        if chosen_ncid:
            for record in records:
                if str(record["NCid"]).strip() == chosen_ncid:
                    return record, [record]

        chosen_plan = override["chosen_plan"]
        if chosen_plan:
            matching_records = [
                record
                for record in records
                if self.normalize_plan(record["Plan"]) == chosen_plan
            ]
            if matching_records:
                return matching_records[0], matching_records

        return None, records

    def resolve_records_by_normalized_nc(self, nc: str) -> tuple[dict | None, list[dict]]:
        records = self.get_records_by_normalized_nc(nc)
        if not records:
            return None, []

        grouped = {}
        for record in records:
            grouped.setdefault(self.normalize_plan(record["Plan"]), []).append(record)

        if len(grouped) == 1:
            group = next(iter(grouped.values()))
            return group[0], group

        override_record, override_records = self.apply_resolution_override(records)
        if override_record is not None:
            return override_record, override_records

        return None, [record for group in grouped.values() for record in group]
