import pandas as pd


DATASET_COLUMN_ALIASES = {
    "NCid": ("NCid",),
    "NC": (
        "NC",
        "NonConformité",
        "NonConformite",
        "NonConformité_cleaned",
        "NonConformite_cleaned",
    ),
    "Plan": (
        "Plan",
        "Action Plan",
        "Action Plan_cleaned",
    ),
}


def _validate_duplicate_ncid_rows(df: pd.DataFrame) -> None:
    duplicate_rows = df[df.duplicated(subset=["NCid"], keep=False)].copy()
    if duplicate_rows.empty:
        return

    conflicting_ncid_values: list[str] = []
    for ncid, group in duplicate_rows.groupby("NCid", sort=False):
        normalized_pairs = {
            (
                str(row["NC"]).strip(),
                str(row["Plan"]).strip(),
            )
            for _, row in group.iterrows()
        }
        if len(normalized_pairs) > 1:
            conflicting_ncid_values.append(str(ncid))

    if conflicting_ncid_values:
        preview = ", ".join(conflicting_ncid_values[:5])
        raise ValueError(
            "Dataset contains conflicting duplicate NCid rows. "
            f"Resolve duplicates before loading. Example NCid values: {preview}"
        )


def _resolve_column_names(df: pd.DataFrame) -> dict[str, str]:
    resolved = {}
    for canonical_name, aliases in DATASET_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                resolved[canonical_name] = alias
                break
        else:
            raise ValueError(
                f"Missing required column for '{canonical_name}'. Accepted names: {aliases}"
            )
    return resolved


def load_dataset(path):
    df = pd.read_excel(path)
    columns = _resolve_column_names(df)

    df = df[[columns["NCid"], columns["NC"], columns["Plan"]]].copy()
    df.columns = ["NCid", "NC", "Plan"]
    df = df.dropna(subset=["NCid"])
    df["NCid"] = df["NCid"].astype(str).str.strip()
    df["NC"] = df["NC"].fillna("").astype(str).str.strip()
    df["Plan"] = df["Plan"].fillna("").astype(str).str.strip()
    _validate_duplicate_ncid_rows(df)
    df = df.drop_duplicates(subset=["NCid"], keep="first").reset_index(drop=True)
    return df
