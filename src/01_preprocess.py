"""
Preparation pipeline for the RPA Projects database.
data/raw/Original.xlsx -> data/processed/DB_RPA_Projects_V2..V9.xlsx

Stages (each one saves an intermediate version, for traceability):
  V2  Sequential IDs + column-name normalisation
  V3  Cleaning and one-hot encoding of 'infrastructure_for_development'
  V4  Cleaning and one-hot encoding of 'system_interaction'
  V5  Ordinal encoding + reformulation of 'frequence_of_the_project'
      + ordinal 'process_area'
  V6  One-hot encoding of 'business_area' and 'developed_by'
  V7  Target variable 'project_duration_ordinal' (5 classes)
  V8  Cardinality reduction of rare dummies
  V9  Post-bivariate removal (depends on the results of 03_bivariate.py)

Methodological notes carried over from earlier versions of the pipeline:
  - Bug 1 (fixed): 'documentation' no longer collapses into the constant 1.
    Now '1 - Yes' -> 1 (documented) and '2 - No' -> 0 (not documented).
  - Bug 2 (fixed): Class 5 ('5 - 160 to 200') is no longer absorbed by
    Class 4. 'project_duration_ordinal' is extracted directly from the class
    index present in the original string, preserving all 5 classes.
  - 'frequence_of_the_project': the original scale mixed 5 ordinal frequency
    levels (1,2,4,5,6) with the non-ordinal "On demand" category (7). It was
    reduced to a contiguous 3-level scale (Daily=1, Weekly=2, Monthly=3) in
    'frequence_grouped', with "On demand" isolated in the binary flag
    'frequence_on_demand'.
  - Cardinality reduction (V8): dummies with n=1 dropped; dummies with n=2
    grouped into 'Other' buckets; 'developed_by' reduced to the
    'developed_by_CD' flag; 'Sourcing' and 'Sourcing/TNA/MDM/AP' merged.
  - Post-bivariate removal (V9): variables with no relevant association with
    the target (|tau-b| or |rank-biserial| < 0.10 and FDR-adjusted p > 0.5),
    plus 'digital_level' (significant chi-squared but negligible tau-b).

NOTE on execution order: the V9 stage depends on the results of the
bivariate analysis (03_bivariate.py), which runs on V8. The list of
variables to remove is fixed below (POST_BIVARIATE_DROPS), reflecting the
results reported in the dissertation; the script can therefore be run in a
single pass. To reproduce the study step by step, use --stop-at-v8, run the
bivariate analysis, then run this script again in full.
"""

import argparse
import re

import numpy as np
import pandas as pd

from config import DB_VERSIONS, PATH_ORIGINAL, ensure_dirs

# ============================================================
# Cleaning maps
# ============================================================
INFRA_MAPPING = {
    "pyhton": "Python",
    "uipath": "UiPath",
    "python": "Python",
    "power automate": "PowerAutomate",
    "sap gui scripting": "GuiScripting",
    "sql": "MySQL",
    "powerbi": "PowerBi",
}

SYSTEM_MAPPING = {
    "sap": "SAP",
    "sap one": "SAP",
    "SAP One": "SAP",
    "coe sap": "SAP",
    "sap o.n.e.": "SAP",
    "outllok": "Outlook",
    "outlook mail": "Outlook",
    "ms outlook": "Outlook",
    "excel sheet": "Excel",
    "excel": "Excel",
    "ms excel": "Excel",
    "share point": "SharePoint",
    "teamsapp": "SharePoint",
    "teams application": "SharePoint",
    "powerbi": "PowerBi",
    "sql database": "MySQL",
    "enterprise scan": "EnterpriseScan",
    "ivanti": "Ivanti",
    "teams": "SharePoint",
    "sql": "MySQL",
    "sharepoint": "SharePoint",
    "email": "Outlook",
    "power bi": "PowerBi",
    "outlook": "Outlook",
    "one stream": "OneStream",
    "ms-excel": "Excel",
    "ms teams": "SharePoint",
    "ms access": "Access",
    "forms": "Forms",
    "fmc": "FMC",
    "flip": "FLIP",
    "extranet": "Extranet",
    "concurapi": "ConcurAPI",
    "blackline": "Blackline",
    "api": "ConcurAPI",
    "google forms": "Google Forms",
    "experian website": "Experian",
    "bank website": "Bank Website",
}

ORDINAL_COLUMNS = [
    "development_complexity",
    "process_complexity",
    "pain_points",
    "project_based_in_rules_",
    "digital_level",
    "structure_of_input_data",
    "knowledge_of_the_business_process",
    "documentation",
]

# Variables removed in V9 based on the bivariate analysis (03_bivariate.py):
# |tau-b| or |rank-biserial| < 0.10 and FDR-adjusted p > 0.5. Special case
# 'digital_level': significant chi-squared but negligible tau-b (-0.04) and
# 79% of observations at level 5 — uninformative for ordinal prediction.
POST_BIVARIATE_DROPS = [
    "system_Blackline",
    "system_SAP",
    "system_PowerBi",
    "business_area_Finance",
    "digital_level",
]


# ============================================================
# Helper functions
# ============================================================
def clean_and_map_multivalued(items, mapping):
    """Clean and normalise a list of comma-separated values."""
    cleaned = [item.strip().lower() for item in items if item.strip()]
    return [mapping.get(item, item) for item in cleaned]


def one_hot_multivalued(df, column, prefix, mapping):
    """
    Normalise a multi-valued column (comma-separated values) and apply
    one-hot encoding with the given prefix.
    """
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.split(",")
        .apply(lambda items: clean_and_map_multivalued(items, mapping))
        .apply(lambda x: x if isinstance(x, list) and len(x) > 1
               else (x[0] if x else ""))
    )
    exploded = df.explode(column)
    one_hot = pd.get_dummies(exploded[column], prefix=prefix)
    one_hot = one_hot.groupby(level=0).sum()
    df = df.drop(columns=[column])
    return pd.concat([df, one_hot], axis=1)


def reorder_columns(df, family_prefixes):
    """Put process_id first, followed by the given dummy families."""
    family_cols = []
    for prefix in family_prefixes:
        family_cols += sorted(c for c in df.columns if c.startswith(prefix))
    other = [c for c in df.columns if c not in ["process_id"] + family_cols]
    return df[["process_id"] + family_cols + other]


def extract_ordinal_value(text):
    """Extract the first number from a string like '2 - Simple'."""
    if pd.isna(text):
        return None
    match = re.search(r"\d+", str(text))
    return int(match.group(0)) if match else None


def map_frequence_grouped(value):
    """1,2 -> 1 (Daily); 4 -> 2 (Weekly); 5,6 -> 3 (Monthly); 7 -> NaN."""
    if pd.isna(value):
        return np.nan
    v = int(re.search(r"\d+", str(value)).group(0))
    if v in (1, 2):
        return 1
    if v == 4:
        return 2
    if v in (5, 6):
        return 3
    return np.nan  # 7 = On demand: handled by the binary flag


def map_on_demand(value):
    if pd.isna(value):
        return 0
    v = int(re.search(r"\d+", str(value)).group(0))
    return 1 if v == 7 else 0


def extract_process_area_ordinal(text):
    """Extract the leading number from 'process_area' ('5 - Global' -> 5)."""
    if pd.isna(text):
        return np.nan
    match = re.search(r"\d+", str(text))
    return int(match.group(0)) if match else np.nan


def clean_category(text):
    # 'Desconhecido' ("Unknown") is kept in Portuguese on purpose: it is a
    # data value, and changing it would alter the generated column names.
    if pd.isna(text):
        return "Desconhecido"
    cleaned = re.sub(r"^\d+\s*-\s*", "", str(text)).strip()
    return cleaned.replace(" ", "_")


def extract_duration_class(text):
    """Extract the duration class index ('5 - 160 to 200' -> 5)."""
    if pd.isna(text):
        return np.nan
    match = re.match(r"\s*(\d+)\s*-", str(text))
    return int(match.group(1)) if match else np.nan


def reduce_rare_dummies(df, prefix, drop_threshold=1, group_threshold=2):
    """
    For a set of dummies sharing the same prefix:
      - drop columns with sum <= drop_threshold (default: 1)
      - group columns with sum == group_threshold into a '<prefix>Other' bucket
    Returns (df, dropped_columns, grouped_columns).
    """
    cols = [c for c in df.columns if c.startswith(prefix)]
    sums = df[cols].sum()

    to_drop = sums[sums <= drop_threshold].index.tolist()
    to_group = sums[sums == group_threshold].index.tolist()

    df.drop(columns=to_drop, inplace=True)

    if to_group:
        other_col = f"{prefix}Other"
        df[other_col] = (df[to_group].sum(axis=1) > 0).astype(int)
        df.drop(columns=to_group, inplace=True)

    return df, to_drop, to_group


# ============================================================
# Pipeline
# ============================================================
def main(stop_at_v8=False):
    ensure_dirs()

    if not PATH_ORIGINAL.exists():
        raise FileNotFoundError(
            f"Source file not found: {PATH_ORIGINAL}\n"
            "The original database is not distributed with this repository "
            "(confidentiality — see README). Place 'Original.xlsx' in "
            "data/raw/ to run the pipeline."
        )

    # --------------------------------------------------------
    # 1) Sequential IDs
    # --------------------------------------------------------
    df = pd.read_excel(PATH_ORIGINAL)
    df["process_id"] = range(1, len(df) + 1)
    df.drop(columns=["Process"], inplace=True)
    df.insert(0, "process_id", df.pop("process_id"))

    # --------------------------------------------------------
    # 2) Column-name normalisation (V2)
    # --------------------------------------------------------
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.strip()
    )
    df = df.rename(columns={
        "infrasestructure_for_development": "infrastructure_for_development"
    })
    df.to_excel(DB_VERSIONS[2], index=False)

    # --------------------------------------------------------
    # 3) 'infrastructure_for_development' -> one-hot (V3)
    # --------------------------------------------------------
    df = one_hot_multivalued(df, "infrastructure_for_development", "infra",
                             INFRA_MAPPING)
    df = reorder_columns(df, ["infra_"])
    df.to_excel(DB_VERSIONS[3], index=False)

    # --------------------------------------------------------
    # 4) 'system_interaction' -> one-hot (V4)
    # --------------------------------------------------------
    df = one_hot_multivalued(df, "system_interaction", "system",
                             SYSTEM_MAPPING)
    df = reorder_columns(df, ["infra_", "system_"])
    df.to_excel(DB_VERSIONS[4], index=False)

    # --------------------------------------------------------
    # 5) Ordinal variable encoding
    # --------------------------------------------------------
    for col in ORDINAL_COLUMNS:
        df[col] = df[col].apply(extract_ordinal_value)

    # BUG FIX 1: 'documentation' as a proper binary variable.
    # Original values: '1 - Yes' and '2 - No'. After extract_ordinal_value we
    # have 1 and 2; we want 1 -> 1 (documented) and 2 -> 0 (not documented).
    df["documentation"] = (df["documentation"] == 1).astype(int)
    assert df["documentation"].nunique() == 2, (
        "Documentation should have 2 distinct values (0 and 1). "
        f"Found: {df['documentation'].unique()}"
    )

    # Reformulation of 'frequence_of_the_project'
    df["frequence_grouped"] = df["frequence_of_the_project"].apply(
        map_frequence_grouped)
    df["frequence_on_demand"] = df["frequence_of_the_project"].apply(
        map_on_demand).astype(int)
    df.drop(columns=["frequence_of_the_project"], inplace=True)

    assert set(df["frequence_grouped"].dropna().unique()) <= {1, 2, 3}, (
        "frequence_grouped should have levels {1,2,3}. "
        f"Found: {sorted(df['frequence_grouped'].dropna().unique())}"
    )
    assert df["frequence_grouped"].isna().sum() == df["frequence_on_demand"].sum(), (
        "NaNs in frequence_grouped must match the 1s in frequence_on_demand."
    )

    # Imputation: on-demand projects have no defined periodic frequency.
    # Impute the mode (Daily=1); the 'frequence_on_demand' flag carries the
    # distinguishing information.
    df["frequence_grouped"] = df["frequence_grouped"].fillna(1).astype(int)

    # 'process_area' -> ordinal (V5)
    df["process_area_ordinal"] = df["process_area"].apply(
        extract_process_area_ordinal)
    df.drop(columns=["process_area"], inplace=True)
    df.to_excel(DB_VERSIONS[5], index=False)

    # --------------------------------------------------------
    # 6-7) Nominal variables -> one-hot (V6)
    # --------------------------------------------------------
    df["developed_by"] = df["developed_by"].apply(clean_category)
    df["business_area"] = (
        df["business_area"].fillna("Desconhecido")
        .apply(lambda x: str(x).replace(" ", "_"))
    )
    df = pd.get_dummies(df, columns=["business_area", "developed_by"],
                        dtype=int)
    df.to_excel(DB_VERSIONS[6], index=False)

    # --------------------------------------------------------
    # 8) Target variable (V7)
    # --------------------------------------------------------
    # BUG FIX 2: the old version extracted the lower bound of the interval and
    # used pd.cut with right=True, causing Class 5 to fall into Class 4.
    # The class index is already at the start of the string — extract it
    # directly.
    df["project_duration_ordinal"] = df["project_duration"].apply(
        extract_duration_class)
    df.drop(columns=["project_duration"], inplace=True)

    assert df["project_duration_ordinal"].nunique() == 5, (
        "Expected 5 classes in project_duration_ordinal. "
        f"Found: {sorted(df['project_duration_ordinal'].dropna().unique())}"
    )
    df.to_excel(DB_VERSIONS[7], index=False)

    # --------------------------------------------------------
    # 9) Cardinality reduction (V8)
    # --------------------------------------------------------
    # 9.1) Merge Sourcing before any other operation
    sourcing_main = "business_area_Sourcing"
    sourcing_ext = [
        c for c in df.columns
        if c.startswith("business_area_") and "Sourcing" in c
        and c != sourcing_main
    ]
    if sourcing_main in df.columns and sourcing_ext:
        for col in sourcing_ext:
            df[sourcing_main] = ((df[sourcing_main] + df[col]) > 0).astype(int)
        df.drop(columns=sourcing_ext, inplace=True)

    # 9.2) Reduce 'developed_by' to the binary flag 'developed_by_CD'
    if "developed_by_IA_Team" in df.columns:
        df.drop(columns=["developed_by_IA_Team"], inplace=True)

    # 9.3) Rare dummies per family
    df, infra_dropped, infra_grouped = reduce_rare_dummies(df, "infra_")
    df, system_dropped, system_grouped = reduce_rare_dummies(df, "system_")
    df, ba_dropped, ba_grouped = reduce_rare_dummies(df, "business_area_")

    # 9.4) Reorder columns. The exact order matters: Random Forest results
    # depend on feature positions (per-node feature sampling), so this
    # replicates the original pipeline's order — id, infra_, system_,
    # remaining variables, business_area_, developed_by_ — to keep results
    # reproducible.
    infra_cols = sorted(c for c in df.columns if c.startswith("infra_"))
    system_cols = sorted(c for c in df.columns if c.startswith("system_"))
    ba_cols = sorted(c for c in df.columns if c.startswith("business_area_"))
    dev_cols = sorted(c for c in df.columns if c.startswith("developed_by_"))
    grouped = infra_cols + system_cols + ba_cols + dev_cols
    other_cols = [c for c in df.columns if c not in ["process_id"] + grouped]
    df = df[["process_id"] + infra_cols + system_cols + other_cols
            + ba_cols + dev_cols]

    print("\n=== Cardinality reduction (V8) ===")
    print("Dropped (n=1):")
    print(f"  infra_: {infra_dropped}")
    print(f"  system_: {system_dropped}")
    print(f"  business_area_: {ba_dropped}")
    print("Grouped into 'Other' (n=2):")
    print(f"  infra_: {infra_grouped}")
    print(f"  system_: {system_grouped}")
    print(f"  business_area_: {ba_grouped}")

    for col in ("documentation", "frequence_grouped", "frequence_on_demand",
                "project_duration_ordinal"):
        print(f"\n=== Final distribution of '{col}' ===")
        print(df[col].value_counts(dropna=False).sort_index())

    for prefix in ("infra_", "system_", "business_area_"):
        print(f"\n=== {prefix} dummies in V8 ===")
        cols = [c for c in df.columns if c.startswith(prefix)]
        print(df[cols].sum().sort_values(ascending=False))

    print(f"\nFinal V8 shape: {df.shape}")
    df.to_excel(DB_VERSIONS[8], index=False)
    print(f"V8 saved to: {DB_VERSIONS[8]}")

    if stop_at_v8:
        print("\n--stop-at-v8: V9 was not generated. Run 03_bivariate.py "
              "and then run this script again without the flag.")
        return

    # --------------------------------------------------------
    # 10) Post-bivariate removal (V9)
    # --------------------------------------------------------
    existing = [c for c in POST_BIVARIATE_DROPS if c in df.columns]
    missing = set(POST_BIVARIATE_DROPS) - set(existing)
    if missing:
        print(f"\n[WARNING] Columns to remove not found in V8: {missing}")

    df.drop(columns=existing, inplace=True)

    print("\n=== Post-bivariate removal (V9) ===")
    print(f"Variables removed ({len(existing)}): {existing}")
    print(f"\nFinal V9 shape: {df.shape}")

    df.to_excel(DB_VERSIONS[9], index=False)
    print(f"V9 saved to: {DB_VERSIONS[9]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preparation pipeline Original.xlsx -> V2..V9.")
    parser.add_argument(
        "--stop-at-v8", action="store_true",
        help="Stop after generating V8 (to reproduce the study step by step: "
             "run the bivariate analysis before generating V9).")
    args = parser.parse_args()
    main(stop_at_v8=args.stop_at_v8)
