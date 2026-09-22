"""
Generate a SYNTHETIC version of DB_RPA_Projects_V9.xlsx for the public demo app.

Why this exists
---------------
The real V9 database is derived from confidential company data (52 real RPA
projects) and must never leave the local machine or the private analysis. To
publish the Streamlit app online, we instead ship a synthetic dataset with the
*same structure* (27 columns, same names, same value ranges) but *invented*
values. The published app trains on this synthetic data, so it demonstrates the
interface and the prediction flow without exposing any real information.

The results reported in the dissertation come from the REAL data, run locally.
The public app is a demo only.

How the synthetic data is built
-------------------------------
- Each feature is sampled independently from the marginal distribution observed
  in the real V9 (percentages only are embedded below, never row-level data).
- The target 'project_duration_ordinal' is generated from a light, noisy
  function of a few complexity features, so that moving the app's sliders
  produces sensible variation. This relationship is invented for the demo and
  does NOT reflect the study's real findings.

Usage
-----
    uv run python src/generate_synthetic_v9.py

Writes data/processed/DB_RPA_Projects_V9.xlsx (overwrites whatever is there).
"""
import numpy as np
import pandas as pd

from config import PATH_V9, ensure_dirs

RANDOM_STATE = 42
N_ROWS = 52

# Column order must match the real V9 exactly (feature position affects the
# Random Forest, per the preprocessing pipeline).
COLUMN_ORDER = [
    "process_id",
    "infra_MySQL", "infra_Other", "infra_Python", "infra_UiPath",
    "system_ConcurAPI", "system_EnterpriseScan", "system_Excel",
    "system_Ivanti", "system_MySQL", "system_Other", "system_Outlook",
    "system_SharePoint",
    "development_complexity", "process_complexity", "pain_points",
    "project_based_in_rules_", "structure_of_input_data",
    "knowledge_of_the_business_process", "documentation",
    "frequence_grouped", "frequence_on_demand", "process_area_ordinal",
    "project_duration_ordinal",
    "business_area_Other", "business_area_Sourcing", "developed_by_CD",
]

# Marginal distributions from the real V9 (proportions only). Each entry maps
# {value: probability}. The target is generated separately (see below).
MARGINALS = {
    "infra_MySQL": {0: 0.9423, 1: 0.0577},
    "infra_Other": {0: 0.9615, 1: 0.0385},
    "infra_Python": {0: 0.7308, 1: 0.2692},
    "infra_UiPath": {0: 0.25, 1: 0.75},
    "system_ConcurAPI": {0: 0.9038, 1: 0.0962},
    "system_EnterpriseScan": {0: 0.8846, 1: 0.1154},
    "system_Excel": {0: 0.2885, 1: 0.7115},
    "system_Ivanti": {0: 0.9423, 1: 0.0577},
    "system_MySQL": {0: 0.8269, 1: 0.1731},
    "system_Other": {0: 0.9231, 1: 0.0769},
    "system_Outlook": {0: 0.1731, 1: 0.8269},
    "system_SharePoint": {0: 0.9231, 1: 0.0385, 2: 0.0385},
    "development_complexity": {1: 0.0192, 2: 0.2692, 3: 0.3846, 4: 0.25, 5: 0.0769},
    "process_complexity": {1: 0.0769, 2: 0.5385, 3: 0.3077, 4: 0.0577, 5: 0.0192},
    "pain_points": {1: 0.0962, 2: 0.2308, 3: 0.4423, 4: 0.2308},
    "project_based_in_rules_": {3: 0.0385, 4: 0.2308, 5: 0.7308},
    "structure_of_input_data": {3: 0.2115, 4: 0.1538, 5: 0.6346},
    "knowledge_of_the_business_process": {1: 0.0769, 2: 0.25, 3: 0.3269, 4: 0.3462},
    "documentation": {0: 0.8077, 1: 0.1923},
    "frequence_grouped": {1: 0.8077, 2: 0.0769, 3: 0.1154},
    "frequence_on_demand": {0: 0.7692, 1: 0.2308},
    "process_area_ordinal": {1: 0.1346, 2: 0.0385, 3: 0.5, 4: 0.0385, 5: 0.2885},
    "business_area_Other": {0: 0.9615, 1: 0.0385},
    "business_area_Sourcing": {0: 0.8077, 1: 0.1923},
    "developed_by_CD": {0: 0.9423, 1: 0.0577},
}


def sample_column(rng, spec, n):
    """Sample n values from a {value: probability} marginal."""
    values = list(spec.keys())
    probs = np.array(list(spec.values()), dtype=float)
    probs = probs / probs.sum()  # renormalise (rounding safety)
    return rng.choice(values, size=n, p=probs)


def make_target(rng, df):
    """
    Invented, noisy target for the demo: longer duration tends to go with
    higher development complexity, more pain points and higher process
    complexity. NOT a real finding of the study.
    """
    score = (
        1.1 * df["development_complexity"]
        + 0.7 * df["pain_points"]
        + 0.6 * df["process_complexity"]
        + rng.normal(0, 1.2, size=len(df))
    )
    # Map the continuous score to 5 ordinal classes by quantiles.
    ranks = score.rank(method="first")
    classes = pd.qcut(ranks, q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    return classes


def main():
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_STATE)

    data = {"process_id": range(1, N_ROWS + 1)}
    for col, spec in MARGINALS.items():
        data[col] = sample_column(rng, spec, N_ROWS)

    df = pd.DataFrame(data)
    df["project_duration_ordinal"] = make_target(rng, df)

    # Enforce the exact column order of the real V9.
    df = df[COLUMN_ORDER]

    df.to_excel(PATH_V9, index=False)
    print(f"Synthetic V9 written to: {PATH_V9}")
    print(f"Shape: {df.shape}")
    print("\nTarget class distribution:")
    print(df["project_duration_ordinal"].value_counts().sort_index())
    print("\nReminder: this file is SYNTHETIC. Commit it to the repo for the "
          "public demo; keep the real V9 local and never commit it.")


if __name__ == "__main__":
    main()