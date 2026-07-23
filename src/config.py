"""
Central project configuration.

All paths are relative to the repository root, so the pipeline runs on any
machine after cloning, provided 'Original.xlsx' is placed in data/raw/
(the file is not included in the repository for confidentiality reasons —
see README).
"""

from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"

PATH_ORIGINAL = DATA_RAW / "Original.xlsx"

# Intermediate database versions (traceability of the preprocessing steps)
DB_VERSIONS = {
    v: DATA_PROCESSED / f"DB_RPA_Projects_V{v}.xlsx" for v in range(2, 10)
}
PATH_V8 = DB_VERSIONS[8]
PATH_V9 = DB_VERSIONS[9]

# Figure subfolders per pipeline stage
FIG_UNIVARIATE = FIGURES_DIR / "univariate"
FIG_BIVARIATE = FIGURES_DIR / "bivariate"
FIG_MODELING = FIGURES_DIR / "modeling"


def ensure_dirs():
    """Create the data and output folders if they do not exist yet."""
    for d in (DATA_RAW, DATA_PROCESSED, FIG_UNIVARIATE, FIG_BIVARIATE,
              FIG_MODELING, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Study constants
# ---------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 3          # stratified cross-validation (k=3, cf. dissertation §3.6)
ALPHA = 0.05          # significance level (with FDR correction)

TARGET = "project_duration_ordinal"

# ---------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------
COLOR_ORDINAL = "#2E5C8A"
COLOR_BINARY = "#3A8D8C"
COLOR_TARGET = "#C97B0F"
COLOR_FAMILY = {
    "infra": "#5B7FB0",
    "system": "#7BA05B",
    "business_area": "#A06B5B",
    "developed_by": "#8A5B7F",
}
MODEL_COLORS = ["#2E5C8A", "#3A8D8C", "#C97B0F", "#7BA05B"]

# ---------------------------------------------------------------------
# Matplotlib style shared by every script
# ---------------------------------------------------------------------
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_plot_style():
    """Apply the common plotting style (serif fonts, 300 DPI, subtle grid)."""
    plt.rcParams.update(PLOT_STYLE)
