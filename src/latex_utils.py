"""
Shared helpers for generating LaTeX (booktabs) tables.

These consolidate functions that were previously duplicated across the
univariate and bivariate analysis scripts.

Note: table *content* produced elsewhere (captions, column headers) is
intentionally kept in Portuguese so that generated tables match the
dissertation.
"""

import pandas as pd


def tex_escape(s):
    """Escape underscores for use in LaTeX."""
    return str(s).replace("_", r"\_")


def fmt_p(p):
    """Format a p-value: '<0.001' below the threshold, 3 decimals above."""
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_val(v, decimals=3):
    """Format a numeric value with fixed decimals ('--' if NaN)."""
    if pd.isna(v):
        return "--"
    return f"{v:.{decimals}f}"


def df_to_latex(df, label, caption):
    """Convert a DataFrame into a booktabs LaTeX table (no tabularx)."""
    n_cols = len(df.columns)
    col_spec = "l" + "r" * (n_cols - 1)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)
