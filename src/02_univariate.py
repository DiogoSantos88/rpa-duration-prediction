"""
Univariate analysis of V8 (dissertation §3.3).

Generates:
  - Individual PNGs per variable (300 DPI) in outputs/figures/univariate/
  - Summary PDF with all charts
  - LaTeX (booktabs) tables with descriptive statistics in outputs/tables/

Note: figure labels/titles and LaTeX captions are intentionally in
Portuguese so that generated artefacts match the dissertation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from config import (
    COLOR_BINARY,
    COLOR_FAMILY,
    COLOR_ORDINAL,
    COLOR_TARGET,
    FIG_UNIVARIATE,
    PATH_V8,
    TABLES_DIR,
    TARGET,
    apply_plot_style,
    ensure_dirs,
)
from latex_utils import df_to_latex, tex_escape

ensure_dirs()
apply_plot_style()

OUT_PDF = FIG_UNIVARIATE / "univariada_resumo.pdf"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df = pd.read_excel(PATH_V8)
n_total = len(df)

ordinal_vars = [
    "development_complexity",
    "process_complexity",
    "pain_points",
    "project_based_in_rules_",
    "digital_level",
    "structure_of_input_data",
    "knowledge_of_the_business_process",
    "frequence_grouped",
    "process_area_ordinal",
]
binary_vars = [
    "documentation",
    "frequence_on_demand",
    "developed_by_CD",
]
infra_dummies = sorted(c for c in df.columns if c.startswith("infra_"))
system_dummies = sorted(c for c in df.columns if c.startswith("system_"))
business_area_dummies = sorted(
    c for c in df.columns if c.startswith("business_area_"))

PRETTY = {
    "development_complexity": "Development Complexity",
    "process_complexity": "Process Complexity",
    "pain_points": "Pain Points",
    "project_based_in_rules_": "Project Based in Rules",
    "digital_level": "Digital Level",
    "structure_of_input_data": "Structure of Input Data",
    "knowledge_of_the_business_process": "Knowledge of the Business Process",
    "frequence_grouped": "Frequency (Grouped)",
    "process_area_ordinal": "Process Area (Ordinal)",
    "project_duration_ordinal": "Project Duration (Target)",
    "documentation": "Documentation",
    "frequence_on_demand": "Frequency: On Demand",
    "developed_by_CD": "Developed by CD",
}

FREQ_LABELS = {1: "Daily", 2: "Weekly", 3: "Monthly"}
DURATION_LABELS = {
    1: "1\n(1-40h)",
    2: "2\n(41-80h)",
    3: "3\n(81-120h)",
    4: "4\n(121-160h)",
    5: "5\n(161-200h)",
}


# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------
def plot_ordinal(ax, series, title, color, custom_labels=None):
    """Bar chart of an ordinal variable with annotated mean and median."""
    counts = series.value_counts().sort_index()
    total = len(series.dropna())
    max_count = counts.max()

    ax.bar(counts.index.astype(int), counts.values, color=color,
           edgecolor="black", linewidth=0.5)

    mean = series.mean()
    median = series.median()
    ax.vlines(mean, 0, max_count * 1.05, color="red", linestyle="--",
              linewidth=1.2, label=f"Média = {mean:.2f}")
    ax.vlines(median, 0, max_count * 1.05, color="black", linestyle=":",
              linewidth=1.2, label=f"Mediana = {median:.1f}")

    for x, c in zip(counts.index, counts.values):
        pct = 100 * c / total
        ax.text(x, c + max_count * 0.03, f"{int(c)}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Nível")
    ax.set_ylabel("Frequência")
    if custom_labels:
        ax.set_xticks(list(custom_labels.keys()))
        ax.set_xticklabels(list(custom_labels.values()), fontsize=9)
    else:
        ax.set_xticks(sorted(counts.index.astype(int).unique()))
    ax.set_ylim(0, max_count * 1.40)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def plot_binary(ax, series, title, color):
    """Bar chart of a 0/1 binary variable."""
    counts = series.value_counts().sort_index()
    total = len(series)
    for v in (0, 1):
        if v not in counts.index:
            counts[v] = 0
    counts = counts.sort_index()
    bars = ax.bar(["0", "1"], counts.values, color=color, edgecolor="black",
                  linewidth=0.5)
    bars[0].set_alpha(0.5)
    for i, c in enumerate(counts.values):
        pct = 100 * c / total
        ax.text(i, c + 0.5, f"{int(c)}\n({pct:.1f}%)", ha="center",
                va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frequência")
    ax.set_ylim(0, max(counts.values) * 1.25)


def plot_dummy_family(ax, df, dummies, title, color):
    """Horizontal bar chart with the counts of a dummy family."""
    sums = df[dummies].sum().sort_values(ascending=True)
    labels = [d.split("_", 1)[1] for d in sums.index]
    bars = ax.barh(labels, sums.values, color=color, edgecolor="black",
                   linewidth=0.5)
    total = len(df)
    for bar, val in zip(bars, sums.values):
        pct = 100 * val / total
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{int(val)} ({pct:.1f}%)", va="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Nº de projetos")
    ax.set_xlim(0, max(sums.values) * 1.20)


def freq_custom_labels(var):
    if var != "frequence_grouped":
        return None
    return {k: f"{k}\n({v})" for k, v in FREQ_LABELS.items()}


# ---------------------------------------------------------------------
# Individual PNGs
# ---------------------------------------------------------------------
print(f"Generating charts in {FIG_UNIVARIATE}...")

# 1) Target
fig, ax = plt.subplots(figsize=(7, 4.5))
plot_ordinal(ax, df[TARGET], PRETTY[TARGET], COLOR_TARGET,
             custom_labels=DURATION_LABELS)
ax.set_xlabel("Classe (intervalo em horas)")
fig.savefig(FIG_UNIVARIATE / f"00_{TARGET}.png")
plt.close(fig)

# 2) Ordinal variables
for var in ordinal_vars:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_ordinal(ax, df[var], PRETTY[var], COLOR_ORDINAL,
                 custom_labels=freq_custom_labels(var))
    fig.savefig(FIG_UNIVARIATE / f"01_{var}.png")
    plt.close(fig)

# 3) Binary variables
for var in binary_vars:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_binary(ax, df[var], PRETTY[var], COLOR_BINARY)
    fig.savefig(FIG_UNIVARIATE / f"02_{var}.png")
    plt.close(fig)

# 4) Dummy families
fig, ax = plt.subplots(figsize=(7, 3.5))
plot_dummy_family(ax, df, infra_dummies, "Infrastructure for Development",
                  COLOR_FAMILY["infra"])
fig.savefig(FIG_UNIVARIATE / "03_infra_family.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5.5))
plot_dummy_family(ax, df, system_dummies, "System Interaction",
                  COLOR_FAMILY["system"])
fig.savefig(FIG_UNIVARIATE / "03_system_family.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3))
plot_dummy_family(ax, df, business_area_dummies, "Business Area",
                  COLOR_FAMILY["business_area"])
fig.savefig(FIG_UNIVARIATE / "03_business_area_family.png")
plt.close(fig)


# ---------------------------------------------------------------------
# Summary PDF
# ---------------------------------------------------------------------
print(f"Generating summary PDF in {OUT_PDF}...")

with PdfPages(OUT_PDF) as pdf:
    # Cover page
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.5, 0.85, "Análise Univariada", ha="center", fontsize=22,
             weight="bold")
    fig.text(0.5, 0.80, "DB_RPA_Projects_V8", ha="center", fontsize=14,
             color="gray")
    fig.text(0.5, 0.74, f"n = {n_total} projetos    |    "
             f"{df.shape[1]} variáveis", ha="center", fontsize=11)
    fig.text(0.1, 0.60, "Conteúdo:", fontsize=12, weight="bold")
    fig.text(0.12, 0.55,
             "1. Target: project_duration_ordinal\n"
             "2. Variáveis ordinais (9)\n"
             "3. Variáveis binárias (3)\n"
             "4. Famílias de dummies (infra, system, business_area)",
             fontsize=11)
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)

    # 1. Target
    fig, ax = plt.subplots(figsize=(8.27, 5))
    plot_ordinal(ax, df[TARGET], PRETTY[TARGET], COLOR_TARGET,
                 custom_labels=DURATION_LABELS)
    ax.set_xlabel("Classe (intervalo em horas)")
    fig.suptitle("1. Variável-alvo", fontsize=13, weight="bold", y=1.02)
    pdf.savefig(fig)
    plt.close(fig)

    # 2. Ordinal variables — 4 per page
    chunks = [ordinal_vars[i:i + 4] for i in range(0, len(ordinal_vars), 4)]
    for ci, chunk in enumerate(chunks):
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 9))
        axes = axes.flatten()
        for ax, var in zip(axes, chunk):
            plot_ordinal(ax, df[var], PRETTY[var], COLOR_ORDINAL,
                         custom_labels=freq_custom_labels(var))
        for ax in axes[len(chunk):]:
            ax.axis("off")
        fig.suptitle(f"2. Variáveis ordinais ({ci*4+1}–{ci*4+len(chunk)})",
                     fontsize=13, weight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # 3. Binary variables
    fig, axes = plt.subplots(1, 3, figsize=(8.27, 4))
    for ax, var in zip(axes, binary_vars):
        plot_binary(ax, df[var], PRETTY[var], COLOR_BINARY)
    fig.suptitle("3. Variáveis binárias", fontsize=13, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # 4. Dummy families
    for suffix, dummies, family, height in (
        ("a", infra_dummies, "infra", 3.5),
        ("b", system_dummies, "system", 6),
        ("c", business_area_dummies, "business_area", 3),
    ):
        fig, ax = plt.subplots(figsize=(8.27, height))
        plot_dummy_family(ax, df, dummies,
                          {"infra": "Infrastructure for Development",
                           "system": "System Interaction",
                           "business_area": "Business Area"}[family],
                          COLOR_FAMILY[family])
        fig.suptitle(f"4{suffix}. Família {family}_", fontsize=13,
                     weight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"  PDF generated: {OUT_PDF}")


# ---------------------------------------------------------------------
# Descriptive statistics tables (LaTeX booktabs)
# ---------------------------------------------------------------------
print(f"Generating LaTeX tables in {TABLES_DIR}...")


def descriptive_table_ordinal(df, vars_list, label, caption):
    rows = []
    for v in vars_list:
        s = df[v]
        rows.append({
            "Variável": tex_escape(v),
            "n": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Min": int(s.min()) if not s.isna().all() else np.nan,
            "Max": int(s.max()) if not s.isna().all() else np.nan,
            "Média": f"{s.mean():.2f}",
            "Mediana": f"{s.median():.1f}",
            "Desvio Padrão": f"{s.std():.2f}",
            "Níveis distintos": int(s.nunique()),
        })
    return df_to_latex(pd.DataFrame(rows), label, caption)


def descriptive_table_binary(df, vars_list, label, caption):
    rows = []
    n = len(df)
    for v in vars_list:
        s = df[v]
        ones = int(s.sum())
        zeros = int((s == 0).sum())
        rows.append({
            "Variável": tex_escape(v),
            "n": int(s.notna().sum()),
            "0": zeros,
            "1": ones,
            "\\% (=1)": f"{100*ones/n:.1f}",
        })
    return df_to_latex(pd.DataFrame(rows), label, caption)


def descriptive_table_dummies(df, dummies, label, caption):
    rows = []
    n = len(df)
    for d in dummies:
        ones = int(df[d].sum())
        rows.append({
            "Categoria": tex_escape(d.split("_", 1)[1]),
            "n (=1)": ones,
            "\\% (=1)": f"{100*ones/n:.1f}",
        })
    out_df = pd.DataFrame(rows).sort_values("n (=1)", ascending=False)
    return df_to_latex(out_df, label, caption)


tables = [
    ("01_target.tex", descriptive_table_ordinal(
        df, [TARGET], "tab:univariada-target",
        "Estatísticas descritivas da variável-alvo.")),
    ("02_ordinais.tex", descriptive_table_ordinal(
        df, ordinal_vars, "tab:univariada-ordinais",
        "Estatísticas descritivas das variáveis ordinais.")),
    ("03_binarias.tex", descriptive_table_binary(
        df, binary_vars, "tab:univariada-binarias",
        "Estatísticas descritivas das variáveis binárias.")),
    ("04_infra.tex", descriptive_table_dummies(
        df, infra_dummies, "tab:univariada-infra",
        "Distribuição da família \\texttt{infra\\_} "
        "(Infrastructure for Development).")),
    ("05_system.tex", descriptive_table_dummies(
        df, system_dummies, "tab:univariada-system",
        "Distribuição da família \\texttt{system\\_} (System Interaction).")),
    ("06_business_area.tex", descriptive_table_dummies(
        df, business_area_dummies, "tab:univariada-business-area",
        "Distribuição da família \\texttt{business\\_area\\_}.")),
]
for filename, tex in tables:
    (TABLES_DIR / filename).write_text(tex, encoding="utf-8")

print(f"  Tables generated in {TABLES_DIR}")
print("\n=== Done ===")
print(f"  PNGs:    {FIG_UNIVARIATE}/")
print(f"  Tables:  {TABLES_DIR}/")
print(f"  PDF:     {OUT_PDF}")
