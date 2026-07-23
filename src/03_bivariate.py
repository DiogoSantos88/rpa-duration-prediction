"""
Bivariate analysis of V8 against the target variable (dissertation §3.5).

For each feature:
  - Pearson chi-squared test (G-test as a robust alternative for sparse
    2xK tables)
  - Directional association measure suited to the feature type:
      * ordinal feature vs ordinal target -> Kendall tau-b +
        Goodman-Kruskal gamma
      * binary feature vs ordinal target -> Mann-Whitney U + rank-biserial
  - FDR correction (Benjamini-Hochberg) over the p-values

Outputs:
  - Consolidated LaTeX table (outputs/tables/07_bivariada.tex)
  - CSV with the full results
  - Association-strength summary chart
  - Boxplots/heatmaps for the significant variables

Note: figure labels/titles and LaTeX captions are intentionally in
Portuguese so that generated artefacts match the dissertation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from config import (
    ALPHA,
    FIG_BIVARIATE,
    PATH_V8,
    TABLES_DIR,
    TARGET,
    apply_plot_style,
    ensure_dirs,
)
from latex_utils import fmt_p, fmt_val, tex_escape

ensure_dirs()
apply_plot_style()

RESULTS_CSV = TABLES_DIR / "bivariada_resultados.csv"


# ---------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------
def benjamini_hochberg(pvals, alpha=ALPHA):
    """
    Benjamini-Hochberg (FDR) correction.
    Returns (rejected, p_adjusted), statsmodels-compatible interface.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    p_adjusted = np.empty(n, dtype=float)
    p_adjusted[order] = np.clip(adj, 0, 1)
    rejected = p_adjusted < alpha
    return rejected, p_adjusted


def goodman_kruskal_gamma(x, y):
    """
    Goodman-Kruskal gamma: directional association between ordinal
    variables, between -1 (perfect inverse) and +1 (perfect direct).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (x[i] - x[j]) * (y[i] - y[j])
            if d > 0:
                concordant += 1
            elif d < 0:
                discordant += 1
    if concordant + discordant == 0:
        return np.nan
    return (concordant - discordant) / (concordant + discordant)


def chi2_or_gtest(feature, y, expected_threshold=5):
    """
    Pearson chi-squared; if >20% of cells have expected frequency below the
    threshold and the table is 2xK, uses the likelihood-ratio test (G-test)
    as a robust alternative for sparse tables.
    """
    contingency = pd.crosstab(feature, y)
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    low_exp_pct = 100 * (expected < expected_threshold).sum() / expected.size

    if (low_exp_pct > 20) and (contingency.shape[0] == 2):
        try:
            res = stats.chi2_contingency(contingency,
                                         lambda_="log-likelihood")
            return {
                "test_used": "G-test (chi2 LR)",
                "statistic": res[0],
                "p_value": res[1],
                "low_expected_pct": low_exp_pct,
                "dof": res[2],
            }
        except Exception:
            pass

    return {
        "test_used": "Chi2 Pearson",
        "statistic": chi2,
        "p_value": p_chi2,
        "low_expected_pct": low_exp_pct,
        "dof": dof,
    }


def mann_whitney_with_effect(group0, group1):
    """Two-sided Mann-Whitney U + rank-biserial (effect size, in [-1, 1])."""
    if len(group0) == 0 or len(group1) == 0:
        return None, None, None
    u, p = stats.mannwhitneyu(group0, group1, alternative="two-sided")
    rb = 1 - (2 * u) / (len(group0) * len(group1))
    return u, p, rb


def classify_strength(value):
    """Qualitative strength classification (thresholds adapted from Rea & Parker, 1992)."""
    a = abs(value) if value is not None and not np.isnan(value) else 0
    if a < 0.10:
        return "Negligenciável"
    if a < 0.20:
        return "Fraca"
    if a < 0.40:
        return "Moderada"
    if a < 0.60:
        return "Forte"
    return "Muito forte"


# ---------------------------------------------------------------------
# Load data and classify features
# ---------------------------------------------------------------------
df = pd.read_excel(PATH_V8)
y = df[TARGET]

ordinal_features = [
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
binary_features = (
    ["documentation", "frequence_on_demand", "developed_by_CD"]
    + sorted(c for c in df.columns if c.startswith("infra_"))
    + sorted(c for c in df.columns if c.startswith("system_"))
    + sorted(c for c in df.columns if c.startswith("business_area_"))
)
all_features = ordinal_features + binary_features

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
results = []
for var in all_features:
    x = df[var]
    record = {
        "variavel": var,
        "tipo": "ordinal" if var in ordinal_features else "binaria",
    }

    chi_res = chi2_or_gtest(x, y)
    record.update({
        "chi2_test": chi_res["test_used"],
        "chi2_stat": chi_res["statistic"],
        "chi2_p": chi_res["p_value"],
        "chi2_low_exp_pct": chi_res["low_expected_pct"],
    })

    if var in ordinal_features:
        tau, p_tau = stats.kendalltau(x, y)
        record.update({
            "measure_name": "Kendall tau-b",
            "measure_value": tau,
            "measure_p": p_tau,
            "gamma": goodman_kruskal_gamma(x.values, y.values),
            "strength": classify_strength(tau),
        })
    else:
        g0 = y[x == 0]
        g1 = y[x == 1]
        _, p_mw, rb = mann_whitney_with_effect(g0, g1)
        record.update({
            "measure_name": "Mann-Whitney U",
            "measure_value": rb,
            "measure_p": p_mw,
            "gamma": np.nan,
            "strength": classify_strength(rb),
        })

    results.append(record)

results_df = pd.DataFrame(results)

# ---------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------
mask_chi2 = results_df["chi2_p"].notna()
_, p_fdr_chi2 = benjamini_hochberg(results_df.loc[mask_chi2, "chi2_p"].values)
results_df.loc[mask_chi2, "chi2_p_fdr"] = p_fdr_chi2

mask_meas = results_df["measure_p"].notna()
_, p_fdr_meas = benjamini_hochberg(
    results_df.loc[mask_meas, "measure_p"].values)
results_df.loc[mask_meas, "measure_p_fdr"] = p_fdr_meas

results_df["sig_chi2"] = results_df["chi2_p_fdr"] < ALPHA
results_df["sig_measure"] = results_df["measure_p_fdr"] < ALPHA

results_df["abs_measure"] = results_df["measure_value"].abs()
results_df = results_df.sort_values(
    "abs_measure", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------
print("\n=== Results (sorted by association strength) ===\n")
display_cols = [
    "variavel", "tipo", "measure_name", "measure_value", "measure_p_fdr",
    "gamma", "chi2_test", "chi2_p_fdr", "strength", "sig_chi2", "sig_measure",
]
with pd.option_context("display.max_rows", None, "display.width", 200,
                       "display.max_colwidth", 30):
    print(results_df[display_cols].to_string(index=False))

print(f"\nSignificant variables after FDR (chi2):    "
      f"{results_df['sig_chi2'].sum()}/{len(results_df)}")
print(f"Significant variables after FDR (measure): "
      f"{results_df['sig_measure'].sum()}/{len(results_df)}")

# ---------------------------------------------------------------------
# Summary chart: sorted association strength
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 0.28 * len(results_df) + 1.5))


def bar_color(row):
    if row["tipo"] == "ordinal":
        return "#2E5C8A" if row["sig_measure"] else "#A8B8CC"
    return "#3A8D8C" if row["sig_measure"] else "#B8CCCB"


colors = results_df.apply(bar_color, axis=1).tolist()
y_pos = np.arange(len(results_df))[::-1]
ax.barh(y_pos, results_df["measure_value"], color=colors, edgecolor="black",
        linewidth=0.4)

for yp, row in zip(y_pos, results_df.itertuples()):
    sig_marker = " *" if row.sig_measure else ""
    ax.text(row.measure_value + (0.015 if row.measure_value >= 0 else -0.015),
            yp, f"{row.measure_value:.3f}{sig_marker}", va="center",
            ha="left" if row.measure_value >= 0 else "right", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels(results_df["variavel"].str.replace("_", " "), fontsize=8)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("Força de associação (Kendall τ-b para ordinais; "
              "rank-biserial para binárias)")
ax.set_title(f"Associação de cada feature com {TARGET}\n"
             "(barras saturadas = significativas após FDR; * = p_FDR < 0.05)",
             fontsize=10)
ax.set_xlim(-0.85, 0.85)

legend = [
    Patch(facecolor="#2E5C8A", label="Ordinal (significativa)"),
    Patch(facecolor="#A8B8CC", label="Ordinal (não significativa)"),
    Patch(facecolor="#3A8D8C", label="Binária (significativa)"),
    Patch(facecolor="#B8CCCB", label="Binária (não significativa)"),
]
ax.legend(handles=legend, loc="lower right", fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(FIG_BIVARIATE / "forca_associacao_geral.png")
plt.close(fig)
print(f"\nSummary chart saved to "
      f"{FIG_BIVARIATE / 'forca_associacao_geral.png'}")

# ---------------------------------------------------------------------
# Detailed charts for significant variables
# ---------------------------------------------------------------------
sig_vars = results_df[results_df["sig_measure"]].copy()

# Boxplots for significant ordinal features
sig_ordinal = sig_vars[sig_vars["tipo"] == "ordinal"]["variavel"].tolist()
if sig_ordinal:
    n = len(sig_ordinal)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    rng = np.random.default_rng(0)  # reproducible jitter
    for ax, var in zip(axes, sig_ordinal):
        data_by_class = [df.loc[y == k, var].values for k in sorted(y.unique())]
        ax.boxplot(data_by_class, tick_labels=sorted(y.unique()), patch_artist=True,
                   boxprops=dict(facecolor="#2E5C8A", alpha=0.6),
                   medianprops=dict(color="red", linewidth=1.5))
        for k_idx, k in enumerate(sorted(y.unique()), start=1):
            vals = df.loc[y == k, var].values
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full_like(vals, k_idx, dtype=float) + jitter, vals,
                       alpha=0.5, s=15, color="black")
        tau_val = sig_vars.loc[sig_vars["variavel"] == var,
                               "measure_value"].iloc[0]
        ax.set_xlabel(f"{TARGET} (classe)")
        ax.set_ylabel(var)
        ax.set_title(f"{var}\n(τ-b = {tau_val:.3f})")
    for ax in axes[len(sig_ordinal):]:
        ax.axis("off")
    fig.suptitle("Boxplots: variáveis ordinais com associação significativa",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_BIVARIATE / "boxplots_ordinais_significativas.png")
    plt.close(fig)
    print(f"Boxplots saved: {len(sig_ordinal)} significant ordinal "
          "variables")

# Heatmaps for significant binary features
sig_binary = sig_vars[sig_vars["tipo"] == "binaria"]["variavel"].tolist()
if sig_binary:
    n = len(sig_binary)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, var in zip(axes, sig_binary):
        ct = pd.crosstab(df[var], y, normalize="index") * 100
        ax.imshow(ct.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(ct.columns)))
        ax.set_xticklabels(ct.columns)
        ax.set_yticks(range(len(ct.index)))
        ax.set_yticklabels(ct.index)
        ax.set_xlabel(TARGET)
        ax.set_ylabel(var)
        rb = sig_vars.loc[sig_vars["variavel"] == var,
                          "measure_value"].iloc[0]
        ax.set_title(f"{var}\n(rank-biserial = {rb:.3f})")
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                ax.text(j, i, f"{ct.values[i, j]:.0f}%", ha="center",
                        va="center",
                        color="white" if ct.values[i, j] > 50 else "black",
                        fontsize=9)
    for ax in axes[len(sig_binary):]:
        ax.axis("off")
    fig.suptitle("Heatmaps: variáveis binárias com associação significativa\n"
                 "(% = distribuição da target condicionada à feature)",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_BIVARIATE / "heatmaps_binarias_significativas.png")
    plt.close(fig)
    print(f"Heatmaps saved: {len(sig_binary)} significant binary "
          "variables")

# ---------------------------------------------------------------------
# Consolidated LaTeX table
# ---------------------------------------------------------------------
lines = [
    r"\begin{table}[h]",
    r"\centering",
    r"\small",
    r"\caption{Análise bivariada: cada feature contra "
    r"\texttt{project\_duration\_ordinal}. "
    r"Para variáveis ordinais reporta-se Kendall $\tau$-b e $\gamma$ de "
    r"Goodman-Kruskal; para binárias reporta-se rank-biserial (effect size "
    r"do Mann-Whitney U). Valores-p ajustados pelo método de "
    r"Benjamini-Hochberg (FDR).}",
    r"\label{tab:bivariada}",
    r"\begin{tabular}{llrrrrrl}",
    r"\toprule",
    r"Variável & Tipo & Medida & $p_{\text{FDR}}$ & $\gamma$ & "
    r"$\chi^2$ $p_{\text{FDR}}$ & Sig.\ & Força \\",
    r"\midrule",
]
for _, row in results_df.iterrows():
    sig_str = "$\\ast$" if row["sig_measure"] else ""
    lines.append(
        f"{tex_escape(row['variavel'])} & "
        f"{row['tipo']} & "
        f"{fmt_val(row['measure_value'])} & "
        f"{fmt_p(row['measure_p_fdr'])} & "
        f"{fmt_val(row['gamma'])} & "
        f"{fmt_p(row['chi2_p_fdr'])} & "
        f"{sig_str} & "
        f"{row['strength']} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex_path = TABLES_DIR / "07_bivariada.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"LaTeX table saved to {tex_path}")

results_df.to_csv(RESULTS_CSV, index=False)
print(f"Full results saved to {RESULTS_CSV}")
