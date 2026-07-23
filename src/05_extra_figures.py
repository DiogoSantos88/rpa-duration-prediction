"""
Complementary charts for Chapter 4:

  1. Strip plot — per-fold dispersion of each metric (stability)
  2. Error histogram — distribution of |y_pred - y_true| per model

Uses exactly the same models, folds and metrics as 04_modeling.py
(via evaluation.py and ordinal_model.py), guaranteeing full consistency
with Table 4 of the dissertation.

Note: figure labels/titles are intentionally in Portuguese so that
generated artefacts match the dissertation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from config import (
    FIG_MODELING,
    MODEL_COLORS,
    PATH_V9,
    RANDOM_STATE,
    TARGET,
    apply_plot_style,
    ensure_dirs,
)
from evaluation import evaluate_model_cv
from ordinal_model import OrdinalLogisticRegression

ensure_dirs()
apply_plot_style()

# ---------------------------------------------------------------------
# Data and models (identical to 04_modeling.py)
# ---------------------------------------------------------------------
df = pd.read_excel(PATH_V9)
X_cols = [c for c in df.columns if c not in ["process_id", TARGET]]
X = df[X_cols].values.astype(float)
y = df[TARGET].values.astype(int)
n_total = len(y)

models = [
    ("Random Forest",
     lambda: RandomForestClassifier(
         n_estimators=200, max_depth=5, min_samples_leaf=2,
         class_weight="balanced", random_state=RANDOM_STATE),
     False),
    ("Logistic Regression",
     lambda: LogisticRegression(
         solver="lbfgs", class_weight="balanced", max_iter=2000,
         random_state=RANDOM_STATE),
     True),
    ("SVM (RBF)",
     lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced",
                 random_state=RANDOM_STATE),
     True),
    ("LR Ordinal",
     lambda: OrdinalLogisticRegression(max_iter=300),
     True),
]

all_folds = []
preds_by_model = {}
for name, factory, scale in models:
    print(f"  {name}...")
    fold_df, yt, yp, _ = evaluate_model_cv(factory, X, y, scale=scale)
    fold_df["model"] = name
    all_folds.append(fold_df)
    preds_by_model[name] = (yt, yp)

folds_df = pd.concat(all_folds, ignore_index=True)
print("\nFold dataframe:")
print(folds_df.to_string())

model_names = [name for name, _, _ in models]

# ---------------------------------------------------------------------
# CHART 1 — Strip plot of per-fold dispersion
# ---------------------------------------------------------------------
metrics_info = [
    ("accuracy", "Accuracy", "maior é melhor"),
    ("f1_weighted", "F1 ponderado", "maior é melhor"),
    ("mae_classes", "MAE em classes", "menor é melhor"),
    ("qwk", "Quadratic Weighted Kappa", "maior é melhor"),
]

rng = np.random.default_rng(RANDOM_STATE)  # reproducible jitter
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for ax, (metric_key, metric_label, direction) in zip(axes, metrics_info):
    for i, mname in enumerate(model_names):
        vals = folds_df[folds_df["model"] == mname][metric_key].values
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter([i] * len(vals) + jitter, vals, color=MODEL_COLORS[i],
                   s=70, edgecolor="black", linewidth=0.6, zorder=3,
                   alpha=0.85)
        ax.hlines(vals.mean(), i - 0.18, i + 0.18, color="black",
                  linewidth=2, zorder=2)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in model_names],
                       fontsize=8)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} ({direction})", fontsize=10)

fig.suptitle("Dispersão das métricas entre folds — cada ponto representa "
             "um fold", fontsize=12, weight="bold")
fig.tight_layout()
out1 = FIG_MODELING / "dispersao_folds.png"
fig.savefig(out1)
plt.close(fig)
print(f"\nSaved: {out1}")

# ---------------------------------------------------------------------
# CHART 2 — Histogram of |y_pred - y_true| errors per model
# ---------------------------------------------------------------------
all_errors = {m: np.abs(preds_by_model[m][1] - preds_by_model[m][0])
              for m in model_names}
max_err = max(e.max() for e in all_errors.values())
bins = np.arange(0, max_err + 2) - 0.5

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes = axes.flatten()

for ax, mname, color in zip(axes, model_names, MODEL_COLORS):
    err = all_errors[mname]
    counts, bin_edges = np.histogram(err, bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(centers, counts, width=0.75, color=color, edgecolor="black",
           linewidth=0.6, alpha=0.85)
    for c, count in zip(centers, counts):
        if count > 0:
            pct = 100 * count / n_total
            ax.text(c, count + 0.5, f"{count}\n({pct:.0f}%)", ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(range(int(max_err) + 1))
    ax.set_xlabel("Erro absoluto entre classe prevista e real")
    ax.set_ylabel("Número de projetos")
    ax.set_title(mname, fontsize=10)
    ax.set_ylim(0, max(counts) * 1.3)

fig.suptitle("Distribuição dos erros de classificação por modelo",
             fontsize=12, weight="bold")
fig.tight_layout()
out2 = FIG_MODELING / "distribuicao_erros.png"
fig.savefig(out2)
plt.close(fig)
print(f"Saved: {out2}")

# Error summary statistics
print("\nAbsolute error summary:")
for mname in model_names:
    err = all_errors[mname]
    n0 = int(np.sum(err == 0))
    n1 = int(np.sum(err == 1))
    n2plus = int(np.sum(err >= 2))
    print(f"  {mname:25s}  err=0: {n0:3d} ({100*n0/n_total:.0f}%)   "
          f"err=1: {n1:3d} ({100*n1/n_total:.0f}%)   "
          f"err>=2: {n2plus:3d} ({100*n2plus/n_total:.0f}%)")
