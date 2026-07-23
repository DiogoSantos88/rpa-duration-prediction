"""
Predictive modelling on V9 (dissertation §3.6 and Chapter 4).

Models:
  - Random Forest (multiclass, non-ordinal)
  - Multinomial Logistic Regression (non-ordinal)
  - SVM with RBF kernel (non-ordinal)
  - Ordinal Logistic Regression (proportional odds; own implementation
    in ordinal_model.py — see that module's docstring)

Metrics: accuracy, weighted F1, class-distance MAE, Quadratic Weighted Kappa.
Stratified cross-validation with k=3 (shared in evaluation.py).

Outputs:
  - Comparative summary CSV + feature-importance CSV (RF)
  - Aggregated confusion matrices, metric comparison and feature
    importance charts in outputs/figures/modeling/

Note: figure labels/titles and CSV headers are intentionally in Portuguese
so that generated artefacts match the dissertation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from config import (
    FIG_MODELING,
    MODEL_COLORS,
    PATH_V9,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET,
    apply_plot_style,
    ensure_dirs,
)
from evaluation import evaluate_model_cv
from ordinal_model import OrdinalLogisticRegression

ensure_dirs()
apply_plot_style()

SUMMARY_CSV = TABLES_DIR / "modelos_resumo.csv"
IMPORTANCE_CSV = TABLES_DIR / "importancia_rf.csv"

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
df = pd.read_excel(PATH_V9)
X_cols = [c for c in df.columns if c not in ["process_id", TARGET]]
X = df[X_cols].values.astype(float)
y = df[TARGET].values.astype(int)

print(f"Dataset: {df.shape}")
print(f"Features: {len(X_cols)}")
print(f"Target classes: {sorted(np.unique(y))}")
print(f"Class distribution:\n{pd.Series(y).value_counts().sort_index()}")

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
models_config = {
    "Random Forest": dict(
        factory=lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        scale=False,
    ),
    "Logistic Regression": dict(
        factory=lambda: LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        scale=True,
    ),
    "SVM (RBF)": dict(
        factory=lambda: SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        scale=True,
    ),
    "Logistic Regression Ordinal": dict(
        factory=lambda: OrdinalLogisticRegression(max_iter=300),
        scale=True,
    ),
}

# ---------------------------------------------------------------------
# Cross-validation evaluation
# ---------------------------------------------------------------------
all_results = {}

print("\n" + "=" * 70)
print("STRATIFIED CROSS-VALIDATION EVALUATION (k=3)")
print("=" * 70)

for name, cfg in models_config.items():
    print(f"\n--- {name} ---")
    fold_df, y_true_all, y_pred_all, fitted = evaluate_model_cv(
        cfg["factory"], X, y, scale=cfg["scale"])
    all_results[name] = {
        "folds": fold_df,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "fitted": fitted,
    }
    summary = fold_df[["accuracy", "f1_weighted", "mae_classes",
                       "qwk"]].agg(["mean", "std"])
    print(summary.round(3).to_string())

# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPARATIVE SUMMARY")
print("=" * 70)

summary_rows = []
for name, res in all_results.items():
    f = res["folds"]
    summary_rows.append({
        "Modelo": name,
        "Accuracy (média)": f["accuracy"].mean(),
        "Accuracy (dp)": f["accuracy"].std(),
        "F1 ponderado (média)": f["f1_weighted"].mean(),
        "F1 ponderado (dp)": f["f1_weighted"].std(),
        "MAE classes (média)": f["mae_classes"].mean(),
        "MAE classes (dp)": f["mae_classes"].std(),
        "QWK (média)": f["qwk"].mean(),
        "QWK (dp)": f["qwk"].std(),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.round(3).to_string(index=False))
summary_df.to_csv(SUMMARY_CSV, index=False)

# ---------------------------------------------------------------------
# Aggregated confusion matrices
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("CONFUSION MATRICES (aggregated over all folds)")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()
classes_unique = sorted(np.unique(y))

for ax, (name, res) in zip(axes, all_results.items()):
    cm = confusion_matrix(res["y_true"], res["y_pred"],
                          labels=classes_unique)
    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(classes_unique)))
    ax.set_yticks(range(len(classes_unique)))
    ax.set_xticklabels(classes_unique)
    ax.set_yticklabels(classes_unique)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(name)
    cm_max = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm_max / 2 else "black",
                    fontsize=10)
    print(f"\n{name}:")
    print(pd.DataFrame(cm, index=classes_unique, columns=classes_unique))

fig.suptitle("Matrizes de Confusão (agregadas, k=3 folds)", fontsize=13,
             weight="bold")
fig.tight_layout()
fig.savefig(FIG_MODELING / "matrizes_confusao.png")
plt.close(fig)
print(f"\nChart saved: {FIG_MODELING / 'matrizes_confusao.png'}")

# ---------------------------------------------------------------------
# Visual metric comparison
# ---------------------------------------------------------------------
metrics_to_plot = [
    ("accuracy", "Accuracy", "maior é melhor"),
    ("f1_weighted", "F1 ponderado", "maior é melhor"),
    ("mae_classes", "MAE em classes", "menor é melhor"),
    ("qwk", "Quadratic Weighted Kappa", "maior é melhor"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()
model_names = list(all_results.keys())

for ax, (metric, title, direction) in zip(axes, metrics_to_plot):
    means = [all_results[m]["folds"][metric].mean() for m in model_names]
    stds = [all_results[m]["folds"][metric].std() for m in model_names]
    ax.bar(range(len(model_names)), means, yerr=stds, color=MODEL_COLORS,
           edgecolor="black", linewidth=0.5, capsize=5,
           error_kw={"linewidth": 1})
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in model_names],
                       fontsize=8)
    ax.set_ylabel(title)
    ax.set_title(f"{title} ({direction})", fontsize=10)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02 * max(abs(min(means)), 1), f"{m:.3f}",
                ha="center", fontsize=8)

fig.suptitle("Comparação de modelos por métrica (média ± dp sobre 3 folds)",
             fontsize=12, weight="bold")
fig.tight_layout()
fig.savefig(FIG_MODELING / "comparacao_metricas.png")
plt.close(fig)
print(f"Chart saved: {FIG_MODELING / 'comparacao_metricas.png'}")

# ---------------------------------------------------------------------
# Feature importance (Random Forest, aggregated over folds)
# ---------------------------------------------------------------------
rf_importances = np.zeros(len(X_cols))
for m in all_results["Random Forest"]["fitted"]:
    rf_importances += m.feature_importances_
rf_importances /= len(all_results["Random Forest"]["fitted"])

imp_df = pd.DataFrame({"feature": X_cols, "importance": rf_importances})
imp_df = imp_df.sort_values("importance", ascending=False)

print("\n" + "=" * 70)
print("TOP 10 FEATURES — RANDOM FOREST IMPORTANCE")
print("=" * 70)
print(imp_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 7))
top = imp_df.head(15).iloc[::-1]
ax.barh(top["feature"].str.replace("_", " "), top["importance"],
        color="#2E5C8A", edgecolor="black", linewidth=0.4)
ax.set_xlabel("Importância média (Random Forest, agregada sobre 3 folds)")
ax.set_title("Top 15 features por importância no Random Forest", fontsize=11)
for i, v in enumerate(top["importance"]):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_MODELING / "importancia_rf.png")
plt.close(fig)
print(f"\nChart saved: {FIG_MODELING / 'importancia_rf.png'}")

imp_df.to_csv(IMPORTANCE_CSV, index=False)

print("\n" + "=" * 70)
print("MODELLING COMPLETE")
print("=" * 70)
