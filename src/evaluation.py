"""
Shared model evaluation: metrics and stratified cross-validation.

Consolidates code that was previously duplicated between the modelling
script and the extra-figures script, guaranteeing that both use exactly
the same procedure (same folds, same imputation, same scaling).
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import N_SPLITS, RANDOM_STATE


def compute_metrics(y_true, y_pred):
    """Accuracy, weighted F1, class-distance MAE and Quadratic Weighted Kappa."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0),
        "mae_classes": mean_absolute_error(y_true, y_pred),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def evaluate_model_cv(model_factory, X, y, n_splits=N_SPLITS, scale=False,
                      random_state=RANDOM_STATE):
    """
    Evaluate a model with stratified cross-validation.

    Parameters
    ----------
    model_factory : callable
        Zero-argument function returning a fresh sklearn-like estimator.
    X, y : arrays
        Features and target variable.
    scale : bool
        If True, apply StandardScaler (fitted on each fold's training data
        only, preventing information leakage).

    Returns
    -------
    (fold_df, y_true_all, y_pred_all, fitted_models)
        fold_df: DataFrame with per-fold metrics;
        y_true_all / y_pred_all: predictions aggregated over all folds;
        fitted_models: list with the model fitted on each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    fitted = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Median imputation (fitted on training data) — defensive: V9 has no
        # NaNs, but this keeps the pipeline robust to future changes.
        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = compute_metrics(y_test, y_pred)
        m["fold"] = fold_idx
        fold_metrics.append(m)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        fitted.append(model)

    fold_df = pd.DataFrame(fold_metrics)
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    return fold_df, y_true_all, y_pred_all, fitted
