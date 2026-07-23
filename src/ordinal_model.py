"""
Ordinal Logistic Regression (proportional odds / cumulative link model).

Direct implementation of McCullagh's (1980) proportional odds model,
estimated by maximum likelihood with scipy's L-BFGS-B optimiser. This is
the exact formulation described in the dissertation (§2.3.6, Eqs. 2.5-2.6):

    P(Y <= k | x) = sigmoid(theta_k - beta @ x)

with a single coefficient vector beta shared across classes and K-1
ordered cutpoints theta_1 < ... < theta_{K-1} (ordering enforced by
parameterising the increments on the log scale).

Note: this is *not* the same model as mord's LogisticAT ("All-Threshold"
loss), which yields different results on this dataset. This implementation
is the one that produced the results reported in the dissertation.
"""

import numpy as np
from scipy.optimize import minimize


class OrdinalLogisticRegression:
    """Proportional-odds ordinal logistic regression (sklearn-like API)."""

    def __init__(self, max_iter=300, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _sigmoid(z):
        # Numerically stable sigmoid: clip the argument so exp() never
        # overflows. np.where would evaluate both branches (triggering
        # overflow warnings), so we clip instead.
        z = np.clip(z, -709, 709)  # exp is finite within this range (float64)
        return 1.0 / (1.0 + np.exp(-z))

    def _neg_log_likelihood(self, params, X, y, n_classes):
        n_features = X.shape[1]
        beta = params[:n_features]
        theta_1 = params[n_features]
        deltas = np.exp(params[n_features + 1:])
        cutoffs = np.concatenate([[theta_1], theta_1 + np.cumsum(deltas)])
        linear = X @ beta
        ll = 0.0
        eps = 1e-12
        for k in range(n_classes):
            mask = (y == k)
            if not np.any(mask):
                continue
            if k == 0:
                p = self._sigmoid(cutoffs[0] - linear[mask])
            elif k == n_classes - 1:
                p = 1 - self._sigmoid(cutoffs[-1] - linear[mask])
            else:
                p = (self._sigmoid(cutoffs[k] - linear[mask])
                     - self._sigmoid(cutoffs[k - 1] - linear[mask]))
            ll += np.sum(np.log(np.clip(p, eps, 1.0)))
        return -ll

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])
        n_features = X.shape[1]
        init = np.concatenate([np.zeros(n_features), [-1.0],
                               np.zeros(self.n_classes_ - 2)])
        result = minimize(
            self._neg_log_likelihood, init,
            args=(X, y_idx, self.n_classes_),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        self.beta_ = result.x[:n_features]
        theta_1 = result.x[n_features]
        deltas = np.exp(result.x[n_features + 1:])
        self.cutoffs_ = np.concatenate([[theta_1],
                                        theta_1 + np.cumsum(deltas)])
        return self

    def predict_proba(self, X):
        """Class probability matrix, one row per observation."""
        X = np.asarray(X, dtype=float)
        linear = X @ self.beta_
        probs = np.zeros((X.shape[0], self.n_classes_))
        for k in range(self.n_classes_):
            if k == 0:
                probs[:, k] = self._sigmoid(self.cutoffs_[0] - linear)
            elif k == self.n_classes_ - 1:
                probs[:, k] = 1 - self._sigmoid(self.cutoffs_[-1] - linear)
            else:
                probs[:, k] = (self._sigmoid(self.cutoffs_[k] - linear)
                               - self._sigmoid(self.cutoffs_[k - 1] - linear))
        probs = np.clip(probs, 0, 1)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
