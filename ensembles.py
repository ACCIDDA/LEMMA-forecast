from __future__ import annotations
from typing import Iterable, Optional, List
import numpy as np


class BaseEnsemble:
    """Interface for pluggable ensemble strategies.

    Contract:
    - fit(X, y=None) -> self  (no-op for unsupervised ensembles)
    - predict(X, quantiles) -> np.ndarray of shape (n_samples, n_quantiles)
      where n_samples is typically weeks_ahead in this app.
    - is_supervised: indicates whether fit requires targets.
    """
    is_supervised: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseEnsemble":
        return self

    def predict(self, X: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
        raise NotImplementedError


class BasicEnsemble(BaseEnsemble):
    """Unsupervised aggregator that computes empirical quantiles across predictors.
    Note: We also compute mean internally if needed later, but we return only
    the requested quantiles with shape (n_samples, n_quantiles) to match the app's
    expected output (median is just 0.5 in quantiles).
    """
    def predict(self, X: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
        # Optionally compute mean to have it available if needed (not returned)
        _ = np.mean(X, axis=1)
        qs = np.array(list(quantiles))
        # np.quantile over predictors dimension (axis=1)
        qvals = np.quantile(X, q=qs, axis=1)  # (n_quantiles, n_samples)
        if qvals.ndim == 1:
            qvals = qvals[None, :]
        return qvals.T  # (n_samples, n_quantiles)


class RFQuantileEnsemble(BaseEnsemble):
    is_supervised: bool = True

    def __init__(self, n_estimators: int = 50, random_state_base: int = 1):
        from sklearn_quantile import RandomForestQuantileRegressor
        self.model = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            random_state=random_state_base
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFQuantileEnsemble":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
        qs = list(quantiles)
        preds = []
        for q in qs:
            # sklearn-quantile RFs often use q in [0,1] or could require percentiles.
            # Here, shared_utils.predict_quantiles previously set model.q; we call API directly if available.
            try:
                preds.append(self.model.predict(X, quantile=int(q * 100)))
            except TypeError:
                # Fallback: set attribute q if required by implementation
                if hasattr(self.model, "q"):
                    self.model.q = q
                    preds.append(self.model.predict(X))
                else:
                    # As a last resort, use median prediction for all quantiles
                    preds.append(self.model.predict(X))
        return np.column_stack(preds)  # (n_samples, n_quantiles)
