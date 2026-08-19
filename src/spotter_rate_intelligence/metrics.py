from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y - p)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "wape_pct": float(abs_error.sum() / np.abs(y).sum() * 100.0),
        "mape_pct": float(np.mean(abs_error / np.clip(np.abs(y), 1e-9, None)) * 100.0),
        "r2": float(r2_score(y, p)),
        "median_ae": float(np.median(abs_error)),
        "p95_ae": float(np.quantile(abs_error, 0.95)),
    }
