from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter_rate_intelligence.features import feature_columns
from spotter_rate_intelligence.metrics import regression_metrics
from spotter_rate_intelligence.model import RateModelBundle


def drift_report(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    numeric = [
        "distance", "weight", "market_index", "quote_signal",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    ]
    for column in numeric:
        tr = pd.to_numeric(train[column], errors="coerce")
        va = pd.to_numeric(validation[column], errors="coerce")
        tr_std = float(tr.std())
        rows.append({
            "feature": column,
            "type": "numeric",
            "train_mean": float(tr.mean()),
            "validation_mean": float(va.mean()),
            "standardized_mean_shift": float((va.mean() - tr.mean()) / tr_std) if tr_std > 0 else 0.0,
            "train_missing_pct": float(tr.isna().mean() * 100),
            "validation_missing_pct": float(va.isna().mean() * 100),
        })

    tr_eq = train["equipment"].value_counts(normalize=True)
    va_eq = validation["equipment"].value_counts(normalize=True)
    levels = sorted(set(tr_eq.index) | set(va_eq.index))
    tvd = 0.5 * sum(abs(float(tr_eq.get(level, 0)) - float(va_eq.get(level, 0))) for level in levels)
    rows.append({
        "feature": "equipment",
        "type": "categorical",
        "train_mean": np.nan,
        "validation_mean": np.nan,
        "standardized_mean_shift": float(tvd),
        "train_missing_pct": float(train["equipment"].isna().mean() * 100),
        "validation_missing_pct": float(validation["equipment"].isna().mean() * 100),
    })
    return pd.DataFrame(rows)


def error_slices(oof: pd.DataFrame) -> pd.DataFrame:
    frame = oof.copy()
    frame["distance_band"] = pd.cut(
        frame["distance"],
        bins=[-np.inf, 500, 1000, 2000, np.inf],
        labels=["<=500", "501-1000", "1001-2000", ">2000"],
    )
    rate_cut = frame["actual_rate"].quantile(0.99)
    frame["target_segment"] = np.where(frame["actual_rate"] >= rate_cut, "top_1pct_rate", "regular_99pct")

    rows: list[dict] = []
    for slice_type, column in [
        ("fold", "fold"),
        ("equipment", "equipment"),
        ("distance_band", "distance_band"),
        ("target_segment", "target_segment"),
    ]:
        for value, group in frame.groupby(column, observed=True):
            metrics = regression_metrics(group["actual_rate"], group["ensemble_prediction"])
            rows.append({
                "slice_type": slice_type,
                "slice_value": str(value),
                "rows": int(len(group)),
                **metrics,
            })
    return pd.DataFrame(rows)


def feature_importance(model: RateModelBundle) -> pd.DataFrame:
    columns, _ = feature_columns(model.include_market_signals)
    lgb_gain = np.asarray(model.lgb_estimator.booster_.feature_importance(importance_type="gain"), dtype=float)
    lgb_gain = lgb_gain / lgb_gain.sum() if lgb_gain.sum() else lgb_gain
    cat = np.asarray(model.cat_estimator.get_feature_importance(), dtype=float)
    cat = cat / cat.sum() if cat.sum() else cat
    result = pd.DataFrame({
        "feature": columns,
        "lightgbm_gain_share": lgb_gain,
        "catboost_importance_share": cat,
    })
    result["mean_importance_share"] = result[["lightgbm_gain_share", "catboost_importance_share"]].mean(axis=1)
    return result.sort_values("mean_importance_share", ascending=False).reset_index(drop=True)


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train_test.csv")
    validation = pd.read_csv(ROOT / "data" / "validation.csv")
    oof = pd.read_csv(ROOT / "reports" / "core_temporal_oof_predictions.csv")
    model = RateModelBundle.load(ROOT / "artifacts" / "champion_model.joblib")

    drift_report(train, validation).to_csv(ROOT / "reports" / "drift_report.csv", index=False)
    error_slices(oof).to_csv(ROOT / "reports" / "error_slices.csv", index=False)
    feature_importance(model).to_csv(ROOT / "reports" / "feature_importance.csv", index=False)
    print("Wrote drift_report.csv, error_slices.csv, and feature_importance.csv")


if __name__ == "__main__":
    main()
