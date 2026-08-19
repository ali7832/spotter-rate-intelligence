from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_quality import summarize_quality
from .features import build_city_lookup
from .metrics import regression_metrics
from .model import RateModelBundle, train_ensemble_bundle


def temporal_folds() -> list[tuple[str, str, str]]:
    """Forward validation windows that mirror the real future-scoring setup."""
    return [
        ("august", "2025-08-01", "2025-08-31"),
        ("september", "2025-09-01", "2025-09-30"),
        ("october", "2025-10-01", "2025-10-31"),
    ]


def equipment_rpm_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    rpm = train["posted_rate"] / train["distance"]
    medians = train.assign(_rpm=rpm).groupby("equipment")["_rpm"].median()
    global_median = float(rpm.median())
    return test["distance"].to_numpy() * test["equipment"].map(medians).fillna(global_median).to_numpy()


def run_temporal_benchmark(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run leakage-resistant forward backtests and return OOF predictions.

    The output keeps component predictions so the ensemble decision is directly
    evidenced rather than asserted in prose.
    """
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    rows: list[dict] = []
    oof_parts: list[pd.DataFrame] = []

    for fold_name, start, end in temporal_folds():
        train = data[data["date"] < pd.Timestamp(start)].copy()
        test = data[(data["date"] >= pd.Timestamp(start)) & (data["date"] <= pd.Timestamp(end))].copy()

        baseline = equipment_rpm_baseline(train, test)
        rows.append({
            "fold": fold_name,
            "model": "business_baseline_equipment_median_rpm",
            **regression_metrics(test["posted_rate"], baseline),
        })

        lookup = build_city_lookup(train)
        bundle = train_ensemble_bundle(
            train,
            lookup,
            include_market_signals=True,
            metadata={"fold": fold_name, "validation": "forward_temporal"},
        )
        parts = bundle.predict_components(test)
        ensemble = bundle.predict(test)

        rows.append({"fold": fold_name, "model": "lightgbm_l1", **regression_metrics(test["posted_rate"], parts["lightgbm"])})
        rows.append({"fold": fold_name, "model": "catboost_mae", **regression_metrics(test["posted_rate"], parts["catboost"])})
        rows.append({"fold": fold_name, "model": "ensemble_50_50", **regression_metrics(test["posted_rate"], ensemble)})

        oof = pd.DataFrame({
            "load_id": test["load_id"].to_numpy(),
            "date": test["date"].to_numpy(),
            "fold": fold_name,
            "equipment": test["equipment"].to_numpy(),
            "distance": test["distance"].to_numpy(),
            "actual_rate": test["posted_rate"].to_numpy(),
            "lightgbm_prediction": parts["lightgbm"],
            "catboost_prediction": parts["catboost"],
            "ensemble_prediction": ensemble,
        })
        oof["absolute_error"] = (oof["actual_rate"] - oof["ensemble_prediction"]).abs()
        oof_parts.append(oof)

    metrics = pd.DataFrame(rows)
    oof_predictions = pd.concat(oof_parts, ignore_index=True)
    calibration = {
        "p80": float(oof_predictions["absolute_error"].quantile(0.80)),
        "p90": float(oof_predictions["absolute_error"].quantile(0.90)),
        "p95": float(oof_predictions["absolute_error"].quantile(0.95)),
    }
    return metrics, oof_predictions, calibration


def run_cold_start_benchmark(frame: pd.DataFrame) -> dict:
    """Simulate unseen-city traffic on the latest development month.

    Eight cities are removed from the historical categorical universe. Test rows
    still retain their route coordinates, so the benchmark measures whether the
    geographic representation generalizes beyond memorized city/lane IDs.
    """
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    held_out = ["Lexington", "Bakersfield", "Richmond", "Oklahoma City", "Atlanta", "Mobile", "Baton Rouge", "Hartford"]
    before_october = data[data["date"] < pd.Timestamp("2025-10-01")]
    train = before_october[
        ~before_october["pickup"].isin(held_out) & ~before_october["delivery"].isin(held_out)
    ].copy()
    october = data[data["date"] >= pd.Timestamp("2025-10-01")]
    test = october[october["pickup"].isin(held_out) | october["delivery"].isin(held_out)].copy()

    lookup = build_city_lookup(train)
    core = train_ensemble_bundle(train, lookup, include_market_signals=False, metadata={"cold_start": True, "feature_set": "core"})
    full = train_ensemble_bundle(train, lookup, include_market_signals=True, metadata={"cold_start": True, "feature_set": "full"})
    core_pred = core.predict(test)
    full_pred = full.predict(test)
    return {
        "held_out_cities": held_out,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "metrics": {
            "core_champion_ensemble": regression_metrics(test["posted_rate"], core_pred),
            "full_signal_challenger": regression_metrics(test["posted_rate"], full_pred),
        },
    }


def run_fallback_benchmark(frame: pd.DataFrame) -> dict:
    """Measure the reduced-feature model on the latest temporal holdout."""
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    train = data[data["date"] < pd.Timestamp("2025-10-01")].copy()
    test = data[(data["date"] >= pd.Timestamp("2025-10-01")) & (data["date"] <= pd.Timestamp("2025-10-31"))].copy()
    lookup = build_city_lookup(train)
    model = train_ensemble_bundle(train, lookup, include_market_signals=False, metadata={"fallback_test": True})
    prediction = model.predict(test)
    return {
        "fold": "october",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "removed_signals": ["market_index", "quote_signal"],
        "metrics": regression_metrics(test["posted_rate"], prediction),
    }


def create_data_audit(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    train_dates = pd.to_datetime(train["date"])
    validation_dates = pd.to_datetime(validation["date"])
    train_cities = set(train["pickup"]) | set(train["delivery"])
    validation_cities = set(validation["pickup"]) | set(validation["delivery"])
    train_lanes = set(zip(train["pickup"], train["delivery"]))
    validation_lanes = list(zip(validation["pickup"], validation["delivery"]))
    unseen_city_mask = ~validation["pickup"].isin(train_cities) | ~validation["delivery"].isin(train_cities)
    rpm = train["posted_rate"] / train["distance"]
    return {
        "train_shape": list(train.shape),
        "validation_shape": list(validation.shape),
        "train_date_range": [str(train_dates.min().date()), str(train_dates.max().date())],
        "validation_date_range": [str(validation_dates.min().date()), str(validation_dates.max().date())],
        "train_quality": summarize_quality(train).to_dict(),
        "validation_quality": summarize_quality(validation).to_dict(),
        "target": {
            "mean": float(train["posted_rate"].mean()),
            "median": float(train["posted_rate"].median()),
            "p01": float(train["posted_rate"].quantile(0.01)),
            "p99": float(train["posted_rate"].quantile(0.99)),
            "p999": float(train["posted_rate"].quantile(0.999)),
            "max": float(train["posted_rate"].max()),
            "rate_per_mile_median": float(rpm.median()),
            "rate_per_mile_p999": float(rpm.quantile(0.999)),
        },
        "train_city_count": int(len(train_cities)),
        "validation_city_count": int(len(validation_cities)),
        "unseen_validation_cities": sorted(validation_cities - train_cities),
        "unseen_city_row_pct": float(unseen_city_mask.mean() * 100.0),
        "unseen_lane_row_pct": float(np.mean([lane not in train_lanes for lane in validation_lanes]) * 100.0),
        "market_index": {
            "train_mean": float(train["market_index"].mean()),
            "validation_mean": float(validation["market_index"].mean()),
            "train_missing_pct": float(train["market_index"].isna().mean() * 100.0),
            "validation_missing_pct": float(validation["market_index"].isna().mean() * 100.0),
        },
        "important_interpretation": [
            "The evaluation data is strictly later in time than the labeled development data.",
            "A non-trivial share of evaluation rows contains cities/lanes not seen in training.",
            "Weight contains missing and negative values, so cleaning behavior must be explicit and reproducible.",
            "The December chart file lacks market_index and quote_signal, requiring a deliberate fallback path.",
        ],
    }


def write_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
