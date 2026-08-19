from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter_rate_intelligence.features import build_city_lookup
from spotter_rate_intelligence.model import train_ensemble_bundle


def _core_interval_calibration() -> dict[str, float]:
    paths = sorted((ROOT / "reports").glob("core_oof_*.csv"))
    if not paths:
        fallback = ROOT / "reports" / "uncertainty_calibration.json"
        return json.loads(fallback.read_text()) if fallback.exists() else {}
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if "absolute_error" not in frame:
        return {}
    return {
        "p80": float(frame["absolute_error"].quantile(0.80)),
        "p90": float(frame["absolute_error"].quantile(0.90)),
        "p95": float(frame["absolute_error"].quantile(0.95)),
    }


def _selection_evidence() -> dict:
    path = ROOT / "reports" / "model_ablation_benchmark.csv"
    if not path.exists():
        return {}
    metrics = pd.read_csv(path)
    result = {}
    for model in ["business_baseline_equipment_median_rpm", "ensemble_50_50", "core_ensemble_50_50"]:
        subset = metrics[metrics["model"].eq(model)]
        if len(subset):
            result[model] = {
                "mean_mae": float(subset["mae"].mean()),
                "mean_wape_pct": float(subset["wape_pct"].mean()),
                "mean_rmse": float(subset["rmse"].mean()),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.train, parse_dates=["date"])
    lookup = build_city_lookup(frame)
    calibration = _core_interval_calibration()

    common = {
        "training_rows": len(frame),
        "training_start": str(frame["date"].min().date()),
        "training_end": str(frame["date"].max().date()),
        "target": "posted_rate",
        "validation_strategy": "forward temporal backtesting: Aug/Sep/Oct",
        "model_family": "LightGBM L1 + CatBoost MAE ensemble",
        "blend_weight_lgb": 0.5,
        "blend_weight_catboost": 0.5,
    }

    champion = train_ensemble_bundle(
        frame,
        lookup,
        include_market_signals=False,
        metadata={
            **common,
            "role": "champion_core",
            "market_signal_policy": "excluded after temporal ablation improved future validation",
        },
        interval_calibration=calibration,
    )
    challenger = train_ensemble_bundle(
        frame,
        lookup,
        include_market_signals=True,
        metadata={
            **common,
            "role": "challenger_full_signals",
            "market_signal_policy": "offline challenger only due temporal drift/generalization penalty",
        },
        interval_calibration=calibration,
    )

    champion.save(ROOT / "artifacts" / "champion_model.joblib")
    challenger.save(ROOT / "artifacts" / "challenger_full_model.joblib")
    (ROOT / "artifacts" / "city_lookup.json").write_text(
        json.dumps({k: list(v) for k, v in lookup.items()}, indent=2), encoding="utf-8"
    )

    metadata = {
        **common,
        "selected_champion": "core_ensemble_50_50",
        "selection_reason": "The reduced-feature core model beat the full-signal model on every forward validation month, indicating market_index/quote_signal were non-stationary and hurt future generalization.",
        "interval_calibration": calibration,
        "selection_evidence": _selection_evidence(),
    }
    (ROOT / "reports" / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved temporally selected champion core model and full-signal challenger.")


if __name__ == "__main__":
    main()
