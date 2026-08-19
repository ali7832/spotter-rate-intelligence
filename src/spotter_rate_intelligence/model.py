from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from .features import engineer_features, feature_columns, fill_coordinates


@dataclass
class RateModelBundle:
    """Serializable champion/fallback model bundle.

    The bundle deliberately ensembles two different tree learners:
    - LightGBM with an L1 objective for robust numeric/categorical splits.
    - CatBoost with an MAE objective for strong native categorical handling.

    The blend reduced temporal MAE versus either learner alone in the development
    backtests, while keeping a single reusable feature/inference contract.
    """

    lgb_estimator: Any
    cat_estimator: Any
    include_market_signals: bool
    category_maps: dict[str, dict[str, int]]
    city_lookup: dict[str, tuple[float, float]]
    known_cities: set[str]
    known_lanes: set[str]
    blend_weight_lgb: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    interval_calibration: dict[str, float] = field(default_factory=dict)

    def _enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        filled = fill_coordinates(frame, self.city_lookup)
        return engineer_features(filled, include_market_signals=self.include_market_signals)

    def _transform_lgb(self, enriched: pd.DataFrame) -> pd.DataFrame:
        columns, categorical = feature_columns(self.include_market_signals)
        x = enriched[columns].copy()
        for column in categorical:
            mapping = self.category_maps[column]
            x[column] = x[column].astype("string").map(mapping).fillna(-1).astype("int32")
        return x

    def _transform_cat(self, enriched: pd.DataFrame) -> pd.DataFrame:
        columns, categorical = feature_columns(self.include_market_signals)
        x = enriched[columns].copy()
        for column in categorical:
            # CatBoost categorical columns must be consistently string-like.
            x[column] = x[column].astype("string").fillna("__MISSING__").astype(str)
        return x

    def predict_components(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        enriched = self._enrich(frame)
        lgb_x = self._transform_lgb(enriched)
        cat_x = self._transform_cat(enriched)
        lgb_prediction = np.asarray(self.lgb_estimator.predict(lgb_x), dtype=float)
        cat_prediction = np.asarray(self.cat_estimator.predict(cat_x), dtype=float)
        return {
            "lightgbm": np.clip(lgb_prediction, 1.0, None),
            "catboost": np.clip(cat_prediction, 1.0, None),
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        parts = self.predict_components(frame)
        w = float(self.blend_weight_lgb)
        prediction = w * parts["lightgbm"] + (1.0 - w) * parts["catboost"]
        return np.clip(prediction, 1.0, None)

    def ood_flags(self, frame: pd.DataFrame) -> pd.DataFrame:
        pickup = frame["pickup"].astype("string")
        delivery = frame["delivery"].astype("string")
        lane = pickup + "__" + delivery
        flags = pd.DataFrame(index=frame.index)
        flags["unseen_pickup"] = ~pickup.isin(self.known_cities)
        flags["unseen_delivery"] = ~delivery.isin(self.known_cities)
        flags["unseen_lane"] = ~lane.isin(self.known_lanes)

        distance = pd.to_numeric(frame.get("distance"), errors="coerce")
        low = self.metadata.get("distance_p001")
        high = self.metadata.get("distance_p999")
        flags["distance_outlier"] = False
        if low is not None and high is not None:
            flags["distance_outlier"] = (distance < float(low)) | (distance > float(high))

        # A deliberately interpretable OOD score. It is not presented as a
        # calibrated probability; it is a risk flag for product behavior.
        flags["ood_score"] = (
            flags["unseen_pickup"].astype(int)
            + flags["unseen_delivery"].astype(int)
            + flags["unseen_lane"].astype(int)
            + flags["distance_outlier"].astype(int)
        )
        return flags

    def prediction_interval(self, prediction: np.ndarray, frame: pd.DataFrame, coverage: str = "p90") -> tuple[np.ndarray, np.ndarray]:
        width = float(self.interval_calibration.get(coverage, self.interval_calibration.get("p90", 250.0)))
        # Widen the empirical interval for OOD traffic. This is intentionally
        # transparent and conservative rather than pretending to be perfectly
        # calibrated outside the development distribution.
        flags = self.ood_flags(frame)
        multiplier = 1.0 + 0.20 * flags["ood_score"].clip(upper=3).to_numpy(dtype=float)
        half_width = width * multiplier
        p = np.asarray(prediction, dtype=float)
        return np.clip(p - half_width, 1.0, None), p + half_width

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RateModelBundle":
        return joblib.load(path)


def _category_maps(frame: pd.DataFrame, categorical: list[str]) -> dict[str, dict[str, int]]:
    return {
        column: {
            value: index
            for index, value in enumerate(frame[column].astype("string").dropna().unique().tolist())
        }
        for column in categorical
    }


def train_ensemble_bundle(
    frame: pd.DataFrame,
    city_lookup: dict[str, tuple[float, float]],
    include_market_signals: bool,
    random_seed: int = 42,
    metadata: dict[str, Any] | None = None,
    interval_calibration: dict[str, float] | None = None,
    blend_weight_lgb: float = 0.5,
) -> RateModelBundle:
    enriched = fill_coordinates(frame, city_lookup)
    enriched = engineer_features(enriched, include_market_signals=include_market_signals)
    columns, categorical = feature_columns(include_market_signals)

    maps = _category_maps(enriched, categorical)
    lgb_x = enriched[columns].copy()
    for column in categorical:
        lgb_x[column] = lgb_x[column].astype("string").map(maps[column]).fillna(-1).astype("int32")

    lgb = LGBMRegressor(
        objective="regression_l1",
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.95,
        colsample_bytree=0.95,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb.fit(lgb_x, frame["posted_rate"], categorical_feature=categorical)

    cat_x = enriched[columns].copy()
    for column in categorical:
        cat_x[column] = cat_x[column].astype("string").fillna("__MISSING__").astype(str)
    cat_indices = [columns.index(column) for column in categorical]
    cat = CatBoostRegressor(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=random_seed,
        verbose=False,
        thread_count=-1,
        l2_leaf_reg=5.0,
        random_strength=0.2,
        allow_writing_files=False,
    )
    cat.fit(cat_x, frame["posted_rate"], cat_features=cat_indices)

    cities = set(frame["pickup"].astype(str)) | set(frame["delivery"].astype(str))
    lanes = set(frame["pickup"].astype(str) + "__" + frame["delivery"].astype(str))
    final_metadata = dict(metadata or {})
    final_metadata.setdefault("distance_p001", float(frame["distance"].quantile(0.001)))
    final_metadata.setdefault("distance_p999", float(frame["distance"].quantile(0.999)))
    final_metadata.setdefault("ensemble", "0.5*LightGBM_L1 + 0.5*CatBoost_MAE")

    return RateModelBundle(
        lgb_estimator=lgb,
        cat_estimator=cat,
        include_market_signals=include_market_signals,
        category_maps=maps,
        city_lookup=city_lookup,
        known_cities=cities,
        known_lanes=lanes,
        blend_weight_lgb=blend_weight_lgb,
        metadata=final_metadata,
        interval_calibration=dict(interval_calibration or {}),
    )


# Backward-compatible name for earlier scripts/imports.
train_lgbm_bundle = train_ensemble_bundle
