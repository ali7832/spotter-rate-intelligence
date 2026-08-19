from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_quality import sanitize_inputs, validate_core_columns
from .model import RateModelBundle


@dataclass
class PredictionResult:
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    model_used: str


class RatePredictor:
    """Production inference wrapper around the temporally validated champion.

    The core model intentionally excludes market_index and quote_signal because
    forward backtests showed materially better future generalization without
    those drifting signals. A full-signal model may be kept as an offline
    challenger, but it is not used for production predictions until evidence
    shows that it outperforms the champion on future data.
    """

    def __init__(self, champion_model: RateModelBundle, challenger_model: RateModelBundle | None = None):
        self.champion_model = champion_model
        self.challenger_model = challenger_model

    def predict(self, frame: pd.DataFrame) -> PredictionResult:
        missing_core = validate_core_columns(frame)
        if missing_core:
            raise ValueError(f"Missing required columns: {', '.join(missing_core)}")

        clean, diagnostics = sanitize_inputs(frame)
        valid_mask = diagnostics["status"].ne("INVALID_INPUT")

        predictions = pd.DataFrame(index=frame.index)
        predictions["predicted_rate"] = np.nan
        predictions["lower_rate_p90"] = np.nan
        predictions["upper_rate_p90"] = np.nan
        predictions["model_used"] = "none"

        if valid_mask.any():
            subset = clean.loc[valid_mask]
            pred = self.champion_model.predict(subset)
            lo, hi = self.champion_model.prediction_interval(pred, subset)
            predictions.loc[valid_mask, "predicted_rate"] = pred
            predictions.loc[valid_mask, "lower_rate_p90"] = lo
            predictions.loc[valid_mask, "upper_rate_p90"] = hi
            predictions.loc[valid_mask, "model_used"] = "champion_core"

            ood = self.champion_model.ood_flags(subset)
            for column in ood.columns:
                diagnostics.loc[valid_mask, column] = ood[column].to_numpy()

            # Presence of optional market signals is useful for monitoring even
            # though the champion deliberately does not consume them.
            diagnostics.loc[valid_mask, "market_signals_present"] = (
                {"market_index", "quote_signal"}.issubset(clean.columns)
                and clean.loc[valid_mask, "market_index"].notna().to_numpy()
                & clean.loc[valid_mask, "quote_signal"].notna().to_numpy()
            )

        if "ood_score" in diagnostics:
            ood_mask = valid_mask & diagnostics["ood_score"].fillna(0).gt(0)
            diagnostics.loc[ood_mask & diagnostics["status"].eq("SUCCESS"), "status"] = "RECOVERED_WITH_WARNING"
            diagnostics.loc[ood_mask, "issues"] = diagnostics.loc[ood_mask, "issues"].where(
                diagnostics.loc[ood_mask, "issues"].eq(""), diagnostics.loc[ood_mask, "issues"] + "; "
            ) + "limited historical coverage / out-of-distribution features"

        return PredictionResult(
            predictions=predictions,
            diagnostics=diagnostics,
            model_used="champion_core",
        )
