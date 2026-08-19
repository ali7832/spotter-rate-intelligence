from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import REQUIRED_CORE_COLUMNS, VALID_EQUIPMENT

_EQUIPMENT_NORMALIZATION = {
    "dry van": "Dry Van",
    "dryvan": "Dry Van",
    "van": "Dry Van",
    "reefer": "Reefer",
    "refrigerated": "Reefer",
    "flatbed": "Flatbed",
    "flat bed": "Flatbed",
}


@dataclass
class QualitySummary:
    rows: int
    missing_weight: int
    negative_weight: int
    missing_market_index: int
    missing_quote_signal: int
    invalid_distance: int
    unknown_equipment: int
    invalid_date: int
    duplicate_load_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_equipment(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    normalized = raw.str.lower().map(_EQUIPMENT_NORMALIZATION)
    return normalized.fillna(raw)


def summarize_quality(frame: pd.DataFrame) -> QualitySummary:
    weight = pd.to_numeric(frame.get("weight"), errors="coerce")
    distance = pd.to_numeric(frame.get("distance"), errors="coerce")
    equipment = normalize_equipment(frame.get("equipment", pd.Series(index=frame.index, dtype="string")))
    date = pd.to_datetime(frame.get("date"), errors="coerce")
    load_id = frame.get("load_id")
    return QualitySummary(
        rows=len(frame),
        missing_weight=int(weight.isna().sum()),
        negative_weight=int((weight < 0).fillna(False).sum()),
        missing_market_index=int(pd.to_numeric(frame.get("market_index"), errors="coerce").isna().sum()) if "market_index" in frame else len(frame),
        missing_quote_signal=int(pd.to_numeric(frame.get("quote_signal"), errors="coerce").isna().sum()) if "quote_signal" in frame else len(frame),
        invalid_distance=int((distance.isna() | (distance <= 0)).sum()),
        unknown_equipment=int((~equipment.isin(VALID_EQUIPMENT)).sum()),
        invalid_date=int(date.isna().sum()),
        duplicate_load_id=int(load_id.duplicated().sum()) if load_id is not None else 0,
    )


def validate_core_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_CORE_COLUMNS if column not in frame.columns]


def sanitize_inputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize recoverable issues and return row-level diagnostics.

    The function is intentionally conservative: it never invents required business
    values such as pickup, delivery, date, or distance.
    """
    result = frame.copy()
    diagnostics = pd.DataFrame(index=result.index)
    diagnostics["status"] = "SUCCESS"
    diagnostics["issues"] = ""

    def warn(mask: pd.Series, message: str) -> None:
        if not mask.any():
            return
        diagnostics.loc[mask & diagnostics["status"].eq("SUCCESS"), "status"] = "RECOVERED_WITH_WARNING"
        diagnostics.loc[mask, "issues"] = diagnostics.loc[mask, "issues"].where(
            diagnostics.loc[mask, "issues"].eq(""), diagnostics.loc[mask, "issues"] + "; "
        ) + message

    def reject(mask: pd.Series, message: str) -> None:
        if not mask.any():
            return
        diagnostics.loc[mask, "status"] = "INVALID_INPUT"
        diagnostics.loc[mask, "issues"] = diagnostics.loc[mask, "issues"].where(
            diagnostics.loc[mask, "issues"].eq(""), diagnostics.loc[mask, "issues"] + "; "
        ) + message

    for column in ["distance", "weight", "market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if "equipment" in result:
        original = result["equipment"].astype("string")
        result["equipment"] = normalize_equipment(result["equipment"])
        warn(original.fillna("").str.strip().ne(result["equipment"].astype("string").fillna("")), "equipment normalized")
        reject(~result["equipment"].isin(VALID_EQUIPMENT), "unknown equipment")

    if "date" in result:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        reject(result["date"].isna(), "invalid date")

    if "distance" in result:
        reject(result["distance"].isna() | (result["distance"] <= 0), "distance must be positive")

    if "pickup" in result:
        reject(result["pickup"].isna() | result["pickup"].astype("string").str.strip().eq(""), "pickup is required")
    if "delivery" in result:
        reject(result["delivery"].isna() | result["delivery"].astype("string").str.strip().eq(""), "delivery is required")

    if "weight" in result:
        missing_weight = result["weight"].isna()
        negative_weight = (result["weight"] < 0).fillna(False)
        warn(missing_weight, "weight missing; model missing-value path used")
        result["_weight_was_negative"] = negative_weight.astype("int8")
        warn(negative_weight, "negative weight converted to absolute magnitude and flagged")
        result.loc[negative_weight, "weight"] = result.loc[negative_weight, "weight"].abs()

    return result, diagnostics
