from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.8

BASE_CATEGORICAL = ["pickup", "delivery", "equipment", "lane"]


def build_city_lookup(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    pieces = []
    if {"pickup", "pickup_lat", "pickup_lon"}.issubset(frame.columns):
        pieces.append(frame[["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}))
    if {"delivery", "delivery_lat", "delivery_lon"}.issubset(frame.columns):
        pieces.append(frame[["delivery", "delivery_lat", "delivery_lon"]].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}))
    if not pieces:
        return {}
    coords = pd.concat(pieces, ignore_index=True).dropna().groupby("city")[["lat", "lon"]].median()
    return {str(city): (float(row.lat), float(row.lon)) for city, row in coords.iterrows()}


def fill_coordinates(frame: pd.DataFrame, city_lookup: dict[str, tuple[float, float]]) -> pd.DataFrame:
    result = frame.copy()
    for city_col, lat_col, lon_col in [
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ]:
        if lat_col not in result:
            result[lat_col] = np.nan
        if lon_col not in result:
            result[lon_col] = np.nan
        missing = result[lat_col].isna() | result[lon_col].isna()
        if missing.any():
            mapped = result.loc[missing, city_col].map(city_lookup)
            result.loc[missing, lat_col] = [value[0] if isinstance(value, tuple) else np.nan for value in mapped]
            result.loc[missing, lon_col] = [value[1] if isinstance(value, tuple) else np.nan for value in mapped]
    return result


def _haversine(frame: pd.DataFrame) -> pd.Series:
    lat1 = np.radians(frame["pickup_lat"].astype(float))
    lon1 = np.radians(frame["pickup_lon"].astype(float))
    lat2 = np.radians(frame["delivery_lat"].astype(float))
    lon2 = np.radians(frame["delivery_lon"].astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return EARTH_RADIUS_MILES * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def engineer_features(frame: pd.DataFrame, include_market_signals: bool = True) -> pd.DataFrame:
    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["weight_missing"] = x["weight"].isna().astype("int8")
    detected_negative = (pd.to_numeric(x["weight"], errors="coerce") < 0).fillna(False)
    if "_weight_was_negative" in x.columns:
        detected_negative = detected_negative | pd.to_numeric(x["_weight_was_negative"], errors="coerce").fillna(0).astype(bool)
    x["weight_negative"] = detected_negative.astype("int8")
    x["weight_clean"] = pd.to_numeric(x["weight"], errors="coerce").abs()
    x["day_of_week"] = x["date"].dt.dayofweek.astype("float")
    x["day_of_year"] = x["date"].dt.dayofyear.astype("float")
    x["month"] = x["date"].dt.month.astype("float")
    x["week_of_year"] = x["date"].dt.isocalendar().week.astype("float")
    x["is_weekend"] = (x["date"].dt.dayofweek >= 5).astype("int8")
    x["doy_sin"] = np.sin(2.0 * np.pi * x["day_of_year"] / 365.25)
    x["doy_cos"] = np.cos(2.0 * np.pi * x["day_of_year"] / 365.25)
    x["lat_delta"] = x["delivery_lat"] - x["pickup_lat"]
    x["lon_delta"] = x["delivery_lon"] - x["pickup_lon"]
    x["haversine_miles"] = _haversine(x)
    x["route_factor"] = x["distance"] / x["haversine_miles"].replace(0, np.nan)
    x["distance_sqrt"] = np.sqrt(x["distance"].clip(lower=0))
    x["distance_log1p"] = np.log1p(x["distance"].clip(lower=0))
    x["lane"] = x["pickup"].astype("string") + "__" + x["delivery"].astype("string")

    if include_market_signals:
        x["market_missing"] = x["market_index"].isna().astype("int8")
        x["quote_missing"] = x["quote_signal"].isna().astype("int8")
        x["market_quote"] = x["market_index"] * x["quote_signal"]
        x["distance_market"] = x["distance"] * x["market_index"]
        x["distance_quote"] = x["distance"] * x["quote_signal"]

    return x


def feature_columns(include_market_signals: bool = True) -> tuple[list[str], list[str]]:
    numeric = [
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
        "distance_sqrt", "distance_log1p", "weight_clean", "weight_missing",
        "weight_negative", "day_of_week", "day_of_year", "month", "week_of_year",
        "is_weekend", "doy_sin", "doy_cos", "lat_delta", "lon_delta",
        "haversine_miles", "route_factor",
    ]
    if include_market_signals:
        numeric += [
            "market_index", "quote_signal", "market_missing", "quote_missing",
            "market_quote", "distance_market", "distance_quote",
        ]
    return BASE_CATEGORICAL + numeric, BASE_CATEGORICAL
