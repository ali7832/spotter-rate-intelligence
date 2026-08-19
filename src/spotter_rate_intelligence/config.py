from __future__ import annotations

TARGET = "posted_rate"
ID_COLUMN = "load_id"

REQUIRED_TRAIN_COLUMNS = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
    "date", "market_index", "quote_signal", "posted_rate",
]

REQUIRED_FULL_INFERENCE_COLUMNS = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
    "date", "market_index", "quote_signal",
]

REQUIRED_CORE_COLUMNS = [
    "pickup", "delivery", "distance", "equipment", "weight", "date",
]

VALID_EQUIPMENT = ("Dry Van", "Reefer", "Flatbed")

FULL_MODEL_NAME = "lgbm_l1_full"
FALLBACK_MODEL_NAME = "lgbm_l1_core"
RANDOM_SEED = 42
