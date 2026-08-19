from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from .inference import RatePredictor
from .model import RateModelBundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "static"
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "artifacts"))
MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "50000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))


class RateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pickup: str = Field(min_length=1)
    delivery: str = Field(min_length=1)
    distance: float
    equipment: str
    weight: float | None = None
    date: str
    market_index: float | None = None
    quote_signal: float | None = None
    pickup_lat: float | None = None
    pickup_lon: float | None = None
    delivery_lat: float | None = None
    delivery_lon: float | None = None


class BatchRequest(BaseModel):
    rows: list[RateRequest] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def get_predictor() -> RatePredictor:
    champion_path = MODEL_DIR / "champion_model.joblib"
    challenger_path = MODEL_DIR / "challenger_full_model.joblib"
    if not champion_path.exists():
        raise RuntimeError(f"Champion model artifact not found under {MODEL_DIR}")
    champion = RateModelBundle.load(champion_path)
    challenger = RateModelBundle.load(challenger_path) if challenger_path.exists() else None
    return RatePredictor(champion, challenger)


def _result_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = get_predictor().predict(frame)
    merged = pd.concat([result.predictions, result.diagnostics], axis=1)
    return merged.where(pd.notna(merged), None).to_dict(orient="records")


app = FastAPI(
    title="Spotter Rate Intelligence",
    version="0.2.0",
    description="ML Engineering Assessment Prototype - freight rate prediction service.",
)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        predictor = get_predictor()
        _ = predictor.champion_model.metadata
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model not ready: {exc}") from exc
    return {"status": "ready"}


@app.get("/v1/model/info")
def model_info() -> dict[str, Any]:
    predictor = get_predictor()
    return {
        "service": "Spotter Rate Intelligence",
        "prototype": True,
        "champion": predictor.champion_model.metadata,
        "challenger": predictor.challenger_model.metadata if predictor.challenger_model else None,
        "interval_calibration": predictor.champion_model.interval_calibration,
    }


@app.post("/v1/predict")
def predict_one(request: RateRequest) -> dict[str, Any]:
    frame = pd.DataFrame([request.model_dump()])
    record = _result_records(frame)[0]
    return record


@app.post("/v1/predict/batch")
def predict_batch(request: BatchRequest) -> dict[str, Any]:
    frame = pd.DataFrame([row.model_dump() for row in request.rows])
    records = _result_records(frame)
    valid = sum(record.get("status") != "INVALID_INPUT" for record in records)
    return {
        "row_count": len(records),
        "predicted_rows": valid,
        "invalid_rows": len(records) - valid,
        "results": records,
    }


@app.post("/v1/predict/csv")
async def predict_csv(file: UploadFile = File(...)) -> Response:
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds configured upload limit")
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc
    if len(frame) == 0:
        raise HTTPException(status_code=422, detail="CSV contains no data rows")
    if len(frame) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV has {len(frame):,} rows; synchronous demo limit is {MAX_BATCH_ROWS:,}. Large jobs should use the async batch worker.",
        )

    try:
        result = get_predictor().predict(frame)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    output = frame.copy()
    for column in result.predictions.columns:
        output[column] = result.predictions[column]
    for column in result.diagnostics.columns:
        # Avoid duplicate column names if an uploaded file already has a field
        # with one of our diagnostic names.
        output[f"diagnostic_{column}"] = result.diagnostics[column]

    buffer = io.StringIO()
    output.to_csv(buffer, index=False)
    invalid = int(result.diagnostics["status"].eq("INVALID_INPUT").sum())
    headers = {
        "Content-Disposition": 'attachment; filename="spotter_rate_predictions.csv"',
        "X-Row-Count": str(len(output)),
        "X-Invalid-Rows": str(invalid),
    }
    return Response(content=buffer.getvalue(), media_type="text/csv", headers=headers)
