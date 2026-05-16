"""
FastAPI micro-backend exposing the champion AQI model.

Endpoints:
  GET  /health
  GET  /forecast      → 72h AQI forecast for the configured city
  POST /predict       → predict from an arbitrary feature payload
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from features.api_client import OpenWeatherClient
from features.feature_engineering import build_feature_frame

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

app = FastAPI(title="Pearls AQI Predictor", version="1.0.0")


def _load_champion():
    meta = json.loads((ARTIFACTS / "leaderboard.json").read_text())
    if meta["champion"] == "lstm":
        from tensorflow.keras.models import load_model  # type: ignore
        return meta, load_model(MODELS_DIR / "champion_lstm.keras"), joblib.load(MODELS_DIR / "champion_scaler.joblib")
    return meta, joblib.load(MODELS_DIR / "champion.joblib"), None


class PredictRequest(BaseModel):
    features: dict[str, Any]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/forecast")
def forecast() -> dict:
    try:
        meta, model, scaler = _load_champion()
    except FileNotFoundError as e:
        raise HTTPException(500, f"Model not trained yet: {e}")

    client = OpenWeatherClient.from_env()
    current = client.fetch_current()
    forecast_df = client.fetch_forecast()
    combined = pd.concat([current, forecast_df]).drop_duplicates("timestamp").sort_values("timestamp")
    feats = build_feature_frame(combined)
    X = feats[meta["feature_columns"]].tail(72)

    if scaler is not None:
        import numpy as np
        seq = scaler.transform(X)
        seq = seq.reshape(seq.shape[0], 1, seq.shape[1]).repeat(model.input_shape[1], axis=1)
        preds = model.predict(seq, verbose=0).ravel()
    else:
        preds = model.predict(X)

    timestamps = feats["timestamp"].tail(72).astype(str).tolist()
    return {
        "champion": meta["champion"],
        "forecast": [{"timestamp": t, "predicted_aqi": float(p)} for t, p in zip(timestamps, preds)],
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    meta, model, scaler = _load_champion()
    X = pd.DataFrame([req.features])[meta["feature_columns"]]
    if scaler is not None:
        import numpy as np
        seq = scaler.transform(X)
        seq = seq.reshape(seq.shape[0], 1, seq.shape[1]).repeat(model.input_shape[1], axis=1)
        pred = float(model.predict(seq, verbose=0).ravel()[0])
    else:
        pred = float(model.predict(X)[0])
    return {"predicted_aqi": pred, "champion": meta["champion"]}
