"""
FastAPI inference backend — Pearls AQI Predictor.

Endpoints:
  GET  /health            → liveness check
  GET  /forecast          → 72h AQI forecast using champion model + live Open-Meteo data
  POST /predict           → single-row prediction from arbitrary feature payload
  GET  /leaderboard       → benchmark metrics from the last training run

The champion model is loaded once at startup and cached in module-level state.
Supports Ridge / RF / XGBoost (champion.joblib) and LSTM (champion_lstm.keras).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from features.api_client import OpenMeteoClient
from features.feature_engineering import build_feature_frame
from features.synthetic_data import generate_synthetic_raw

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

app = FastAPI(title="Pearls AQI Predictor", version="2.0.0")

_champion_cache: dict = {}


def _load_champion() -> dict:
    if _champion_cache:
        return _champion_cache

    meta_path = ARTIFACTS / "leaderboard.json"
    if not meta_path.exists():
        raise FileNotFoundError("leaderboard.json not found. Run training pipeline first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    name = meta["champion"]

    if name == "lstm":
        from tensorflow.keras.models import load_model  # type: ignore
        _champion_cache["model"] = load_model(MODELS_DIR / "champion_lstm.keras")
        _champion_cache["scaler"] = joblib.load(MODELS_DIR / "champion_scaler.joblib")
    else:
        artifact = joblib.load(MODELS_DIR / "champion.joblib")
        _champion_cache["model"] = artifact["model"]
        _champion_cache["scaler"] = artifact.get("scaler")

    _champion_cache["meta"] = meta
    _champion_cache["name"] = name
    logger.info("Champion model loaded: %s", name)
    return _champion_cache


def _predict_rows(model, scaler, name: str, meta: dict, X: pd.DataFrame) -> np.ndarray:
    cols = meta["feature_columns"]
    available = [c for c in cols if c in X.columns]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        for c in missing:
            X[c] = 0.0
    X = X[cols]

    if name == "lstm":
        seq_len = model.input_shape[1]
        X_s = scaler.transform(X)
        X_seq = X_s.reshape(X_s.shape[0], 1, X_s.shape[1]).repeat(seq_len, axis=1)
        return model.predict(X_seq, verbose=0).ravel()

    if name == "ridge" and scaler is not None:
        return model.predict(scaler.transform(X))

    return model.predict(X)


# ---------------------------------------------------------------- Endpoints
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_dir": str(MODELS_DIR)}


@app.get("/leaderboard")
def leaderboard() -> dict:
    lb_path = ARTIFACTS / "leaderboard.json"
    if not lb_path.exists():
        raise HTTPException(404, "No leaderboard found. Run training pipeline.")
    return json.loads(lb_path.read_text(encoding="utf-8"))


@app.get("/forecast")
def forecast() -> dict:
    try:
        champion = _load_champion()
    except FileNotFoundError as exc:
        raise HTTPException(500, f"Model not trained: {exc}")

    try:
        client = OpenMeteoClient()
        raw = client.fetch_combined_forecast(forecast_days=4)
    except Exception as exc:
        logger.warning("Open-Meteo unavailable (%s) — using synthetic forecast.", exc)
        raw = generate_synthetic_raw(days=4)

    if raw.empty:
        raise HTTPException(500, "No forecast data available from Open-Meteo.")

    feats = build_feature_frame(raw)
    if feats.empty:
        raise HTTPException(500, "Feature engineering produced no rows for forecast.")

    X = feats.tail(72)
    meta = champion["meta"]
    preds = _predict_rows(
        champion["model"], champion["scaler"],
        champion["name"], meta, X.copy()
    )
    timestamps = feats["timestamp"].tail(72).astype(str).tolist()

    return {
        "champion": meta["champion"],
        "champion_metrics": meta["metrics"],
        "forecast": [
            {"timestamp": t, "predicted_aqi": float(p)}
            for t, p in zip(timestamps, preds)
        ],
    }


class PredictRequest(BaseModel):
    features: dict[str, Any]


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        champion = _load_champion()
    except FileNotFoundError as exc:
        raise HTTPException(500, f"Model not trained: {exc}")

    X = pd.DataFrame([req.features])
    meta = champion["meta"]
    preds = _predict_rows(
        champion["model"], champion["scaler"],
        champion["name"], meta, X
    )
    return {
        "predicted_aqi": float(preds[0]),
        "champion": meta["champion"],
    }
