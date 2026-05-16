"""
Feature engineering for AQI forecasting.

Produces:
  * calendar features (hour, day_of_week, month, is_weekend)
  * rolling means / stds for each pollutant over configurable windows
  * AQI change rate over 3/6/12 hour windows
  * lagged target column for supervised learning (next 72h mean AQI)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    # cyclical encodings — let linear models see the periodicity
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    cfg = _cfg()
    df = df.sort_values("timestamp").copy()
    pollutants = [p for p in cfg["features"]["pollutants"] if p in df.columns]
    for window in cfg["features"]["rolling_windows_hours"]:
        for p in pollutants:
            df[f"{p}_rollmean_{window}h"] = df[p].rolling(window, min_periods=1).mean()
            df[f"{p}_rollstd_{window}h"] = df[p].rolling(window, min_periods=1).std().fillna(0)
    return df


def add_aqi_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    for h in (3, 6, 12):
        df[f"aqi_change_{h}h"] = df["aqi"] - df["aqi"].shift(h)
    df = df.fillna({c: 0 for c in df.columns if c.startswith("aqi_change_")})
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = mean AQI over the next `target_horizon_hours` hours."""
    cfg = _cfg()
    horizon = cfg["features"]["target_horizon_hours"]
    df = df.sort_values("timestamp").copy()
    # forward-looking rolling mean (shifted by -horizon)
    df["target_aqi_next_72h"] = (
        df["aqi"].shift(-1).rolling(horizon, min_periods=1).mean()
    )
    return df


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline from raw OpenWeather frame to model-ready frame."""
    df = add_calendar_features(raw)
    df = add_rolling_features(df)
    df = add_aqi_change_rate(df)
    df = add_target(df)
    df = df.dropna(subset=["target_aqi_next_72h"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # quick smoke test with synthetic data
    rng = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    demo = pd.DataFrame({
        "timestamp": rng,
        "aqi": np.random.randint(1, 6, len(rng)),
        "pm2_5": np.random.rand(len(rng)) * 80,
        "pm10": np.random.rand(len(rng)) * 120,
        "no2": np.random.rand(len(rng)) * 40,
        "so2": np.random.rand(len(rng)) * 20,
        "o3":  np.random.rand(len(rng)) * 100,
        "co":  np.random.rand(len(rng)) * 500,
        "nh3": np.random.rand(len(rng)) * 10,
        "no":  np.random.rand(len(rng)) * 5,
    })
    print(build_feature_frame(demo).head())
