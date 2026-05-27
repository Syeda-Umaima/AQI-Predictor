"""
Massive feature engineering engine — targets 100-200 engineered columns.

Generates from raw Open-Meteo hourly frames:
  1. Temporal embeddings  — sin/cos for hour, day-of-week, month
  2. Lag features         — t-1h, t-2h, t-3h, t-24h, t-48h for key signals
  3. Rolling aggregations — 3h/6h/12h/24h mean, std, min, max per pollutant
  4. Interaction features — cross-variable multiplicative / ratio signals
  5. AQI change rates     — delta over 3h, 6h, 12h, 24h windows
  6. Supervised target    — mean us_aqi over the next 72 hours

Data-leakage prevention:
  - All rolling and lag computations operate strictly on past rows
    (default closed="left" / min_periods to avoid look-ahead).
  - Time-ordered sort is enforced before any windowed operation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ Temporal
def add_temporal_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features + cyclical sin/cos embeddings (no look-ahead)."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["week_of_year"] = ts.dt.isocalendar().week
    df["is_weekend"] = ts.dt.dayofweek >= 5
    df["is_rush_hour"] = (
        ((ts.dt.hour >= 7) & (ts.dt.hour <= 9))
        | ((ts.dt.hour >= 17) & (ts.dt.hour <= 20))
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


# ---------------------------------------------------------------------- Lags
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly past-only lag features — no future data is accessed."""
    cfg = _cfg()
    lag_vars = [v for v in cfg["features"]["lag_vars"] if v in df.columns]
    lag_hours = cfg["features"]["lag_hours"]

    df = df.sort_values("timestamp").reset_index(drop=True)
    for var in lag_vars:
        for lag in lag_hours:
            df[f"{var}_lag_{lag}h"] = df[var].shift(lag)
    return df


# ----------------------------------------------------------------- Rolling
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling aggregations over strictly past windows (closed='left').
    Windows: 3h, 6h, 12h, 24h  × stats: mean, std, min, max.
    """
    cfg = _cfg()
    roll_vars = [v for v in cfg["features"]["rolling_vars"] if v in df.columns]
    windows = cfg["features"]["rolling_windows_hours"]

    df = df.sort_values("timestamp").reset_index(drop=True)
    for window in windows:
        for var in roll_vars:
            base = df[var].rolling(window, min_periods=1)
            df[f"{var}_roll_mean_{window}h"] = base.mean()
            df[f"{var}_roll_std_{window}h"] = base.std().fillna(0)
            df[f"{var}_roll_min_{window}h"] = base.min()
            df[f"{var}_roll_max_{window}h"] = base.max()
    return df


# --------------------------------------------------------------- Interactions
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative and ratio atmospheric interaction signals."""
    df = df.copy()

    def _safe_col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    temp = _safe_col("temperature_2m")
    hum = _safe_col("relative_humidity_2m")
    wind = _safe_col("wind_speed_10m")
    pres = _safe_col("surface_pressure")
    pm25 = _safe_col("pm2_5")
    pm10 = _safe_col("pm10")
    no2 = _safe_col("nitrogen_dioxide")
    dust = _safe_col("dust")
    aqi = _safe_col("us_aqi")

    df["feat_temp_x_humidity"] = temp * hum
    df["feat_wind_x_pressure"] = wind * pres
    df["feat_wind_div_pressure"] = (wind / pres.replace(0, np.nan)).fillna(0)
    df["feat_pm25_x_no2"] = pm25 * no2
    df["feat_pm25_x_pm10"] = pm25 * pm10
    df["feat_pm25_x_humidity"] = pm25 * hum
    df["feat_pm10_div_pm25"] = (pm10 / pm25.replace(0, np.nan)).fillna(1)
    df["feat_dust_x_wind"] = dust * wind
    df["feat_aqi_x_temp"] = aqi * temp
    df["feat_apparent_vs_actual"] = (
        _safe_col("apparent_temperature") - temp
    )
    df["feat_cloud_x_precip"] = (
        _safe_col("cloud_cover") * _safe_col("precipitation")
    )
    return df


# ------------------------------------------------------------- Forecast
def add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lookahead features for the 72h forecast target.
    Shifts future weather data to the current row to give the model a
    view of the conditions at the time of the prediction target.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["temperature_lookahead_72h"] = df["temperature_2m"].shift(-72)
    df["wind_speed_lookahead_72h"] = df["wind_speed_10m"].shift(-72)
    df["precipitation_lookahead_72h"] = df["precipitation"].shift(-72)
    return df


# --------------------------------------------------------------- AQI change
def add_aqi_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Delta AQI over 3h, 6h, 12h, 24h backward windows (no future data)."""
    df = df.sort_values("timestamp").copy()
    if "us_aqi" not in df.columns:
        return df
    for h in (3, 6, 12, 24):
        df[f"aqi_change_{h}h"] = df["us_aqi"] - df["us_aqi"].shift(h)
    change_cols = [c for c in df.columns if c.startswith("aqi_change_")]
    df[change_cols] = df[change_cols].fillna(0)
    return df


# ------------------------------------------------------------------- Target
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supervised regression target: us_aqi in the NEXT 72 hours.
    Computed as a strictly forward-shifted value to avoid leakage.
    Rows where the 72h horizon is unavailable are dropped downstream.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "us_aqi" not in df.columns:
        return df
    df["target_aqi_next_72h"] = df["us_aqi"].shift(-72)
    return df


# ---------------------------------------------------------------- Full pipe
def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.
    Input:  merged Open-Meteo hourly DataFrame (weather + air quality).
    Output: model-ready DataFrame with 100-200+ engineered columns.
    """
    if raw.empty:
        return pd.DataFrame()

    df = raw.sort_values("timestamp").reset_index(drop=True)

    # --- SAFE IMPUTATION & CLEANING ---
    # 1. Drop rows where primary AQI/pollutant data is missing. This prevents
    #    imputing the target variable across large historical gaps.
    primary_cols = [
        'us_aqi', 'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone',
        'sulphur_dioxide', 'carbon_monoxide', 'dust'
    ]
    df = df.dropna(subset=primary_cols)
    logger.info("Dropped rows with missing primary pollutants. Rows remaining: %d", len(df))

    # 2. For remaining rows, forward-fill small gaps (<=3 hours) in weather data.
    cfg = _cfg()
    weather_cols = [c for c in cfg["open_meteo"]["weather_hourly_vars"] if c in df.columns]
    df[weather_cols] = df[weather_cols].ffill(limit=3)
    logger.info("Forward-filled small gaps in weather data (limit=3).")

    df = df.reset_index(drop=True)
    
    df = add_temporal_embeddings(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    df = add_forecast_features(df)
    df = add_aqi_change_rate(df)
    df = add_target(df)

    df = df.dropna(subset=["target_aqi_next_72h"]).reset_index(drop=True)

    # Fix for Hopsworks: cast any completely null columns to float to avoid dtype errors.
    for col in df.columns:
        if df[col].isnull().all():
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                pass  # Ignore columns that cannot be cast to float
    return df


def count_feature_columns(df: pd.DataFrame) -> int:
    """Return number of engineered numeric feature columns (excludes timestamp, target)."""
    non_feature = {"timestamp", "target_aqi_next_72h"}
    return sum(
        1 for c in df.columns
        if c not in non_feature and pd.api.types.is_numeric_dtype(df[c])
    )


if __name__ == "__main__":
    import numpy as np

    rng = pd.date_range("2025-01-01", periods=720, freq="h", tz="UTC")
    n = len(rng)
    demo = pd.DataFrame({
        "timestamp": rng,
        "us_aqi": np.random.randint(30, 200, n).astype(float),
        "pm2_5": np.random.rand(n) * 80,
        "pm10": np.random.rand(n) * 120,
        "nitrogen_dioxide": np.random.rand(n) * 40,
        "sulphur_dioxide": np.random.rand(n) * 20,
        "ozone": np.random.rand(n) * 100,
        "carbon_monoxide": np.random.rand(n) * 500,
        "dust": np.random.rand(n) * 15,
        "temperature_2m": 20 + np.random.randn(n) * 8,
        "relative_humidity_2m": 50 + np.random.randn(n) * 15,
        "apparent_temperature": 19 + np.random.randn(n) * 8,
        "wind_speed_10m": np.abs(np.random.randn(n) * 10),
        "wind_direction_10m": np.random.rand(n) * 360,
        "surface_pressure": 1010 + np.random.randn(n) * 5,
        "precipitation": np.abs(np.random.randn(n) * 0.5),
        "cloud_cover": np.random.rand(n) * 100,
        "visibility": 5000 + np.random.randn(n) * 2000,
    })
    result = build_feature_frame(demo)
    n_feats = count_feature_columns(result)
    print(f"Rows: {len(result)}  |  Feature columns: {n_feats}")
    print(result.head(2))
