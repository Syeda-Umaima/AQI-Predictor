"""
Feature engineering pipeline for AQI forecasting.
Processes raw weather and pollutant data into predictive features.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

def _cfg() -> dict:
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encoding of time variables."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 23.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 23.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 11.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 11.0)
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Historical lag features."""
    cfg = _cfg()
    lag_vars = [v for v in cfg["features"]["lag_vars"] if v in df.columns]
    lags = cfg["features"]["lag_hours"]

    df = df.sort_values("timestamp").reset_index(drop=True)
    for lag in lags:
        for var in lag_vars:
            df[f"{var}_lag_{lag}h"] = df[var].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling aggregations over past windows."""
    cfg = _cfg()
    roll_vars = [v for v in cfg["features"]["rolling_vars"] if v in df.columns]
    windows = cfg["features"]["rolling_windows_hours"]

    df = df.sort_values("timestamp").reset_index(drop=True)
    for window in windows:
        for var in roll_vars:
            base = df[var].shift(1).rolling(window, min_periods=1)
            df[f"{var}_roll_mean_{window}h"] = base.mean()
            df[f"{var}_roll_std_{window}h"] = base.std().fillna(0)
            df[f"{var}_roll_min_{window}h"] = base.min()
            df[f"{var}_roll_max_{window}h"] = base.max()
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Atmospheric interaction signals."""
    df = df.copy()

    def _safe_col(name: str) -> pd.Series:
        return df[name].shift(1) if name in df.columns else pd.Series(np.nan, index=df.index)

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
    df["feat_temp_humidity_index"] = (temp * hum / 100.0).fillna(0)
    df["feat_temp_humidity_ratio"] = (temp / hum.replace(0, np.nan)).fillna(0)
    df["feat_wind_x_pressure"] = wind * pres
    df["feat_wind_humidity_interaction"] = wind * hum
    df["feat_wind_div_pressure"] = (wind / pres.replace(0, np.nan)).fillna(0)
    df["feat_pm25_x_no2"] = pm25 * no2
    df["feat_pm25_x_pm10"] = pm25 * pm10
    df["feat_pm25_x_humidity"] = pm25 * hum
    df["feat_pm10_div_pm25"] = (pm10 / pm25.replace(0, np.nan)).fillna(1)
    df["feat_dust_x_wind"] = dust * wind
    df["feat_aqi_x_temp"] = aqi * temp
    df["feat_aqi_div_wind"] = (aqi / wind.replace(0, np.nan)).fillna(aqi)
    df["feat_pm25_div_wind"] = (pm25 / wind.replace(0, np.nan)).fillna(pm25)
    df["feat_apparent_vs_actual"] = (_safe_col("apparent_temperature") - temp)
    df["feat_cloud_x_precip"] = (_safe_col("cloud_cover") * _safe_col("precipitation"))
    df["feat_temp_gradient"] = temp.diff().fillna(0)
    df["feat_pressure_gradient"] = pres.diff().fillna(0)
    return df

def add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather lookahead features."""
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["temperature_target_hour"] = df["temperature_2m"].shift(-1)
    df["relative_humidity_target_hour"] = df["relative_humidity_2m"].shift(-1)
    df["wind_speed_target_hour"] = df["wind_speed_10m"].shift(-1)
    df["precipitation_target_hour"] = df["precipitation"].shift(-1)
    if "cloud_cover" in df.columns:
        df["cloud_cover_target_hour"] = df["cloud_cover"].shift(-1)
    return df

def add_aqi_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """AQI change rate features."""
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "us_aqi" not in df.columns: return df
    for h in (3, 6, 12, 24):
        df[f"aqi_change_{h}h"] = df["us_aqi"].shift(1) - df["us_aqi"].shift(h + 1)
    change_cols = [c for c in df.columns if c.startswith("aqi_change_")]
    df[change_cols] = df[change_cols].fillna(0)
    return df

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Supervised target: us_aqi at T+1."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "us_aqi" not in df.columns: return df
    df["target_aqi_next_1h"] = df["us_aqi"].shift(-1)
    return df

def build_feature_frame(raw: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """End-to-end feature engineering pipeline."""
    if raw.empty: return pd.DataFrame()
    df = raw.sort_values("timestamp").reset_index(drop=True)
    primary_cols = ['us_aqi', 'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide', 'carbon_monoxide', 'dust']
    df = df.dropna(subset=primary_cols)
    weather_cols = [c for c in _cfg()["open_meteo"]["weather_hourly_vars"] if c in df.columns]
    df[weather_cols] = df[weather_cols].ffill(limit=3)
    
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    df = add_forecast_features(df)
    df = add_aqi_change_rate(df)
    if include_target:
        df = add_target(df)
        df = df.dropna(subset=["target_aqi_next_1h"]).reset_index(drop=True)
    return df.reset_index(drop=True)

def count_feature_columns(df: pd.DataFrame) -> int:
    """Count numeric feature columns."""
    non_feature = {"timestamp", "target_aqi_next_1h", "target_aqi_next_72h"}
    return sum(1 for c in df.columns if c not in non_feature and pd.api.types.is_numeric_dtype(df[c]))
