"""
Realistic synthetic AQI / pollutant history for Hyderabad, Sindh.

Used when OpenWeather's paid history endpoint is unavailable on the free tier.
Correlations and seasonality mirror typical Upper Sindh patterns:
  * winter PM peaks (inversion + regional haze)
  * diurnal rush-hour NO2 / PM bumps
  * PM2.5–PM10–NO2 coupling
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

# Typical hourly baselines for Hyderabad metro (µg/m³ unless noted)
_BASELINES = {
    "pm2_5": 38.0,
    "pm10": 72.0,
    "no2": 22.0,
    "so2": 8.0,
    "o3": 48.0,
    "co": 420.0,   # µg/m³
    "nh3": 6.0,
    "no": 4.0,
}


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _aqi_from_pm25(pm25: float) -> int:
    """Map PM2.5 (µg/m³) to OpenWeather-style AQI bucket 1–5."""
    if pm25 <= 12:
        return 1
    if pm25 <= 35:
        return 2
    if pm25 <= 55:
        return 3
    if pm25 <= 150:
        return 4
    return 5


def generate_synthetic_raw(days: int | None = None, end: datetime | None = None) -> pd.DataFrame:
    cfg = _cfg()
    city = cfg["city"]["name"]
    n_days = days if days is not None else cfg["backfill"]["days"]
    end = end or datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=n_days)
    timestamps = pd.date_range(start, end, freq="h", inclusive="left", tz=timezone.utc)
    n = len(timestamps)

    rng = np.random.default_rng(42)
    hour = timestamps.hour.values
    month = timestamps.month.values

    # Seasonal: higher particulates Nov–Feb (Sindh winter haze)
    winter = np.isin(month, [11, 12, 1, 2]).astype(float)
    summer = np.isin(month, [4, 5, 6]).astype(float)
    season_factor = 1.0 + 0.35 * winter - 0.12 * summer

    # Diurnal rush peaks
    rush = (
        np.exp(-0.5 * ((hour - 8) / 2) ** 2)
        + np.exp(-0.5 * ((hour - 18) / 2.5) ** 2)
    )

    # Correlated pollutant draws
    z = rng.standard_normal((n, 4))
    pm25 = _BASELINES["pm2_5"] * season_factor * (1 + 0.25 * rush) * np.exp(0.15 * z[:, 0])
    pm10 = pm25 * (1.6 + 0.08 * z[:, 1]) + rng.normal(0, 5, n)
    no2 = _BASELINES["no2"] * (1 + 0.4 * rush) * np.exp(0.12 * z[:, 2])
    so2 = _BASELINES["so2"] * season_factor * np.exp(0.1 * rng.standard_normal(n))
    o3 = _BASELINES["o3"] * (1 + 0.2 * summer) * np.exp(0.1 * z[:, 3])
    co = _BASELINES["co"] * (1 + 0.2 * rush) * np.exp(0.08 * rng.standard_normal(n))
    nh3 = _BASELINES["nh3"] * np.exp(0.1 * rng.standard_normal(n))
    no = _BASELINES["no"] * (1 + 0.15 * rush) * np.exp(0.1 * rng.standard_normal(n))

    pollutants = np.clip(
        np.column_stack([pm25, pm10, no2, so2, o3, co, nh3, no]),
        a_min=0.1,
        a_max=None,
    )
    col_names = cfg["features"]["pollutants"]
    df = pd.DataFrame(pollutants, columns=col_names)
    df["timestamp"] = timestamps
    df["aqi"] = [_aqi_from_pm25(v) for v in df["pm2_5"]]
    df = df[["timestamp", "aqi", *col_names]].sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Generated %d hours of synthetic AQI data for %s (free-tier history fallback).",
        len(df),
        city,
    )
    return df
