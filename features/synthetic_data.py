"""
Realistic synthetic hourly data for Hyderabad, Sindh, Pakistan.

Mirrors the Open-Meteo combined schema (weather + air quality).
Used when the API or backfill is unavailable during offline/CI runs.

Seasonal patterns:
  * Winter (Nov-Feb): higher PM from inversions and regional haze
  * Summer (Apr-Jun): elevated ozone + dust from heat and dry winds
  * Diurnal rush-hour (07-09h, 17-20h) bumps on NO2, PM, CO
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PM25_BASE = 38.0
_PM10_BASE = 72.0
_NO2_BASE = 22.0
_SO2_BASE = 8.0
_O3_BASE = 48.0
_CO_BASE = 420.0
_DUST_BASE = 12.0
_TEMP_BASE = 28.0
_HUM_BASE = 55.0
_WIND_BASE = 8.0
_PRES_BASE = 1009.0
_VIS_BASE = 6000.0


def _us_aqi_from_pm25(pm25: float) -> float:
    """Approximate US EPA AQI from PM2.5 (µg/m³) — continuous piecewise."""
    breakpoints = [
        (0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return i_lo + (pm25 - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    return 500.0


def generate_synthetic_raw(days: int = 30, end: datetime | None = None) -> pd.DataFrame:
    """Generate `days` hours of realistic synthetic Open-Meteo-schema data."""
    end_dt = end or datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)
    timestamps = pd.date_range(start_dt, end_dt, freq="h", inclusive="left", tz=timezone.utc)
    n = len(timestamps)

    rng = np.random.default_rng(42)
    hour = timestamps.hour.values
    month = timestamps.month.values

    winter = np.isin(month, [11, 12, 1, 2]).astype(float)
    summer = np.isin(month, [4, 5, 6]).astype(float)
    season_pm = 1.0 + 0.40 * winter - 0.10 * summer
    season_o3 = 1.0 - 0.10 * winter + 0.25 * summer

    rush = (
        np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
        + np.exp(-0.5 * ((hour - 18) / 2.5) ** 2)
    )

    z = rng.standard_normal((n, 8))
    pm25 = np.clip(_PM25_BASE * season_pm * (1 + 0.30 * rush) * np.exp(0.15 * z[:, 0]), 1, 500)
    pm10 = np.clip(pm25 * (1.7 + 0.09 * z[:, 1]) + rng.normal(0, 5, n), 1, 600)
    no2 = np.clip(_NO2_BASE * (1 + 0.50 * rush) * np.exp(0.12 * z[:, 2]), 0.5, 200)
    so2 = np.clip(_SO2_BASE * season_pm * np.exp(0.10 * rng.standard_normal(n)), 0.1, 100)
    o3 = np.clip(_O3_BASE * season_o3 * np.exp(0.10 * z[:, 3]), 1, 180)
    co = np.clip(_CO_BASE * (1 + 0.20 * rush) * np.exp(0.08 * rng.standard_normal(n)), 50, 2000)
    dust = np.clip(_DUST_BASE * (1 + 0.20 * summer) * np.exp(0.12 * z[:, 4]), 0.1, 100)

    us_aqi = np.array([_us_aqi_from_pm25(v) for v in pm25])

    temp = _TEMP_BASE + 10 * np.sin(2 * np.pi * (month - 1) / 12) + 5 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.normal(0, 2, n)
    humidity = np.clip(_HUM_BASE - 15 * summer + rng.normal(0, 8, n), 10, 99)
    wind = np.clip(_WIND_BASE + rng.normal(0, 3, n), 0.1, 40)
    pressure = _PRES_BASE + rng.normal(0, 3, n)
    precip = np.clip(rng.exponential(0.3, n) * (humidity > 70), 0, 50)
    cloud = np.clip(30 + rng.normal(0, 25, n), 0, 100)
    visibility = np.clip(_VIS_BASE - pm25 * 30 + rng.normal(0, 500, n), 500, 20000)
    apparent_temp = temp - 2 + rng.normal(0, 1, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_2m": temp,
        "relative_humidity_2m": humidity,
        "apparent_temperature": apparent_temp,
        "wind_speed_10m": wind,
        "wind_direction_10m": rng.uniform(0, 360, n),
        "surface_pressure": pressure,
        "precipitation": precip,
        "cloud_cover": cloud,
        "visibility": visibility,
        "pm2_5": pm25,
        "pm10": pm10,
        "nitrogen_dioxide": no2,
        "sulphur_dioxide": so2,
        "ozone": o3,
        "carbon_monoxide": co,
        "dust": dust,
        "us_aqi": us_aqi,
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Generated %d hours of synthetic data for Hyderabad, Pakistan.", n)
    return df
