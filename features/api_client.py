"""
Open-Meteo API client for weather + air quality data.

Fetches from two free, no-key-required endpoints:
  - https://api.open-meteo.com/v1/forecast       (weather)
  - https://air-quality-api.open-meteo.com/v1/air-quality  (pollutants + US AQI)
  - https://archive-api.open-meteo.com/v1/archive (historical weather)

Returns tidy, hourly pandas DataFrames indexed by UTC timestamp.
No API key required — Open-Meteo is fully open access.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class OpenMeteoClient:
    """Thin wrapper around the Open-Meteo REST endpoints with retry logic."""

    def __init__(self) -> None:
        self._cfg = _load_config()
        self._city = self._cfg["city"]
        self._om = self._cfg["open_meteo"]

    # ------------------------------------------------------------------ HTTP
    def _get(self, url: str, params: dict) -> dict:
        last_exc: Optional[Exception] = None
        for attempt in range(self._om["max_retries"]):
            try:
                r = requests.get(url, params=params, timeout=self._om["request_timeout_s"])
                r.raise_for_status()
                return r.json()
            except requests.RequestException as exc:
                last_exc = exc
                sleep_s = self._om["backoff_base_s"] ** (attempt + 1)
                logger.warning("Request to %s failed (%s). Retry in %.1fs", url, exc, sleep_s)
                time.sleep(sleep_s)
        raise RuntimeError(f"Open-Meteo request failed after retries: {last_exc}")

    # -------------------------------------------------------- Base params
    def _base_params(self) -> dict:
        return {
            "latitude": self._city["latitude"],
            "longitude": self._city["longitude"],
            "timezone": "UTC",
        }

    # -------------------------------------------------------- Weather
    def _parse_hourly(self, data: dict, extra_rename: dict | None = None) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()
        df = pd.DataFrame(hourly)
        df["timestamp"] = pd.to_datetime(df["time"], utc=True)
        df = df.drop(columns=["time"])
        if extra_rename:
            df = df.rename(columns=extra_rename)
        return df.sort_values("timestamp").reset_index(drop=True)

    def fetch_weather_forecast(self, forecast_days: int = 3) -> pd.DataFrame:
        """Fetch hourly weather forecast for the next `forecast_days` days."""
        params = {
            **self._base_params(),
            "hourly": ",".join(self._om["weather_hourly_vars"]),
            "forecast_days": forecast_days,
        }
        data = self._get(self._om["weather_url"], params)
        return self._parse_hourly(data)

    def fetch_weather_archive(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical hourly weather via the archive endpoint."""
        params = {
            **self._base_params(),
            "hourly": ",".join(self._om["weather_hourly_vars"]),
            "start_date": start_date,
            "end_date": end_date,
        }
        data = self._get(self._om["weather_archive_url"], params)
        return self._parse_hourly(data)

    # -------------------------------------------------------- Air quality
    def fetch_air_quality_forecast(self, forecast_days: int = 3) -> pd.DataFrame:
        """Fetch hourly air quality forecast."""
        params = {
            **self._base_params(),
            "hourly": ",".join(self._om["air_quality_hourly_vars"]),
            "forecast_days": forecast_days,
        }
        data = self._get(self._om["air_quality_url"], params)
        return self._parse_hourly(data)

    def fetch_air_quality_past(self, past_days: int) -> pd.DataFrame:
        """Fetch historical air quality using past_days parameter (max ~92 days)."""
        params = {
            **self._base_params(),
            "hourly": ",".join(self._om["air_quality_hourly_vars"]),
            "past_days": past_days,
            "forecast_days": 0,
        }
        data = self._get(self._om["air_quality_url"], params)
        return self._parse_hourly(data)

    # -------------------------------------------------------- Combined
    def fetch_combined_forecast(self, forecast_days: int = 4) -> pd.DataFrame:
        """Merge weather + air quality into one hourly DataFrame."""
        weather = self.fetch_weather_forecast(forecast_days=forecast_days)
        aq = self.fetch_air_quality_forecast(forecast_days=forecast_days)
        if weather.empty and aq.empty:
            return pd.DataFrame()
        if weather.empty:
            return aq
        if aq.empty:
            return weather
        merged = pd.merge(weather, aq, on="timestamp", how="outer").sort_values("timestamp")
        return merged.reset_index(drop=True)

    def fetch_combined_historical(self, days: int = 30) -> pd.DataFrame:
        """Fetch `days` of merged historical weather + air quality."""
        end = datetime.now(tz=timezone.utc).date()
        start = end - timedelta(days=days)
        weather = self.fetch_weather_archive(
            start_date=start.isoformat(), end_date=end.isoformat()
        )
        aq = self.fetch_air_quality_past(past_days=min(days, 92))
        if weather.empty and aq.empty:
            return pd.DataFrame()
        if weather.empty:
            return aq
        if aq.empty:
            return weather
        merged = pd.merge(weather, aq, on="timestamp", how="outer").sort_values("timestamp")
        return merged.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = OpenMeteoClient()
    df = client.fetch_combined_forecast()
    print(df.head())
    print(f"Columns: {list(df.columns)}")
