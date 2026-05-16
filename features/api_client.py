"""
OpenWeather API client.

- Reads the API key from the OPENWEATHER_API_KEY environment variable.
- Retries with exponential backoff.
- Returns tidy pandas DataFrames ready for feature engineering.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class OpenWeatherClient:
    api_key: str
    base_cfg: dict
    city_cfg: dict

    @classmethod
    def from_env(cls) -> "OpenWeatherClient":
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENWEATHER_API_KEY is not set. Add it to your .env or the "
                "Lovable Secrets panel before running the pipeline."
            )
        cfg = _load_config()
        return cls(api_key=api_key, base_cfg=cfg["openweather"], city_cfg=cfg["city"])

    # ------------------------------------------------------------------ HTTP
    def _request(self, url: str, params: dict) -> dict:
        params = {**params, "appid": self.api_key}
        last_exc: Optional[Exception] = None
        for attempt in range(self.base_cfg["max_retries"]):
            try:
                r = requests.get(url, params=params, timeout=self.base_cfg["request_timeout_s"])
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code in (401, 403):
                    raise
                last_exc = e
                sleep_s = self.base_cfg["backoff_base_s"] ** (attempt + 1)
                logger.warning("Request failed (%s). Retry in %.1fs", e, sleep_s)
                time.sleep(sleep_s)
            except requests.RequestException as e:
                last_exc = e
                sleep_s = self.base_cfg["backoff_base_s"] ** (attempt + 1)
                logger.warning("Request failed (%s). Retry in %.1fs", e, sleep_s)
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenWeather request failed after retries: {last_exc}")

    # --------------------------------------------------------------- Endpoints
    def fetch_current(self) -> pd.DataFrame:
        data = self._request(
            self.base_cfg["current_url"],
            {"lat": self.city_cfg["latitude"], "lon": self.city_cfg["longitude"]},
        )
        return self._to_frame(data["list"])

    def fetch_forecast(self) -> pd.DataFrame:
        data = self._request(
            self.base_cfg["forecast_url"],
            {"lat": self.city_cfg["latitude"], "lon": self.city_cfg["longitude"]},
        )
        return self._to_frame(data["list"])

    def fetch_history(self, start_unix: int, end_unix: int) -> pd.DataFrame:
        data = self._request(
            self.base_cfg["history_url"],
            {
                "lat": self.city_cfg["latitude"],
                "lon": self.city_cfg["longitude"],
                "start": start_unix,
                "end": end_unix,
            },
        )
        return self._to_frame(data["list"])

    # ----------------------------------------------------------------- Helpers
    @staticmethod
    def _to_frame(records: list[dict]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        rows = []
        for r in records:
            row = {"timestamp": datetime.fromtimestamp(r["dt"], tz=timezone.utc),
                   "aqi": r["main"]["aqi"]}
            row.update(r["components"])
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = OpenWeatherClient.from_env()
    print(client.fetch_current().head())
