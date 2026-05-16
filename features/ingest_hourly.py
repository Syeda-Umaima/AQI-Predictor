"""
Hourly feature ingestion — current conditions + 72h forecast only.

Appends engineered rows to the local Feature Store (Hopsworks or Parquet).
Designed for GitHub Actions cron; avoids expensive 30-day history backfill.
"""
from __future__ import annotations

import logging

import pandas as pd

from features.api_client import OpenWeatherClient
from features.feature_engineering import build_feature_frame
from features.feature_store import push_to_store

logger = logging.getLogger(__name__)


def run_hourly_ingest() -> pd.DataFrame:
    client = OpenWeatherClient.from_env()
    current = client.fetch_current()
    forecast = client.fetch_forecast()

    if current.empty and forecast.empty:
        raise RuntimeError("OpenWeather returned no current or forecast data.")

    raw = pd.concat([current, forecast]).drop_duplicates("timestamp").sort_values("timestamp")
    logger.info("Fetched %d hourly rows (current + forecast).", len(raw))

    features = build_feature_frame(raw)
    if features.empty:
        raise RuntimeError("Feature engineering produced no rows after dropna.")

    push_to_store(features)
    logger.info("Hourly ingest complete: appended %d engineered rows.", len(features))
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_hourly_ingest()
