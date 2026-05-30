"""
Hourly feature ingestion — designed to run on a cron / GitHub Actions schedule.

Fetches the next 3 days of combined weather + air quality forecast from
Open-Meteo, engineers features, and appends them to the Feature Store.
Also ingests the most recent past_days=2 to capture any newly available
air-quality measurements since the last run.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from features.api_client import OpenMeteoClient
from features.feature_engineering import build_feature_frame, count_feature_columns
from features.feature_store import get_latest_timestamp, push_to_store

logger = logging.getLogger(__name__)


def run_hourly_ingest() -> pd.DataFrame:
    client = OpenMeteoClient()

    forecast = client.fetch_combined_forecast(forecast_days=4)

    latest_ts = get_latest_timestamp()
    if latest_ts is None:
        recent_aq = client.fetch_air_quality_past(past_days=2)
    else:
        today = datetime.now(tz=timezone.utc).date()
        window_start = max(
            latest_ts.date(),
            today - timedelta(days=2),
        )

        # If the latest data in the DB is from today or later, no need to fetch.
        if window_start >= today:
            logger.info("Data is already up to date (DB latest: %s >= Today: %s). Skipping ingestion.", window_start, today)
            return pd.DataFrame()

        recent_aq = client.fetch_air_quality_archive(
            start_date=str(window_start),
            end_date=str(today),
        )

    if not recent_aq.empty:
        recent_wx = client.fetch_weather_archive(
            start_date=str(recent_aq["timestamp"].min().date()),
            end_date=str(recent_aq["timestamp"].max().date()),
        )
        if not recent_wx.empty:
            recent = pd.merge(recent_wx, recent_aq, on="timestamp", how="outer")
        else:
            recent = recent_aq
    else:
        recent = pd.DataFrame()

    frames = [f for f in [recent, forecast] if not f.empty]
    if not frames:
        raise RuntimeError("Open-Meteo returned no data during hourly ingest.")

    raw = (
        pd.concat(frames)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    logger.info("Hourly ingest: %d raw rows (recent + forecast).", len(raw))

    features = build_feature_frame(raw)
    if features.empty:
        raise RuntimeError("Feature engineering produced no rows after hourly ingest.")

    n_feats = count_feature_columns(features)
    # Explicitly cast us_aqi to float for MongoDB storage consistency.
    if "us_aqi" in features.columns:
        features["us_aqi"] = features["us_aqi"].astype(float)

    push_to_store(features)
    logger.info(
        "Hourly ingest complete: appended %d rows × %d feature columns.",
        len(features), n_feats,
    )
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_hourly_ingest()
