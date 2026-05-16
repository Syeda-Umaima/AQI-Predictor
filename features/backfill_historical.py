"""
Backfill ~30 days of historical AQI into the local Feature Store.

Tries OpenWeather history first. On free-tier 401/403 or empty responses,
falls back to realistic synthetic data for the configured city (Hyderabad).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

from features.api_client import OpenWeatherClient
from features.feature_engineering import build_feature_frame
from features.feature_store import push_to_store
from features.synthetic_data import generate_synthetic_raw

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_history_unavailable(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (401, 403, 404, 429)
    msg = str(exc).lower()
    return any(k in msg for k in ("401", "403", "unauthorized", "subscription"))


def fetch_history_from_api(client: OpenWeatherClient) -> pd.DataFrame | None:
    """Return concatenated raw history, or None if history API is unusable."""
    cfg = _cfg()
    end = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=cfg["backfill"]["days"])
    chunk = timedelta(hours=cfg["backfill"]["chunk_hours"])

    all_chunks: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        logger.info("Fetching history %s → %s", cursor.isoformat(), chunk_end.isoformat())
        try:
            df = client.fetch_history(int(cursor.timestamp()), int(chunk_end.timestamp()))
        except Exception as e:
            if _is_history_unavailable(e):
                logger.warning(
                    "OpenWeather history API unavailable on free tier (%s). "
                    "Will use synthetic fallback.",
                    e,
                )
                return None
            raise
        if not df.empty:
            all_chunks.append(df)
        cursor = chunk_end

    if not all_chunks:
        return None
    return pd.concat(all_chunks).drop_duplicates("timestamp").sort_values("timestamp")


def run_backfill() -> pd.DataFrame:
    cfg = _cfg()
    city = cfg["city"]["name"]
    raw: pd.DataFrame | None = None

    try:
        client = OpenWeatherClient.from_env()
        raw = fetch_history_from_api(client)
    except RuntimeError as e:
        logger.warning("Could not initialise OpenWeather client (%s). Using synthetic data.", e)
    except Exception as e:
        if _is_history_unavailable(e):
            logger.warning("History fetch failed (%s). Using synthetic data.", e)
        else:
            raise

    if raw is None or raw.empty:
        logger.warning(
            "No historical rows from OpenWeather for %s — generating synthetic training data.",
            city,
        )
        raw = generate_synthetic_raw()

    features = build_feature_frame(raw)
    push_to_store(features)
    logger.info("Backfill complete: %d engineered rows for %s.", len(features), city)
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_backfill()
