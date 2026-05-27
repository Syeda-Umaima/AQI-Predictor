"""
Historical backfill pipeline.

Fetches 30 days of merged Open-Meteo weather + air quality data for
Hyderabad, Pakistan, runs the full feature engineering pipeline, and
provisions the Hopsworks Feature Group (or Parquet fallback) with the result.

Falls back to realistic synthetic data if Open-Meteo is unreachable.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from features.api_client import OpenMeteoClient
from features.feature_engineering import build_feature_frame, count_feature_columns
from features.feature_store import push_to_store
from features.synthetic_data import generate_synthetic_raw

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_historical_raw() -> pd.DataFrame | None:
    """Attempt to fetch combined historical data from Open-Meteo."""
    cfg = _cfg()
    days = 730  # Increase from 30 to 730 days (2 years) to fix data scarcity.
    try:
        client = OpenMeteoClient()
        df = client.fetch_combined_historical(days=days)
        if df.empty:
            logger.warning("Open-Meteo returned empty historical data.")
            return None
        logger.info("Fetched %d raw hourly rows from Open-Meteo (%d days).", len(df), days)
        return df
    except Exception as exc:
        logger.warning("Open-Meteo historical fetch failed: %s", exc)
        return None


def run_backfill() -> pd.DataFrame:
    """
    Full backfill:
      1. Fetch raw history from Open-Meteo (or generate synthetic fallback).
      2. Run feature engineering.
      3. Push to Hopsworks Feature Group (or Parquet fallback).
    """
    raw = fetch_historical_raw()

    if raw is None or raw.empty:
        logger.warning(
            "No historical data from Open-Meteo — generating synthetic training data."
        )
        cfg = _cfg()
        raw = generate_synthetic_raw(days=cfg["backfill"]["days"])

    features = build_feature_frame(raw)
    if features.empty:
        raise RuntimeError("Feature engineering produced an empty DataFrame after backfill.")

    n_feat_cols = count_feature_columns(features)
    logger.info(
        "Backfill engineered %d rows × %d feature columns.",
        len(features), n_feat_cols,
    )

    push_to_store(features)
    logger.info("Backfill complete.")
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run_backfill()
    print(f"\nBackfill complete: {len(df)} rows, {df.shape[1]} columns.")
