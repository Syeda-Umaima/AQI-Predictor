"""
Historical backfill pipeline.

Fetches two years of merged Open-Meteo weather + air quality data for
Hyderabad, Pakistan, runs the full feature engineering pipeline, and
writes engineered features into MongoDB Atlas collection `features_v2`.

Falls back to realistic synthetic data if Open-Meteo is unreachable.
"""
from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path

import certifi
import pandas as pd
from dotenv import load_dotenv
import yaml
from pymongo import MongoClient

from features.api_client import OpenMeteoClient
from features.feature_engineering import build_feature_frame, count_feature_columns
from features.synthetic_data import generate_synthetic_raw
from training.logging_config import setup_logging

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
load_dotenv(dotenv_path=ROOT / ".env", override=False)


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mongo_client() -> MongoClient:
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGO_URI is required in .env for MongoDB backfill.")

    ca = certifi.where()
    client = MongoClient(
        uri,
        tls=True,
        tlsCAFile=ca,
        tlsInsecure=True,
        serverSelectionTimeoutMS=10000,
    )
    client.admin.command("ping")
    return client


def _mongo_collection():
    return _mongo_client()["aqi_predictor"]["features_v2"]


def fetch_historical_raw() -> pd.DataFrame | None:
    """Attempt to fetch combined historical data from Open-Meteo."""
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
      3. Insert engineered features into MongoDB Atlas.
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

    if "us_aqi" in features.columns:
        features["us_aqi"] = features["us_aqi"].astype("float64")

    n_feat_cols = count_feature_columns(features)
    logger.info(
        "Backfill engineered %d rows × %d feature columns.",
        len(features), n_feat_cols,
    )

    collection = _mongo_collection()
    logger.info("Dropping existing MongoDB collection 'features_v2' before backfill.")
    collection.drop()
    if not features.empty:
        collection.insert_many(features.to_dict("records"))
    logger.info("Inserted %d rows into MongoDB collection 'features_v2'.", len(features))
    logger.info("Backfill complete.")
    return features


if __name__ == "__main__":
    setup_logging()
    logger.info("Starting historical data backfill...")
    df = run_backfill()
    logger.info("Backfill process finished successfully.")
    print(f"\n✅ Backfill complete: {len(df)} rows, {df.shape[1]} columns.")
