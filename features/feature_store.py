"""
MongoDB Atlas Feature Store.

Uses the MongoDB URI configured in .env to read and write feature data.
This module replaces the old Hopsworks feature-store integration.
"""
from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import ReplaceOne

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env", override=False)

MONGO_DB = "aqi_predictor"
MONGO_FEATURE_COLLECTION = "features_v2"


def _mongo_uri() -> str:
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri:
        raise EnvironmentError("MONGO_URI is required in .env for MongoDB access.")
    return uri


def _mongo_client() -> MongoClient:
    uri = _mongo_uri()
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


def _mongo_db():
    return _mongo_client()[MONGO_DB]


def _feature_collection():
    return _mongo_db()[MONGO_FEATURE_COLLECTION]


# ---------------------------------------------------------------- Push
def push_to_store(df: pd.DataFrame) -> None:
    """Insert engineered features into MongoDB Atlas feature store."""
    if df.empty:
        logger.info("No rows to push to MongoDB feature store.")
        return

    collection = _feature_collection()
    records = df.copy()
    if "_id" in records.columns:
        records = records.drop(columns=["_id"])

    docs = records.to_dict("records")
    operations = [
        ReplaceOne({"timestamp": doc["timestamp"]}, doc, upsert=True)
        for doc in docs
    ]
    if operations:
        collection.bulk_write(operations, ordered=False)
    logger.info(
        "Pushed %d rows to MongoDB collection %s.",
        len(docs), MONGO_FEATURE_COLLECTION,
    )


# ---------------------------------------------------------------- Load
def load_features() -> pd.DataFrame:
    """Load all engineered features from MongoDB Atlas feature collection."""
    collection = _feature_collection()
    docs = list(collection.find({}))
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info(
        "Loaded %d rows from MongoDB collection %s.",
        len(df), MONGO_FEATURE_COLLECTION,
    )
    return df


def load_recent_features(hours: int = 96) -> pd.DataFrame:
    """Load the most recent `hours` rows from MongoDB."""
    df = load_features()
    if df.empty:
        return df
    df = df.sort_values("timestamp").tail(hours).reset_index(drop=True)
    return df


def get_latest_timestamp() -> pd.Timestamp | None:
    """Return the latest timestamp stored in the MongoDB feature collection."""
    collection = _feature_collection()
    doc = collection.find_one(sort=[("timestamp", -1)])
    if not doc or "timestamp" not in doc:
        return None
    return pd.to_datetime(doc["timestamp"], utc=True)
