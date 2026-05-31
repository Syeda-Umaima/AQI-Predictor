"""
MongoDB Atlas Feature Store with Production Resilience.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pymongo.operations import ReplaceOne
from features.mongo_utils import get_database, mongo_retry

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

MONGO_DB = "aqi_predictor"
MONGO_FEATURE_COLLECTION = "features_v2"

def _feature_collection():
    return get_database(MONGO_DB)[MONGO_FEATURE_COLLECTION]

@mongo_retry()
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

@mongo_retry()
def load_features() -> pd.DataFrame:
    """Load all engineered features from MongoDB Atlas feature collection."""
    collection = _feature_collection()
    
    # Use projection and cursor for efficiency
    cursor = collection.find({}, {"_id": 0})
    docs = list(cursor)
    
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
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

@mongo_retry()
def get_latest_timestamp() -> pd.Timestamp | None:
    """Return the latest timestamp stored in the MongoDB feature collection."""
    collection = _feature_collection()
    doc = collection.find_one(sort=[("timestamp", -1)])
    if not doc or "timestamp" not in doc:
        return None
    return pd.to_datetime(doc["timestamp"], utc=True)
