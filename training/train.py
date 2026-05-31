"""
ML training pipeline with cloud persistence and production resilience.
"""
from __future__ import annotations

import json
import logging
import io
import datetime
from pathlib import Path

import gridfs
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb

from features.mongo_utils import get_database, mongo_retry

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MONGO_DB_NAME = "aqi_predictor"
FEATURE_COLLECTION = "features_v2"
MODEL_METADATA_COLLECTION = "model_metadata"
TARGET = "target_aqi_next_1h"

def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@mongo_retry(max_retries=3, delay=5.0)
def load_training_frame() -> pd.DataFrame:
    """Load features from MongoDB with iterative fetching for maximum resilience."""
    db = get_database(MONGO_DB_NAME)
    collection = db[FEATURE_COLLECTION]
    
    logger.info(f"Fetching features from collection: {FEATURE_COLLECTION}...")
    
    # Use a projection to exclude _id early and set a reasonable batch size
    cursor = collection.find({}, {"_id": 0}).batch_size(2000)
    
    rows = []
    try:
        for i, doc in enumerate(cursor):
            rows.append(doc)
            if (i + 1) % 5000 == 0:
                logger.info(f"Loaded {i + 1} rows so far...")
    except Exception as e:
        logger.error(f"Error during iterative cursor fetch: {e}")
        raise
    
    if not rows:
        logger.warning("No data found in MongoDB for training.")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    logger.info(f"Successfully loaded {len(df)} rows.")
    return df

def time_ordered_split(df: pd.DataFrame, test_size: float):
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def feature_columns(df: pd.DataFrame) -> list[str]:
    drop_set = {"timestamp", TARGET, "target", "us_aqi", "target_aqi_next_72h"}
    pollutants = {'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide', 'carbon_monoxide', 'dust'}
    drop_set.update(pollutants)
    valid_cols = []
    for c in df.columns:
        if c in drop_set or "target" in c.lower(): continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique() > 1: valid_cols.append(c)
    return valid_cols

def score(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

@mongo_retry()
def persist_champion(leaderboard: dict, feature_cols: list[str]) -> str:
    """Stream champion model and metadata directly to MongoDB GridFS."""
    champ_name = min(leaderboard, key=lambda k: leaderboard[k]["metrics"]["rmse"])
    champ = leaderboard[champ_name]
    
    artifact = {
        "model": champ["model"],
        "scaler": champ.get("scaler"),
        "type": champ_name,
        "use_log_y": champ.get("use_log_y", False)
    }
    
    db = get_database(MONGO_DB_NAME)
    fs = gridfs.GridFS(db)
    
    # Serialize to buffer
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    artifact_bytes = buffer.getvalue()
    
    timestamp = datetime.datetime.utcnow()
    run_id = f"run_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    # Store in GridFS
    file_id = fs.put(
        artifact_bytes,
        filename=f"champion_{champ_name}.joblib",
        run_id=run_id,
        model_type=champ_name,
        timestamp=timestamp
    )
    
    # Prepare metadata for upsert
    meta = {
        "champion": champ_name,
        "metrics": champ["metrics"],
        "feature_columns": feature_cols,
        "leaderboard": {k: v["metrics"] for k, v in leaderboard.items()},
        "timestamp": timestamp,
        "file_id": file_id,
        "run_id": run_id
    }
    
    # Upsert latest metadata
    db[MODEL_METADATA_COLLECTION].replace_one(
        {"type": "latest_champion"},
        {**meta, "type": "latest_champion"},
        upsert=True
    )
    
    # Also save a history record
    db[MODEL_METADATA_COLLECTION].insert_one({**meta, "type": "history"})
    
    logger.info(f"Champion '{champ_name}' persisted to Cloud. RunID: {run_id}")
    return champ_name

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_training_frame()
    if df.empty:
        logger.error("No data found for training.")
        return
    
    logger.info(f"Fetched {len(df)} rows.")
    
    nan_threshold = 0.5
    high_nan_cols = [c for c in df.columns if df[c].isna().sum() / len(df) > nan_threshold]
    if high_nan_cols:
        df = df.drop(columns=high_nan_cols)
    
    cols = feature_columns(df)
    df = df.dropna(subset=[TARGET] + cols).reset_index(drop=True)
    
    if df.shape[0] < 100:
        logger.error("Insufficient data for training after cleaning.")
        return
        
    train_df, test_df = time_ordered_split(df, 0.2)
    X_train, y_train = train_df[cols], train_df[TARGET]
    X_test, y_test = test_df[cols], test_df[TARGET]
    
    # Feature Selection
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1).fit(X_train, np.log1p(y_train))
    cols = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False).head(80).index.tolist()
    X_train, X_test = X_train[cols], X_test[cols]
    
    leaderboard = {}
    model_configs = {
        "ridge": {"estimator": Ridge(alpha=5.0), "use_scaler": True, "use_log_y": True},
        "xgboost": {"estimator": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4), "use_scaler": False, "use_log_y": True},
        "lightgbm": {"estimator": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4), "use_scaler": False, "use_log_y": True}
    }
    
    for name, cfg in model_configs.items():
        scaler = StandardScaler().fit(X_train) if cfg["use_scaler"] else None
        Xt, Xv = (scaler.transform(X_train), scaler.transform(X_test)) if scaler else (X_train, X_test)
        model = cfg["estimator"].fit(Xt, np.log1p(y_train))
        preds = np.expm1(model.predict(Xv))
        leaderboard[name] = {"model": model, "scaler": scaler, "metrics": score(y_test, preds), "use_log_y": True}
    
    persist_champion(leaderboard, cols)

if __name__ == "__main__":
    main()
