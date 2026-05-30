"""
ML training pipeline with strict leakage prevention.
Ridge, Random Forest, XGBoost, LightGBM, and LSTM.
"""
from __future__ import annotations

import certifi
import json
import logging
from pathlib import Path
import io
import os
import datetime

import gridfs
import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
load_dotenv(dotenv_path=ROOT / ".env", override=False)

MONGO_DB = "aqi_predictor"
FEATURE_COLLECTION = "features_v2"
MODEL_METADATA_COLLECTION = "model_metadata"
TARGET = "target_aqi_next_1h"

def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _mongo_client() -> MongoClient:
    uri = os.getenv("MONGO_URI", "").strip()
    ca = certifi.where()
    return MongoClient(uri, tls=True, tlsCAFile=ca, tlsInsecure=True, serverSelectionTimeoutMS=10000)

def load_training_frame() -> pd.DataFrame:
    client = _mongo_client()
    collection = client[MONGO_DB][FEATURE_COLLECTION]
    rows = list(collection.find())
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "_id" in df.columns: df = df.drop(columns=["_id"])
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

def train_lstm(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    feature_scaler = StandardScaler().fit(X_train)
    y_train_log = np.log1p(y_train)
    target_scaler = StandardScaler().fit(y_train_log.values.reshape(-1, 1))
    
    seq_len = 24
    Xtr, Xte = feature_scaler.transform(X_train), feature_scaler.transform(X_test)
    ytr_scaled = target_scaler.transform(y_train_log.values.reshape(-1, 1))
    
    def to_sequences(X, y):
        xs, ys = [], []
        for i in range(seq_len, len(X)):
            xs.append(X[i-seq_len:i])
            ys.append(y.iloc[i])
        return np.array(xs), np.array(ys)
    
    Xtr_seq, ytr_seq = to_sequences(Xtr, pd.Series(ytr_scaled.ravel()))
    Xte_seq, yte_seq = to_sequences(Xte, y_test.reset_index(drop=True))
    
    model = models.Sequential([
        layers.Input(shape=(seq_len, Xtr.shape[1])),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr_seq, ytr_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=0, callbacks=[EarlyStopping(patience=5)])
    
    preds_scaled = model.predict(Xte_seq, verbose=0)
    preds = np.expm1(target_scaler.inverse_transform(preds_scaled).ravel())
    return {"model": model, "scaler": feature_scaler, "metrics": score(yte_seq, preds)}

def promote_champion(leaderboard: dict, feature_cols: list[str]) -> str:
    champ_name = min(leaderboard, key=lambda k: leaderboard[k]["metrics"]["rmse"])
    champ = leaderboard[champ_name]
    
    artifact = {
        "model": champ["model"],
        "scaler": champ.get("scaler"),
        "type": champ_name,
        "use_log_y": champ.get("use_log_y", False)
    }
    
    if champ_name == "lstm":
        champ["model"].save(MODELS_DIR / "champion_lstm.keras")
        joblib.dump(champ.get("scaler"), MODELS_DIR / "champion_scaler.joblib")
    else:
        joblib.dump(artifact, MODELS_DIR / "champion.joblib")
        
    meta = {
        "champion": champ_name,
        "metrics": champ["metrics"],
        "feature_columns": feature_cols,
        "leaderboard": {k: v["metrics"] for k, v in leaderboard.items()},
        "timestamp": datetime.datetime.utcnow()
    }
    
    db = _mongo_client()[MONGO_DB]
    fs = gridfs.GridFS(db)
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    file_id = fs.put(buffer.getvalue(), filename=f"champion_{champ_name}.joblib", champion=champ_name, timestamp=meta["timestamp"])
    
    db[MODEL_METADATA_COLLECTION].insert_one({**meta, "file_id": file_id})
    return champ_name

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_training_frame()
    if df.empty:
        logger.error("Fetched 0 rows from feature store.")
        return
    
    logger.info("Fetched %d rows from feature store.", len(df))
    
    # 1. SMART COLUMN DROPPING: Remove columns with > 50% NaN values
    nan_threshold = 0.5
    high_nan_cols = [c for c in df.columns if df[c].isna().sum() / len(df) > nan_threshold]
    if high_nan_cols:
        logger.warning("Dropping columns with >50%% NaNs: %s", high_nan_cols)
        df = df.drop(columns=high_nan_cols)
    
    logger.info("Rows remaining after dropping high-NaN columns: %d", len(df))
    
    # 2. Identify features and clean rows
    cols = feature_columns(df)
    df = df.dropna(subset=[TARGET] + cols).reset_index(drop=True)
    
    # 3. DATA VALIDATION GUARD: Ensure we have data for training
    if df.shape[0] == 0:
        raise ValueError(
            "The training dataset is empty after data cleaning. "
            "Please ensure historical data has been backfilled into the collection."
        )
    
    logger.info("Final rows for training: %d", len(df))
    
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
        
    try: leaderboard["lstm"] = train_lstm(X_train, y_train, X_test, y_test)
    except Exception as e: logger.warning(f"LSTM failed: {e}")
    
    promote_champion(leaderboard, cols)

if __name__ == "__main__":
    main()
