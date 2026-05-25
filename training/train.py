"""
Multi-model training pipeline with strict data-leakage prevention.

Models trained and benchmarked:
  1. Ridge Regression  (baseline linear model)
  2. Random Forest     (tree ensemble)
  3. XGBoost           (gradient boosting)
  4. TensorFlow LSTM   (sequence model on 24h lookback windows)

Data-leakage guarantees:
  - Features are sorted by timestamp before splitting.
  - Train / test split is strictly time-ordered (no shuffle).
  - Rolling statistics are computed inside feature_engineering with past-only
    windows; no future rows are visible during any rolling computation.
  - StandardScaler is fit ONLY on the training set; the fitted scaler is then
    applied to the test set — never re-fit on test data.

Outputs:
  - models/champion.joblib        (or champion_lstm.keras + champion_scaler.joblib)
  - artifacts/leaderboard.json    (full benchmark summary)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features.feature_store import load_features

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

TARGET = "target_aqi_next_72h"


def _cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- Data layer
def load_training_frame() -> pd.DataFrame:
    return load_features()


def time_ordered_split(df: pd.DataFrame, test_size: float):
    """
    Strictly chronological train/test split — no data leakage.
    Future rows are NEVER present in the training set.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric columns except timestamp, target, and constant features."""
    drop = {"timestamp", TARGET}
    valid_cols = []
    for c in df.columns:
        if c not in drop and pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique() > 1:
                valid_cols.append(c)
    return valid_cols


# ------------------------------------------------------------------- Metrics
def score(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ------------------------------------------------------------------- Models
def _fit_scaler_train_only(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler exclusively on training data to prevent leakage."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def train_ridge(X_train, y_train, X_test, y_test) -> dict:
    logger.info("Training Ridge Regression …")
    scaler = _fit_scaler_train_only(X_train)
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(Xtr_s, y_train)
    preds = model.predict(Xte_s)
    return {"model": model, "scaler": scaler, "metrics": score(y_test, preds)}


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    logger.info("Training Random Forest …")
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {"model": model, "scaler": None, "metrics": score(y_test, preds)}


def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    logger.info("Training XGBoost …")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {"model": model, "scaler": None, "metrics": score(y_test, preds)}


def train_lstm(X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Keras LSTM on a sliding 24h window.
    Scaler fit on training data only — test data transformed with train scaler.
    """
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore

    cfg = _cfg()["training"]["lstm"]
    seq_len = cfg["sequence_length"]

    scaler = _fit_scaler_train_only(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    def to_sequences(X: np.ndarray, y):
        xs, ys = [], []
        y_arr = y.values if hasattr(y, "values") else np.array(y)
        for i in range(seq_len, len(X)):
            xs.append(X[i - seq_len:i])
            ys.append(y_arr[i])
        return np.array(xs), np.array(ys)

    Xtr_seq, ytr_seq = to_sequences(Xtr, y_train.reset_index(drop=True))
    Xte_seq, yte_seq = to_sequences(Xte, y_test.reset_index(drop=True))

    if len(Xtr_seq) < 10:
        raise ValueError("Not enough data to train LSTM (need > seq_len rows in training set).")

    model = models.Sequential([
        layers.Input(shape=(seq_len, Xtr.shape[1])),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        Xtr_seq, ytr_seq,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=0.1,
        verbose=0,
    )
    preds = model.predict(Xte_seq, verbose=0).ravel()
    return {"model": model, "scaler": scaler, "metrics": score(yte_seq, preds)}


# ---------------------------------------------------------------- Champion
def promote_champion(leaderboard: dict, feature_cols: list[str]) -> str:
    champ_name = min(leaderboard, key=lambda k: leaderboard[k]["metrics"]["rmse"])
    champ = leaderboard[champ_name]
    logger.info(
        "Champion: %s  (RMSE=%.4f, MAE=%.4f, R²=%.4f)",
        champ_name,
        champ["metrics"]["rmse"],
        champ["metrics"]["mae"],
        champ["metrics"]["r2"],
    )

    if champ_name == "lstm":
        champ["model"].save(MODELS_DIR / "champion_lstm.keras")
        joblib.dump(champ["scaler"], MODELS_DIR / "champion_scaler.joblib")
    else:
        artifact = {"model": champ["model"], "scaler": champ.get("scaler")}
        joblib.dump(artifact, MODELS_DIR / "champion.joblib")

    meta = {
        "champion": champ_name,
        "metrics": champ["metrics"],
        "feature_columns": feature_cols,
        "leaderboard": {k: v["metrics"] for k, v in leaderboard.items()},
    }
    (ARTIFACTS / "leaderboard.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved leaderboard to %s", ARTIFACTS / "leaderboard.json")

    try:
        import os
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if api_key:
            import hopsworks  # type: ignore
            cfg = _cfg()["hopsworks"]
            project_name = os.getenv("HOPSWORKS_PROJECT", cfg["project"])
            project = hopsworks.login(
                project=project_name,
                api_key_value=api_key,
                host="eu-west.cloud.hopsworks.ai",
            )
            mr = project.get_model_registry()
            registered = mr.python.create_model(
                name=cfg["model_registry_name"],
                metrics=champ["metrics"],
                description=f"Champion AQI model ({champ_name})",
            )
            registered.save(str(MODELS_DIR))
            logger.info("Champion registered in Hopsworks Model Registry.")
    except Exception as exc:
        logger.warning("Model Registry push skipped (%s) — local copy kept in %s.", exc, MODELS_DIR)

    return champ_name


# -------------------------------------------------------------------- Driver
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _cfg()["training"]

    df = load_training_frame()
    if df.empty:
        raise RuntimeError("Feature store is empty. Run backfill first.")
 
    # Drop rows with NaNs (rolling window edges and lookahead targets)
    logger.info("Before cleaning NaNs: %d rows", len(df))
    df = df.dropna().reset_index(drop=True)
    logger.info("After cleaning NaNs: %d rows", len(df))
 
    if df.empty:
        raise RuntimeError("DataFrame is empty after dropping NaNs. Check feature engineering.")
 
    train_df, test_df = time_ordered_split(df, cfg["test_size"])
    cols = feature_columns(df)
 
    logger.info(
        "Split: %d train rows / %d test rows / %d feature columns.",
        len(train_df), len(test_df), len(cols),
    )

    X_train, y_train = train_df[cols], train_df[TARGET]
    X_test, y_test = test_df[cols], test_df[TARGET]

    leaderboard: dict = {}
    leaderboard["ridge"] = train_ridge(X_train, y_train, X_test, y_test)
    leaderboard["random_forest"] = train_random_forest(X_train, y_train, X_test, y_test)
    leaderboard["xgboost"] = train_xgboost(X_train, y_train, X_test, y_test)

    try:
        leaderboard["lstm"] = train_lstm(X_train, y_train, X_test, y_test)
    except Exception as exc:
        logger.warning("LSTM training skipped: %s", exc)

    print("\n=== Model Leaderboard ===")
    board = pd.DataFrame({k: v["metrics"] for k, v in leaderboard.items()}).T
    print(board.round(4).sort_values("rmse"))

    promote_champion(leaderboard, cols)
    print(f"\nChampion model saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
