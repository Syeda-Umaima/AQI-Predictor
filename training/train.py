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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb

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


def train_lstm(X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Keras LSTM on a sliding 24h window.
    Scaler fit on training data only — test data transformed with train scaler.
    """
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore

    cfg = _cfg()["training"]["lstm"]
    scaler = StandardScaler().fit(X_train)

    seq_len = 24  # Fix: Hardcode sequence length to prevent config error

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
        Xtr_seq, ytr_seq, # type: ignore
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
 
    logger.info("Initial dataframe shape: %s", df.shape)

    # 1. Identify valid feature candidates
    drop_cols = {"timestamp", TARGET}
    feature_candidates = [
        c for c in df.columns 
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # 2. Filter out columns that are entirely NaN or have zero variance (constant)
    valid_features = []
    for c in feature_candidates:
        if df[c].isna().all():
            logger.warning("Dropping column '%s' because it is entirely NaN.", c)
            continue
        if df[c].nunique() <= 1:
            logger.warning("Dropping column '%s' because it has zero variance (constant).", c)
            continue
        valid_features.append(c)

    # 3. Retain only essential columns and drop rows with NaNs in valid features/target
    keep_cols = ["timestamp", TARGET] + valid_features
    df = df[keep_cols]
    
    df = df.dropna(subset=[TARGET] + valid_features).reset_index(drop=True)
    logger.info("Dataframe shape after cleaning: %s", df.shape)
 
    if df.empty:
        raise RuntimeError("DataFrame is empty after processing. Check feature engineering or target alignment.")
 
    # 4. Proceed with time-ordered split and training
    cols = feature_columns(df)
    train_df, test_df = time_ordered_split(df, cfg["test_size"])

    # 5. Final training on full feature set
    logger.info(
        "Split: %d train rows / %d test rows / %d feature columns.",
        len(train_df), len(test_df), len(cols),
    )
    
    X_train, y_train = train_df[cols], train_df[TARGET]
    X_test, y_test = test_df[cols], test_df[TARGET]

    # 6. Simplified Model Training & Evaluation Loop
    leaderboard: dict = {}
    tscv = TimeSeriesSplit(n_splits=3)

    model_configs = {
        "ridge": {
            "estimator": Ridge(),
            "params": {"alpha": [0.1, 1.0, 10.0]},
            "use_scaler": True,
        },
        "random_forest": {
            "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200, 300], "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 1.0],
            },
            "use_scaler": False,
        },
        "xgboost": {
            "estimator": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            "params": {
                "n_estimators": [200, 400, 600], "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0], "min_child_weight": [1, 3, 5],
            },
            "use_scaler": False,
        },
        "lightgbm": {
            "estimator": lgb.LGBMRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 300, 500], "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 7, -1], "num_leaves": [15, 31, 63],
            },
            "use_scaler": False,
        }
    }

    for name, config in model_configs.items():
        logger.info("Training %s ...", name)

        # A. Handle scaling
        scaler = StandardScaler().fit(X_train) if config["use_scaler"] else None
        X_train_final = scaler.transform(X_train) if scaler else X_train
        X_test_final = scaler.transform(X_test) if scaler else X_test

        # B. Tune model
        search = RandomizedSearchCV(
            estimator=config["estimator"],
            param_distributions=config["params"],
            n_iter=15 if name != "ridge" else 3,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train_final, y_train)
        
        # C. Predict and score
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test_final)
        metrics = score(y_test, y_pred)
        
        # D. Store results
        leaderboard[name] = {"model": best_model, "scaler": scaler, "metrics": metrics}

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
