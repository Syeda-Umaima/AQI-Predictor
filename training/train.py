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
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    Keras LSTM on a sliding 72h window.
    Scaler fit on training data only — test data transformed with train scaler.
    """
    import tensorflow as tf
    import numpy as np
    np.random.seed(42)
    tf.random.set_seed(42)
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    feature_scaler = StandardScaler().fit(X_train)
    target_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))

    seq_len = 24  # Fix: 24h lookback for 72h ahead AQI forecasting

    Xtr = feature_scaler.transform(X_train)
    Xte = feature_scaler.transform(X_test)
    ytr_scaled = target_scaler.transform(y_train.values.reshape(-1, 1))

    def to_sequences(X: np.ndarray, y):
        xs, ys = [], []
        y_arr = y.values if hasattr(y, "values") else np.array(y)
        for i in range(seq_len, len(X)):
            xs.append(X[i - seq_len:i])
            ys.append(y_arr[i])
        return np.array(xs), np.array(ys)

    Xtr_seq, ytr_seq_scaled = to_sequences(Xtr, pd.Series(ytr_scaled.ravel()))
    Xte_seq, yte_seq = to_sequences(Xte, y_test.reset_index(drop=True))

    if len(Xtr_seq) < 10:
        raise ValueError("Not enough data to train LSTM (need > seq_len rows in training set).")

    model = models.Sequential([
        layers.Input(shape=(seq_len, Xtr.shape[1])),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        min_lr=1e-5,
        verbose=1,
    )
    
    model.fit(
        Xtr_seq, ytr_seq_scaled,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        verbose=0,
        callbacks=[early_stopping, reduce_lr],
    )
    preds_scaled = model.predict(Xte_seq, verbose=0)
    preds = target_scaler.inverse_transform(preds_scaled).ravel().flatten()
    yte_seq_flat = np.asarray(yte_seq).ravel().flatten()
    return {"model": model, "scaler": feature_scaler, "metrics": score(yte_seq_flat, preds)}


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
        joblib.dump(champ.get("scaler"), MODELS_DIR / "champion_scaler.joblib")
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

    # 5. Define direct-fit models with robust hyperparameters for our small wide time-series dataset
    model_configs = {
        "ridge": {
            "estimator": TransformedTargetRegressor(
                regressor=Ridge(alpha=10.0),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "use_scaler": True,
        },
        "random_forest": {
            "estimator": TransformedTargetRegressor(
                regressor=RandomForestRegressor(
                    n_estimators=150,
                    max_depth=5,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "use_scaler": False,
        },
        "xgboost": {
            "estimator": TransformedTargetRegressor(
                regressor=XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.4,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "use_scaler": False,
        },
        "lightgbm": {
            "estimator": TransformedTargetRegressor(
                regressor=lgb.LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.4,
                    random_state=42,
                    n_jobs=-1,
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "use_scaler": False,
        }
    }

    # 6. Final training on full feature set
    logger.info(
        "Split: %d train rows / %d test rows / %d feature columns.",
        len(train_df), len(test_df), len(cols),
    )
    
    X_train, y_train = train_df[cols], train_df[TARGET]
    X_test, y_test = test_df[cols], test_df[TARGET]

    # 7. Model Training & Evaluation Loop
    leaderboard: dict = {}

    for name, config in model_configs.items():
        logger.info("Training %s ...", name)

        # A. Handle scaling
        scaler = StandardScaler().fit(X_train) if config["use_scaler"] else None
        X_train_final = scaler.transform(X_train) if scaler else X_train
        X_test_final = scaler.transform(X_test) if scaler else X_test

        # B. Direct fit on the full training split
        model = config["estimator"]
        model.fit(X_train_final, y_train)

        # C. Predict and score
        y_pred = model.predict(X_test_final)
        metrics = score(y_test, y_pred)

        # D. Store results
        leaderboard[name] = {"model": model, "scaler": scaler, "metrics": metrics}

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
