"""
Multi-model training pipeline.

Trains Ridge, Random Forest, XGBoost, and a Keras LSTM on the engineered
feature store, scores them on a time-ordered holdout, and promotes the
lowest-RMSE model as the champion in the local Hopsworks Model Registry
(or a local `models/` folder when Hopsworks is unavailable).
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
    df = df.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * (1 - test_size))
    return df.iloc[:split], df.iloc[split:]


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"timestamp", TARGET}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


# ------------------------------------------------------------------- Metrics
def score(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ------------------------------------------------------------------- Models
def train_classical_models(X_train, y_train, X_test, y_test) -> dict:
    results = {}
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "xgboost": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
        ),
    }
    for name, model in candidates.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {"model": model, "metrics": score(y_test, preds)}
    return results


def train_lstm(X_train, y_train, X_test, y_test) -> dict:
    """Sequence model on a sliding window of recent feature rows."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    cfg = _cfg()["training"]["lstm"]
    seq_len = cfg["sequence_length"]

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    def to_sequences(X, y):
        xs, ys = [], []
        for i in range(seq_len, len(X)):
            xs.append(X[i - seq_len:i])
            ys.append(y.iloc[i] if hasattr(y, "iloc") else y[i])
        return np.array(xs), np.array(ys)

    Xtr_seq, ytr_seq = to_sequences(Xtr, y_train.reset_index(drop=True))
    Xte_seq, yte_seq = to_sequences(Xte, y_test.reset_index(drop=True))

    model = models.Sequential([
        layers.Input(shape=(seq_len, Xtr.shape[1])),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr_seq, ytr_seq, epochs=cfg["epochs"], batch_size=cfg["batch_size"], verbose=0)
    preds = model.predict(Xte_seq, verbose=0).ravel()
    return {
        "model": model,
        "scaler": scaler,
        "metrics": score(yte_seq, preds),
    }


# ---------------------------------------------------------------- Champion
def promote_champion(leaderboard: dict, X_test_cols: list[str]) -> str:
    champ_name = min(leaderboard, key=lambda k: leaderboard[k]["metrics"]["rmse"])
    champ = leaderboard[champ_name]
    logger.info("Champion: %s  (RMSE=%.4f)", champ_name, champ["metrics"]["rmse"])

    # Persist locally
    if champ_name == "lstm":
        champ["model"].save(MODELS_DIR / "champion_lstm.keras")
        joblib.dump(champ["scaler"], MODELS_DIR / "champion_scaler.joblib")
    else:
        joblib.dump(champ["model"], MODELS_DIR / "champion.joblib")

    meta = {
        "champion": champ_name,
        "metrics": champ["metrics"],
        "feature_columns": X_test_cols,
        "leaderboard": {k: v["metrics"] for k, v in leaderboard.items()},
    }
    (ARTIFACTS / "leaderboard.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Best-effort push to local Hopsworks Model Registry
    try:
        import hopsworks  # type: ignore
        cfg = _cfg()["hopsworks"]
        project = hopsworks.login(host=cfg["host"], project=cfg["project"],
                                  api_key_value=cfg["api_key_value"])
        mr = project.get_model_registry()
        model_dir = MODELS_DIR
        registered = mr.python.create_model(
            name=cfg["model_registry_name"],
            metrics=champ["metrics"],
            description=f"Champion AQI model ({champ_name})",
        )
        registered.save(str(model_dir))
        logger.info("Champion registered in local Hopsworks Model Registry.")
    except Exception as e:
        logger.warning("Model Registry unavailable (%s) — kept local copy in %s.", e, MODELS_DIR)

    return champ_name


# -------------------------------------------------------------------- Driver
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _cfg()["training"]
    df = load_training_frame()
    train_df, test_df = time_ordered_split(df, cfg["test_size"])
    cols = feature_columns(df)
    X_train, y_train = train_df[cols], train_df[TARGET]
    X_test, y_test = test_df[cols], test_df[TARGET]

    leaderboard = train_classical_models(X_train, y_train, X_test, y_test)
    try:
        leaderboard["lstm"] = train_lstm(X_train, y_train, X_test, y_test)
    except Exception as e:
        logger.warning("LSTM training skipped: %s", e)

    print("\n=== Leaderboard ===")
    print(pd.DataFrame({k: v["metrics"] for k, v in leaderboard.items()}).T.round(4))

    promote_champion(leaderboard, cols)


if __name__ == "__main__":
    main()
