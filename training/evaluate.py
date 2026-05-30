"""
Champion model evaluation — SHAP and LIME explainability with Cloud Persistence.
"""
from __future__ import annotations

import json
import logging
import io
import datetime
from pathlib import Path

import gridfs
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from training.train import load_training_frame, time_ordered_split
from features.mongo_utils import get_database, mongo_retry

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MONGO_DB_NAME = "aqi_predictor"
MODEL_METADATA_COLLECTION = "model_metadata"

@mongo_retry()
def load_latest_champion():
    """Load latest champion artifact from MongoDB GridFS."""
    db = get_database(MONGO_DB_NAME)
    meta = db[MODEL_METADATA_COLLECTION].find_one({"type": "latest_champion"})
    if not meta:
        raise FileNotFoundError("No champion metadata found in Cloud. Train models first.")
    
    fs = gridfs.GridFS(db)
    raw_artifact = fs.get(meta["file_id"]).read()
    artifact = joblib.load(io.BytesIO(raw_artifact))
    
    return meta["champion"], artifact["model"], artifact.get("scaler"), meta

def _build_predict_fn(name: str, model, scaler, cols: list[str], use_log_y: bool = False):
    def predict_fn(arr: np.ndarray) -> np.ndarray:
        df_tmp = pd.DataFrame(arr, columns=cols)
        if scaler:
            arr_final = scaler.transform(df_tmp)
        else:
            arr_final = df_tmp
        
        preds = model.predict(arr_final)
        if use_log_y:
            preds = np.expm1(preds)
        return preds
    return predict_fn

@mongo_retry()
def persist_xai_artifact(buf: io.BytesIO, filename: str, run_id: str):
    """Save XAI plots to GridFS."""
    db = get_database(MONGO_DB_NAME)
    fs = gridfs.GridFS(db)
    fs.put(buf.getvalue(), filename=filename, run_id=run_id, type="xai_plot")

def run_shap(name: str, model, scaler, meta: dict, X_test: pd.DataFrame, run_id: str) -> None:
    cols = meta["feature_columns"]
    X_test_sub = X_test[cols].copy()
    use_log_y = meta.get("use_log_y", False)

    try:
        # Use a background sample for KernelExplainer if TreeExplainer fails or not applicable
        predict_fn = _build_predict_fn(name, model, scaler, cols, use_log_y)
        background = shap.sample(X_test_sub, min(100, len(X_test_sub)))
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_test_sub.iloc[:50], nsamples=100)

        # Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_sub.iloc[:50], show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        persist_xai_artifact(buf, "shap_summary.png", run_id)
        plt.close()

        logger.info("SHAP artifacts persisted to Cloud.")
    except Exception as exc:
        logger.warning(f"SHAP failed: {exc}")

def run_lime(name: str, model, scaler, meta: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, run_id: str) -> None:
    cols = meta["feature_columns"]
    use_log_y = meta.get("use_log_y", False)
    
    try:
        predict_fn = _build_predict_fn(name, model, scaler, cols, use_log_y)
        
        # Robust LIME initialization
        # Ensure training data passed to LIME has positive variance for scaling
        training_data = X_train[cols].values
        
        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=cols,
            mode="regression"
        )
        
        # Explain the most recent instance
        instance = X_test[cols].iloc[-1].values
        exp = explainer.explain_instance(instance, predict_fn, num_features=10)
        
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME Explanation: {name}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        persist_xai_artifact(buf, "lime_explanation.png", run_id)
        plt.close()
        
        logger.info("LIME artifacts persisted to Cloud.")
    except Exception as exc:
        logger.warning(f"LIME failed: {exc}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        name, model, scaler, meta = load_latest_champion()
        run_id = meta["run_id"]
        cols = meta["feature_columns"]
        
        df = load_training_frame()
        if df.empty: return
        
        train_df, test_df = time_ordered_split(df.dropna(subset=cols), 0.2)
        
        run_shap(name, model, scaler, meta, test_df, run_id)
        run_lime(name, model, scaler, meta, train_df, test_df, run_id)
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")

if __name__ == "__main__":
    main()
