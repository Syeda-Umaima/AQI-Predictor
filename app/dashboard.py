"""
Streamlit dashboard for the AQI Predictor.
Optimized for Cloud Deployment with Caching and Resilience.
"""
from __future__ import annotations

import io
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import gridfs
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from dotenv import load_dotenv

from features.api_client import OpenMeteoClient
from features.feature_engineering import build_feature_frame
from features.mongo_utils import get_database, mongo_retry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

load_dotenv(dotenv_path=ROOT / ".env", override=False)
MONGO_DB_NAME = "aqi_predictor"
MODEL_METADATA_COLLECTION = "model_metadata"

CITY = CONFIG["city"]["name"]
HAZARD_THRESHOLD = CONFIG["dashboard"]["hazardous_aqi_threshold"]

US_AQI_BANDS = [
    (0, 50, "Good", "#22c55e"),
    (51, 100, "Moderate", "#eab308"),
    (101, 150, "Unhealthy for Sensitive Groups", "#f97316"),
    (151, 200, "Unhealthy", "#ef4444"),
    (201, 300, "Very Unhealthy", "#a855f7"),
    (301, 500, "Hazardous", "#7f1d1d"),
]

def _aqi_label_color(aqi: float) -> tuple[str, str]:
    for lo, hi, label, color in US_AQI_BANDS:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "#7f1d1d"

@st.cache_resource(ttl=600)
def _load_all_models_cached():
    """Load all distinct models from Cloud GridFS."""
    db = get_database(MONGO_DB_NAME)
    fs = gridfs.GridFS(db)
    
    distinct_models = db[MODEL_METADATA_COLLECTION].distinct("champion")
    models_dict = {}
    
    for name in distinct_models:
        meta = db[MODEL_METADATA_COLLECTION].find_one({"champion": name, "type": "history"}, sort=[("timestamp", -1)])
        if meta and "file_id" in meta:
            raw = fs.get(meta["file_id"]).read()
            artifact = joblib.load(io.BytesIO(raw))
            models_dict[name] = {
                "model": artifact["model"],
                "scaler": artifact.get("scaler"),
                "use_log_y": artifact.get("use_log_y", False),
                "feature_columns": meta["feature_columns"]
            }
    return models_dict

@st.cache_resource(ttl=600)
def _load_champion_cached():
    """Load latest champion from Cloud."""
    db = get_database(MONGO_DB_NAME)
    meta = db[MODEL_METADATA_COLLECTION].find_one({"type": "latest_champion"})
    if not meta:
        return None

    fs = gridfs.GridFS(db)
    raw_model = fs.get(meta["file_id"]).read()
    artifact = joblib.load(io.BytesIO(raw_model))
    return {
        "champion": meta["champion"],
        "metrics": meta["metrics"],
        "feature_columns": meta["feature_columns"],
        "model": artifact["model"],
        "scaler": artifact.get("scaler"),
        "use_log_y": artifact.get("use_log_y", False),
    }

@st.cache_data(ttl=600)
def _load_features_cached():
    from features.feature_store import load_features
    return load_features()

def _predict_rows(model, scaler, name: str, feature_columns: list[str], X: pd.DataFrame, use_log_y: bool = False) -> pd.Series:
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    # Basic prediction logic (handling scalers and log transforms)
    if name == "ridge" and scaler is not None:
        preds_raw = model.predict(scaler.transform(X))
    else:
        preds_raw = model.predict(X)
    
    preds = np.asarray(preds_raw, dtype=float)
    if use_log_y:
        preds = np.expm1(preds)
    return pd.Series(np.maximum(preds, 0))

def _get_live_accuracy_leaderboard(models: dict[str, dict], latest_data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if latest_data.empty or len(latest_data) < 2:
        return pd.DataFrame(), next(iter(models.keys()))
    
    actual = latest_data["us_aqi"].iloc[-1]
    features_t_minus_1 = latest_data.iloc[[-2]]
    
    results = []
    for name, m_info in models.items():
        pred = _predict_rows(m_info["model"], m_info["scaler"], name, m_info["feature_columns"], features_t_minus_1, m_info["use_log_y"]).iloc[0]
        results.append({"Model": name, "Live Error": abs(actual - pred), "Prediction": pred})
    
    df_acc = pd.DataFrame(results).sort_values("Live Error")
    return df_acc, df_acc["Model"].iloc[0]

def _build_recursive_forecast(champion: dict, raw: pd.DataFrame, history: pd.DataFrame) -> list[dict]:
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    max_lag = max(CONFIG["features"]["lag_hours"])
    prior_aqi = []
    if not history.empty and "us_aqi" in history.columns:
        recent_aqi = history.sort_values("timestamp")["us_aqi"].ffill().dropna().tolist()
        if recent_aqi:
            prior_aqi = recent_aqi[-max_lag:]

    if not prior_aqi:
        prior_aqi = [float(raw["us_aqi"].ffill().iloc[0])] * max_lag
    if len(prior_aqi) < max_lag:
        prior_aqi = [prior_aqi[0]] * (max_lag - len(prior_aqi)) + prior_aqi

    state = deque(prior_aqi, maxlen=max_lag)
    forecast_feats = build_feature_frame(raw, include_target=False)
    
    local_tz = ZoneInfo(CONFIG["city"]["timezone"])
    start_local = datetime.now(tz=local_tz).replace(minute=0, second=0, microsecond=0)
    timestamps = pd.date_range(start_local + timedelta(hours=1), periods=72, freq="H", tz=local_tz)

    forecast = []
    for index in range(min(72, len(forecast_feats))):
        row = forecast_feats.iloc[index].copy()
        for lag in CONFIG["features"]["lag_hours"]:
            row[f"us_aqi_lag_{lag}h"] = state[-lag]

        X_row = pd.DataFrame([row])
        pred = _predict_rows(champion["model"], champion["scaler"], champion["champion"], champion["feature_columns"], X_row, champion.get("use_log_y", False)).iloc[0]
        forecast.append({"timestamp": timestamps[index].isoformat(), "predicted_aqi": pred})
        state.append(pred)
    return forecast

st.set_page_config(page_title="AQI Predictor — Hyderabad", page_icon="🌫️", layout="wide")
st.sidebar.markdown("### 🌫️ AQI Predictor")

page = st.sidebar.radio("Navigate", ["Real-Time Forecast", "EDA & Analysis", "Model Diagnostics & XAI", "Historical Overview"])

if page == "Real-Time Forecast":
    st.title("🌫️ Real-Time AQI Forecast")
    try:
        models = _load_all_models_cached()
        history = _load_features_cached()
        
        if not models:
            st.warning("Model registry is empty. Run training pipeline first.")
            st.stop()
            
        acc_df, best_model_name = _get_live_accuracy_leaderboard(models, history.tail(5))
        st.sidebar.subheader("🎯 Live Model Accuracy")
        st.sidebar.table(acc_df[["Model", "Live Error"]].set_index("Model"))
        
        champion = models[best_model_name]
        champion["champion"] = best_model_name
        
        with st.spinner("Generating 72h forecast..."):
            client = OpenMeteoClient()
            raw_forecast = client.fetch_combined_forecast(forecast_days=6)
            forecast_data = _build_recursive_forecast(champion, raw_forecast, history)
            
        df = pd.DataFrame(forecast_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        current_aqi = df["predicted_aqi"].iloc[0]
        max_aqi = df["predicted_aqi"].max()
        label, color = _aqi_label_color(current_aqi)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Live Predicted AQI", f"{current_aqi:.1f}", label)
        c2.metric("72h Max Forecast", f"{max_aqi:.1f}")
        c3.metric("Active Model", best_model_name.upper())
        
        fig = px.line(df, x="timestamp", y="predicted_aqi", title=f"72-Hour Forecast — {CITY}")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Forecast engine offline: {e}")

elif page == "EDA & Analysis":
    st.title("📊 Exploratory Data Analysis")
    df = _load_features_cached()
    if df.empty:
        st.info("Feature store is synchronizing from cloud...")
    else:
        st.subheader("Diurnal Trends")
        df["hour"] = df["timestamp"].dt.hour
        fig = px.box(df, x="hour", y="us_aqi", title="AQI Distribution by Hour")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Diagnostics & XAI":
    st.title("🧪 Model Diagnostics")
    try:
        db = get_database(MONGO_DB_NAME)
        fs = gridfs.GridFS(db)
        meta = db[MODEL_METADATA_COLLECTION].find_one({"type": "latest_champion"})
        
        if not meta:
            st.info("Model diagnostics are currently synchronizing from the cloud...")
        else:
            st.subheader("Global Feature Importance (SHAP)")
            # Try to fetch SHAP plot from GridFS
            shap_file = db["fs.files"].find_one({"run_id": meta["run_id"], "filename": "shap_summary.png"})
            if shap_file:
                st.image(fs.get(shap_file["_id"]).read())
            else:
                st.info("Generating SHAP interpretations...")
    except Exception as e:
        st.error(f"XAI Module Error: {e}")

else:
    st.title("📅 Historical Overview")
    df = _load_features_cached()
    if not df.empty:
        st.line_chart(df.set_index("timestamp")["us_aqi"])
        st.dataframe(df.tail(100))
