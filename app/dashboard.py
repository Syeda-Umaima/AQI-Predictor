from __future__ import annotations

"""
Streamlit dashboard for the AQI Predictor.
High-Performance Cloud Edition: Caching, Resilience, and Real-Time MLOps.
"""

import os
import sys

# Force Python to recognize the root project directory for Streamlit Cloud deployment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# --- Config & Initialization ---
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

# --- Cached Cloud Data Access ---

@st.cache_resource(ttl=900)
def _load_cloud_registry():
    """Load all distinct models and the latest champion metadata from Cloud."""
    try:
        db = get_database(MONGO_DB_NAME)
        # Fetch latest champion metadata
        latest_meta = db[MODEL_METADATA_COLLECTION].find_one({"type": "latest_champion"})
        # Fetch all distinct historical models for dynamic routing
        distinct_models = db[MODEL_METADATA_COLLECTION].distinct("champion")
        
        models_dict = {}
        fs = gridfs.GridFS(db)
        
        for name in distinct_models:
            m_meta = db[MODEL_METADATA_COLLECTION].find_one(
                {"champion": name, "type": "history"}, 
                sort=[("timestamp", -1)]
            )
            if m_meta and "file_id" in m_meta:
                try:
                    raw = fs.get(m_meta["file_id"]).read()
                    artifact = joblib.load(io.BytesIO(raw))
                    models_dict[name] = {
                        "model": artifact["model"],
                        "scaler": artifact.get("scaler"),
                        "use_log_y": artifact.get("use_log_y", False),
                        "feature_columns": m_meta["feature_columns"],
                        "metrics": m_meta.get("metrics", {})
                    }
                except Exception:
                    continue
        
        return {"latest_meta": latest_meta, "models_dict": models_dict}
    except Exception as e:
        st.error(f"Cloud Registry Error: {e}")
        return {"latest_meta": None, "models_dict": {}}

@st.cache_data(ttl=900)
def _load_feature_store_cached():
    """Load historical features with fail-fast logic."""
    try:
        from features.feature_store import load_features
        return load_features()
    except Exception as e:
        st.error(f"Feature Store Offline: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=900)
def _fetch_xai_image(run_id: str, filename: str):
    """Securely fetch XAI plots from GridFS with verification."""
    if not run_id or run_id == "Unknown Run":
        return None
    try:
        db = get_database(MONGO_DB_NAME)
        fs = gridfs.GridFS(db)
        file_doc = db["fs.files"].find_one({"run_id": run_id, "filename": filename})
        if file_doc:
            return fs.get(file_doc["_id"]).read()
    except Exception:
        pass
    return None

# --- Prediction & Logic ---

def _predict_rows(model, scaler, name: str, feature_columns: list[str], X: pd.DataFrame, use_log_y: bool = False) -> pd.Series:
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    if name == "ridge" and scaler is not None:
        preds_raw = model.predict(scaler.transform(X))
    else:
        preds_raw = model.predict(X)
    
    preds = np.asarray(preds_raw, dtype=float)
    if use_log_y:
        preds = np.expm1(preds)
    return pd.Series(np.maximum(preds, 0))

def _calculate_live_metrics(models: dict[str, dict], history: pd.DataFrame):
    """Calculate realistic drift/accuracy using last 24h data."""
    if not models or history.empty or len(history) < 25:
        return pd.DataFrame(), "N/A"
    
    # Last 24 hours of actual data
    recent = history.sort_values("timestamp").tail(25).reset_index(drop=True)
    results = []
    
    for name, m_info in models.items():
        # Predict on T-1 to compare with T
        try:
            X_batch = recent.iloc[:-1] # Features
            y_actual = recent["us_aqi"].iloc[1:].values # Actual targets
            
            y_pred = _predict_rows(
                m_info["model"], m_info["scaler"], name, 
                m_info["feature_columns"], X_batch, m_info["use_log_y"]
            ).values
            
            mae = np.mean(np.abs(y_actual - y_pred))
            results.append({"Model": name, "Live MAE": mae, "Baseline Test MAE": m_info["metrics"].get("mae", 0)})
        except Exception:
            continue
            
    df_acc = pd.DataFrame(results).sort_values("Live MAE")
    best_model = df_acc["Model"].iloc[0] if not df_acc.empty else "N/A"
    return df_acc, best_model

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
        pred = _predict_rows(
            champion["model"], champion["scaler"], champion["champion"], 
            champion["feature_columns"], X_row, champion.get("use_log_y", False)
        ).iloc[0]
        forecast.append({"timestamp": timestamps[index].isoformat(), "predicted_aqi": pred})
        state.append(pred)
    return forecast

# --- UI Setup ---
st.set_page_config(page_title="AQI Predictor — Hyderabad", page_icon="🌫️", layout="wide")

# --- Sidebar ---
st.sidebar.markdown("### 🌫️ AQI Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("**US AQI Scale Guide**")
st.sidebar.markdown(
    """
    🟢 **0-50:** Good  
    🟡 **51-100:** Moderate  
    🟠 **101-150:** Sensitive  
    🔴 **151-200:** Unhealthy  
    🟣 **201-300:** Very Unhealthy  
    🟤 **301+:** Hazardous  
    """
)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", ["Real-Time Forecast", "EDA & Analysis", "Model Diagnostics & XAI", "Historical Overview"])

# --- App Execution ---
registry = _load_cloud_registry()
models = registry["models_dict"]
history = _load_feature_store_cached()

if not models or history.empty:
    st.error("Cloud System Synchronizing. Please ensure backfill and training pipelines have completed.")
    st.stop()

# Dynamic Routing & Metrics
acc_df, best_model_name = _calculate_live_metrics(models, history)
st.sidebar.subheader("🎯 Live Model Performance")
if not acc_df.empty:
    st.sidebar.table(acc_df.set_index("Model").style.format("{:.2f}"))
st.sidebar.info(f"Dynamic Routing: Using **{best_model_name}**")

# Champion model for forecast
champion = models[best_model_name]
champion["champion"] = best_model_name

if page == "Real-Time Forecast":
    st.title("🌫️ Real-Time AQI Forecast")
    st.caption("Note: Predictions are based on regional satellite data (Open-Meteo) and may slightly differ from hyper-local ground sensors.")
    
    with st.spinner("Generating 72-hour recursive forecast..."):
        client = OpenMeteoClient()
        raw_forecast = client.fetch_combined_forecast(forecast_days=6)
        forecast_data = _build_recursive_forecast(champion, raw_forecast, history)
        
    df = pd.DataFrame(forecast_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    current_aqi = df["predicted_aqi"].iloc[0]
    max_aqi = df["predicted_aqi"].max()
    label, color = _aqi_label_color(current_aqi)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Prediction", f"{current_aqi:.1f}", label)
    c2.metric("72h Max Forecast", f"{max_aqi:.1f}")
    c3.metric("72h Average", f"{df['predicted_aqi'].mean():.1f}")
    c4.metric("Active Model", best_model_name.upper())
    
    if max_aqi >= HAZARD_THRESHOLD:
        st.error(f"⚠️ **HAZARDOUS AQI ALERT** — Forecast peak of **{max_aqi:.0f}** exceeds threshold.", icon="🚨")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_aqi"], mode="lines+markers", name="Forecast", line=dict(color="#3b82f6", width=2), fill="tozeroy"))
    for lo, hi, band_label, band_color in US_AQI_BANDS:
        fig.add_hrect(y0=lo, y1=min(hi, max_aqi + 20), fillcolor=band_color, opacity=0.05, annotation_text=band_label, annotation_position="right", line_width=0)
    fig.update_layout(title=f"72-Hour AQI Forecast — {CITY}", xaxis_title="Time (Local)", yaxis_title="US AQI", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Forecast Detail Table")
    display_df = df.copy().rename(columns={"timestamp": "Time", "predicted_aqi": "AQI"})
    display_df["Category"] = display_df["AQI"].apply(lambda v: _aqi_label_color(v)[0])
    st.dataframe(display_df.set_index("Time"), use_container_width=True)

elif page == "EDA & Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("Correlation Heatmap")
    corr_cols = [c for c in ["us_aqi", "pm2_5", "pm10", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"] if c in history.columns]
    fig_corr = px.imshow(history[corr_cols].corr(), text_auto=".2f", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Diurnal AQI Cycle")
        history["hour"] = history["timestamp"].dt.hour
        fig_hour = px.box(history, x="hour", y="us_aqi", title="AQI by Hour of Day")
        st.plotly_chart(fig_hour, use_container_width=True)
    with col2:
        st.subheader("Seasonal Trends")
        history["month"] = history["timestamp"].dt.month
        fig_month = px.box(history, x="month", y="us_aqi", title="AQI by Month")
        st.plotly_chart(fig_month, use_container_width=True)

elif page == "Model Diagnostics & XAI":
    st.title("🧪 Model Diagnostics & Explainability")
    latest_meta = registry["latest_meta"]
    if not latest_meta:
        st.info("Visual explanation currently synchronizing.")
    else:
        st.subheader("🏆 Model Performance Leaderboard (Test Metrics)")
        lb_df = pd.DataFrame(latest_meta["leaderboard"]).T.sort_values("rmse")
        st.dataframe(lb_df.style.format("{:.4f}"), use_container_width=True)
        
        st.success(f"🏆 **Champion Model:** {latest_meta['champion'].upper()}")
        
        st.markdown("---")
        st.subheader("Global & Local Interpretation")
        run_id = latest_meta.get("run_id", "Unknown Run")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**SHAP Global Importance**")
            shap_bytes = _fetch_xai_image(run_id, "shap_summary.png")
            if shap_bytes:
                st.image(shap_bytes, use_container_width=True)
            else:
                st.info("SHAP plot not found for this training run. They will generate on the next training cycle.")
        
        with col2:
            st.markdown("**LIME Local Explanation**")
            lime_bytes = _fetch_xai_image(run_id, "lime_explanation.png")
            if lime_bytes:
                st.image(lime_bytes, use_container_width=True)
            else:
                st.info("LIME plot not found for this training run. They will generate on the next training cycle.")

else:
    st.title("📅 Historical Overview")
    st.subheader("Historical US AQI Trend")
    fig_hist = px.line(history, x="timestamp", y="us_aqi", title="Feature Store Timeseries")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.subheader("Raw Feature Data (Latest 100 Rows)")
    st.dataframe(history.sort_values("timestamp", ascending=False).head(100), use_container_width=True)
