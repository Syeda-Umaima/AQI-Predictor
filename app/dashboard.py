"""
Streamlit dashboard for the AQI Predictor.

Features:
- Real-time 72h recursive forecast with dynamic model routing.
- Live accuracy leaderboard based on the most recent actual data point.
- Exploratory Data Analysis (EDA) of atmospheric trends.
- Model diagnostics and explainability (SHAP/LIME).
"""
from __future__ import annotations

import certifi
import io
import os
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
from pymongo import MongoClient

from features.api_client import OpenMeteoClient
from features.feature_engineering import build_feature_frame

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

load_dotenv(dotenv_path=ROOT / ".env", override=False)
MONGO_DB = "aqi_predictor"
MONGO_FEATURE_COLLECTION = "features_v2"
MONGO_METADATA_COLLECTION = "model_metadata"

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

def _render_sidebar_logo() -> None:
    st.sidebar.markdown("### 🌫️ AQI Predictor")

st.set_page_config(
    page_title="AQI Predictor — Hyderabad",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_render_sidebar_logo()
st.sidebar.markdown(f"**City:** {CITY}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Real-Time Forecast",
        "EDA & Analysis",
        "Model Diagnostics & XAI",
        "Historical Overview",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**US AQI Scale**\n"
    + "\n".join(f"- {lo}–{hi}: {lbl}" for lo, hi, lbl, _ in US_AQI_BANDS)
)

def _mongo_client() -> MongoClient:
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGO_URI is required in .env for dashboard connectivity.")
    ca = certifi.where()
    client = MongoClient(uri, tls=True, tlsCAFile=ca, tlsInsecure=True, serverSelectionTimeoutMS=10000)
    return client

def _load_all_models() -> dict[str, dict]:
    """Load all available models from GridFS for dynamic routing."""
    client = _mongo_client()
    db = client[MONGO_DB]
    fs = gridfs.GridFS(db)
    
    # Get all distinct model names from metadata
    distinct_models = db[MONGO_METADATA_COLLECTION].distinct("champion")
    models_dict = {}
    
    for name in distinct_models:
        meta = db[MONGO_METADATA_COLLECTION].find_one({"champion": name}, sort=[("timestamp", -1)])
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

def _load_champion_from_gridfs() -> dict:
    client = _mongo_client()
    db = client[MONGO_DB]
    metadata = db[MONGO_METADATA_COLLECTION].find_one(sort=[("timestamp", -1)])
    if not metadata:
        raise RuntimeError("No champion metadata found.")

    fs = gridfs.GridFS(db)
    raw_model = fs.get(metadata["file_id"]).read()
    artifact = joblib.load(io.BytesIO(raw_model))
    return {
        "champion": metadata["champion"],
        "metrics": metadata["metrics"],
        "feature_columns": metadata["feature_columns"],
        "model": artifact["model"],
        "scaler": artifact.get("scaler"),
        "use_log_y": artifact.get("use_log_y", False),
    }

def _predict_rows(model, scaler, name: str, feature_columns: list[str], X: pd.DataFrame, use_log_y: bool = False) -> pd.Series:
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    if name == "lstm":
        seq_len = model.input_shape[1]
        X_s = scaler.transform(X)
        X_seq = X_s.reshape(X_s.shape[0], 1, X_s.shape[1]).repeat(seq_len, axis=1)
        preds_raw = model.predict(X_seq, verbose=0).ravel()
    elif name == "ridge" and scaler is not None:
        preds_raw = model.predict(scaler.transform(X))
    else:
        preds_raw = model.predict(X)
    preds = np.asarray(preds_raw, dtype=float)
    if use_log_y:
        preds = np.expm1(preds)
    return pd.Series(np.maximum(preds, 0))

def _get_live_accuracy_leaderboard(models: dict[str, dict], latest_data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Calculate live absolute error for all models and return the best one."""
    if latest_data.empty or len(latest_data) < 2:
        return pd.DataFrame(), next(iter(models.keys()))
    
    # Latest actual AQI is at index -1
    actual = latest_data["us_aqi"].iloc[-1]
    # Features are from index -2 (Time t-1)
    features_t_minus_1 = latest_data.iloc[[-2]]
    
    results = []
    for name, m_info in models.items():
        pred = _predict_rows(
            m_info["model"], m_info["scaler"], name, 
            m_info["feature_columns"], features_t_minus_1, m_info["use_log_y"]
        ).iloc[0]
        error = abs(actual - pred)
        results.append({"Model": name, "Live Error": round(error, 2), "Prediction": round(pred, 1)})
    
    df_acc = pd.DataFrame(results).sort_values("Live Error")
    best_model = df_acc["Model"].iloc[0]
    return df_acc, best_model

def _build_recursive_forecast(champion: dict, raw: pd.DataFrame) -> list[dict]:
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    max_lag = max(CONFIG["features"]["lag_hours"])
    history = _load_feature_store()
    prior_aqi = []
    if history is not None and not history.empty and "us_aqi" in history.columns:
        recent_aqi = history.sort_values("timestamp")["us_aqi"].ffill().dropna().tolist()
        if recent_aqi:
            prior_aqi = recent_aqi[-max_lag:]

    if not prior_aqi:
        prior_aqi = [float(raw["us_aqi"].ffill().iloc[0])] * max_lag

    if len(prior_aqi) < max_lag:
        prior_aqi = [prior_aqi[0]] * (max_lag - len(prior_aqi)) + prior_aqi

    state = deque(prior_aqi, maxlen=max_lag)
    forecast_feats = build_feature_frame(raw, include_target=False)
    forecast_feats = forecast_feats.sort_values("timestamp").reset_index(drop=True)

    local_tz = ZoneInfo(CONFIG["city"]["timezone"])
    start_local = datetime.now(tz=local_tz).replace(minute=0, second=0, microsecond=0)
    timestamps = pd.date_range(start_local + timedelta(hours=1), periods=72, freq="H", tz=local_tz)

    forecast = []
    for index in range(min(72, len(forecast_feats))):
        row = forecast_feats.iloc[index].copy()
        for lag in CONFIG["features"]["lag_hours"]:
            col = f"us_aqi_lag_{lag}h"
            if col in row.index:
                row[col] = state[-lag] if lag <= len(state) else state[0]

        X_row = pd.DataFrame([row]).reset_index(drop=True)
        pred_series = _predict_rows(
            champion["model"],
            champion["scaler"],
            champion["champion"],
            champion["feature_columns"],
            X_row,
            use_log_y=champion.get("use_log_y", False),
        )
        predicted_aqi = float(pred_series.iloc[0])
        forecast.append({"timestamp": timestamps[index].isoformat(), "predicted_aqi": predicted_aqi})
        state.append(predicted_aqi)
    return forecast

def _load_forecast_with_routing() -> dict:
    models = _load_all_models()
    history = _load_feature_store()
    
    if history is not None and not history.empty:
        acc_df, best_model_name = _get_live_accuracy_leaderboard(models, history.tail(5))
        st.sidebar.subheader("🎯 Live Model Accuracy")
        st.sidebar.table(acc_df.set_index("Model"))
        st.sidebar.info(f"Dynamic Routing: Using **{best_model_name}**")
        champion = models[best_model_name]
        champion["champion"] = best_model_name
    else:
        champion = _load_champion_from_gridfs()
        best_model_name = champion["champion"]

    client = OpenMeteoClient()
    raw = client.fetch_combined_forecast(forecast_days=6)
    forecast = _build_recursive_forecast(champion, raw)

    return {
        "champion": best_model_name,
        "forecast": forecast,
    }

def _load_leaderboard() -> dict | None:
    client = _mongo_client()
    db = client[MONGO_DB]
    metadata = db[MONGO_METADATA_COLLECTION].find_one(sort=[("timestamp", -1)])
    if not metadata: return None
    return {"champion": metadata["champion"], "metrics": metadata["metrics"], "leaderboard": metadata["leaderboard"]}

def _load_feature_store() -> pd.DataFrame | None:
    try:
        from features.feature_store import load_features
        return load_features()
    except Exception as e:
        st.error(f"Failed to load feature store: {e}")
        return None

def _load_recent_feature_sample(limit: int = 120) -> pd.DataFrame | None:
    try:
        from features.feature_store import load_recent_features
        sample = load_recent_features(hours=limit * 2)
        if sample.empty: return None
        return sample.sort_values("timestamp").tail(limit).reset_index(drop=True)
    except Exception as e:
        st.warning(f"Unable to load recent features: {e}")
        return None

def _predict_for_explainer(champion: dict, X: pd.DataFrame) -> np.ndarray:
    return _predict_rows(champion["model"], champion["scaler"], champion["champion"], champion["feature_columns"], X, champion.get("use_log_y", False)).to_numpy()

def _build_explainability(champion: dict) -> tuple[go.Figure, go.Figure] | tuple[None, None]:
    try:
        import shap
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        st.warning(f"Explainability unavailable: {exc}")
        return None, None

    sample = _load_recent_feature_sample(limit=120)
    if sample is None or sample.empty: return None, None

    X = sample.reindex(columns=champion["feature_columns"], fill_value=0.0).fillna(0)
    X_np = X.to_numpy(dtype=float)

    try:
        model_for_shap = champion["model"]
        if hasattr(model_for_shap, "estimators_"):
            base_estimators = [est for est in getattr(model_for_shap, "estimators_") if hasattr(est, "predict")]
            if base_estimators: model_for_shap = base_estimators[0]

        explainer = shap.Explainer(model_for_shap, X_np, feature_names=champion["feature_columns"])
        shap_values = explainer(X_np)
        shap_mean = np.abs(shap_values.values if not isinstance(shap_values.values, list) else shap_values.values[0]).mean(axis=0)
        shap_df = pd.DataFrame({"feature": champion["feature_columns"], "importance": shap_mean}).sort_values("importance", ascending=False).head(20)
        fig_shap = px.bar(shap_df, x="importance", y="feature", orientation="h", title="Global Feature Importance (SHAP)")
        fig_shap.update_layout(yaxis=dict(autorange="reversed"), height=500)
    except Exception as exc:
        st.warning(f"SHAP failed: {exc}")
        fig_shap = None

    try:
        explainer = LimeTabularExplainer(training_data=X_np, feature_names=champion["feature_columns"], mode="regression")
        instance = X_np[-1]
        lime_exp = explainer.explain_instance(instance, lambda x: _predict_for_explainer(champion, pd.DataFrame(x, columns=champion["feature_columns"])), num_features=10)
        lime_df = pd.DataFrame(lime_exp.as_list(), columns=["Feature", "Weight"])
        fig_lime = px.bar(lime_df, x="Weight", y="Feature", orientation="h", title="Local Explanation (LIME)")
        fig_lime.update_layout(yaxis=dict(autorange="reversed"), height=500)
    except Exception as exc:
        st.warning(f"LIME failed: {exc}")
        fig_lime = None

    return fig_shap, fig_lime

if page == "Real-Time Forecast":
    st.title("🌫️ Real-Time AQI Forecast")
    with st.spinner("Executing dynamic model routing and fetching forecast..."):
        data = _load_forecast_with_routing()

    df = pd.DataFrame(data["forecast"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    current_aqi = float(df["predicted_aqi"].iloc[0])
    max_aqi = float(df["predicted_aqi"].max())
    label, color = _aqi_label_color(current_aqi)

    if max_aqi >= HAZARD_THRESHOLD:
        st.error(f"⚠️ **HAZARDOUS AQI ALERT** — Forecast peak of **{max_aqi:.0f}** exceeds threshold.", icon="🚨")
    elif max_aqi >= 101:
        st.warning(f"⚠️ Forecast AQI peaks at **{max_aqi:.0f}** — unhealthy for sensitive groups.", icon="⚠️")
    else:
        st.success(f"✅ Air quality looks acceptable. Peak: **{max_aqi:.0f}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Predicted AQI", f"{current_aqi:.1f}", label)
    c2.metric("72h Maximum", f"{max_aqi:.1f}")
    c3.metric("72h Average", f"{df['predicted_aqi'].mean():.1f}")
    c4.metric("Active Model", data["champion"].upper())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_aqi"], mode="lines+markers", name="Forecast", line=dict(color="#3b82f6", width=2), fill="tozeroy"))
    for lo, hi, band_label, band_color in US_AQI_BANDS:
        fig.add_hrect(y0=lo, y1=min(hi, max_aqi + 20), fillcolor=band_color, opacity=0.05, annotation_text=band_label, annotation_position="right", line_width=0)
    fig.update_layout(title=f"AQI Forecast — {CITY}", xaxis_title="Time (UTC)", yaxis_title="US AQI", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Data")
    display_df = df.rename(columns={"timestamp": "Time", "predicted_aqi": "AQI"})
    display_df["Category"] = display_df["AQI"].apply(lambda v: _aqi_label_color(v)[0])
    st.dataframe(display_df.set_index("Time"))

elif page == "EDA & Analysis":
    st.title("📊 Exploratory Data Analysis")
    df_fs = _load_feature_store()
    if df_fs is None or df_fs.empty:
        st.warning("Feature store empty.")
        st.stop()
    
    corr_cols = [c for c in ["us_aqi", "pm2_5", "pm10", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"] if c in df_fs.columns]
    if len(corr_cols) > 1:
        st.subheader("Correlation Matrix")
        fig = px.imshow(df_fs[corr_cols].corr(), text_auto=".2f", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        df_fs["hour"] = pd.to_datetime(df_fs["timestamp"]).dt.hour
        fig = px.line(df_fs.groupby("hour")["us_aqi"].mean().reset_index(), x="hour", y="us_aqi", title="Diurnal Cycle", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_fs["month"] = pd.to_datetime(df_fs["timestamp"]).dt.month
        fig = px.line(df_fs.groupby("month")["us_aqi"].mean().reset_index(), x="month", y="us_aqi", title="Monthly Seasonality", markers=True)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Diagnostics & XAI":
    st.title("🧪 Model Diagnostics & Explainability")
    meta = _load_leaderboard()
    if meta:
        st.subheader("Offline Leaderboard")
        st.dataframe(pd.DataFrame(meta["leaderboard"]).T.round(4).sort_values("rmse"))
    
    st.markdown("---")
    st.subheader("Model Interpretation")
    champion = _load_champion_from_gridfs()
    fig_shap, fig_lime = _build_explainability(champion)
    if fig_shap: st.plotly_chart(fig_shap, use_container_width=True)
    if fig_lime: st.plotly_chart(fig_lime, use_container_width=True)

else:
    st.title("📅 Historical Data")
    df_fs = _load_feature_store()
    if df_fs is not None:
        df_fs["timestamp"] = pd.to_datetime(df_fs["timestamp"], utc=True)
        fig = px.line(df_fs.sort_values("timestamp"), x="timestamp", y="us_aqi", title="Historical US AQI")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Feature Store Sample")
        st.dataframe(df_fs.tail(100))
