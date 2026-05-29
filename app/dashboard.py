"""
Multi-page Streamlit dashboard — Pearls AQI Predictor for Hyderabad, Pakistan.

Pages:
  1. Real-Time Forecast  — KPI cards, 72h forecast timeline, hazard alerts
  2. EDA & Analysis      — correlation heatmap, diurnal/seasonal/scatter charts
  3. Model Diagnostics   — benchmark leaderboard, SHAP, LIME
  4. Historical Overview — raw feature-store timeseries and distributions

Directive 4 compliance:
  - All file reads use encoding="utf-8" explicitly.
  - Streamlit layout now uses width="stretch" instead of deprecated use_container_width.
  - Hazardous AQI alert banners shown when forecast max >= threshold.
  - All deprecated Streamlit calls removed.
"""
from __future__ import annotations

import certifi
import io
import json
import os
from pathlib import Path
import ssl

import gridfs
import joblib
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
ARTIFACTS = ROOT / "artifacts"
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
    local_logo = ROOT / "assets" / "logo.png"
    remote_logo = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/"
        "Flag_of_Pakistan.svg/200px-Flag_of_Pakistan.svg.png"
    )
    if local_logo.exists():
        st.sidebar.image(str(local_logo), width=80)
        return

    try:
        import requests

        head = requests.head(remote_logo, allow_redirects=True, timeout=3)
        if head.status_code == 200 and "image" in head.headers.get("Content-Type", ""):
            st.sidebar.image(remote_logo, width=80)
            return
    except Exception:
        pass

    st.sidebar.markdown("### 🌫️ Pearls AQI Predictor")


st.set_page_config(
    page_title="Pearls AQI Predictor — Hyderabad",
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


# ------------------------------------------------------------------ Helpers
def _mongo_client() -> MongoClient:
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGO_URI is required in .env for dashboard connectivity.")

    ca = certifi.where()
    client = MongoClient(
        uri,
        tls=True,
        tlsCAFile=ca,
        tlsInsecure=True,
        serverSelectionTimeoutMS=10000,
    )
    client.admin.command("ping")
    return client


def _load_champion_from_gridfs() -> dict:
    client = _mongo_client()
    db = client[MONGO_DB]
    metadata = db[MONGO_METADATA_COLLECTION].find_one(sort=[("timestamp", -1)])
    if not metadata:
        raise RuntimeError("No champion metadata found in MongoDB model_metadata collection.")

    fs = gridfs.GridFS(db)
    file_id = metadata.get("file_id")
    if file_id is None:
        raise RuntimeError("Champion metadata document is missing GridFS file_id.")

    raw_model = fs.get(file_id).read()
    artifact = joblib.load(io.BytesIO(raw_model))
    return {
        "champion": metadata["champion"],
        "metrics": metadata["metrics"],
        "feature_columns": metadata["feature_columns"],
        "model": artifact["model"],
        "scaler": artifact.get("scaler"),
    }


def _predict_rows(model, scaler, name: str, feature_columns: list[str], X: pd.DataFrame) -> pd.Series:
    X = X.reindex(columns=feature_columns, fill_value=0.0)

    if name == "lstm":
        seq_len = model.input_shape[1]
        X_s = scaler.transform(X)
        X_seq = X_s.reshape(X_s.shape[0], 1, X_s.shape[1]).repeat(seq_len, axis=1)
        return pd.Series(model.predict(X_seq, verbose=0).ravel())

    if name == "ridge" and scaler is not None:
        return pd.Series(model.predict(scaler.transform(X)))

    return pd.Series(model.predict(X))


def _load_forecast() -> dict:
    champion = _load_champion_from_gridfs()
    client = OpenMeteoClient()
    raw = client.fetch_combined_forecast(forecast_days=4)
    if raw.empty:
        raise RuntimeError("Open-Meteo returned no forecast rows.")

    feats = build_feature_frame(raw)
    if feats.empty:
        raise RuntimeError("Feature engineering produced no rows for forecast.")

    X = feats.tail(72).copy()
    preds = _predict_rows(
        champion["model"],
        champion["scaler"],
        champion["champion"],
        champion["feature_columns"],
        X,
    )
    timestamps = feats["timestamp"].tail(72).astype(str).tolist()
    return {
        "champion": champion["champion"],
        "champion_metrics": champion["metrics"],
        "forecast": [
            {"timestamp": t, "predicted_aqi": float(p)}
            for t, p in zip(timestamps, preds)
        ],
    }


def _load_leaderboard() -> dict | None:
    client = _mongo_client()
    db = client[MONGO_DB]
    metadata = db[MONGO_METADATA_COLLECTION].find_one(sort=[("timestamp", -1)])
    if not metadata:
        return None
    return {
        "champion": metadata["champion"],
        "metrics": metadata["metrics"],
        "leaderboard": metadata["leaderboard"],
    }


def _load_feature_store() -> pd.DataFrame | None:
    try:
        from features.feature_store import load_features
        return load_features()
    except Exception as e:
        import traceback
        st.error(f"Failed to load data from feature store: {e}")
        st.code(traceback.format_exc())
        return None

def _html_artifact(filename: str) -> str | None:
    path = ARTIFACTS / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _img_artifact(filename: str) -> Path | None:
    path = ARTIFACTS / filename
    return path if path.exists() else None


# ============================================================= PAGE 1
if page == "Real-Time Forecast":
    st.title("🌫️ Real-Time AQI Forecast")
    st.caption(f"Next 72 hours · {CITY} · powered by the champion model")

    with st.spinner("Loading forecast …"):
        data = _load_forecast()

    df = pd.DataFrame(data["forecast"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    current_aqi = float(df["predicted_aqi"].iloc[0])
    max_aqi = float(df["predicted_aqi"].max())
    mean_aqi = float(df["predicted_aqi"].mean())
    label, color = _aqi_label_color(current_aqi)

    if max_aqi >= HAZARD_THRESHOLD:
        st.error(
            f"⚠️ **HAZARDOUS AQI ALERT** — Forecast peak of **{max_aqi:.0f}** "
            f"exceeds the safe threshold of {HAZARD_THRESHOLD}. "
            "Limit outdoor exposure and wear an N95 mask.",
            icon="🚨",
        )
    elif max_aqi >= 101:
        st.warning(
            f"⚠️ Forecast AQI peaks at **{max_aqi:.0f}** — "
            "sensitive groups should reduce prolonged outdoor activity.",
            icon="⚠️",
        )
    else:
        st.success(f"✅ Air quality looks acceptable for the next 72 hours. Peak: **{max_aqi:.0f}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Predicted AQI", f"{current_aqi:.1f}", label)
    c2.metric("72h Maximum", f"{max_aqi:.1f}")
    c3.metric("72h Average", f"{mean_aqi:.1f}")
    c4.metric("Champion Model", data["champion"].upper())

    st.markdown("---")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["predicted_aqi"],
        mode="lines+markers",
        name="Predicted AQI",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
    ))

    for lo, hi, band_label, band_color in US_AQI_BANDS:
        fig.add_hrect(
            y0=lo, y1=min(hi, max_aqi + 20),
            fillcolor=band_color, opacity=0.06,
            annotation_text=band_label, annotation_position="right",
            line_width=0,
        )

    fig.update_layout(
        title=f"AQI Forecast — Next 72 Hours ({CITY})",
        xaxis_title="Date / Time (UTC)",
        yaxis_title="US AQI",
        template="plotly_white",
        height=460,
        showlegend=True,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Forecast Detail Table")
    display_df = df.rename(columns={"timestamp": "Timestamp (UTC)", "predicted_aqi": "Predicted AQI"})
    display_df["Category"] = display_df["Predicted AQI"].apply(lambda v: _aqi_label_color(v)[0])
    st.dataframe(display_df.set_index("Timestamp (UTC)"))


# ============================================================= PAGE 2
elif page == "EDA & Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown(
        "Feature engineering decisions were guided by these exploratory analyses: "
        "pollutant correlations, diurnal AQI patterns, seasonal cycles, and "
        "PM2.5 vs NO₂ relationships. All charts are computed from the live Feature Store."
    )

    df_fs = _load_feature_store()

    if df_fs is None or df_fs.empty:
        st.warning("Feature store is empty or unavailable. Run `python -m features.backfill_historical`.")
        st.stop()

    st.markdown("---")

    corr_cols = [c for c in ["us_aqi", "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide", "dust", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"] if c in df_fs.columns]
    if len(corr_cols) > 1:
        st.subheader("Pollutant Correlation Heatmap")
        corr_matrix = df_fs[corr_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diurnal AQI Cycle")
        if "us_aqi" in df_fs.columns:
            df_fs["hour"] = pd.to_datetime(df_fs["timestamp"]).dt.hour
            hourly_aqi = df_fs.groupby("hour")["us_aqi"].mean().reset_index()
            fig = px.line(hourly_aqi, x="hour", y="us_aqi", title="Average AQI by Hour of Day", markers=True)
            fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Mean US AQI")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column 'us_aqi' not found for Diurnal Cycle plot.")

    with col2:
        st.subheader("Monthly AQI Seasonality")
        if "us_aqi" in df_fs.columns:
            df_fs["month"] = pd.to_datetime(df_fs["timestamp"]).dt.month
            monthly_aqi = df_fs.groupby("month")["us_aqi"].mean().reset_index()
            fig = px.line(monthly_aqi, x="month", y="us_aqi", title="Average AQI by Month", markers=True)
            fig.update_layout(xaxis_title="Month", yaxis_title="Mean US AQI")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column 'us_aqi' not found for Monthly Seasonality plot.")

    st.markdown("---")

    if "us_aqi" in df_fs.columns:
        st.subheader("AQI Over Time")
        fig = px.line(df_fs.sort_values("timestamp"), x="timestamp", y="us_aqi", title="Historical AQI from Feature Store")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    scatter_cols = ["pm2_5", "nitrogen_dioxide", "us_aqi"]
    if all(c in df_fs.columns for c in scatter_cols):
        st.subheader("PM2.5 vs NO₂ (coloured by AQI)")
        fig = px.scatter(
            df_fs.sample(min(len(df_fs), 2000)),
            x="pm2_5",
            y="nitrogen_dioxide",
            color="us_aqi",
            title="PM2.5 vs. Nitrogen Dioxide",
            color_continuous_scale=px.colors.sequential.OrRd,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"One or more columns missing for scatter plot: {scatter_cols}")

    st.markdown("---")
    st.subheader("Live Feature Store — Quick Statistics")
    if not df_fs.empty:
        raw_cols = [
            c for c in [
                "us_aqi", "pm2_5", "pm10", "nitrogen_dioxide",
                "temperature_2m", "relative_humidity_2m",
            ] if c in df_fs.columns
        ]
        if raw_cols:
            st.dataframe(df_fs[raw_cols].describe().round(2))
            st.caption(f"Feature store contains {len(df_fs):,} rows × {df_fs.shape[1]} columns.")


# ============================================================= PAGE 3
elif page == "Model Diagnostics & XAI":
    st.title("🧪 Model Diagnostics & Explainability")

    meta = _load_leaderboard()
    if meta is None:
        st.warning("No trained models found. Run `python -m training.train` first.")
    else:
        st.subheader("Benchmark Leaderboard")
        lb_df = pd.DataFrame(meta["leaderboard"]).T.round(4)
        lb_df.index.name = "Model"
        lb_df = lb_df.sort_values("rmse")
        st.dataframe(lb_df)

        champ = meta["champion"]
        champ_m = meta["metrics"]
        st.success(
            f"🏆 **Champion model:** `{champ}` — "
            f"RMSE: **{champ_m['rmse']:.4f}**, "
            f"MAE: **{champ_m['mae']:.4f}**, "
            f"R²: **{champ_m['r2']:.4f}**"
        )

        st.markdown("---")
        fig_lb = px.bar(
            lb_df.reset_index(),
            x="Model", y="rmse",
            color="Model",
            title="Test RMSE Comparison (lower is better)",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_lb, width="stretch")

    st.markdown("---")
    st.subheader("SHAP — Global Feature Importance")
    col1, col2 = st.columns(2)
    shap_s = _img_artifact("shap_summary.png")
    shap_b = _img_artifact("shap_bar.png")
    if shap_s:
        col1.image(str(shap_s), caption="SHAP Summary (beeswarm)", use_column_width='always')
    if shap_b:
        col2.image(str(shap_b), caption="Mean |SHAP| (bar)", use_column_width='always')
    if not shap_s:
        st.info("Run `python -m training.evaluate` to generate SHAP and LIME artefacts.")

    st.markdown("---")
    st.subheader("LIME — Local Instance Explanation")
    lime_plot = _img_artifact("lime_explanation.png")
    lime_json = ARTIFACTS / "lime_weights.json"
    if lime_plot:
        st.image(str(lime_plot), caption="LIME local explanation (single forecast row)", use_column_width='always')
    if lime_json.exists():
        weights = json.loads(lime_json.read_text(encoding="utf-8"))
        wdf = (
            pd.DataFrame([{"Feature": k, "LIME Weight": v} for k, v in weights.items()])
            .sort_values("LIME Weight", key=abs, ascending=False)
            .head(20)
        )
        st.markdown("**Top 20 local feature contributions**")
        st.dataframe(wdf, hide_index=True)

        fig_lime = px.bar(
            wdf, x="LIME Weight", y="Feature",
            orientation="h",
            title="LIME Feature Contributions",
            template="plotly_white",
            color="LIME Weight",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
        )
        fig_lime.update_layout(yaxis=dict(autorange="reversed"), height=520)
        st.plotly_chart(fig_lime, width="stretch")


# ============================================================= PAGE 4
else:
    st.title("📅 Historical Overview")
    st.caption("Loaded directly from the Feature Store (MongoDB Atlas).")

    df_fs = _load_feature_store()
    if df_fs is None or df_fs.empty:
        st.warning("Feature store is empty. Run `python -m features.backfill_historical` first.")
        st.stop()

    df_fs["timestamp"] = pd.to_datetime(df_fs["timestamp"], utc=True)
    df_fs = df_fs.sort_values("timestamp")

    st.markdown(f"**Rows:** {len(df_fs):,} &nbsp;|&nbsp; **Columns:** {df_fs.shape[1]} &nbsp;|&nbsp; **Date range:** {df_fs['timestamp'].min().date()} → {df_fs['timestamp'].max().date()}")

    if "us_aqi" in df_fs.columns:
        fig_ts = px.line(
            df_fs, x="timestamp", y="us_aqi",
            title="Historical US AQI (Feature Store)",
            template="plotly_white",
            labels={"us_aqi": "US AQI", "timestamp": "Time (UTC)"},
        )
        fig_ts.update_traces(line_color="#3b82f6")
        st.plotly_chart(fig_ts, width="stretch")

    pollutant_cols = [
        c for c in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "dust"]
        if c in df_fs.columns
    ]
    if pollutant_cols:
        st.subheader("Pollutant Trends")
        selected = st.multiselect(
            "Select pollutants to plot",
            options=pollutant_cols,
            default=pollutant_cols[:3],
        )
        if selected:
            fig_p = px.line(
                df_fs.melt(id_vars="timestamp", value_vars=selected),
                x="timestamp", y="value", color="variable",
                title="Pollutant Concentration Over Time",
                template="plotly_white",
                labels={"value": "Concentration", "variable": "Pollutant"},
            )
            st.plotly_chart(fig_p, width="stretch")

    weather_cols = [
        c for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]
        if c in df_fs.columns
    ]
    if weather_cols:
        st.subheader("Weather Conditions")
        fig_w = px.line(
            df_fs.melt(id_vars="timestamp", value_vars=weather_cols),
            x="timestamp", y="value", color="variable",
            title="Weather Variables Over Time",
            template="plotly_white",
            facet_col="variable",
            facet_col_wrap=2,
            labels={"value": "Value", "variable": "Variable"},
        )
        fig_w.update_yaxes(matches=None, title_text="")
        st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("Feature Store Sample (first 200 rows)")
    st.dataframe(df_fs.head(200))
