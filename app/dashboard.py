"""
Multi-page Streamlit dashboard for the Pearls AQI Predictor.

Pages:
  1. Real-Time Forecast — current AQI KPI + 3-day forecast line chart
  2. EDA & Thought Process — correlation heatmap + seasonality plots
  3. Model Diagnostics & XAI — benchmark leaderboard + SHAP + LIME
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
AQI_COLORS = {1: "#22c55e", 2: "#84cc16", 3: "#eab308", 4: "#f97316", 5: "#ef4444"}

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")

page = st.sidebar.radio(
    "Navigate",
    ["Real-Time Forecast", "EDA & Thought Process", "Model Diagnostics & XAI"],
)
st.sidebar.markdown(f"**City:** {CONFIG['city']['name']}")


# ---------------------------------------------------------- helpers
def _load_forecast():
    """Live forecast via FastAPI; fall back to a synthetic preview if unavailable."""
    try:
        import requests
        r = requests.get("http://localhost:8000/forecast", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        ts = pd.date_range(pd.Timestamp.now("UTC"), periods=72, freq="h")
        return {
            "champion": "demo",
            "forecast": [{"timestamp": str(t), "predicted_aqi": 2 + (i % 4) * 0.5}
                         for i, t in enumerate(ts)],
        }


# ============================================================ PAGE 1
if page == "Real-Time Forecast":
    st.title("🌫️ Real-Time AQI Forecast")
    st.caption("Next 72 hours · powered by the champion model")

    data = _load_forecast()
    df = pd.DataFrame(data["forecast"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    current_aqi = float(df["predicted_aqi"].iloc[0])
    current_bucket = max(1, min(5, round(current_aqi)))

    threshold = CONFIG["dashboard"]["hazardous_aqi_threshold"]
    if df["predicted_aqi"].max() >= threshold:
        st.error("⚠️ HAZARDOUS AQI forecast in the next 72 hours. Limit outdoor exposure.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current predicted AQI", f"{current_aqi:.2f}", AQI_LABELS[current_bucket])
    c2.metric("72h max", f"{df['predicted_aqi'].max():.2f}")
    c3.metric("Champion model", data["champion"])

    fig = px.line(df, x="timestamp", y="predicted_aqi", markers=True,
                  title="Predicted AQI — next 72 hours")
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, width="stretch")

# ============================================================ PAGE 2
elif page == "EDA & Thought Process":
    st.title("📊 EDA & Thought Process")
    st.markdown(
        "We started by exploring how pollutants relate to each other and how "
        "AQI fluctuates over time. This drove the feature-engineering choices "
        "(rolling means, cyclical hour/day encodings, change-rate features)."
    )

    heatmap = ARTIFACTS / "corr_heatmap.png"
    if heatmap.exists():
        st.subheader("Pollutant correlation heatmap")
        st.image(str(heatmap))
    else:
        st.info("Run `python data/eda_notebook_scaffold.py` to generate EDA artefacts.")

    for title, file in [
        ("Diurnal AQI cycle", "hourly_seasonality.html"),
        ("Monthly AQI cycle", "monthly_seasonality.html"),
        ("PM2.5 vs NO2", "pm25_vs_no2.html"),
        ("AQI over time", "aqi_timeseries.html"),
    ]:
        path = ARTIFACTS / file
        if path.exists():
            st.subheader(title)
            st.components.v1.html(path.read_text(encoding="utf-8"), height=480, scrolling=True)

# ============================================================ PAGE 3
else:
    st.title("🧪 Model Diagnostics & Explainability")

    lb_path = ARTIFACTS / "leaderboard.json"
    if not lb_path.exists():
        st.warning("Train the models first: `python -m training.train`.")
    else:
        meta = json.loads(lb_path.read_text(encoding="utf-8"))
        st.subheader("Benchmark leaderboard")
        st.dataframe(pd.DataFrame(meta["leaderboard"]).T.round(4), width="stretch")
        st.success(f"Champion model: **{meta['champion']}**")

    st.subheader("SHAP — global feature importance")
    col1, col2 = st.columns(2)
    summary = ARTIFACTS / "shap_summary.png"
    bar = ARTIFACTS / "shap_bar.png"
    if summary.exists():
        col1.image(str(summary), caption="SHAP summary (beeswarm)")
    if bar.exists():
        col2.image(str(bar), caption="Mean |SHAP value|")
    if not summary.exists():
        st.info("Run `python -m training.evaluate` to generate SHAP and LIME artefacts.")

    st.subheader("LIME — local instance explanation")
    lime_plot = ARTIFACTS / "lime_explanation.png"
    lime_json = ARTIFACTS / "lime_weights.json"
    if lime_plot.exists():
        st.image(str(lime_plot), caption="LIME local explanation (single forecast row)")
    if lime_json.exists():
        lime_weights = json.loads(lime_json.read_text(encoding="utf-8"))
        st.markdown("**Top local feature contributions**")
        st.dataframe(
            pd.DataFrame(
                [{"feature": k, "weight": v} for k, v in lime_weights.items()]
            ).sort_values("weight", key=abs, ascending=False),
            width="stretch",
            hide_index=True,
        )
    elif not lime_plot.exists():
        st.info("Run `python -m training.evaluate` to generate LIME plots.")
