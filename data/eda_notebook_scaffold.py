"""
EDA scaffold for the Pearls AQI Predictor.

Run as a script (`python data/eda_notebook_scaffold.py`) or paste cells into
a notebook. Every function returns a Plotly/Matplotlib figure so the
dashboard can re-use them directly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def load_features() -> pd.DataFrame:
    """Load the engineered feature frame from the local feature store."""
    from features.feature_store import load_features as _load
    return _load()


# --------------------------------------------------------------------- Plots
def plot_correlation_heatmap(df: pd.DataFrame):
    pollutants = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "nh3", "no", "aqi"]
    pollutants = [p for p in pollutants if p in df.columns]
    corr = df[pollutants].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Pollutant correlation heatmap")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "corr_heatmap.png", dpi=140)
    return fig


def plot_hourly_seasonality(df: pd.DataFrame):
    hourly = df.groupby("hour")["aqi"].mean().reset_index()
    fig = px.line(hourly, x="hour", y="aqi", markers=True,
                  title="Average AQI by hour of day (diurnal cycle)")
    fig.update_layout(template="plotly_white")
    fig.write_html(ARTIFACTS / "hourly_seasonality.html")
    return fig


def plot_monthly_seasonality(df: pd.DataFrame):
    monthly = df.groupby("month")["aqi"].mean().reset_index()
    fig = px.bar(monthly, x="month", y="aqi",
                 title="Average AQI by month (seasonal cycle)")
    fig.update_layout(template="plotly_white")
    fig.write_html(ARTIFACTS / "monthly_seasonality.html")
    return fig


def plot_pm25_vs_no2(df: pd.DataFrame):
    fig = px.scatter(df, x="pm2_5", y="no2", color="aqi",
                     title="PM2.5 vs NO2 coloured by AQI",
                     opacity=0.6, color_continuous_scale="Turbo")
    fig.update_layout(template="plotly_white")
    fig.write_html(ARTIFACTS / "pm25_vs_no2.html")
    return fig


def plot_aqi_timeseries(df: pd.DataFrame):
    fig = px.line(df, x="timestamp", y="aqi", title="AQI over time")
    fig.update_layout(template="plotly_white")
    fig.write_html(ARTIFACTS / "aqi_timeseries.html")
    return fig


def run_full_eda() -> None:
    df = load_features()
    plot_correlation_heatmap(df)
    plot_hourly_seasonality(df)
    plot_monthly_seasonality(df)
    plot_pm25_vs_no2(df)
    plot_aqi_timeseries(df)
    print(f"EDA artefacts written to {ARTIFACTS}/")


if __name__ == "__main__":
    run_full_eda()
