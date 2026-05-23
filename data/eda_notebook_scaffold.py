"""
EDA scaffold for the Pearls AQI Predictor — Open-Meteo schema.

Run as a script:  python -m data.eda_notebook_scaffold
Or paste cells into a Jupyter notebook.

Generates into artifacts/:
  corr_heatmap.png
  hourly_seasonality.html
  monthly_seasonality.html
  pm25_vs_no2.html
  aqi_timeseries.html
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

POLLUTANT_COLS = [
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "carbon_monoxide", "dust", "us_aqi",
]

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "surface_pressure", "precipitation",
]


def load_features() -> pd.DataFrame:
    from features.feature_store import load_features as _load
    return _load()


def plot_correlation_heatmap(df: pd.DataFrame):
    cols = [c for c in POLLUTANT_COLS + WEATHER_COLS if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, ax=ax, linewidths=0.3,
    )
    ax.set_title("Pollutant + Weather Correlation Heatmap — Hyderabad AQI")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "corr_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"Saved corr_heatmap.png")
    return fig


def plot_hourly_seasonality(df: pd.DataFrame):
    if "hour" not in df.columns:
        df = df.copy()
        df["hour"] = pd.to_datetime(df["timestamp"], utc=True).dt.hour
    hourly = df.groupby("hour")["us_aqi"].mean().reset_index()
    fig = px.line(
        hourly, x="hour", y="us_aqi", markers=True,
        title="Average US AQI by Hour of Day (Diurnal Cycle)",
        labels={"us_aqi": "Mean US AQI", "hour": "Hour (UTC)"},
        template="plotly_white",
    )
    fig.write_html(ARTIFACTS / "hourly_seasonality.html")
    print("Saved hourly_seasonality.html")
    return fig


def plot_monthly_seasonality(df: pd.DataFrame):
    if "month" not in df.columns:
        df = df.copy()
        df["month"] = pd.to_datetime(df["timestamp"], utc=True).dt.month
    monthly = df.groupby("month")["us_aqi"].mean().reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["month_name"] = monthly["month"].map(month_names)
    fig = px.bar(
        monthly, x="month_name", y="us_aqi",
        title="Average US AQI by Month (Seasonal Cycle)",
        labels={"us_aqi": "Mean US AQI", "month_name": "Month"},
        template="plotly_white",
        color="us_aqi",
        color_continuous_scale="RdYlGn_r",
    )
    fig.write_html(ARTIFACTS / "monthly_seasonality.html")
    print("Saved monthly_seasonality.html")
    return fig


def plot_pm25_vs_no2(df: pd.DataFrame):
    if "pm2_5" not in df.columns or "nitrogen_dioxide" not in df.columns:
        print("pm2_5 or nitrogen_dioxide column missing — skipping scatter.")
        return None
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="pm2_5", y="nitrogen_dioxide",
        color="us_aqi",
        title="PM2.5 vs NO₂ (coloured by US AQI)",
        labels={
            "pm2_5": "PM2.5 (µg/m³)",
            "nitrogen_dioxide": "NO₂ (µg/m³)",
            "us_aqi": "US AQI",
        },
        opacity=0.6,
        color_continuous_scale="Turbo",
        template="plotly_white",
    )
    fig.write_html(ARTIFACTS / "pm25_vs_no2.html")
    print("Saved pm25_vs_no2.html")
    return fig


def plot_aqi_timeseries(df: pd.DataFrame):
    fig = px.line(
        df.sort_values("timestamp"), x="timestamp", y="us_aqi",
        title="US AQI Over Time — Hyderabad, Pakistan",
        labels={"us_aqi": "US AQI", "timestamp": "Time (UTC)"},
        template="plotly_white",
    )
    fig.update_traces(line_color="#3b82f6")
    fig.write_html(ARTIFACTS / "aqi_timeseries.html")
    print("Saved aqi_timeseries.html")
    return fig


def run_full_eda() -> None:
    print("Loading features …")
    df = load_features()
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
    plot_correlation_heatmap(df)
    plot_hourly_seasonality(df)
    plot_monthly_seasonality(df)
    plot_pm25_vs_no2(df)
    plot_aqi_timeseries(df)
    print(f"\nAll EDA artefacts saved to {ARTIFACTS}/")


if __name__ == "__main__":
    run_full_eda()
