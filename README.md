# Pearls AQI Predictor

A production-grade, serverless-style MLOps pipeline that forecasts the Air Quality
Index (AQI) for the next **72 hours** in a configurable city. Built around a
**local Hopsworks** Feature Store / Model Registry, multi-model benchmarking,
SHAP explainability, automated GitHub Actions pipelines, and a polished
multi-page Streamlit dashboard.

> **Why this project stands out**
> - Local-first MLOps (no cloud lock-in during the Hopsworks outage)
> - Four models benchmarked head-to-head (Ridge, Random Forest, XGBoost, Keras LSTM)
> - SHAP explainability baked into evaluation
> - Multi-page Streamlit dashboard built for both technical and non-technical judges
> - Hourly + daily CI/CD via GitHub Actions

---

## 1. Architecture

```
OpenWeather API ──► feature_pipeline (hourly)  ──► Local Hopsworks FS
                                                      │
                                                      ▼
                                        training_pipeline (daily)
                                                      │
                                ┌─────────────────────┼──────────────────────┐
                                ▼                     ▼                      ▼
                          Ridge / RF / XGB     Keras LSTM             SHAP report
                                └────────┬────────────┘
                                         ▼
                            Champion model → Local Model Registry
                                         ▼
                            FastAPI inference + Streamlit dashboard
```

## 2. Folder layout

```
.github/workflows/    feature_pipeline.yml, training_pipeline.yml
config/               config.yaml
data/                 eda_notebook_scaffold.py
features/             api_client.py, feature_engineering.py, backfill_historical.py
training/             train.py, evaluate.py
app/                  main.py (FastAPI), dashboard.py (Streamlit)
requirements.txt
.env.example
```

## 3. Local Hopsworks setup (Docker)

The Hopsworks Cloud is currently unavailable, so we run the full feature store
and model registry locally inside Docker:

```bash
docker run -d \
  -p 8080:8080 -p 443:443 \
  --name hopsworks \
  logicalclocks/hopsworks-sandbox:latest
```

Wait ~3 minutes for the sandbox to boot, then verify `https://localhost`.

All Python scripts log in via:

```python
import hopsworks
project = hopsworks.login(
    host="localhost",
    project="local_project",
    api_key_value="offline",
)
```

> If the sandbox image is unreachable at evaluation time, every script falls
> back to a local **Parquet feature store** under `./.local_fs/` so the demo
> never breaks. This is governed by `config/config.yaml: hopsworks.fallback_to_parquet`.

## 4. Secrets — OpenWeather API key

Never hardcode keys. The API client reads from `OPENWEATHER_API_KEY`.

**Locally:** copy `.env.example` → `.env` and fill in the value.

**In Lovable:** open the **Secrets** panel (left sidebar → Cloud → Secrets),
click *Add secret*, name it `OPENWEATHER_API_KEY`, paste your key, and save.
It is then exposed as an env var to any backend code.

**On GitHub Actions:** go to *Settings → Secrets and variables → Actions*
and add `OPENWEATHER_API_KEY` — the workflow files already reference it.

## 5. Run locally (Cursor / any IDE)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in OPENWEATHER_API_KEY

# 1. Backfill ~30 days of history into the local Feature Store
python -m features.backfill_historical

# 2. Train + benchmark all 4 models, register the champion
python -m training.train

# 3. Generate SHAP explainability artefacts
python -m training.evaluate

# 4. Launch the dashboard
streamlit run app/dashboard.py

# 5. (Optional) Run the FastAPI inference layer
uvicorn app.main:app --reload --port 8000
```

## 6. Multi-model thought process

| Model            | Why it's in the bake-off                                       |
|------------------|----------------------------------------------------------------|
| Ridge Regression | Linear baseline — anchors the benchmark, fast to retrain       |
| Random Forest    | Captures non-linear pollutant interactions, robust to outliers |
| XGBoost          | State-of-the-art tabular learner, strong on engineered features|
| Keras LSTM       | Models temporal dependencies in the 24h pollutant sequence     |

We score each on **RMSE, MAE, R²** against a held-out 20% time-ordered split
(no shuffling — leakage matters for time series). The lowest-RMSE model is
dynamically promoted to the local Hopsworks Model Registry as the **champion**.

## 7. Explainability

`training/evaluate.py` runs SHAP on the champion (TreeExplainer for tree models,
KernelExplainer fallback for the LSTM) and writes a summary plot to
`artifacts/shap_summary.png`. The dashboard surfaces this directly to judges.
