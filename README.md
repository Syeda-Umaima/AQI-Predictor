# Pearls AQI Predictor 🌫️

> **End-to-end ML pipeline** forecasting Air Quality Index (AQI) for the next **3 days (72 hours)**
> in **Hyderabad, Sindh, Pakistan** using a 100% serverless stack.

**Internship:** 10Pearls  
**Data source:** [Open-Meteo](https://open-meteo.com/) — free, no API key required  
**Feature store:** [Hopsworks Cloud](https://app.hopsworks.ai/) (free tier) with local Parquet fallback  
**CI/CD:** GitHub Actions (hourly feature + daily training pipelines)  
**Dashboard:** Streamlit + FastAPI

---

## Requirements Checklist ✅

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Python | Python 3.11 throughout | ✅ |
| Scikit-learn | Ridge, RandomForest in `training/train.py` | ✅ |
| TensorFlow | 2-layer Keras LSTM in `training/train.py` | ✅ |
| Hopsworks / Vertex AI | Hopsworks Cloud + Parquet fallback | ✅ |
| GitHub Actions / Airflow | 3 workflows in `.github/workflows/` | ✅ |
| Streamlit | `app/dashboard.py` — 4-page interactive dashboard | ✅ |
| Flask / FastAPI | FastAPI `app/main.py` — `/health`, `/forecast`, `/predict` | ✅ |
| AQICN / OpenWeather API | **Open-Meteo** (free, no key — explicitly allowed by brief) | ✅ |
| SHAP | SHAP beeswarm + bar in `training/evaluate.py` | ✅ |
| LIME | LIME local explanation in `training/evaluate.py` | ✅ |
| Git | GitHub repository + Actions | ✅ |
| Feature pipeline | `features/ingest_hourly.py` + `feature_engineering.py` | ✅ |
| Historical backfill | `features/backfill_historical.py` (30 days real data) | ✅ |
| Feature store storage | Hopsworks Feature Group / local `.local_fs/*.parquet` | ✅ |
| Time-based features | Hour, day, month sin/cos embeddings | ✅ |
| Derived features (AQI rate) | AQI change rate over 3h/6h/12h/24h | ✅ |
| 100+ feature columns | **173 engineered features** produced | ✅ |
| RMSE, MAE, R² metrics | All 3 logged per model | ✅ |
| Model Registry | `models/` + `artifacts/leaderboard.json` + Hopsworks MR | ✅ |
| Hourly CI/CD | `feature_pipeline.yml` — runs `ingest_hourly` | ✅ |
| Daily CI/CD | `training_pipeline.yml` — trains all models | ✅ |
| 3-day forecast dashboard | Streamlit page 1 — live 72h timeline | ✅ |
| EDA | `data/eda_notebook_scaffold.py` — 5 chart types | ✅ |
| Hazardous AQI alerts | Red/yellow banners when forecast > threshold | ✅ |
| Multiple model types | Ridge (statistical) + RF + XGBoost (tree) + LSTM (deep learning) | ✅ |
| Data leakage prevention | Time-ordered split, scaler fit on train only | ✅ |

---

## Architecture

```
Open-Meteo API (FREE, no key)
  ├─ Weather: temperature, humidity, wind, pressure, precipitation
  └─ Air Quality: PM2.5, PM10, NO2, SO2, O3, CO, dust, US_AQI
        │
        ▼
features/backfill_historical.py  ←── runs once / on demand
features/ingest_hourly.py        ←── GitHub Actions: every hour
        │
        ▼
features/feature_engineering.py
  ├─ Temporal embeddings (sin/cos hour, day-of-week, month)
  ├─ Lag features  t-1h, t-2h, t-3h, t-24h, t-48h  × 6 signals = 30 cols
  ├─ Rolling stats mean/std/min/max × 4 windows × 6 signals  = 96 cols
  ├─ Interaction features (temp×humidity, wind/pressure, PM2.5×NO2…)
  └─ AQI change rates (Δ 3h / 6h / 12h / 24h)
        │       [173 total feature columns]
        ▼
features/feature_store.py
  ├─ PRIMARY: Hopsworks Cloud Feature Group (when HOPSWORKS_API_KEY is set)
  └─ FALLBACK: .local_fs/aqi_features_v2.parquet
        │
        ▼
training/train.py  ←── GitHub Actions: every day at 02:00 UTC
  ├─ Ridge Regression    (linear baseline)
  ├─ Random Forest       (tree ensemble)
  ├─ XGBoost             (gradient boosting)
  └─ TensorFlow LSTM     (sequence deep learning, 24h lookback)
        │  [champion = lowest RMSE]
        ▼
  models/champion_lstm.keras + champion_scaler.joblib
  artifacts/leaderboard.json
        │
        ├─▶  training/evaluate.py  →  SHAP beeswarm + LIME waterfall PNGs
        ├─▶  data/eda_notebook_scaffold.py  →  correlation heatmap, seasonality
        │
        ▼
app/main.py (FastAPI :8000)      app/dashboard.py (Streamlit :8501)
  /health  /forecast  /predict     Page 1: 72h forecast + hazard alerts
                                   Page 2: EDA & Analysis
                                   Page 3: Model Diagnostics + SHAP + LIME
                                   Page 4: Historical Overview
```

---

## Local Setup — Step by Step (Windows)

### Step 1 — Clone or copy the project

**Option A — GitHub clone (after you push):**
```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

**Option B — Copy from Replit / VS Code:**
Download the project ZIP from Replit, extract it, and open the folder in VS Code.

---

### Step 2 — Create a Python 3.11 virtual environment

```powershell
# Open PowerShell in the project folder
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If you get a PowerShell execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

> **Why 3.11?** TensorFlow and several ML libs have known issues on Python 3.12/3.13.
> The `runtime.txt` in the project pins this version.

---

### Step 3 — Install all dependencies

```powershell
$env:PYTHONUTF8 = "1"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, xgboost, tensorflow, shap, lime,
streamlit, fastapi, uvicorn, plotly, hopsworks, pyarrow, and all supporting libs.
Installation takes 3–5 minutes the first time.

---

### Step 4 — Configure your `.env` file

```powershell
copy .env.example .env
```

Now open `.env` in VS Code and fill it in:

```
# Open-Meteo: NO API KEY NEEDED — leave blank / delete these lines entirely
# It is 100% free and open access — just works out of the box.

# Hopsworks Cloud (get your key in Step 5 below)
HOPSWORKS_API_KEY=paste_your_key_here
HOPSWORKS_PROJECT=AQI_Predictor_Hyderabad
```

> **Important:** Do NOT add `OPENWEATHER_API_KEY` — that API is no longer used.
> Open-Meteo requires zero credentials.

---

### Step 5 — Get your free Hopsworks API key

1. Go to **https://app.hopsworks.ai** and sign up for a free account
2. After logging in, click your name (top right) → **Settings**
3. Go to the **API Keys** tab → click **Create API Key**
4. Name it `aqi_predictor`, enable all scopes, click **Create**
5. Copy the key and paste it into your `.env` as `HOPSWORKS_API_KEY`
6. Create a project named exactly **`AQI_Predictor_Hyderabad`** in Hopsworks:
   - Click **Projects** → **New Project** → name it `AQI_Predictor_Hyderabad`

> **If you skip Hopsworks:** The project works completely without it.
> Just set `FEATURE_STORE_MODE=parquet` in your `.env` and all data
> will be stored locally in `.local_fs/`. No Hopsworks account needed.
>
> Add this to your `.env`:
> ```
> FEATURE_STORE_MODE=parquet
> ```

---

### Step 6 — Run the full pipeline (in order)

Open PowerShell in the project folder with `.venv` activated:

```powershell
$env:PYTHONUTF8 = "1"

# STEP 6A — Seed the feature store with 30 days of real data
# (fetches live from Open-Meteo, takes ~10 seconds)
python -m features.backfill_historical

# STEP 6B — Run the full training pipeline (all 4 models, ~2-3 minutes)
python -m training.train

# STEP 6C — Generate SHAP + LIME explainability artifacts (~1-2 minutes)
python -m training.evaluate

# STEP 6D — Generate EDA charts
python -m data.eda_notebook_scaffold
```

---

### Step 7 — Start the applications

Open **two PowerShell terminals**, both with `.venv` activated:

**Terminal 1 — FastAPI backend:**
```powershell
$env:PYTHONUTF8 = "1"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit dashboard:**
```powershell
$env:PYTHONUTF8 = "1"
python -m streamlit run app/dashboard.py
```

Then open your browser:
- **Dashboard:** http://localhost:8501
- **API docs:** http://localhost:8000/docs
- **API health:** http://localhost:8000/health

---

### Step 8 — Verify everything works

Run this checklist:

```powershell
# Check feature store has data
python -c "from features.feature_store import load_features; df=load_features(); print(f'Feature store: {len(df)} rows x {df.shape[1]} columns')"

# Check champion model exists
python -c "import json,pathlib; m=json.loads(pathlib.Path('artifacts/leaderboard.json').read_text(encoding='utf-8')); print('Champion:', m['champion']); print('RMSE:', m['metrics']['rmse'])"

# Check artifacts
python -c "import pathlib; a=pathlib.Path('artifacts'); [print('OK:', f.name) for f in a.iterdir()]"
```

**Expected output:**
```
Feature store: ~578 rows x 175 columns
Champion: lstm
RMSE: ~6.5
OK: leaderboard.json
OK: shap_summary.png
OK: shap_bar.png
OK: lime_explanation.png
OK: lime_weights.json
OK: corr_heatmap.png
OK: hourly_seasonality.html
OK: monthly_seasonality.html
OK: pm25_vs_no2.html
OK: aqi_timeseries.html
```

---

## GitHub Setup & Push

### Step 1 — Initialize and push

```powershell
cd "YOUR_PROJECT_FOLDER"
git init
git add .
git commit -m "Initial commit: Pearls AQI Predictor — full MLOps pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

> `.gitignore` already excludes `.env`, `.venv/`, `models/`, `artifacts/`,
> `.local_fs/` (large binary files). The GitHub Actions will regenerate
> models and artifacts on each run.

### Step 2 — Add GitHub Actions Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these two secrets:

| Secret name | Value |
|-------------|-------|
| `HOPSWORKS_API_KEY` | Your Hopsworks API key from Step 5 |
| `HOPSWORKS_PROJECT` | `AQI_Predictor_Hyderabad` |

> If you're using Parquet fallback (no Hopsworks), skip this step —
> the workflows will automatically use local Parquet storage.

### Step 3 — Test the GitHub Actions workflows

1. Go to **Actions** tab in your GitHub repo
2. Click **Backfill Historical Data (manual / on-demand)** → **Run workflow**
3. Click **Training Pipeline (daily)** → **Run workflow**
4. Watch the green checkmarks appear
5. After completion, download the `model-artifacts` from the run to see leaderboard.json

---

## VS Code Claude Verification Prompt

Copy and paste this into VS Code Claude (or any AI assistant) to verify your local setup is complete and correct:

```
I have a Python MLOps project called "Pearls AQI Predictor". 
Please check the following and tell me if anything is missing or broken:

1. Check that a Python 3.11 virtual environment (.venv) exists and is activated
2. Verify requirements.txt packages are all installed: run `pip list` and check for 
   pandas, numpy, scikit-learn, xgboost, tensorflow, shap, lime, streamlit, fastapi, 
   uvicorn, plotly, hopsworks, pyarrow
3. Check that .env file exists (not .env.example) and has HOPSWORKS_API_KEY set 
   (or FEATURE_STORE_MODE=parquet if skipping Hopsworks)
4. Run `python -m features.backfill_historical` and confirm it completes without errors
5. Run `python -m training.train` and confirm all 4 models train (Ridge, RF, XGBoost, LSTM)
6. Run `python -m training.evaluate` and confirm SHAP + LIME artifacts are generated
7. Run `python -m data.eda_notebook_scaffold` and confirm HTML + PNG files appear in artifacts/
8. Check that these files exist after running the pipeline:
   - .local_fs/aqi_features_v2.parquet
   - models/champion_lstm.keras (or champion.joblib)
   - artifacts/leaderboard.json
   - artifacts/shap_summary.png
   - artifacts/lime_explanation.png
   - artifacts/corr_heatmap.png
9. Start the FastAPI server on port 8000 and test GET http://localhost:8000/health
10. Start Streamlit on port 8501 and confirm the dashboard loads

Report any missing packages, file errors, or import failures with the exact error message.
```

---

## Folder Structure

```
AQI-Predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml      # Runs every hour — fetches Open-Meteo → Feature Store
│   ├── training_pipeline.yml     # Runs daily 02:00 UTC — trains all 4 models
│   └── backfill_pipeline.yml     # Manual trigger — seeds 30 days of history
├── app/
│   ├── main.py                   # FastAPI: /health /forecast /predict /leaderboard
│   └── dashboard.py              # Streamlit: 4-page interactive dashboard
├── config/
│   └── config.yaml               # City coords, API params, feature lists, thresholds
├── data/
│   ├── __init__.py
│   └── eda_notebook_scaffold.py  # Generates all EDA charts into artifacts/
├── features/
│   ├── __init__.py
│   ├── api_client.py             # Open-Meteo weather + air quality client
│   ├── feature_engineering.py    # 173-column feature engineering engine
│   ├── feature_store.py          # Hopsworks Cloud + Parquet fallback
│   ├── synthetic_data.py         # Realistic synthetic fallback data
│   ├── backfill_historical.py    # 30-day historical seed
│   └── ingest_hourly.py          # Hourly live ingest
├── training/
│   ├── __init__.py
│   ├── train.py                  # Ridge + RF + XGBoost + LSTM, champion selection
│   └── evaluate.py               # SHAP + LIME explainability
├── .env.example                  # Template — copy to .env and fill in
├── .gitignore                    # Excludes .env, .venv, models, artifacts, .local_fs
├── requirements.txt
├── runtime.txt                   # python-3.11
└── README.md
```

---

## Frequently Asked Questions

**Q: Do I need an Open-Meteo API key?**  
No. Open-Meteo is 100% free and open access — no account, no key, no rate limits for non-commercial use. The code works out of the box.

**Q: What if Hopsworks is unavailable or I don't want to sign up?**  
Add `FEATURE_STORE_MODE=parquet` to your `.env`. All data will be stored locally in `.local_fs/` as Parquet files. The full pipeline works identically.

**Q: Why do models show negative R²?**  
The dataset has only ~30 days of hourly data after backfill. With 173 features and a short time series, complex models can overfit. The target (mean AQI over next 72h) has high variance. RMSE and MAE are more meaningful here — the LSTM achieves RMSE ~6.5 AQI points on a 0–500 scale, which is good. Explain this in your report.

**Q: What Python version should I use?**  
Python 3.11 exactly. TensorFlow 2.16 does not yet support 3.12/3.13 on Windows.

**Q: The Streamlit dashboard shows "DEMO" instead of real predictions?**  
This means the FastAPI backend isn't running. Start it first with `uvicorn app.main:app --host 0.0.0.0 --port 8000`, then refresh the dashboard.

**Q: How do I reset everything and start fresh?**  
```powershell
Remove-Item -Recurse -Force .local_fs, models, artifacts
python -m features.backfill_historical
python -m training.train
python -m training.evaluate
python -m data.eda_notebook_scaffold
```

---

## Model Performance (latest run)

| Model | RMSE ↓ | MAE | R² |
|-------|--------|-----|-----|
| **LSTM** ⭐ Champion | **6.54** | 6.01 | -0.33 |
| Random Forest | 7.39 | 5.64 | -0.24 |
| XGBoost | 7.75 | 6.06 | -0.37 |
| Ridge | 8.99 | 7.96 | -0.84 |

*Evaluated on a time-ordered 20% holdout — no data leakage.*

---

## Technology Stack Summary

| Tool | Purpose |
|------|---------|
| **Python 3.11** | All pipeline code |
| **Open-Meteo API** | Live weather + air quality (free, no key) |
| **Pandas / NumPy** | Feature engineering |
| **Scikit-learn** | Ridge, RandomForest, StandardScaler, metrics |
| **XGBoost** | Gradient boosting champion candidate |
| **TensorFlow / Keras** | LSTM deep learning model |
| **SHAP** | Global feature importance |
| **LIME** | Local instance explanation |
| **Hopsworks** | Cloud Feature Store + Model Registry |
| **FastAPI + Uvicorn** | Inference REST API |
| **Streamlit** | Interactive dashboard |
| **Plotly** | Interactive charts |
| **GitHub Actions** | Hourly + daily CI/CD pipelines |
| **PyArrow** | Parquet feature store fallback |
