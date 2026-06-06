# Internship Project Report: Data-Driven AQI Forecasting with Automated Retraining and Prediction Pipelines


**Project Title:**  Pearls AQI Predictor

**Target Location:** Hyderabad, Sindh, Pakistan

**Developed By:** Syeda Umaima

**Live Application:** https://aqi-predictor-hyd-sindh.streamlit.app/

## 1. Executive Project Overview

This report documents the engineering design, system architecture, and operational maturity of a production-grade, self-sustaining AQI forecasting system for Hyderabad, Sindh, Pakistan. The system addresses urban air quality degradation by delivering highly localized, 72-hour recursive forecasts of the Air Quality Index (AQI) using a fully serverless, decoupled MLOps stack.

**Core Inference Target:** The primary forecast target is the **1-hour ahead US-AQI** value. A recursive multi-step engine then propagates that prediction forward to produce a full **72-hour forecast horizon**.

**Architectural Topology:** The system avoids dedicated, always-on infrastructure through a three-node serverless design:

<img src="assets/Diagram 2.png">

The pipeline is intentionally self-sustaining: CI/CD workflows in GitHub Actions service hourly feature ingestion and daily model retraining; MongoDB Atlas acts as both feature store and binary model registry via GridFS; and Streamlit Cloud serves live inference and explainability visualizations.

---

## 2. Streamlit UI/UX Functional Design Architecture

The Streamlit dashboard is the production-grade user touchpoint for the AQI system. It is designed as a multi-page application with four analytical views. It uses dynamic data fetching from MongoDB Atlas to provide fast page loads and real-time updates.


### 2.1 Real-Time Forecast
The primary view displays live forecasting metrics and automated health risk tiers. It uses a recursive model loop to project hourly US-AQI trends up to 3 days out, backed by a structured tabular log.

<figure>
  <img src="assets/Real%20Time%20Forecast%201.png" alt="Real-Time Forecast Interface">
  <figcaption><i>Figure 1: Live prediction metrics, active model status, and the 72-hour forecast trend line.</i></figcaption>
</figure>

<br>

<figure>
  <img src="assets/Real%20Time%20Forecast%202.png" alt="Real-Time Forecast Interface">
  <figcaption><i>Figure 2: Forecast Detail Table mapping exact future hourly timestamps to calculated EPA hazard categories.</i></figcaption>
</figure>

### 2.2 EDA & Analysis

This section displays historical data relationships and seasonal air quality patterns. It gives researchers instant insight into how weather conditions influence local pollution levels.

<figure>
  <img src="assets/Exploratory Data Analysis 1.png" alt="EDA & Analysis Interface">
  <figcaption><i>Figure 3: Interactive correlation heatmap revealing strong dependencies between PM₂.₅, temperature, and wind speed. line.</i></figcaption>
</figure>

<br>

<figure>
  <img src="assets/Exploratory Data Analysis 2.png" alt="EDA & Analysis Interface">
  <figcaption><i>Figure 4: Seasonal Trends and Diurnal AQI Cycles mapping out predictable hourly and monthly pollution patterns.</i></figcaption>
</figure>

### 2.3 Model Diagnostics & XAI

This workspace handles model evaluation and transparency. It displays the production model leaderboard and hosts the pre-rendered Explainable AI charts fetched directly from GridFS.

<figure>
  <img src="assets/Model Diagnostics & XAI 1.png" alt="Model Diagnostics & XAI">
  <figcaption><i>Figure 5: Test metric leaderboard ranking XGBoost, LightGBM, Ridge, and TensorFlow models to tag the Champion Model.</i></figcaption>
</figure>

<br>

<figure>
  <img src="assets/Model Diagnostics & XAI 2.png" alt="Model Diagnostics & XAI">
  <figcaption><i>Figure 6: Global SHAP feature importance distributions alongside local instance LIME charts for individual forecasts.</i></figcaption>
</figure>


### 2.4 Historical Overview

This module acts as the system's empirical record. It charts long-term historical air quality trends and provides direct visibility into the feature store's data collection layer.

<figure>
  <img src="assets/Historical Overview 1.png" alt="Historical Overview">
  <figcaption><i>Figure 7: Multi-year historical timeline tracking continuous air quality variations across Hyderabad.</i></figcaption>
</figure>

<br>

<figure>
  <img src="assets/Historical Overview 2.png" alt="Historical Overview">
  <figcaption><i>Figure 8: Tabular registry exposing the latest 100 raw feature rows stored inside the MongoDB collection.</i></figcaption>
</figure>

---

## 3. Regional Environmental Context & Value Proposition
Hyderabad, Sindh, faces severe air pollution due to urbanization, waste burning, desert dust, and dense traffic along the National Highway. Low wind speeds before the monsoon season frequently trap these pollutants close to the ground.

### 3.1 Ground-Level Sensor Constraints

Traditional air monitoring in Hyderabad suffers from critical limitations:

- **No Reference Infrastructure:** Publicly accessible high-fidelity monitoring stations are completely absent.

- **High Maintenance Overhead:** Power cuts and dust build-up disable basic field sensors.

- **Zero Predictive Context:** Public monitors only report historical data without future risk warnings.

### 3.2 The MLOps Solution

This system addresses these gaps by combining satellite-derived observations with available ground station data via the Open-Meteo API. By processing multi-pollutant variables — PM₂.₅, PM₁₀, O₃, NO₂, SO₂, CO — alongside co-dependent meteorological features (relative humidity, temperature, planetary boundary layer height, wind vector components), the system replaces expensive physical sensor infrastructure with a virtual forecasting array.

The resulting pipeline enables vulnerable populations, healthcare providers, and urban planners to shift from reactive responses to **proactive risk mitigation up to 72 hours before acute smog or dust events**.

---

## 4. Data Lifecycle & Historical Architecture

### 4.1 End-to-End Data Flow
The system is built as a decoupled, time-safe data architecture where ingestion, storage, training, and inference are separated but connected through MongoDB Atlas.

  <img src="assets/Diagram 1.png" alt="Data Flow Diagram">

### 4.2 Production Feature Store

The feature store is the central temporal source of truth.

* **`features_v2`** stores engineered hourly rows keyed by `timestamp`.
* Ingestion is idempotent with `ReplaceOne(..., upsert=True)`.
* A unique index on **timestamp** enforces single-record temporal anchors.
* Repeated hourly jobs produce the same database state without duplicates.

  <img src="assets/Production Feature Store.png" alt="Real-Time Forecast Interface">
  
### 4.3 PyMongo Implementation

The ingestion engine is implemented in `feature_store.py` using PyMongo bulk writes.

The code ensures:

* high throughput
* non-blocking batch writes
* deterministic temporal consistency
* explicit retry and connection hardening for Atlas

### 4.4 Data Movement and Storage

Historical observations are transformed into the engineered feature matrix, then persisted into MongoDB Atlas. The system reads from the same collection for both training and live inference, which enforces a single source of truth and avoids data drift between pipelines.

---

## 5. Exploratory Data Analysis & Feature Engineering Taxonomy

### 5.1 EDA Insights

The EDA phase produced operationally relevant signals that shaped model design:

* **Humidity Effect:** Relative humidity strongly correlates with PM₂.₅ accumulation, especially in Hyderabad’s pre-monsoon season.
* **Planetary Boundary Layer (PBL) Dynamics:** Lower PBL height concentrates pollutants near ground level, causing morning AQI spikes.
* **Wind Vector Behavior:** Decomposed wind into U/V components showed that weak northeasterly flows contribute to smog stagnation.
* **Seasonal and Diurnal Patterns:** Hourly and monthly cycles are strong signals, justifying cyclical feature engineering.

### 5.2 Feature Engineering Taxonomy

`feature_engineering.py` produces over 80 engineered features across these categories:

* **Temporal encoding:** `hour`, `day_of_week`, `month`, plus sine/cosine transforms
* **Lagged values:** `*_lag_{n}h` for pollutant and meteorological variables
* **Rolling statistics:** `*_roll_mean_{w}h`, `*_roll_std_{w}h`, `*_roll_min_{w}h`, `*_roll_max_{w}h`
* **Interaction features:** `feat_temp_x_humidity`, `feat_pm25_x_no2`, `feat_wind_x_pressure`
* **Forecast lead features:** `temperature_target_hour`, `relative_humidity_target_hour`
* **Change rates:** delta over 3h, 6h, 12h, 24h windows
* **Supervised target:** `target_aqi_next_1h` derived from the next-hour US-AQI value

### 5.3 Operational Validation

The feature engineering pipeline is explicitly designed to capture:

* pollutant inertia
* atmospheric transition dynamics
* short-term autoregressive behavior
* weather-driven AQI triggers

This ensures the model learns from both immediate history and meteorological context.

---

## 6. Champion-Challenger Training Framework

### 6.1 Training Architecture

The training engine in `train.py` executes daily and is built for temporal safety.

* historical data is loaded via `load_training_frame()`
* splits are created using a time-ordered process (`time_ordered_split`)
* a validation window is held out for hyperparameter evaluation
* a final chronologically later test window measures real-world performance

This avoids lookahead bias and preserves the sequential forecasting scenario.

### 6.2 Feature Selection 

A `RandomForestRegressor` is trained on log-transformed labels to compute feature importance. The pipeline truncates to the top 80 features, which reduces noise and improves stability for downstream model candidates.

### 6.3 Candidate Model Portfolio

The daily evaluation compares multiple learners:

* **Ridge Regression** — stable linear baseline with scaler and log-target transform
* **XGBoost Regressor** — powerful non-linear tree boosting
* **LightGBM Regressor** — fast gradient boosting suited for tabular data
* **TensorFlow MLP** — deep learning candidate for complex atmospheric regimes

> **Note:** champion persistence is restricted to `joblib`-compatible models for reliable GridFS serialization and recovery.
.

### 6.4 Evaluation Metrics
The system evaluates models using standard regression metrics:

- **RMSE**: `sqrt(mean_squared_error(y_true, y_pred))`
- **MAE**: `mean_absolute_error(y_true, y_pred)`
- **R²**: `r2_score(y_true, y_pred)`

Champion selection is based on lowest RMSE on the held-out test data, with leaderboard metadata persisted for traceability.

### 6.5 Champion Persistence
The winning model is persisted into MongoDB GridFS with:

* serialized model/scaler bundle
* feature column list
* training metrics
* timestamp and run_id
* historical lineage record

This enables auditability, rollback, and transparent model routing.

  <img src="assets/Model Registry .png" alt="Real-Time Forecast Interface">
  
---

## 7. Explainable AI (XAI) Cloud Integration 

### 7.1 SHAP Global Attribution

After champion selection, `evaluate.py` reconstructs the active model and computes SHAP explanations.

* uses KernelExplainer on sampled test data
* generates a global feature importance summary
* stores the resulting visualization in GridFS

### 7.2 LIME Local Explanations

Local interpretability is provided by `LimeTabularExplainer`.

* builds sparse surrogate explanations around a recent test instance
* highlights the top 10 contributing features
* persists the artifact to GridFS for dashboard consumption

### 7.3 Operational Design
Both SHAP and LIME artifacts are precomputed during training, not at inference time. This design:

* avoids runtime compute overhead
* improves dashboard responsiveness
* preserves transparency without sacrificing performance

---

## 8. CI/CD & MLOps Automation

### 8.1 GitHub Actions Workflows

The operational stack is automated through GitHub Actions workflows in workflows:

**`feature_pipeline.yml`**

* hourly cron schedule
* ingests Open-Meteo and pollutant data
* computes engineered features
* upserts into MongoDB Atlas via MONGO_URI from GitHub Secrets

**`training_pipeline.yml`**

* daily cron schedule
* loads feature store history
* trains and evaluates candidate models
* persists champion artifact and XAI visualizations
* prunes old registry entries

**`backfill_pipeline.yml`**

* manual dispatch
* enables historical recovery without affecting live ingestion

  <img src="assets/GitHub Actions.png" alt="Real-Time Forecast Interface">
  
### 8.2 Runner and Dependency Strategy

The workflows are configured for Ubuntu runners, Python 3.11, and dependency caching. This creates consistent environments while minimizing repeated installation overhead.

### 8.3 Secure Database Connectivity
MongoDB access is handled through secure secrets and hardened client options:

* `serverSelectionTimeoutMS=300000`
* `retryWrites=True`
* `retryReads=True`
* `maxPoolSize=50`

This makes GitHub Actions and Streamlit access more resilient against Atlas connectivity issues.

---

## 9. Centralized Model Registry & Free-Tier Storage Management

### 9.1 Registry Design

MongoDB Atlas serves as both:

* feature store (`features_v2`)
* a model registry with GridFS

The registry stores:

* `latest_champion` metadata
* historical run records
* serialized joblib models
* XAI images

### 9.2 Self-Cleaning Retention Engine
The registry enforces a rolling retention policy to preserve free-tier viability.

This routine:

* deletes older model binaries
* removes orphan GridFS chunks
* retains the most recent 3 runs
* keeps storage under the 512MB free-tier limit

### 9.3 Auditability
The model registry stores both active champion pointers and historical lineage documents, enabling:

* rollback
* traceability
* production routing with explainability artifacts

---

## 10. Engineering Blocker Log 

### Blocker 1: Open-Meteo API Timeouts and Rate Limits
**Problem:** Long historical fetches from Open-Meteo were unstable and subject to network timeouts.

**Resolution:** A synthetic raw data generation fallback loop was implemented to preserve data continuity during backfill. The fallback produces valid placeholder records when API ingestion fails, ensuring that the feature store remains healthy and training pipelines do not stall.

### Blocker 2: GitHub Actions Dynamic IP Firewall Blocks
**Problem:** MongoDB Atlas connection failures occurred from GitHub Actions due to IP / firewall restrictions and intermittent `ReplicaSetNoPrimary` errors.

**Resolution:** The pipeline was hardened by centralizing MongoDB credentials in GitHub Secrets, using robust MongoClient options with `serverSelectionTimeoutMS=300000`, `retryWrites=True`, `retryReads=True`, and `maxPoolSize=50`. This reduced handshake failures and improved reliability under GitHub Actions concurrency.

### Blocker 3: Free-Tier Storage Exhaustion
**Problem:** Feature and model artifacts threatened to exceed MongoDB Atlas free-tier storage limits.

**Resolution:** A cascading pruning loop was engineered to retain only the latest 3 runs. Old model binaries and XAI artifacts are removed from GridFS, and stale history documents are pruned using projection queries for `_id` and `run_id`. This keeps storage within budget while preserving the most recent operational history.

---

## 11. Technical Key Takeaways & MLOps Maturity Evaluation

### 11.1 Key Engineering Learnings
* **Decoupled serverless** architecture with clean feature, training, and inference boundaries
* **Idempotent ingestion** via PyMongo bulk ReplaceOne upserts
* **Time-safe modeling** through chronological train/test splits
* **Champion-challenger automation** for daily model refresh
* **Precomputed SHAP/LIME** for production-grade explainability
* **Automated retention** for free-tier MongoDB viability
* **Defensive connectivity** for GitHub Actions and Atlas

### 11.2 MLOps Maturity Evaluation
The final system demonstrates strong production readiness by combining:

* automated feature collection
* temporal leakage protection
* model lineage and auditability
* operational transparency
* cloud-ready CI/CD orchestration

---
<br>

*Developed as part of the 10Pearls Data Science Internship Program.* 