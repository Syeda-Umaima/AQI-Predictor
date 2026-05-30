# AQI Predictor 🌫️

Professional end-to-end machine learning pipeline for real-time Air Quality Index (AQI) forecasting. This system provides a 72-hour recursive forecast for Hyderabad, Pakistan, utilizing a fully cloud-native, production-ready stack.

## 🚀 Key Production Features

### 1. Cloud-Native Persistence (MongoDB Atlas)

- **Feature Store**: All engineered features are stored and indexed in MongoDB Atlas, replacing local CSV/JSON files.
- **Model Registry (GridFS)**: Trained models, scalers, and XAI artifacts (SHAP/LIME plots) are streamed directly to MongoDB GridFS. This ensures that GitHub Action runners can finish their tasks without losing artifacts, and the dashboard can pull the latest assets from anywhere.
- **Metadata Management**: Model performance metrics and run IDs are stored in a dedicated metadata collection, enabling dynamic routing and historical tracking.

### 2. Production Resilience & MLOps

- **Network Fault Tolerance**: Centralized MongoDB utility with custom retry decorators and standardized production parameters (connection pooling, timeouts, TLS).
- **Dynamic Model Routing**: The dashboard performs real-time validation against the latest live data point, automatically routing users to the specific model (XGBoost, LightGBM, etc.) currently exhibiting the lowest absolute error.
- **Robust XAI Pipeline**: Evaluation scripts are hardened against data variance issues (e.g., LIME scale parameter errors) and ensure that SHAP/LIME artifacts are always generated and synced to the cloud.

### 3. High-Performance Dashboard (Streamlit)

- **Advanced Caching**: Utilizes `st.cache_resource` and `st.cache_data` with TTL parameters to minimize database latency and prevent infinite loading.
- **Polished Fallback States**: Implements graceful UI degradation—if cloud assets are still synchronizing, the user is presented with helpful status messages instead of empty screens or raw errors.

## 🛠️ Tech Stack

- **Data Source**: [Open-Meteo API](https://open-meteo.com/)
- **Database**: MongoDB Atlas (Feature Store & Model Registry)
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP, LIME
- **Frontend**: Streamlit
- **CI/CD**: GitHub Actions

## 📖 Deployment & Workflow

### 1. Local Setup

```powershell
# Install dependencies
pip install -r requirements.txt

# Configure environment
# Add MONGO_URI to your .env file
```

### 2. Execution Flow

1. **Feature Engineering**: `python -m features.backfill_historical` (Seeds the Cloud Feature Store).
2. **Model Training**: `python -m training.train` (Fits models and pushes Champion to GridFS).
3. **Model Evaluation**: `python -m training.evaluate` (Generates XAI artifacts and pushes to GridFS).
4. **Dashboard**: `streamlit run app/dashboard.py` (Connects to Cloud for live forecasting).

### 3. GitHub Actions

The repository is pre-configured with two primary workflows:

- **Hourly Feature Pipeline**: Automatically fetches latest weather/pollutant data and updates the Cloud Feature Store.
- **Daily Training Pipeline**: Retrains all models, performs champion promotion, and refreshes XAI diagnostics in the Cloud Model Registry.
