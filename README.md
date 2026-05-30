# AQI Predictor 🌫️

Professional end-to-end machine learning pipeline for real-time Air Quality Index (AQI) forecasting. This system provides a 72-hour recursive forecast for Hyderabad, Pakistan, utilizing a fully cloud-native, production-ready stack.

## 🚀 Project Overview

The **AQI Predictor** is an enterprise-grade MLOps project that automates the entire lifecycle of a machine learning model—from data ingestion and feature engineering to model training, evaluation, and live deployment. The system is designed to be resilient, scalable, and highly transparent through Explainable AI (XAI).

### **Core Architecture**

- **Automated Ingestion**: GitHub Actions trigger hourly pipelines to fetch meteorological and pollutant data from the **Open-Meteo API**.
- **Feature Store**: All engineered features (lags, rolling windows, atmospheric interactions) are persisted in **MongoDB Atlas**.
- **Recursive Forecasting**: A 1-hour ahead predictive model is used recursively to generate a robust 72-hour forecast, dynamically handling atmospheric drift.
- **Dynamic Routing**: The production dashboard evaluates all registered models (XGBoost, LightGBM, Random Forest, etc.) against the latest live data point and automatically routes traffic to the specific model with the lowest real-time error.
- **Cloud Model Registry**: Trained model binaries and XAI artifacts (SHAP/LIME) are stored in **MongoDB GridFS** for instant global availability.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Optimized with advanced caching and responsive UI)
- **Database**: [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) (Feature Store & Model Registry)
- **ML Frameworks**: Scikit-learn, XGBoost, LightGBM
- **XAI**: SHAP, LIME (Global and local interpretability)
- **Orchestration**: GitHub Actions (CI/CD)
- **Data Source**: Open-Meteo Satellite & Ground Station API

## 📖 Local Setup & Development

### 1. Prerequisites

- Python 3.11+
- A MongoDB Atlas Cluster (Free Tier is sufficient)

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd aqi-predictor

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory and add your MongoDB connection string:

```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/?retryWrites=true&w=majority
```

### 4. Running the Pipelines

```bash
# Seed the Feature Store with historical data
python -m features.backfill_historical

# Train models and push champion to Cloud
python -m training.train

# Generate XAI diagnostics
python -m training.evaluate

# Launch the Dashboard
streamlit run app/dashboard.py
```

## 🌐 Deployment

This project is configured for continuous deployment to **Hugging Face Spaces**.

- GitHub Actions automatically update the Cloud Feature Store every hour.
- The dashboard pulls the latest "Champion" model and XAI plots directly from MongoDB GridFS.
- **Zero-Downtime Retraining**: When the daily training pipeline promotes a new champion, the dashboard reflects the update instantly without a restart.

---

_Note: Predictions are based on regional satellite data and may differ from hyper-local ground sensors._
