# AQI Predictor 🌫️

Professional end-to-end machine learning pipeline for real-time Air Quality Index (AQI) forecasting. This system provides a 72-hour recursive forecast for Hyderabad, Pakistan, utilizing a fully cloud-native, production-ready stack.

## 🌐 Live Application

Explore the real-time dashboard and predictive analytics here:  
**[AQI Predictor — Hyderabad Live Dashboard](https://aqi-predictor-hyd-sindh.streamlit.app/)**

## 🚀 Project Overview

The **AQI Predictor** is an enterprise-grade MLOps project that automates the entire lifecycle of a machine learning model—from data ingestion and feature engineering to model training, evaluation, and live deployment. The system is designed to be resilient, scalable, and highly transparent through Explainable AI (XAI).

### **Key Features**

- **Feature Pipeline Development**: Automated ingestion of weather and pollutant data from Open-Meteo API.
- **Historical Data Backfill**: 2-year comprehensive historical dataset generation (730 days) for robust training.
- **Multi-Model Training Pipeline**: Automated training and evaluation of Ridge, XGBoost, LightGBM, and Deep Learning (TensorFlow) models.
- **Cloud Feature Store & Registry**: Centralized MongoDB Atlas storage for features and GridFS for model artifacts.
- **Explainable AI (XAI)**: Global (SHAP) and Local (LIME) interpretations persisted to the cloud and rendered in real-time.
- **Automated CI/CD**: Hourly feature ingestion and daily model retraining via GitHub Actions.
- **Production Dashboard**: Interactive Streamlit UI with dynamic model routing and hazardous AQI alerts.

## 📸 Dashboard Gallery

### Real-Time Forecast

|                      Real Time Forecast 1                      |                      Real Time Forecast 2                      |
| :------------------------------------------------------------: | :------------------------------------------------------------: |
| ![Real Time Forecast 1](assets/Real%20Time%20Forecast%201.png) | ![Real Time Forecast 2](assets/Real%20Time%20Forecast%202.png) |

### Exploratory Data Analysis

|                         Exploratory Data Analysis 1                          |                         Exploratory Data Analysis 2                          |
| :--------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
| ![Exploratory Data Analysis 1](assets/Exploratory%20Data%20Analysis%201.png) | ![Exploratory Data Analysis 2](assets/Exploratory%20Data%20Analysis%202.png) |

### Model Diagnostics & XAI

|                          Model Diagnostics & XAI 1                           |                          Model Diagnostics & XAI 2                           |
| :--------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
| ![Model Diagnostics & XAI 1](assets/Model%20Diagnostics%20%26%20XAI%201.png) | ![Model Diagnostics & XAI 2](assets/Model%20Diagnostics%20%26%20XAI%202.png) |

### Historical Overview

|                     Historical Overview 1                      |                     Historical Overview 2                      |
| :------------------------------------------------------------: | :------------------------------------------------------------: |
| ![Historical Overview 1](assets/Historical%20Overview%201.png) | ![Historical Overview 2](assets/Historical%20Overview%202.png) |

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (High-performance caching & responsive design)
- **Database**: [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) (Serverless Feature Store & Model Registry)
- **ML/DL Frameworks**: Scikit-learn, XGBoost, LightGBM, **TensorFlow** (MLP Architecture)
- **Explainability**: SHAP, LIME
- **Orchestration**: GitHub Actions (CI/CD Pipelines)
- **Data Source**: Open-Meteo Satellite & Ground Station API

## 📊 Project Audit Status

| Requirement             | Status | Details                                                          |
| :---------------------- | :----: | :--------------------------------------------------------------- |
| **Serverless Stack**    |   ✅   | GitHub Actions + MongoDB Atlas + Streamlit Cloud                 |
| **Feature Pipeline**    |   ✅   | Hourly automated ingestion with 80+ engineered features          |
| **Historical Backfill** |   ✅   | 2-year historical data ingestion completed                       |
| **Training Pipeline**   |   ✅   | Daily retraining with champion model promotion                   |
| **Model Diversity**     |   ✅   | Statistical (Ridge), Boosting (XGB/LGBM), and Deep Learning (TF) |
| **Explainability**      |   ✅   | Integrated SHAP Summary and LIME Explanations                    |
| **CI/CD Automation**    |   ✅   | 100% automated via YAML workflows                                |
| **Real-Time Dashboard** |   ✅   | 72-hour recursive forecast with live drift metrics               |

## 📖 Local Setup & Development

### 1. Prerequisites

- Python 3.11+
- MongoDB Atlas Cluster (Free Tier)

### 2. Installation

```bash
git clone <your-repo-url>
cd aqi-predictor
python -m venv .venv
# Activate: .\.venv\Scripts\Activate (Win) or source .venv/bin/activate (Unix)
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file:

```env
MONGO_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/aqi_predictor
```

### 4. Execution

```bash
python -m features.backfill_historical  # Backfill 2 years
python -m training.train                # Train & Select Champion
python -m training.evaluate             # Generate XAI Artifacts
streamlit run app/dashboard.py          # Launch Dashboard
```

---
