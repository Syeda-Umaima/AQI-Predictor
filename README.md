# AQI Predictor 🌫️

Professional end-to-end machine learning pipeline for real-time Air Quality Index (AQI) forecasting. This system provides a 72-hour recursive forecast for Hyderabad, Pakistan, utilizing a modern serverless stack.

## Tech Stack
- **Data Source**: [Open-Meteo API](https://open-meteo.com/) (Free, No-Key)
- **Feature Store & Model Registry**: MongoDB Atlas
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, TensorFlow (LSTM)
- **CI/CD**: GitHub Actions (Hourly Feature Pipeline, Daily Training Pipeline)
- **Dashboard**: Streamlit

## Key Features
- **Dynamic Model Routing**: Real-time validation system that selects the most accurate model based on the latest available data point.
- **Recursive Forecasting**: 1-hour ahead predictive architecture used recursively to generate a robust 72-hour outlook.
- **Explainability**: Integrated SHAP and LIME interpretations to provide transparency into model predictions.
- **Leakage Prevention**: Strict time-ordered splitting and feature engineering to ensure valid performance metrics.

## Local Development

### 1. Prerequisites
- Python 3.11
- MongoDB Atlas account (for Feature Store)

### 2. Setup
```powershell
# Clone repository
git clone <your-repo-url>
cd aqi-predictor

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```
MONGO_URI=your_mongodb_connection_string
```

### 4. Run Pipelines
```powershell
# Initialize feature store with historical data
python -m features.backfill_historical

# Train all models and select champion
python -m training.train
```

### 5. Start Dashboard
```powershell
streamlit run app/dashboard.py
```

## Repository Structure
- `app/`: Streamlit dashboard and dynamic routing logic.
- `features/`: API integration, feature engineering, and store management.
- `training/`: Model training, hyperparameter optimization, and champion promotion.
- `config/`: Centralized project configuration.
- `.github/workflows/`: CI/CD pipeline definitions.
