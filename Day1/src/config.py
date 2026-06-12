# ══════════════════════════════════════════
# FILE: src/config.py
# Load tất cả config từ .env vào một nơi
# ══════════════════════════════════════════
import os
from dotenv import load_dotenv

load_dotenv()

# MLflow
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "house-price-prediction")

# App
APP_ENV   = os.getenv("APP_ENV", "development")
APP_PORT  = int(os.getenv("APP_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Paths
MODEL_PATH   = os.getenv("MODEL_PATH", "artifacts/best_model.pkl")
SCALER_PATH  = os.getenv("SCALER_PATH", "artifacts/scaler.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "artifacts/metrics.json")

# Training hyperparams
TEST_SIZE    = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))
