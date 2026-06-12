# ══════════════════════════════════════════
# FILE: src/serve.py — FastAPI Model Serving
# Chạy: uvicorn src.serve:app --reload
# Docs: http://localhost:8000/docs
# ══════════════════════════════════════════
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np

from src.config import MODEL_PATH, SCALER_PATH

app = FastAPI(title="House Price API", version="1.0")

# Load model + scaler khi startup
try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        SCALER = pickle.load(f)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    MODEL, SCALER = None, None
    print("⚠️  No model found — run main.py first")


class HouseFeatures(BaseModel):
    """8 features của California Housing dataset."""
    MedInc: float = Field(..., gt=0, description="Median income (×$10k)")
    HouseAge: float = Field(..., ge=0, le=100)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=32, le=42)
    Longitude: float = Field(..., ge=-125, le=-114)

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 3.5, "HouseAge": 20.0, "AveRooms": 5.0,
                "AveBedrms": 1.1, "Population": 800.0, "AveOccup": 2.8,
                "Latitude": 37.5, "Longitude": -122.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price_100k: float
    predicted_price_usd: int
    model_version: str = "1.0"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run main.py first")

    input_array = np.array([[
        features.MedInc, features.HouseAge, features.AveRooms,
        features.AveBedrms, features.Population, features.AveOccup,
        features.Latitude, features.Longitude,
    ]])

    input_scaled = SCALER.transform(input_array)
    price_100k = float(MODEL.predict(input_scaled)[0])

    return PredictionResponse(
        predicted_price_100k=round(price_100k, 4),
        predicted_price_usd=int(price_100k * 100_000),
    )


@app.get("/")
def root():
    return {
        "message": "House Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
