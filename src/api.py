from typing import List
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .loader import get_data
from .model import IsolationForestModel
from .prediction import train_and_evaluate
from .processing import process_data


class DeliveryEstimateRequest(BaseModel):
    product_weight_g: float = Field(..., ge=0, description="Weight in grams")
    product_vol_cm3: float = Field(..., ge=0, description="Package volume in cubic centimeters")
    distance_km: float = Field(..., ge=0, description="Distance from seller to customer")
    customer_lat: float = Field(..., ge=-90.0, le=90.0, description="Customer latitude")
    customer_lng: float = Field(..., ge=-180.0, le=180.0, description="Customer longitude")
    seller_lat: float = Field(..., ge=-90.0, le=90.0, description="Seller latitude")
    seller_lng: float = Field(..., ge=-180.0, le=180.0, description="Seller longitude")
    payment_lag_days: float = Field(..., ge=0, le=60, description="Days between purchase and payment approval")
    is_weekend_order: bool = Field(..., description="True when the purchase falls on a weekend")
    freight_value: float = Field(..., ge=0, description="Freight price paid in BRL")
    purchase_month: int = Field(..., ge=1, le=12, description="Numeric month of purchase")

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "product_weight_g": 1200,
                "product_vol_cm3": 4500,
                "distance_km": 800,
                "customer_lat": -23.55,
                "customer_lng": -46.63,
                "seller_lat": -23.95,
                "seller_lng": -46.33,
                "payment_lag_days": 2,
                "is_weekend_order": False,
                "freight_value": 29.9,
                "purchase_month": 11,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_days: float
    r2_score: float
    mae: float
    warnings: List[str]
    message: str


class PredictionEngine:
    def __init__(self):
        self.ready = False
        self.model = None
        self.features: List[str] = []
        self.metrics = {}
        self.record_count = 0
        self._dataframe = pd.DataFrame()

    def train(self) -> None:
        raw = get_data()
        processed = process_data(raw)
        processed = IsolationForestModel(processed)

        df, model, r2, mae, features = train_and_evaluate(processed)

        self._dataframe = df
        self.model = model
        self.features = features
        self.metrics = {"r2": r2, "mae": mae}
        self.record_count = len(df)
        self.ready = True

    def predict(self, payload: DeliveryEstimateRequest) -> float:
        if not self.ready or self.model is None:
            raise RuntimeError("Prediction engine is not initialized")

        data = payload.dict()
        data["is_weekend_order"] = 1 if data["is_weekend_order"] else 0
        candidate = pd.DataFrame([data])

        try:
            candidate = candidate[self.features]
        except KeyError as exc:
            raise RuntimeError(f"Payload missing expected feature: {exc}") from exc

        prediction = self.model.predict(candidate)
        return float(prediction[0])

    def describe_warnings(self, payload: DeliveryEstimateRequest) -> List[str]:
        warnings = []
        if payload.distance_km > 3000:
            warnings.append("very large distance (>3000 km)")
        if payload.distance_km < 10:
            warnings.append("very small distance (<10 km)")
        if payload.product_weight_g > 20000:
            warnings.append("very high weight (>20 kg)")
        if payload.product_weight_g < 100:
            warnings.append("very low weight (<100 g)")
        if payload.freight_value > 500:
            warnings.append("very high shipping price (>500 BRL)")
        if payload.product_vol_cm3 > 50000:
            warnings.append("very large volume (>50000 cm³)")
        return warnings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    # Startup
    await startup()
    yield
    # Shutdown (if needed)
    pass


app = FastAPI(
    title="Delivery Time Estimation API",
    description="XGBoost-backed prediction service with Pydantic validation",
    version="1.0.0",
    lifespan=lifespan  # Use new lifespan pattern
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PredictionEngine()


async def startup():
    """Initialize prediction engine on startup."""
    try:
        engine.train()  # Changed from engine.initialize()
    except Exception as e:
        print(f"Failed to initialize prediction engine: {e}")


@app.get("/health")
def health() -> dict:
    if not engine.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction engine is still warming up",
        )

    return {
        "status": "ready",
        "records": engine.record_count,
        "r2_score": engine.metrics.get("r2"),
        "mae": engine.metrics.get("mae"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: DeliveryEstimateRequest) -> PredictionResponse:
    if not engine.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction engine is still warming up",
        )

    try:
        predicted_days = engine.predict(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    warnings = engine.describe_warnings(payload)
    return PredictionResponse(
        predicted_days=predicted_days,
        r2_score=engine.metrics.get("r2", 0.0),
        mae=engine.metrics.get("mae", 0.0),
        warnings=warnings,
        message="Validated prediction available",
    )


@app.get("/")
async def root():
    """
    Root endpoint - API information and available routes.
    """
    return {
        "service": "Delivery Time Estimation API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict (POST)",
            "openapi": "/openapi.json"
        }
    }
