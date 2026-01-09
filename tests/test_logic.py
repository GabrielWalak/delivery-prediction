"""
Unit tests for FastAPI endpoints and prediction logic.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_engine():
    """Mock the global engine instance to avoid Kaggle downloads."""
    with patch('src.api.engine') as mock_engine_instance:
        # Configure mock attributes
        mock_engine_instance.ready = True
        mock_engine_instance.record_count = 109635
        mock_engine_instance.metrics = {"r2": 0.4117, "mae": 4.36}
        mock_engine_instance.features = [
            'product_weight_g', 'product_vol_cm3', 'distance_km',
            'customer_lat', 'customer_lng', 'seller_lat', 'seller_lng',
            'payment_lag_days', 'is_weekend_order', 'freight_value', 'purchase_month'
        ]
        
        # Mock predict method
        mock_engine_instance.predict.return_value = 7.5
        
        # Mock describe_warnings method
        mock_engine_instance.describe_warnings.return_value = []
        
        yield mock_engine_instance


def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Delivery Time Estimation API"
    assert "endpoints" in data


def test_health_endpoint():
    """Test health endpoint returns system status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["records"] == 109635
    assert "r2_score" in data
    assert "mae" in data


def test_prediction_endpoint_valid_payload():
    """Test prediction with valid data matching API schema."""
    payload = {
        "product_weight_g": 1200.0,
        "product_vol_cm3": 4500.0,
        "distance_km": 800.0,
        "customer_lat": -23.55,
        "customer_lng": -46.63,
        "seller_lat": -23.95,
        "seller_lng": -46.33,
        "payment_lag_days": 2.0,
        "is_weekend_order": False,
        "freight_value": 29.9,
        "purchase_month": 11
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_days" in data
    assert isinstance(data["predicted_days"], float)
    assert data["predicted_days"] == 7.5
    assert data["r2_score"] == 0.4117
    assert data["mae"] == 4.36


def test_prediction_endpoint_invalid_payload():
    """Test prediction with missing required fields."""
    payload = {
        "product_weight_g": 1200.0,
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_endpoint_invalid_values():
    """Test prediction with out-of-range values."""
    payload = {
        "product_weight_g": -100.0,
        "product_vol_cm3": 4500.0,
        "distance_km": 800.0,
        "customer_lat": -23.55,
        "customer_lng": -46.63,
        "seller_lat": -23.95,
        "seller_lng": -46.33,
        "payment_lag_days": 2.0,
        "is_weekend_order": False,
        "freight_value": 29.9,
        "purchase_month": 11
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_endpoint_invalid_coordinates():
    """Test prediction with invalid latitude."""
    payload = {
        "product_weight_g": 1200.0,
        "product_vol_cm3": 4500.0,
        "distance_km": 800.0,
        "customer_lat": 95.0,  # Invalid: > 90
        "customer_lng": -46.63,
        "seller_lat": -23.95,
        "seller_lng": -46.33,
        "payment_lag_days": 2.0,
        "is_weekend_order": False,
        "freight_value": 29.9,
        "purchase_month": 11
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422