import pytest
from fastapi.testclient import TestClient
from app.main import app 

# --- 1. Testing the Health Check ---
def test_read_health():
    """Vrifying if API is running and models are loaded"""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

# --- 2. Predictions test ---
def test_predict_success():
    """Verifying if we have correct inputs to make predictions."""
    valid_payload = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 5

# --- 3. Test for Pydantic validation(422 error) ---
def test_predict_invalid_data():
    """Verify if the API reject invalid inputs (ex: Type unknown)."""
    invalid_payload = {
        "Type": "Z", # 'Z' doesn't exist in our contract (L, M, H)
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 0
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
        # 422 = Unprocessable Entity (Pydantic validation error)
        assert response.status_code == 422

# --- 4. Explainability test (SHAP) ---
def test_explain_endpoint():
    """Verify if router /explain return des numeric contributions of each feature."""
    payload = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1400,
        "Torque [Nm]": 45.0,
        "Tool wear [min]": 100
    }
    with TestClient(app) as client:
        response = client.post("/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "feature_contributions" in data