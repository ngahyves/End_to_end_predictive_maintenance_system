import os
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import shap
from typing import Dict, Literal
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# Initializing logger
logger = get_logger("API_Serving")
cfg = load_config()

# Global dictionary to hold models in RAM
ML_RESOURCES = {}

# --- DATA MODELS (PYDANTIC) ---

class MachineInput(BaseModel):
    """Data contract for incoming sensor data with validation."""
    type: Literal["L", "M", "H"] = Field(..., alias="Type", example="L") #To force these 3 categories only
    air_temp: float = Field(..., alias="Air temperature [K]", ge=200, le=400)
    process_temp: float = Field(..., alias="Process temperature [K]", ge=200, le=400)
    rpm: int = Field(..., alias="Rotational speed [rpm]", ge=0)
    torque: float = Field(..., alias="Torque [Nm]", ge=0)
    tool_wear: int = Field(..., alias="Tool wear [min]", ge=0)

    class Config:
        populate_by_name = True

class PredictionResult(BaseModel):
    label: int
    probability: float

class APIResponse(BaseModel):
    machine_status: str
    predictions: Dict[str, PredictionResult]
    latency_ms: float

# --- LIFESPAN MANAGEMENT (STARTUP/SHUTDOWN) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing API... Loading local artifacts.")
    try:
        prep_path = "artifacts/preprocessor.joblib"
        model_path = "artifacts/model.pkl"
        
        logger.info(f"Loading preprocessor from: {prep_path}")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Missing: {prep_path}")
        ML_RESOURCES["preprocessor"] = joblib.load(prep_path)
        
        logger.info(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing: {model_path}")
        ML_RESOURCES["model"] = joblib.load(model_path)
        
        # Initializing SHAP
        base_estimator = ML_RESOURCES["model"].estimators_[0]
        ML_RESOURCES["explainer"] = shap.TreeExplainer(base_estimator)
        
        logger.info("SUCCESS: API resources loaded.")
    except Exception as e:
        logger.error(f" FATAL STARTUP ERROR: {e}")
        # Display the content if we have errors
        logger.info(f"Current directory contents: {os.listdir('.')}")
        raise RuntimeError(f"Could not load ML artifacts: {e}")
    yield
    ML_RESOURCES.clear()

app = FastAPI(title="Industrial Maintenance API", lifespan=lifespan)

# --- MONITORING ---
Instrumentator().instrument(app).expose(app)

# --- MIDDLEWARE ---
@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start_time) * 1000
    response.headers["X-Inference-Latency"] = f"{latency:.2f}ms"
    return response

# --- ENDPOINTS ---

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/predict", response_model=APIResponse)
async def predict(data: MachineInput):
    """Performs real-time multi-label inference."""
    start_time = time.perf_counter()
    try:
        # Convert Pydantic model to DataFrame for the pipeline
        input_df = pd.DataFrame([data.dict(by_alias=True)])
        
        # Apply the exact same preprocessing used during training
        X_transformed = ML_RESOURCES["preprocessor"].transform(input_df)
        
        # Run inference
        labels = ML_RESOURCES["model"].predict(X_transformed)[0]
        probabilities = ML_RESOURCES["model"].predict_proba(X_transformed)
        
        target_names = cfg["features"]["targets"]
        final_preds = {}
        is_failure = False

        for i, name in enumerate(target_names):
            prob = probabilities[i][0][1] # Probability of class 1
            final_preds[name] = {
                "label": int(labels[i]),
                "probability": round(float(prob), 4)
            }
            if labels[i] == 1: is_failure = True

        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            "machine_status": "ALERT" if is_failure else "STABLE",
            "predictions": final_preds,
            "latency_ms": round(latency, 2)
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Inference Failure")

@app.post("/explain")
async def explain(data: MachineInput):
    """Returns SHAP contributions for a specific prediction."""
    try:
        input_df = pd.DataFrame([data.dict(by_alias=True)])
        X_transformed = ML_RESOURCES["preprocessor"].transform(input_df)
        feature_names = ML_RESOURCES["preprocessor"].get_feature_names_out()
        
        # 1. SHAP values computing
        shap_vals = ML_RESOURCES["explainer"].shap_values(X_transformed)
        expected_val = ML_RESOURCES["explainer"].expected_value

        # 2. If it is a is a list, we take index 1  (failure probability)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = expected_val[1]

        return {
            "feature_contributions": dict(zip(feature_names, shap_vals[0].tolist())),
            "base_value": float(expected_val)
        }
    except Exception as e:
        logger.error(f"SHAP Error: {e}")
        # returning a specific error for debugging
        raise HTTPException(status_code=500, detail=f"Explainability failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)