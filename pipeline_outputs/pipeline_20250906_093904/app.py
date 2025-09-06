"""
FastAPI application for serving DevEthOps models.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DevEthOps Model Serving API",
    description="API for serving ethical ML models with fairness monitoring",
    version="1.0.0"
)

# Global model variable
model = None
preprocessor = None
model_metadata = {}

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, Any]
    
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int
    probability: float
    confidence: float
    model_version: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, preprocessor, model_metadata
    
    try:
        # Load model (implement actual loading logic)
        model_path = os.getenv("MODEL_PATH", "/app/model/model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
        
        # Load preprocessor
        preprocessor_path = os.getenv("PREPROCESSOR_PATH", "/app/model/preprocessor.joblib")
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        
        # Load metadata
        metadata_path = os.getenv("METADATA_PATH", "/app/model/metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Model metadata loaded")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "DevEthOps Model Serving API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version=model_metadata.get("version", "unknown")
    )

@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint."""
    return HealthResponse(
        status="ready" if model is not None else "not_ready",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version=model_metadata.get("version", "unknown")
    )

@app.get("/startup", response_model=HealthResponse)
async def startup_check():
    """Startup check endpoint."""
    return HealthResponse(
        status="started",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version=model_metadata.get("version", "unknown")
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make single prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Preprocess if preprocessor available
        if preprocessor is not None:
            features_processed = preprocessor.transform(features_df)
        else:
            features_processed = features_df.values
        
        # Make prediction
        prediction = model.predict(features_processed)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_processed)[0]
            probability = float(probabilities[1])  # Positive class probability
            confidence = float(max(probabilities))
        else:
            probability = float(prediction)
            confidence = 1.0
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            confidence=confidence,
            model_version=model_metadata.get("version", "unknown"),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame(request.features)
        
        # Preprocess if preprocessor available
        if preprocessor is not None:
            features_processed = preprocessor.transform(features_df)
        else:
            features_processed = features_df.values
        
        # Make predictions
        predictions = model.predict(features_processed)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_processed)
            positive_probs = probabilities[:, 1].tolist()
            confidences = np.max(probabilities, axis=1).tolist()
        else:
            positive_probs = predictions.tolist()
            confidences = [1.0] * len(predictions)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "prediction": int(pred),
                "probability": positive_probs[i],
                "confidence": confidences[i],
                "model_version": model_metadata.get("version", "unknown"),
                "timestamp": datetime.now().isoformat()
            })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "metadata": model_metadata,
        "preprocessor_available": preprocessor is not None,
        "loaded_at": datetime.now().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint (placeholder)."""
    # TODO: Implement actual Prometheus metrics
    return "# DevEthOps Model Metrics\nmodel_predictions_total 0\n"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
