"""
FastAPI application for serving DevEthOps models with ethical monitoring.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('devethops_predictions_total', 'Total predictions made')
prediction_latency = Histogram('devethops_prediction_duration_seconds', 'Prediction latency')
model_load_gauge = Gauge('devethops_model_loaded', 'Whether model is loaded')
fairness_violations = Counter('devethops_fairness_violations_total', 'Fairness violations detected')

# Initialize FastAPI app
app = FastAPI(
    title="DevEthOps Model Serving API",
    description="API for serving ethical ML models with fairness monitoring and bias detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
loaded_model = None
model_metadata = None
preprocessor = None
model = None
preprocessor = None
model_metadata = {}
fairness_checker = None

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for single predictions."""
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")
    include_explanation: bool = Field(False, description="Include model explanation")
    fairness_check: bool = Field(True, description="Perform fairness check")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    include_explanation: bool = Field(False, description="Include explanations")
    fairness_check: bool = Field(True, description="Perform fairness checks")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Model prediction")
    probability: float = Field(..., description="Prediction probability")
    confidence: float = Field(..., description="Prediction confidence")
    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Prediction timestamp")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Model explanation")
    fairness_metrics: Optional[Dict[str, float]] = Field(None, description="Fairness metrics")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    batch_size: int
    processing_time: float

class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Model version")
    fairness_monitoring: bool = Field(..., description="Fairness monitoring status")

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_type: str
    version: str
    features: List[str]
    target: str
    training_date: Optional[str]
    fairness_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    """Load model and initialize services on startup."""
    global model, preprocessor, model_metadata, fairness_checker
    
    try:
        # Load model
        model_path = os.getenv("MODEL_PATH", "/app/models/model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            model_load_gauge.set(1)
        else:
            logger.warning(f"Model file not found: {model_path}")
            model_load_gauge.set(0)
        
        # Load preprocessor
        preprocessor_path = os.getenv("PREPROCESSOR_PATH", "/app/models/preprocessor.joblib")
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        
        # Load metadata
        metadata_path = os.getenv("METADATA_PATH", "/app/models/metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Model metadata loaded")
        
        # Initialize fairness checker
        try:
            from ..fairness_checks import FairnessChecker
            fairness_checker = FairnessChecker({})
            logger.info("Fairness checker initialized")
        except ImportError:
            logger.warning("Fairness checker not available")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        model_load_gauge.set(0)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DevEthOps Model Serving API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version=model_metadata.get("version", "unknown"),
        fairness_monitoring=fairness_checker is not None
    )

@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint."""
    return HealthResponse(
        status="ready" if model is not None else "not_ready",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version=model_metadata.get("version", "unknown"),
        fairness_monitoring=fairness_checker is not None
    )

@app.get("/startup", response_model=HealthResponse)
async def startup_check():
    """Startup check endpoint."""
    return HealthResponse(
        status="started",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version=model_metadata.get("version", "unknown"),
        fairness_monitoring=fairness_checker is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make single prediction with optional fairness check."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with prediction_latency.time():
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
            
            # Prepare response
            response = PredictionResponse(
                prediction=int(prediction),
                probability=probability,
                confidence=confidence,
                model_version=model_metadata.get("version", "unknown"),
                timestamp=datetime.now().isoformat()
            )
            
            # Add explanation if requested
            if request.include_explanation:
                try:
                    from ..explainability import generate_explanation
                    explanation = generate_explanation(model, features_df, prediction)
                    response.explanation = explanation
                except Exception as e:
                    logger.warning(f"Explanation generation failed: {e}")
            
            # Perform fairness check if requested
            if request.fairness_check and fairness_checker is not None:
                try:
                    fairness_metrics = fairness_checker.check_single_prediction(
                        features_df, prediction
                    )
                    response.fairness_metrics = fairness_metrics
                    
                    # Check for fairness violations
                    if any(metric > 0.1 for metric in fairness_metrics.values()):
                        fairness_violations.inc()
                        
                except Exception as e:
                    logger.warning(f"Fairness check failed: {e}")
            
            # Update metrics
            prediction_counter.inc()
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
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
        
        # Prepare responses
        results = []
        for i, pred in enumerate(predictions):
            pred_response = PredictionResponse(
                prediction=int(pred),
                probability=positive_probs[i],
                confidence=confidences[i],
                model_version=model_metadata.get("version", "unknown"),
                timestamp=datetime.now().isoformat()
            )
            results.append(pred_response)
        
        # Update metrics
        prediction_counter.inc(len(predictions))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionResponse(
            predictions=results,
            batch_size=len(predictions),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get detailed model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type=type(model).__name__,
        version=model_metadata.get("version", "unknown"),
        features=model_metadata.get("features", []),
        target=model_metadata.get("target", "unknown"),
        training_date=model_metadata.get("training_date"),
        fairness_metrics=model_metadata.get("fairness_metrics", {}),
        performance_metrics=model_metadata.get("performance_metrics", {})
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest().decode('utf-8')

@app.get("/metrics/fairness")
async def fairness_metrics():
    """Get current fairness metrics."""
    if fairness_checker is None:
        raise HTTPException(status_code=503, detail="Fairness monitoring not available")
    
    try:
        metrics = fairness_checker.get_current_metrics()
        return {"fairness_metrics": metrics, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Fairness metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get fairness metrics: {str(e)}")

@app.post("/explain")
async def explain_prediction(request: PredictionRequest):
    """Get detailed explanation for a prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_df = pd.DataFrame([request.features])
        
        # Generate explanation
        from ..explainability import run_explainability_analysis
        explanation = run_explainability_analysis(
            model, features_df, model_metadata.get("feature_names", [])
        )
        
        return {
            "explanation": explanation,
            "features": request.features,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app

if __name__ == "__main__":
    uvicorn.run(
        "devethops.api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "production") == "development"
    )
