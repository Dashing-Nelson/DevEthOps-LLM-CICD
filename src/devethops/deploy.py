"""
Model deployment utilities for DevEthOps pipeline.

Handles Docker containerization, Kubernetes manifest generation,
and deployment orchestration with health checks.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import subprocess
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)


class DeploymentError(Exception):
    """Custom exception for deployment errors."""
    pass


class ModelDeployer:
    """
    Handles model deployment tasks including containerization and K8s deployment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model deployer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.deployment_config = config.get('deployment', {})
        
    def create_docker_image(self, model_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Create Docker image for model serving.
        
        Args:
            model_path: Path to saved model
            output_dir: Output directory for Docker files
            
        Returns:
            Docker build results
        """
        logger.info("Creating Docker image for model deployment...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Dockerfile
        dockerfile_content = self._generate_dockerfile()
        dockerfile_path = output_dir / "Dockerfile"
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create requirements.txt
        requirements_content = self._generate_requirements()
        requirements_path = output_dir / "requirements.txt"
        
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        # Create FastAPI serving application
        app_content = self._generate_serving_app()
        app_path = output_dir / "app.py"
        
        with open(app_path, 'w') as f:
            f.write(app_content)
        
        # Create health check script
        health_check_content = self._generate_health_check()
        health_check_path = output_dir / "health_check.py"
        
        with open(health_check_path, 'w') as f:
            f.write(health_check_content)
        
        # Create Docker build script
        build_script_content = self._generate_build_script()
        build_script_path = output_dir / "build.sh"
        
        with open(build_script_path, 'w') as f:
            f.write(build_script_content)
        
        # Make build script executable
        os.chmod(build_script_path, 0o755)
        
        return {
            'dockerfile_path': str(dockerfile_path),
            'requirements_path': str(requirements_path),
            'app_path': str(app_path),
            'health_check_path': str(health_check_path),
            'build_script_path': str(build_script_path),
            'image_tag': self._get_image_tag(),
            'build_command': f"cd {output_dir} && ./build.sh"
        }
    
    def generate_k8s_manifests(self, pipeline_id: str, output_dir: str) -> Dict[str, Any]:
        """
        Generate Kubernetes deployment manifests.
        
        Args:
            pipeline_id: Pipeline identifier
            output_dir: Output directory for manifests
            
        Returns:
            Generated manifest information
        """
        logger.info("Generating Kubernetes manifests...")
        
        output_dir = Path(output_dir)
        k8s_dir = output_dir / "k8s"
        k8s_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate deployment manifest
        deployment_manifest = self._generate_deployment_manifest(pipeline_id)
        deployment_path = k8s_dir / "deployment.yaml"
        
        with open(deployment_path, 'w') as f:
            yaml.dump(deployment_manifest, f, default_flow_style=False)
        
        # Generate service manifest
        service_manifest = self._generate_service_manifest(pipeline_id)
        service_path = k8s_dir / "service.yaml"
        
        with open(service_path, 'w') as f:
            yaml.dump(service_manifest, f, default_flow_style=False)
        
        # Generate configmap manifest
        configmap_manifest = self._generate_configmap_manifest(pipeline_id)
        configmap_path = k8s_dir / "configmap.yaml"
        
        with open(configmap_path, 'w') as f:
            yaml.dump(configmap_manifest, f, default_flow_style=False)
        
        # Generate ingress manifest (optional)
        ingress_manifest = self._generate_ingress_manifest(pipeline_id)
        ingress_path = k8s_dir / "ingress.yaml"
        
        with open(ingress_path, 'w') as f:
            yaml.dump(ingress_manifest, f, default_flow_style=False)
        
        # Generate deployment script
        deploy_script = self._generate_deploy_script(pipeline_id)
        deploy_script_path = k8s_dir / "deploy.sh"
        
        with open(deploy_script_path, 'w') as f:
            f.write(deploy_script)
        
        os.chmod(deploy_script_path, 0o755)
        
        return {
            'deployment_path': str(deployment_path),
            'service_path': str(service_path),
            'configmap_path': str(configmap_path),
            'ingress_path': str(ingress_path),
            'deploy_script_path': str(deploy_script_path),
            'namespace': self._get_namespace(pipeline_id),
            'app_name': self._get_app_name(pipeline_id)
        }
    
    def create_health_checks(self) -> Dict[str, Any]:
        """
        Create health check configuration.
        
        Returns:
            Health check configuration
        """
        return {
            'liveness_probe': {
                'http_get': {
                    'path': '/health',
                    'port': 8000
                },
                'initial_delay_seconds': 30,
                'period_seconds': 10,
                'timeout_seconds': 5,
                'failure_threshold': 3
            },
            'readiness_probe': {
                'http_get': {
                    'path': '/ready',
                    'port': 8000
                },
                'initial_delay_seconds': 5,
                'period_seconds': 5,
                'timeout_seconds': 3,
                'failure_threshold': 3
            },
            'startup_probe': {
                'http_get': {
                    'path': '/startup',
                    'port': 8000
                },
                'initial_delay_seconds': 10,
                'period_seconds': 10,
                'timeout_seconds': 5,
                'failure_threshold': 30
            }
        }
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile content."""
        python_version = self.deployment_config.get('python_version', '3.9')
        
        return f'''# DevEthOps Model Serving Dockerfile
FROM python:{python_version}-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY health_check.py .

# Copy model files (add these during build)
# COPY model/ ./model/

# Create non-root user
RUN useradd -m -u 1001 modeluser
USER modeluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python health_check.py

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt content."""
        return '''# DevEthOps Model Serving Requirements
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
python-multipart==0.0.6
aiofiles==23.2.1

# Optional dependencies
# torch==2.1.0
# transformers==4.35.0
# xgboost==2.0.1

# Monitoring
prometheus-client==0.19.0
'''
    
    def _generate_serving_app(self) -> str:
        """Generate FastAPI serving application."""
        return '''"""
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
    return "# DevEthOps Model Metrics\\nmodel_predictions_total 0\\n"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    def _generate_health_check(self) -> str:
        """Generate health check script."""
        return '''#!/usr/bin/env python3
"""
Health check script for DevEthOps model container.
"""

import requests
import sys
import os

def main():
    """Perform health check."""
    try:
        health_url = "http://localhost:8000/health"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy":
                print("Health check passed")
                sys.exit(0)
            else:
                print(f"Health check failed: {health_data}")
                sys.exit(1)
        else:
            print(f"Health check failed with status {response.status_code}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Health check failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _generate_build_script(self) -> str:
        """Generate Docker build script."""
        image_tag = self._get_image_tag()
        
        return f'''#!/bin/bash
# DevEthOps Model Docker Build Script

set -e

IMAGE_TAG="{image_tag}"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "Building Docker image: $IMAGE_TAG"

docker build \\
    --build-arg BUILD_DATE="$BUILD_DATE" \\
    --build-arg GIT_COMMIT="$GIT_COMMIT" \\
    -t "$IMAGE_TAG" \\
    .

echo "Build complete: $IMAGE_TAG"
echo "To run: docker run -p 8000:8000 $IMAGE_TAG"
'''
    
    def _generate_deployment_manifest(self, pipeline_id: str) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        app_name = self._get_app_name(pipeline_id)
        namespace = self._get_namespace(pipeline_id)
        image_tag = self._get_image_tag()
        
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': app_name,
                'namespace': namespace,
                'labels': {
                    'app': app_name,
                    'pipeline': pipeline_id,
                    'component': 'model-serving'
                }
            },
            'spec': {
                'replicas': self.deployment_config.get('replicas', 2),
                'selector': {
                    'matchLabels': {
                        'app': app_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': app_name,
                            'pipeline': pipeline_id
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'model-server',
                            'image': image_tag,
                            'ports': [{
                                'containerPort': 8000,
                                'name': 'http'
                            }],
                            'env': [
                                {
                                    'name': 'MODEL_PATH',
                                    'value': '/app/model/model.joblib'
                                },
                                {
                                    'name': 'PREPROCESSOR_PATH',
                                    'value': '/app/model/preprocessor.joblib'
                                },
                                {
                                    'name': 'PIPELINE_ID',
                                    'value': pipeline_id
                                }
                            ],
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5,
                                'timeoutSeconds': 3,
                                'failureThreshold': 3
                            },
                            'startupProbe': {
                                'httpGet': {
                                    'path': '/startup',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 10,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 30
                            },
                            'resources': {
                                'requests': {
                                    'memory': '512Mi',
                                    'cpu': '250m'
                                },
                                'limits': {
                                    'memory': '1Gi',
                                    'cpu': '500m'
                                }
                            },
                            'volumeMounts': [{
                                'name': 'model-storage',
                                'mountPath': '/app/model'
                            }]
                        }],
                        'volumes': [{
                            'name': 'model-storage',
                            'configMap': {
                                'name': f'{app_name}-config'
                            }
                        }]
                    }
                }
            }
        }
    
    def _generate_service_manifest(self, pipeline_id: str) -> Dict[str, Any]:
        """Generate Kubernetes service manifest."""
        app_name = self._get_app_name(pipeline_id)
        namespace = self._get_namespace(pipeline_id)
        
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{app_name}-service',
                'namespace': namespace,
                'labels': {
                    'app': app_name,
                    'pipeline': pipeline_id
                }
            },
            'spec': {
                'selector': {
                    'app': app_name
                },
                'ports': [{
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': 8000,
                    'name': 'http'
                }],
                'type': 'ClusterIP'
            }
        }
    
    def _generate_configmap_manifest(self, pipeline_id: str) -> Dict[str, Any]:
        """Generate Kubernetes ConfigMap manifest."""
        app_name = self._get_app_name(pipeline_id)
        namespace = self._get_namespace(pipeline_id)
        
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f'{app_name}-config',
                'namespace': namespace,
                'labels': {
                    'app': app_name,
                    'pipeline': pipeline_id
                }
            },
            'data': {
                'MODEL_CONFIG': json.dumps({
                    'pipeline_id': pipeline_id,
                    'model_version': '1.0.0',
                    'deployment_timestamp': datetime.now().isoformat()
                }),
                'MONITORING_CONFIG': json.dumps({
                    'drift_detection_enabled': True,
                    'fairness_monitoring_enabled': True,
                    'metrics_collection_interval': 60
                })
            }
        }
    
    def _generate_ingress_manifest(self, pipeline_id: str) -> Dict[str, Any]:
        """Generate Kubernetes Ingress manifest."""
        app_name = self._get_app_name(pipeline_id)
        namespace = self._get_namespace(pipeline_id)
        
        return {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f'{app_name}-ingress',
                'namespace': namespace,
                'labels': {
                    'app': app_name,
                    'pipeline': pipeline_id
                },
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'false'
                }
            },
            'spec': {
                'rules': [{
                    'host': f'{app_name}.local',
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': f'{app_name}-service',
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
    
    def _generate_deploy_script(self, pipeline_id: str) -> str:
        """Generate Kubernetes deployment script."""
        namespace = self._get_namespace(pipeline_id)
        
        return f'''#!/bin/bash
# DevEthOps Kubernetes Deployment Script

set -e

NAMESPACE="{namespace}"
PIPELINE_ID="{pipeline_id}"

echo "Deploying DevEthOps model for pipeline: $PIPELINE_ID"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply ConfigMap
echo "Applying ConfigMap..."
kubectl apply -f configmap.yaml

# Apply Deployment
echo "Applying Deployment..."
kubectl apply -f deployment.yaml

# Apply Service
echo "Applying Service..."
kubectl apply -f service.yaml

# Apply Ingress (optional)
echo "Applying Ingress..."
kubectl apply -f ingress.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/{self._get_app_name(pipeline_id)} -n $NAMESPACE

echo "Deployment complete!"
echo "Service endpoint: http://{self._get_app_name(pipeline_id)}.local"
echo "Health check: kubectl get pods -n $NAMESPACE"
'''
    
    def _get_image_tag(self) -> str:
        """Get Docker image tag."""
        registry = self.deployment_config.get('registry', 'devethops')
        image_name = self.deployment_config.get('image_name', 'model-server')
        version = self.deployment_config.get('version', 'latest')
        return f"{registry}/{image_name}:{version}"
    
    def _get_app_name(self, pipeline_id: str) -> str:
        """Get Kubernetes app name."""
        return f"devethops-{pipeline_id[:8]}"
    
    def _get_namespace(self, pipeline_id: str) -> str:
        """Get Kubernetes namespace."""
        return self.deployment_config.get('namespace', 'devethops')


def create_deployment_package(model_path: str, config: Dict[str, Any],
                            pipeline_id: str, output_dir: str) -> Dict[str, Any]:
    """
    Create complete deployment package.
    
    Args:
        model_path: Path to trained model
        config: Configuration dictionary
        pipeline_id: Pipeline identifier
        output_dir: Output directory
        
    Returns:
        Deployment package information
    """
    logger.info(f"Creating deployment package for pipeline: {pipeline_id}")
    
    deployer = ModelDeployer(config)
    
    # Create Docker artifacts
    docker_results = deployer.create_docker_image(model_path, output_dir)
    
    # Create Kubernetes manifests
    k8s_results = deployer.generate_k8s_manifests(pipeline_id, output_dir)
    
    # Create health checks
    health_checks = deployer.create_health_checks()
    
    return {
        'pipeline_id': pipeline_id,
        'docker': docker_results,
        'kubernetes': k8s_results,
        'health_checks': health_checks,
        'deployment_ready': True
    }
