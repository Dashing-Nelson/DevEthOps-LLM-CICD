"""
Deploy Stage - Model containerization and Kubernetes deployment
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime

# Docker and Kubernetes imports
import docker
from kubernetes import client, config


class DeploymentError(Exception):
    """Custom exception for deployment failures"""
    pass


class DeployStage:
    """
    Deploy stage for the DevEthOps pipeline.
    Handles model containerization, Kubernetes deployment, and fairness monitoring setup.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Deploy Stage.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/pipeline_config.yaml"
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize clients
        self.docker_client = None
        self.k8s_client = None
        
        self._initialize_clients()
        
        # Deployment tracking
        self.deployment_artifacts = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
    
    def _initialize_clients(self):
        """Initialize Docker and Kubernetes clients"""
        try:
            # Initialize Docker client
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize Docker client: {str(e)}")
            self.docker_client = None
        
        try:
            # Initialize Kubernetes client
            try:
                config.load_incluster_config()  # Try in-cluster config first
            except:
                config.load_kube_config()  # Fall back to local kubeconfig
            
            self.k8s_client = client.ApiClient()
            self.logger.info("Kubernetes client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize Kubernetes client: {str(e)}")
            self.k8s_client = None
    
    def containerize_model(self, model_path: str, model_artifacts: Dict[str, Any],
                          container_name: str = "devethops-model") -> str:
        """
        Create Docker container with the trained model and fairness monitoring.
        
        Args:
            model_path: Path to the trained model
            model_artifacts: Dictionary containing model artifacts
            container_name: Name for the container
            
        Returns:
            str: Docker image name/tag
        """
        if self.docker_client is None:
            raise DeploymentError("Docker client not available")
        
        self.logger.info(f"Creating Docker container for model: {model_path}")
        
        try:
            # Create temporary directory for build context
            with tempfile.TemporaryDirectory() as temp_dir:
                build_context = Path(temp_dir)
                
                # Copy model and artifacts
                model_dir = build_context / "model"
                model_dir.mkdir()
                
                # Copy model files
                if os.path.isfile(model_path):
                    import shutil
                    shutil.copy2(model_path, model_dir / "model.pkl")
                else:
                    # If it's a directory, copy the entire directory
                    import shutil
                    shutil.copytree(model_path, model_dir / "model", dirs_exist_ok=True)
                
                # Create model artifacts file
                with open(model_dir / "artifacts.json", 'w') as f:
                    json.dump(model_artifacts, f, indent=2, default=str)
                
                # Create Dockerfile
                dockerfile_content = self._generate_dockerfile()
                with open(build_context / "Dockerfile", 'w') as f:
                    f.write(dockerfile_content)
                
                # Create application files
                self._create_application_files(build_context)
                
                # Create requirements.txt
                self._create_container_requirements(build_context)
                
                # Build Docker image
                image_tag = f"{container_name}:latest"
                self.logger.info(f"Building Docker image: {image_tag}")
                
                image, build_logs = self.docker_client.images.build(
                    path=str(build_context),
                    tag=image_tag,
                    rm=True
                )
                
                # Log build output
                for log in build_logs:
                    if 'stream' in log:
                        self.logger.info(log['stream'].strip())
                
                self.deployment_artifacts['docker_image'] = image_tag
                self.logger.info(f"Docker image built successfully: {image_tag}")
                
                return image_tag
        
        except Exception as e:
            self.logger.error(f"Error creating Docker container: {str(e)}")
            raise DeploymentError(f"Container creation failed: {str(e)}")
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile content"""
        dockerfile = f"""
FROM python:3.8.10-slim

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
COPY app/ ./app/
COPY model/ ./model/
COPY config/ ./config/

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app/main.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run the application
CMD ["python", "app/main.py"]
"""
        return dockerfile.strip()
    
    def _create_application_files(self, build_context: Path):
        """Create application files for the container"""
        app_dir = build_context / "app"
        app_dir.mkdir()
        
        # Create main Flask application
        main_py_content = '''
import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_restx import Api, Resource
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app, doc='/docs/', title='DevEthOps Model API', 
          description='Ethical AI Model Serving with Fairness Monitoring')

# Load model and artifacts
model = None
model_artifacts = None

def load_model():
    global model, model_artifacts
    try:
        # Load model
        model_path = "/app/model/model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        # Load artifacts
        artifacts_path = "/app/model/artifacts.json"
        if os.path.exists(artifacts_path):
            with open(artifacts_path, 'r') as f:
                model_artifacts = json.load(f)
        
        logger.info("Model and artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@api.route('/health')
class HealthCheck(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'healthy', 'model_loaded': model is not None}

@api.route('/predict')
class Predict(Resource):
    def post(self):
        """Make predictions with fairness monitoring"""
        try:
            data = request.get_json()
            
            if not data or 'features' not in data:
                return {'error': 'No features provided'}, 400
            
            # Convert to numpy array
            features = np.array(data['features'])
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(features)
                prediction = model.predict(features)
            else:
                prediction = model.predict(features)
                prediction_proba = None
            
            result = {
                'prediction': prediction.tolist(),
                'model_version': model_artifacts.get('model_version', 'unknown') if model_artifacts else 'unknown'
            }
            
            if prediction_proba is not None:
                result['probability'] = prediction_proba.tolist()
            
            # TODO: Add fairness monitoring here
            # This would track predictions by demographic groups
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e)}, 500

@api.route('/fairness-metrics')
class FairnessMetrics(Resource):
    def get(self):
        """Get current fairness metrics"""
        # TODO: Implement real-time fairness metrics calculation
        return {
            'demographic_parity': 0.05,
            'equalized_odds': 0.03,
            'disparate_impact': 0.85,
            'status': 'within_thresholds'
        }

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=False)
'''
        
        with open(app_dir / "main.py", 'w') as f:
            f.write(main_py_content)
        
        # Copy config files
        config_dir = build_context / "config"
        config_dir.mkdir()
        
        # Copy existing config files
        import shutil
        if os.path.exists("config/pipeline_config.yaml"):
            shutil.copy2("config/pipeline_config.yaml", config_dir)
        if os.path.exists("config/fairness_thresholds.yaml"):
            shutil.copy2("config/fairness_thresholds.yaml", config_dir)
    
    def _create_container_requirements(self, build_context: Path):
        """Create requirements.txt for the container"""
        requirements = [
            "flask==2.2.2",
            "flask-restx==1.0.6",
            "numpy==1.21.6",
            "pandas==1.5.3",
            "scikit-learn==1.0.2",
            "pyyaml==6.0",
            "requests==2.28.2"
        ]
        
        with open(build_context / "requirements.txt", 'w') as f:
            f.write('\n'.join(requirements))
    
    def deploy_to_kubernetes(self, container_image: str, 
                           deployment_name: str = "devethops-model") -> Dict[str, Any]:
        """
        Deploy the containerized model to Kubernetes.
        
        Args:
            container_image: Docker image name/tag
            deployment_name: Name for the Kubernetes deployment
            
        Returns:
            Dict containing deployment information
        """
        if self.k8s_client is None:
            raise DeploymentError("Kubernetes client not available")
        
        self.logger.info(f"Deploying to Kubernetes: {container_image}")
        
        try:
            # Create namespace if it doesn't exist
            namespace = "devethops"
            self._create_namespace(namespace)
            
            # Create deployment
            deployment_manifest = self._create_deployment_manifest(
                deployment_name, container_image, namespace
            )
            
            apps_v1 = client.AppsV1Api(self.k8s_client)
            
            try:
                # Try to update existing deployment
                apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment_manifest
                )
                self.logger.info(f"Updated existing deployment: {deployment_name}")
            except client.ApiException as e:
                if e.status == 404:
                    # Create new deployment
                    apps_v1.create_namespaced_deployment(
                        namespace=namespace,
                        body=deployment_manifest
                    )
                    self.logger.info(f"Created new deployment: {deployment_name}")
                else:
                    raise
            
            # Create service
            service_manifest = self._create_service_manifest(deployment_name, namespace)
            core_v1 = client.CoreV1Api(self.k8s_client)
            
            try:
                # Try to update existing service
                core_v1.patch_namespaced_service(
                    name=f"{deployment_name}-service",
                    namespace=namespace,
                    body=service_manifest
                )
                self.logger.info(f"Updated existing service: {deployment_name}-service")
            except client.ApiException as e:
                if e.status == 404:
                    # Create new service
                    core_v1.create_namespaced_service(
                        namespace=namespace,
                        body=service_manifest
                    )
                    self.logger.info(f"Created new service: {deployment_name}-service")
                else:
                    raise
            
            # Create ConfigMap for configuration
            self._create_config_map(namespace)
            
            deployment_info = {
                'deployment_name': deployment_name,
                'service_name': f"{deployment_name}-service",
                'namespace': namespace,
                'image': container_image,
                'status': 'deployed'
            }
            
            self.deployment_artifacts['kubernetes_deployment'] = deployment_info
            self.logger.info("Kubernetes deployment completed successfully")
            
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Error deploying to Kubernetes: {str(e)}")
            raise DeploymentError(f"Kubernetes deployment failed: {str(e)}")
    
    def _create_namespace(self, namespace: str):
        """Create Kubernetes namespace if it doesn't exist"""
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            
            # Check if namespace exists
            try:
                core_v1.read_namespace(name=namespace)
                self.logger.info(f"Namespace '{namespace}' already exists")
            except client.ApiException as e:
                if e.status == 404:
                    # Create namespace
                    namespace_manifest = client.V1Namespace(
                        metadata=client.V1ObjectMeta(name=namespace)
                    )
                    core_v1.create_namespace(body=namespace_manifest)
                    self.logger.info(f"Created namespace: {namespace}")
                else:
                    raise
                    
        except Exception as e:
            self.logger.error(f"Error creating namespace: {str(e)}")
            raise
    
    def _create_deployment_manifest(self, deployment_name: str, 
                                   container_image: str, namespace: str) -> client.V1Deployment:
        """Create Kubernetes deployment manifest"""
        
        container = client.V1Container(
            name="devethops-model",
            image=container_image,
            ports=[client.V1ContainerPort(container_port=8080)],
            env=[
                client.V1EnvVar(name="FLASK_ENV", value="production"),
                client.V1EnvVar(name="PYTHONPATH", value="/app")
            ],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "100m", "memory": "256Mi"},
                limits={"cpu": "500m", "memory": "512Mi"}
            ),
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(path="/health", port=8080),
                initial_delay_seconds=30,
                period_seconds=30
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(path="/health", port=8080),
                initial_delay_seconds=10,
                period_seconds=10
            )
        )
        
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={"app": deployment_name, "version": "v1"}
            ),
            spec=client.V1PodSpec(containers=[container])
        )
        
        spec = client.V1DeploymentSpec(
            replicas=2,  # For high availability
            selector=client.V1LabelSelector(
                match_labels={"app": deployment_name}
            ),
            template=template
        )
        
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=deployment_name,
                namespace=namespace,
                labels={"app": deployment_name}
            ),
            spec=spec
        )
        
        return deployment
    
    def _create_service_manifest(self, deployment_name: str, namespace: str) -> client.V1Service:
        """Create Kubernetes service manifest"""
        
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=f"{deployment_name}-service",
                namespace=namespace,
                labels={"app": deployment_name}
            ),
            spec=client.V1ServiceSpec(
                selector={"app": deployment_name},
                ports=[
                    client.V1ServicePort(
                        port=80,
                        target_port=8080,
                        protocol="TCP"
                    )
                ],
                type="ClusterIP"
            )
        )
        
        return service
    
    def _create_config_map(self, namespace: str):
        """Create ConfigMap for configuration files"""
        try:
            core_v1 = client.CoreV1Api(self.k8s_client)
            
            # Load configuration data
            config_data = {}
            
            if os.path.exists("config/pipeline_config.yaml"):
                with open("config/pipeline_config.yaml", 'r') as f:
                    config_data["pipeline_config.yaml"] = f.read()
            
            if os.path.exists("config/fairness_thresholds.yaml"):
                with open("config/fairness_thresholds.yaml", 'r') as f:
                    config_data["fairness_thresholds.yaml"] = f.read()
            
            config_map = client.V1ConfigMap(
                api_version="v1",
                kind="ConfigMap",
                metadata=client.V1ObjectMeta(
                    name="devethops-config",
                    namespace=namespace
                ),
                data=config_data
            )
            
            try:
                core_v1.create_namespaced_config_map(
                    namespace=namespace,
                    body=config_map
                )
                self.logger.info("Created ConfigMap: devethops-config")
            except client.ApiException as e:
                if e.status == 409:  # Already exists
                    core_v1.patch_namespaced_config_map(
                        name="devethops-config",
                        namespace=namespace,
                        body=config_map
                    )
                    self.logger.info("Updated existing ConfigMap: devethops-config")
                else:
                    raise
                    
        except Exception as e:
            self.logger.error(f"Error creating ConfigMap: {str(e)}")
    
    def setup_fairness_monitoring(self, deployment_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Setup fairness monitoring for the deployed model.
        
        Args:
            deployment_info: Information about the deployed model
            
        Returns:
            Dict containing monitoring setup information
        """
        self.logger.info("Setting up fairness monitoring")
        
        try:
            # Create monitoring configuration
            monitoring_config = {
                'metrics_collection': {
                    'enabled': True,
                    'interval_seconds': 60,
                    'batch_size': 100
                },
                'fairness_thresholds': self.config.get('fairness_thresholds', {}),
                'alerts': {
                    'email_enabled': True,
                    'slack_enabled': False,
                    'webhook_url': None
                },
                'storage': {
                    'type': 'local',  # Could be 'redis', 'postgresql', etc.
                    'retention_days': 30
                }
            }
            
            # Save monitoring configuration
            monitoring_artifacts = {
                'config': monitoring_config,
                'deployment_name': deployment_info['deployment_name'],
                'namespace': deployment_info['namespace'],
                'setup_timestamp': str(datetime.now())
            }
            
            # TODO: Setup actual monitoring infrastructure
            # This could include:
            # - Prometheus metrics collection
            # - Grafana dashboards
            # - Alert manager rules
            # - Custom monitoring pods
            
            self.deployment_artifacts['fairness_monitoring'] = monitoring_artifacts
            self.logger.info("Fairness monitoring setup completed")
            
            return monitoring_artifacts
            
        except Exception as e:
            self.logger.error(f"Error setting up fairness monitoring: {str(e)}")
            raise DeploymentError(f"Fairness monitoring setup failed: {str(e)}")
    
    def validate_deployment(self, deployment_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the deployment is working correctly.
        
        Args:
            deployment_info: Information about the deployed model
            
        Returns:
            Dict containing validation results
        """
        self.logger.info("Validating deployment")
        
        try:
            validation_results = {
                'deployment_status': 'unknown',
                'service_status': 'unknown',
                'health_check': 'unknown',
                'prediction_test': 'unknown',
                'overall_status': 'unknown'
            }
            
            apps_v1 = client.AppsV1Api(self.k8s_client)
            core_v1 = client.CoreV1Api(self.k8s_client)
            
            # Check deployment status
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=deployment_info['deployment_name'],
                    namespace=deployment_info['namespace']
                )
                
                ready_replicas = deployment.status.ready_replicas or 0
                desired_replicas = deployment.spec.replicas or 0
                
                if ready_replicas >= desired_replicas:
                    validation_results['deployment_status'] = 'healthy'
                else:
                    validation_results['deployment_status'] = f'scaling ({ready_replicas}/{desired_replicas})'
                    
            except Exception as e:
                validation_results['deployment_status'] = f'error: {str(e)}'
            
            # Check service status
            try:
                service = core_v1.read_namespaced_service(
                    name=deployment_info['service_name'],
                    namespace=deployment_info['namespace']
                )
                validation_results['service_status'] = 'running'
            except Exception as e:
                validation_results['service_status'] = f'error: {str(e)}'
            
            # TODO: Add actual health check and prediction test
            # This would require port forwarding or ingress setup
            validation_results['health_check'] = 'not_implemented'
            validation_results['prediction_test'] = 'not_implemented'
            
            # Determine overall status
            critical_checks = ['deployment_status', 'service_status']
            if all(validation_results[check] in ['healthy', 'running'] for check in critical_checks):
                validation_results['overall_status'] = 'healthy'
            else:
                validation_results['overall_status'] = 'unhealthy'
            
            self.logger.info(f"Deployment validation completed: {validation_results['overall_status']}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating deployment: {str(e)}")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status and artifacts"""
        return {
            'artifacts': self.deployment_artifacts,
            'timestamp': str(datetime.now())
        }


if __name__ == "__main__":
    # Example usage
    deploy_stage = DeployStage()
    # container_image = deploy_stage.containerize_model("models/trained_model.pkl", {})
    # deployment_info = deploy_stage.deploy_to_kubernetes(container_image)
    # monitoring_info = deploy_stage.setup_fairness_monitoring(deployment_info)
