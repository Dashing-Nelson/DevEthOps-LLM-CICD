#!/bin/bash
# DevEthOps Kubernetes Deployment Script

set -e

NAMESPACE="devethops"
PIPELINE_ID="pipeline_20250906_093904"

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
kubectl wait --for=condition=available --timeout=300s deployment/devethops-pipeline -n $NAMESPACE

echo "Deployment complete!"
echo "Service endpoint: http://devethops-pipeline.local"
echo "Health check: kubectl get pods -n $NAMESPACE"
