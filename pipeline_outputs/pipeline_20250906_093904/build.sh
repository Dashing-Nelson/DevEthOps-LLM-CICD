#!/bin/bash
# DevEthOps Model Docker Build Script

set -e

IMAGE_TAG="devethops/model-server:latest"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "Building Docker image: $IMAGE_TAG"

docker build \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg GIT_COMMIT="$GIT_COMMIT" \
    -t "$IMAGE_TAG" \
    .

echo "Build complete: $IMAGE_TAG"
echo "To run: docker run -p 8000:8000 $IMAGE_TAG"
