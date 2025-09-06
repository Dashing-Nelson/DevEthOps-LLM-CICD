# Multi-stage build for DevEthOps-LLM-CICD
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Copy development requirements
COPY requirements.txt .
RUN pip install pytest pytest-cov black flake8 mypy pre-commit

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Production stage
FROM base as production

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY setup.py .
COPY README.md .
COPY requirements.txt .

# Install package
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash devethops
RUN chown -R devethops:devethops /app
USER devethops

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "devethops.api.app"]

# API serving stage
FROM production as api

# Copy API specific files
COPY src/devethops/api/ ./src/devethops/api/

# Install additional API dependencies
RUN pip install fastapi uvicorn[standard] prometheus-client

# Start API server
CMD ["uvicorn", "devethops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Pipeline stage  
FROM production as pipeline

# Default to running the pipeline
CMD ["python", "scripts/run_pipeline.py"]
