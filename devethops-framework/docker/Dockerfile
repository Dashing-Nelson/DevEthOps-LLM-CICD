# DevEthOps Framework - Production Dockerfile
FROM python:3.8.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p data/datasets data/models data/reports logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEVETHOPS_ENV=production

# Create non-root user for security
RUN useradd -m -u 1000 devethops && \
    chown -R devethops:devethops /app
USER devethops

# Expose port for web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "src/main.py", "--mode", "api"]
