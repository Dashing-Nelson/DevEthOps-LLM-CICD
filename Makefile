# DevEthOps Makefile for common development tasks

.PHONY: help install dev-install test lint format clean build run docker-build docker-run

# Default target
help:
	@echo "DevEthOps Development Commands:"
	@echo "  install      - Install package and dependencies"
	@echo "  dev-install  - Install in development mode with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  test-fast    - Run tests excluding slow tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  run          - Run the pipeline"
	@echo "  api          - Start API server"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  monitor      - Start monitoring stack"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

dev-install:
	pip install -r requirements.txt
	pip install -e .[dev]
	pre-commit install

# Testing
test:
	pytest

test-fast:
	pytest -m "not slow"

test-coverage:
	pytest --cov=src/devethops --cov-report=html

# Code quality
lint:
	flake8 src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

check-format:
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Building
build: clean
	python setup.py sdist bdist_wheel

# Running
run:
	python scripts/run_pipeline.py

api:
	uvicorn devethops.api.app:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t devethops-llm-cicd .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/configs:/app/configs -v $(PWD)/data:/app/data devethops-llm-cicd

docker-api:
	docker-compose up devethops-api

# Monitoring
monitor:
	docker-compose --profile monitoring up -d

monitor-stop:
	docker-compose --profile monitoring down

# Development environment
dev-up:
	docker-compose up -d

dev-down:
	docker-compose down

dev-logs:
	docker-compose logs -f

# Pipeline stages
pipeline-build:
	python scripts/run_pipeline.py --stage build

pipeline-test:
	python scripts/run_pipeline.py --stage test

pipeline-deploy:
	python scripts/run_pipeline.py --stage deploy

pipeline-monitor:
	python scripts/run_pipeline.py --stage monitor

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080
