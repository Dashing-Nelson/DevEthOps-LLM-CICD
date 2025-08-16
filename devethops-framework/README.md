# DevEthOps Framework 🤖⚖️

## Production-Ready Ethical AI Integration for CI/CD Pipelines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io/)

**DevEthOps** is a comprehensive framework that integrates ethical AI practices directly into CI/CD pipelines, providing automated fairness evaluation, bias detection, and explainability analysis for machine learning models, especially Large Language Models (LLMs).

## 🌟 Features

### 🔍 **Automated Ethical AI Checks**
- **Fairness Evaluation**: Demographic parity, disparate impact, equalized odds, individual fairness
- **Bias Detection**: Multi-attribute bias detection with intersectional analysis
- **Explainability Analysis**: SHAP and LIME integration for model interpretability
- **Performance Monitoring**: Continuous drift detection and fairness degradation alerts

### 🚀 **CI/CD Integration**
- **Jenkins Pipeline**: Complete Jenkinsfile with ethical AI stages
- **Docker Support**: Production-ready containerization
- **Kubernetes Deployment**: Scalable orchestration with monitoring
- **Automated Testing**: Unit, integration, and end-to-end fairness tests

### 📊 **Comprehensive Monitoring**
- **Real-time Dashboards**: Grafana integration for fairness metrics
- **Alerting System**: Automatic notifications for ethical violations
- **Audit Trails**: Complete logging of all ethical decisions
- **Reporting**: Detailed fairness and explainability reports

### 🛠 **Technology Stack**
- **Python 3.8+** with PyTorch, Hugging Face Transformers
- **Ethical AI**: AIF360, SHAP, LIME, fairness-indicators
- **ML Stack**: scikit-learn, pandas, numpy
- **Infrastructure**: Docker, Kubernetes, Jenkins/GitLab CI
- **Monitoring**: Prometheus, Grafana, PostgreSQL, Redis

## 📁 Project Structure

```
devethops-framework/
├── src/
│   ├── pipeline/          # Core pipeline stages
│   │   ├── build_stage.py       # Data loading & bias detection
│   │   ├── test_stage.py        # Fairness & explainability tests
│   │   ├── deploy_stage.py      # Model deployment with monitoring
│   │   └── monitor_stage.py     # Continuous monitoring
│   ├── ethical_checks/    # Ethical AI evaluation modules
│   │   ├── fairness_evaluator.py    # Comprehensive fairness analysis
│   │   └── explainability_analyzer.py # SHAP/LIME integration
│   ├── metrics/           # Fairness and performance metrics
│   ├── models/            # Model wrappers and utilities
│   ├── monitoring/        # Pipeline monitoring and reporting
│   └── main.py           # Main pipeline orchestrator
├── config/               # Configuration files
│   ├── pipeline_config.yaml     # Pipeline settings
│   ├── fairness_thresholds.yaml # Fairness criteria
│   └── model_config.yaml        # Model configurations
├── docker/               # Docker configuration
│   ├── Dockerfile               # Production container
│   ├── docker-compose.yml       # Multi-service setup
│   └── init-db.sql             # Database initialization
├── kubernetes/           # Kubernetes deployment
│   ├── deployment.yaml          # Application deployment
│   ├── service.yaml            # Service configuration
│   └── configmap.yaml          # Configuration management
├── jenkins/              # CI/CD pipeline
│   ├── Jenkinsfile             # Complete pipeline definition
│   └── jenkins-agent.yaml      # Kubernetes agent config
├── tests/               # Comprehensive test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test data and utilities
├── data/                # Data directories
├── notebooks/           # Jupyter notebooks for analysis
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8.10+
- Docker and Docker Compose
- Kubernetes cluster (for production deployment)
- Jenkins (for CI/CD pipeline)

### 1. Clone and Setup

```bash
git clone https://github.com/your-org/devethops-framework.git
cd devethops-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure the Framework

```bash
# Edit configuration files
nano config/pipeline_config.yaml
nano config/fairness_thresholds.yaml
nano config/model_config.yaml
```

### 3. Run Local Development

```bash
# Start services with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# Run the pipeline
python src/main.py --mode pipeline --config config/pipeline_config.yaml
```

### 4. Run Tests

```bash
# Run unit tests
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run fairness-specific tests
python -m pytest tests/integration/test_fairness_pipeline.py -v
```

## 🔧 Configuration

### Pipeline Configuration (`config/pipeline_config.yaml`)

```yaml
pipeline:
  name: "DevEthOps Ethical AI Pipeline"
  version: "1.0.0"
  stages: [build, test, deploy, monitor]

build:
  enable_bias_detection: true
  bias_mitigation_strategy: "reweighting"
  
test:
  fairness_tests: true
  explainability_tests: true
  performance_tests: true

deploy:
  containerization: true
  kubernetes_deployment: true
  monitoring_setup: true

monitor:
  drift_detection: true
  fairness_monitoring: true
  alert_thresholds:
    fairness_drop: 0.05
    performance_drop: 0.1
```

### Fairness Thresholds (`config/fairness_thresholds.yaml`)

```yaml
fairness_metrics:
  demographic_parity:
    threshold: 0.1
    weight: 0.3
  disparate_impact:
    threshold: 0.8
    weight: 0.3
  equalized_odds:
    threshold: 0.1
    weight: 0.2
  individual_fairness:
    threshold: 0.05
    weight: 0.2

protected_attributes:
  - "gender"
  - "race" 
  - "age_group"
  - "income_level"
```

## 🧪 Usage Examples

### Basic Pipeline Execution

```python
from src.main import DevEthOpsPipeline

# Initialize pipeline
pipeline = DevEthOpsPipeline(config_path="config/pipeline_config.yaml")

# Run complete pipeline
result = pipeline.run(
    data_path="data/datasets/training_data.csv",
    model_path="models/my_model.pkl"
)

print(f"Pipeline Status: {result['status']}")
print(f"Fairness Score: {result['fairness_score']}")
```

### Fairness Evaluation

```python
from src.ethical_checks.fairness_evaluator import FairnessEvaluator

# Initialize evaluator
evaluator = FairnessEvaluator(config_path="config/fairness_thresholds.yaml")

# Evaluate model fairness
fairness_result = evaluator.evaluate_model_fairness(
    model=my_model,
    X_test=X_test,
    y_test=y_test,
    protected_attributes=protected_attrs
)

print(f"Overall Fairness Score: {fairness_result['overall_fairness_score']}")
print(f"Violations: {fairness_result['violations']}")
```

### Explainability Analysis

```python
from src.ethical_checks.explainability_analyzer import ExplainabilityAnalyzer

# Initialize analyzer
analyzer = ExplainabilityAnalyzer(config={'enable_shap': True, 'enable_lime': True})

# Analyze model explainability
explanation = analyzer.analyze(
    model=my_model,
    X_test=X_test,
    feature_names=feature_names
)

print(f"Explainability Score: {explanation['explainability_score']}")
print(f"Bias Detected: {explanation['bias_detected']}")
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker
docker build -f docker/Dockerfile -t devethops-framework .
docker run -p 8000:8000 devethops-framework
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

# Check deployment status
kubectl get pods -l app=devethops-framework
kubectl get svc devethops-framework-service
```

### Jenkins CI/CD

1. Copy `jenkins/Jenkinsfile` to your repository root
2. Configure Jenkins pipeline to use the Jenkinsfile
3. Set up required credentials:
   - `docker-registry-credentials`
   - `kubeconfig`
   - `devethops-api-key`

## 📊 Monitoring and Alerting

### Prometheus Metrics

The framework exposes metrics on `:8080/metrics`:

- `devethops_fairness_score`: Current fairness score
- `devethops_bias_detected`: Binary bias detection indicator
- `devethops_model_accuracy`: Model accuracy metrics
- `devethops_pipeline_duration`: Pipeline execution time

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/devethops123) to view:

- **Fairness Metrics Dashboard**: Real-time fairness monitoring
- **Performance Dashboard**: Model performance tracking
- **Pipeline Health**: CI/CD pipeline status
- **Alert Dashboard**: Active alerts and violations

## 🧪 Testing

### Test Suite Structure

```bash
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_fairness_metrics.py
│   ├── test_pipeline_stages.py
│   └── test_explainability.py
├── integration/            # Integration tests for workflows
│   ├── test_fairness_pipeline.py
│   └── test_end_to_end.py
└── fixtures/              # Test data and utilities
    └── synthetic_datasets.py
```

### Running Tests

```bash
# Run all tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v              # Unit tests
python -m pytest tests/integration/ -v       # Integration tests
python -m pytest -k "fairness" -v           # Fairness-specific tests

# Run tests with specific markers
python -m pytest -m "slow" -v               # Slow tests
python -m pytest -m "not slow" -v           # Fast tests only
```

## 📖 API Documentation

### REST API Endpoints

When running in API mode (`python src/main.py --mode api`):

- `GET /health` - Health check endpoint
- `POST /evaluate/fairness` - Evaluate model fairness
- `POST /evaluate/explainability` - Analyze model explainability
- `POST /pipeline/run` - Execute complete pipeline
- `GET /metrics` - Prometheus metrics endpoint

### API Usage Example

```python
import requests

# Evaluate fairness via API
response = requests.post('http://localhost:8000/evaluate/fairness', json={
    'model_path': '/path/to/model.pkl',
    'data_path': '/path/to/test_data.csv',
    'protected_attributes': ['gender', 'race']
})

fairness_result = response.json()
print(f"Fairness Score: {fairness_result['overall_score']}")
```

## 🛠 Development

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python -m pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code quality checks
flake8 src/
black src/
isort src/
```

### Code Quality

The project uses:
- **Black** for code formatting
- **Flake8** for linting
- **isort** for import sorting
- **mypy** for type checking
- **pytest** for testing

## 📋 Fairness Metrics Reference

### Demographic Parity
Ensures equal positive prediction rates across protected groups.
```
P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
```

### Disparate Impact
Measures ratio of positive rates between groups.
```
DI = P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged)
```

### Equalized Odds
Ensures equal true positive and false positive rates.
```
P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)
P(Ŷ=1|Y=0,A=0) = P(Ŷ=1|Y=0,A=1)
```

### Individual Fairness
Similar individuals should receive similar predictions.
```
d(M(x₁), M(x₂)) ≤ L·d(x₁, x₂)
```

## 🚨 Troubleshooting

### Common Issues

**Issue**: ImportError for ethical AI libraries
```bash
# Solution: Install specific versions
pip install aif360==0.5.0
pip install shap==0.41.0
pip install lime==0.2.0.1
```

**Issue**: Kubernetes deployment fails
```bash
# Check pod status
kubectl describe pod <pod-name>

# Check logs
kubectl logs -l app=devethops-framework
```

**Issue**: Fairness tests failing
```bash
# Check data quality
python -c "
from tests.fixtures.synthetic_datasets import create_test_datasets
paths = create_test_datasets()
print('Test datasets created:', paths)
"
```

### Performance Optimization

- **Large Datasets**: Use `sample_size` parameter in explainability analysis
- **Memory Issues**: Increase Docker memory limits in `docker-compose.yml`
- **Slow Tests**: Use `pytest -m "not slow"` to skip long-running tests

## 📚 Resources

### Documentation
- [Ethical AI Guide](docs/ethical_ai_guide.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Troubleshooting](docs/troubleshooting.md)

### External Resources
- [AIF360 Documentation](https://aif360.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Fairness-Indicators](https://www.tensorflow.org/tfx/guide/fairness_indicators)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/devethops-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/devethops-framework/discussions)
- **Email**: devethops-support@yourorg.com

## 🎯 Roadmap

- [ ] **v1.1.0**: Advanced bias mitigation techniques
- [ ] **v1.2.0**: MLOps integration (MLflow, Kubeflow)
- [ ] **v1.3.0**: Automated fairness testing for LLMs
- [ ] **v1.4.0**: Multi-language support (R, Scala)
- [ ] **v2.0.0**: Distributed fairness evaluation

---

**Built with ❤️ for Ethical AI**

DevEthOps Framework - Making AI fair, transparent, and accountable in production.
