# DevEthOps-LLM-CICD: Ethical CI/CD Pipeline for Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DevEthOps-LLM-CICD is a comprehensive ethical CI/CD pipeline solution for machine learning models, with specialized support for Large Language Models (LLMs). It integrates automated fairness checks, bias detection, explainability analysis, and ethical monitoring throughout the ML lifecycle.

## Features

- **🔍 Automated Fairness Checks**: Statistical parity, disparate impact, equal opportunity analysis
- **⚖️ Bias Mitigation**: Multiple pre-processing and in-processing bias mitigation techniques
- **💡 Explainability**: SHAP and LIME explanations for model decisions
- **📊 Multi-Model Support**: Tabular (LogReg, RF, XGBoost) and Text models (RoBERTa)
- **🚀 CI/CD Integration**: Automated Build → Test → Deploy → Monitor pipeline
- **📈 Continuous Monitoring**: Drift detection and fairness monitoring in production
- **🐳 Containerized Deployment**: Docker and Kubernetes support
- **📋 Comprehensive Logging**: Detailed audit trails for compliance

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Kubernetes cluster (optional, for K8s deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/Dashing-Nelson/DevEthOps-LLM-CICD
cd DevEthOps-LLM-CICD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Run the complete pipeline
python scripts/run_pipeline.py

# Run with specific configuration
python scripts/run_pipeline.py --config configs/settings.yaml --dataset adult

# Run specific stage
python scripts/run_pipeline.py --stage test --model-type tabular
```

### Configuration

The main configuration is in `configs/settings.yaml`. Key sections include:

- **Models**: Configure which models to train and evaluate
- **Datasets**: Specify datasets and preprocessing options
- **Fairness**: Set fairness thresholds and protected attributes
- **Deployment**: Configure deployment targets and monitoring

```yaml
# Example configuration snippet
pipeline:
  stages: ['build', 'test', 'deploy', 'monitor']
  
models:
  tabular:
    - logistic_regression
    - random_forest
    - xgboost
  text:
    - roberta

fairness:
  protected_attributes: ['gender', 'race', 'age']
  metrics: ['statistical_parity', 'disparate_impact', 'equal_opportunity']
```

## Pipeline Stages

### 1. Build Stage
- Data loading and validation
- Feature engineering and preprocessing
- Model training with hyperparameter optimization
- Initial model validation

### 2. Test Stage
- **Fairness Testing**: Automated bias detection using AIF360
- **Explainability Analysis**: SHAP and LIME explanations
- **Performance Testing**: Accuracy, precision, recall, F1-score
- **Threshold Validation**: Check against configured fairness thresholds

### 3. Deploy Stage
- Model packaging and containerization
- Deployment to staging/production environments
- API endpoint creation with FastAPI
- Health check and smoke test setup

### 4. Monitor Stage
- **Drift Detection**: Data and concept drift monitoring
- **Fairness Monitoring**: Continuous fairness metric tracking
- **Performance Monitoring**: Model accuracy and latency tracking
- **Alerting**: Automated notifications for threshold violations

## Supported Datasets

- **IBM HR Analytics**: Employee attrition prediction
- **Adult Census**: Income prediction benchmark
- **MIMIC-III**: Medical diagnosis prediction (with synthetic data option)

Each dataset includes:
- Automated bias injection for testing
- Protected attribute identification
- Preprocessing pipelines
- Fairness evaluation configurations

## Fairness Metrics

The pipeline evaluates multiple fairness metrics:

- **Statistical Parity**: Equal positive prediction rates across groups
- **Disparate Impact**: Ratio of positive rates (80% rule compliance)
- **Equal Opportunity**: Equal true positive rates across groups
- **Average Odds**: Equal TPR and FPR across groups
- **Demographic Parity**: Outcome independence from protected attributes

## Model Explainability

- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local instance explanations
- **Feature Importance Rankings**: Model-specific importance scores
- **Bias Attribution**: Understanding which features contribute to unfair decisions

## Bias Mitigation Techniques

### Pre-processing
- **Reweighing**: Adjust instance weights to balance groups
- **Optimized Preprocessing**: Transform features to reduce bias
- **Disparate Impact Remover**: Remove disparate impact through feature transformation

### In-processing
- **Adversarial Debiasing**: Train with adversarial fairness constraints
- **Fair Constraint Optimization**: Optimize with fairness constraints

### Post-processing
- **Threshold Optimization**: Adjust decision thresholds per group
- **Calibration**: Ensure prediction probabilities are well-calibrated

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Ethical ML Pipeline

on: [push, pull_request]

jobs:
  ethical-checks:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run fairness checks
      run: python scripts/run_pipeline.py --stage test
    - name: Generate fairness report
      run: python scripts/generate_report.py
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'python scripts/run_pipeline.py --stage build'
            }
        }
        stage('Fairness Test') {
            steps {
                sh 'python scripts/run_pipeline.py --stage test'
                publishHTML([allowMissing: false, 
                           alwaysLinkToLastBuild: true, 
                           keepAll: true, 
                           reportDir: 'reports', 
                           reportFiles: 'fairness_report.html'])
            }
        }
        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh 'python scripts/run_pipeline.py --stage deploy'
            }
        }
    }
}
```

## Monitoring and Alerting

### Drift Detection
- **Statistical Tests**: KS test, PSI (Population Stability Index)
- **Distribution Comparison**: Earth Mover's Distance, KL Divergence
- **Feature-level Monitoring**: Individual feature drift tracking

### Fairness Monitoring
- **Real-time Metrics**: Continuous fairness metric calculation
- **Trend Analysis**: Historical fairness performance tracking
- **Threshold Alerts**: Automated notifications for violations

### Integration with Monitoring Systems
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboarding
- **PagerDuty**: Alert management and escalation

## API Documentation

The deployed model exposes a RESTful API:

```bash
# Prediction endpoint
POST /predict
{
  "features": {"age": 35, "education": "Bachelor", ...},
  "include_explanation": true,
  "fairness_check": true
}

# Health check
GET /health

# Fairness metrics
GET /metrics/fairness

# Model explanation
POST /explain
{
  "features": {"age": 35, "education": "Bachelor", ...},
  "explanation_type": "shap"
}
```

## Configuration Reference

### Core Configuration (`configs/settings.yaml`)
- Pipeline stages and execution order
- Model selection and hyperparameters
- Dataset configurations and preprocessing
- Deployment targets and environments

### Fairness Thresholds (`configs/fairness_thresholds.yaml`)
- Threshold values for fairness metrics
- Protected attribute definitions
- Dataset-specific configurations
- Severity levels and alerting rules

### Logging Configuration (`configs/logging.yaml`)
- Log levels and formats
- Output destinations (file, console, remote)
- Audit trail configurations

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_fairness.py
pytest tests/test_models.py
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/ scripts/

# Lint code
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

### Adding New Models

1. Implement model class in appropriate module (`models_tabular.py` or `models_text.py`)
2. Add configuration in `settings.yaml`
3. Update pipeline orchestrator
4. Add tests and documentation

### Adding New Fairness Metrics

1. Implement metric in `fairness_checks.py`
2. Add threshold configuration in `fairness_thresholds.yaml`
3. Update monitoring and reporting
4. Add tests and documentation

## Docker Deployment

```bash
# Build image
docker build -t devethops-llm-cicd .

# Run container
docker run -p 8000:8000 devethops-llm-cicd

# With custom configuration
docker run -v $(pwd)/configs:/app/configs -p 8000:8000 devethops-llm-cicd
```

## Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=devethops-model

# View logs
kubectl logs -l app=devethops-model
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure all tests pass and fairness checks are satisfied before submitting.

## Compliance and Governance

- **GDPR Compliance**: Right to explanation support through model explainability
- **SOX Compliance**: Comprehensive audit trails and model versioning
- **IEEE Standards**: Alignment with IEEE 2857 and related AI ethics standards
- **Model Cards**: Automated generation of model documentation
- **Fairness Certificates**: Attestation of fairness testing completion

## Troubleshooting

### Common Issues

1. **Fairness Check Failures**
   - Review threshold configurations in `fairness_thresholds.yaml`
   - Check data preprocessing for bias introduction
   - Consider bias mitigation techniques

2. **Model Training Errors**
   - Verify dataset availability and format
   - Check feature preprocessing pipeline
   - Review hyperparameter configurations

3. **Deployment Issues**
   - Ensure Docker/Kubernetes environment is properly configured
   - Check resource requirements and limits
   - Verify network connectivity and permissions

### Debug Mode

```bash
# Run with debug logging
python scripts/run_pipeline.py --log-level DEBUG

# Generate detailed fairness report
python scripts/run_pipeline.py --detailed-report

# Test specific components
python -m pytest tests/test_fairness.py -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [AIF360](https://github.com/Trusted-AI/AIF360) for fairness metrics and bias mitigation
- [SHAP](https://github.com/slundberg/shap) for model explainability
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for LLM support
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities

## Citation

If you use this work in your research, please cite:

```bibtex
@software{alfonso_2025_17165999,
  author       = {Alfonso, Nelson},
  title        = {DevEthOps-LLM-CICD},
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.17165999},
  url          = {https://doi.org/10.5281/zenodo.17165999},
  swhid        = {swh:1:dir:6f89ce10f50da5fbebde1351702df44a6e0f5107
                   ;origin=https://doi.org/10.5281/zenodo.17165998;vi
                   sit=swh:1:snp:5bfe3fa05bf5675392ad4d4fcd8a8c583d30
                   b2f4;anchor=swh:1:rel:b1d0749455050bb6abfc58c65a33
                   02674fc79bdd;path=DevEthOps-LLM-CICD-main
                  },
}
```

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation and FAQ

---

**Note**: This is a framework for ethical ML operations. Always validate fairness results with domain experts and ensure compliance with relevant regulations and organizational policies.
