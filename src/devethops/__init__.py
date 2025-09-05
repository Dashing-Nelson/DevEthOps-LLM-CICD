"""
DevEthOps: Ethical CI/CD Pipeline for LLMs
=============================================

A comprehensive framework for implementing ethical AI/ML pipelines with
automated fairness and explainability checks across Build/Test/Deploy/Monitor stages.

Key Components:
- Fairness assessment using AIF360 metrics
- Explainability analysis with SHAP and LIME
- Bias mitigation techniques
- Model monitoring and drift detection
- CI/CD integration with ethical gates

Supported datasets:
- IBM HR Analytics Employee Attrition
- Adult Census Income (UCI)
- MIMIC-III (subset)
- Synthetic bias injection

Author: DevEthOps Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "DevEthOps Team"

# Core imports for easy access
from .config import load_config
from .pipeline import EthicalMLPipeline
from .fairness_checks import FairnessEvaluator
from .explainability import ExplainabilityAnalyzer

__all__ = [
    "load_config",
    "EthicalMLPipeline", 
    "FairnessEvaluator",
    "ExplainabilityAnalyzer"
]
