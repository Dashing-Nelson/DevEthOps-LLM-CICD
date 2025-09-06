"""
Pytest configuration and fixtures for DevEthOps tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Age': np.random.randint(18, 65, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'MonthlyIncome': np.random.randint(3000, 15000, n_samples),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples),
        'YearsAtCompany': np.random.randint(0, 20, n_samples),
        'Attrition': np.random.choice(['Yes', 'No'], n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def config_dict():
    """Sample configuration dictionary."""
    return {
        'dataset': {
            'name': 'test_data',
            'protected_attributes': ['Gender', 'Age'],
            'target_column': 'Attrition',
            'test_size': 0.2,
            'val_size': 0.1
        },
        'model': {
            'type': 'logistic_regression',
            'hyperparameter_search': False,
            'random_state': 42
        },
        'fairness_thresholds': {
            'disparate_impact': [0.8, 1.25],
            'statistical_parity_difference': 0.1,
            'equal_opportunity_difference': 0.1
        },
        'paths': {
            'data': 'data/',
            'models': 'models/',
            'outputs': 'outputs/'
        }
    }

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_model_metadata():
    """Sample model metadata for testing."""
    return {
        'version': '1.0.0',
        'model_type': 'LogisticRegression',
        'features': ['Age', 'Gender', 'Education', 'MonthlyIncome'],
        'target': 'Attrition',
        'training_date': '2024-01-01T00:00:00',
        'performance_metrics': {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77
        },
        'fairness_metrics': {
            'disparate_impact': 0.95,
            'statistical_parity_difference': 0.05,
            'equal_opportunity_difference': 0.03
        }
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "fairness: marks tests related to fairness checking"
    )
