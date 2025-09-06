"""
Unit tests for DevEthOps pipeline components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from devethops.config import load_config
from devethops.data_loader import load_dataset
from devethops.preprocess import preprocess_pipeline

class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_dataset_success(self, sample_data, config_dict):
        """Test successful data loading."""
        with patch('devethops.data_loader.pd.read_csv', return_value=sample_data):
            features, target = load_dataset(config_dict)
            
            assert isinstance(features, pd.DataFrame)
            assert isinstance(target, pd.Series)
            assert len(features) == len(target)
            assert len(features) == 100
    
    def test_load_dataset_missing_file(self, config_dict):
        """Test data loading with missing file."""
        config_dict['dataset']['file_path'] = 'nonexistent.csv'
        
        with pytest.raises(FileNotFoundError):
            load_dataset(config_dict)
    
    @pytest.mark.unit
    def test_data_validation(self, sample_data, config_dict):
        """Test data validation."""
        # Test with invalid target column
        config_dict['dataset']['target_column'] = 'NonexistentColumn'
        
        with patch('devethops.data_loader.pd.read_csv', return_value=sample_data):
            with pytest.raises(KeyError):
                load_dataset(config_dict)

class TestPreprocessing:
    """Test preprocessing functionality."""
    
    @pytest.mark.unit
    def test_preprocessing_pipeline(self, sample_data, config_dict):
        """Test preprocessing pipeline."""
        with patch('devethops.preprocess.load_dataset', return_value=(sample_data.drop('Attrition', axis=1), sample_data['Attrition'])):
            result = preprocess_pipeline(config_dict)
            
            assert 'X_train' in result
            assert 'X_test' in result
            assert 'y_train' in result
            assert 'y_test' in result
            assert 'preprocessor' in result
    
    def test_categorical_encoding(self, sample_data):
        """Test categorical variable encoding."""
        from devethops.preprocess import encode_categorical_features
        
        categorical_cols = ['Gender', 'Education']
        encoded_data = encode_categorical_features(sample_data, categorical_cols)
        
        # Check that categorical columns are properly encoded
        for col in categorical_cols:
            assert col not in encoded_data.columns or encoded_data[col].dtype in ['int64', 'float64']

class TestFairnessChecks:
    """Test fairness checking functionality."""
    
    @pytest.mark.fairness
    def test_fairness_metrics_calculation(self, sample_data):
        """Test fairness metrics calculation."""
        from devethops.fairness_checks import calculate_fairness_metrics
        
        # Create mock predictions
        y_pred = np.random.choice([0, 1], len(sample_data))
        y_true = sample_data['Attrition'].map({'Yes': 1, 'No': 0}).values
        protected_attr = sample_data['Gender'].map({'Male': 1, 'Female': 0}).values
        
        metrics = calculate_fairness_metrics(y_true, y_pred, protected_attr)
        
        assert 'disparate_impact' in metrics
        assert 'statistical_parity_difference' in metrics
        assert isinstance(metrics['disparate_impact'], float)
    
    @pytest.mark.fairness
    def test_fairness_threshold_validation(self, config_dict):
        """Test fairness threshold validation."""
        from devethops.fairness_checks import validate_fairness_thresholds
        
        metrics = {
            'disparate_impact': 0.75,  # Below threshold
            'statistical_parity_difference': 0.15,  # Above threshold
            'equal_opportunity_difference': 0.05  # Within threshold
        }
        
        violations = validate_fairness_thresholds(metrics, config_dict['fairness_thresholds'])
        
        assert len(violations) == 2  # Two violations expected
        assert 'disparate_impact' in violations
        assert 'statistical_parity_difference' in violations

class TestModelTraining:
    """Test model training functionality."""
    
    @pytest.mark.unit
    def test_model_initialization(self, config_dict):
        """Test model initialization."""
        from devethops.models_tabular import get_model
        
        model = get_model(config_dict['model'])
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    @pytest.mark.slow
    def test_model_training(self, sample_data, config_dict):
        """Test model training process."""
        from devethops.models_tabular import train_tabular_model
        
        # Prepare data
        X = sample_data.drop('Attrition', axis=1)
        y = sample_data['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Mock preprocessor
        mock_preprocessor = Mock()
        mock_preprocessor.transform.return_value = X.select_dtypes(include=[np.number]).values
        
        with patch('devethops.models_tabular.preprocess_pipeline') as mock_preprocess:
            mock_preprocess.return_value = {
                'X_train': X[:80].select_dtypes(include=[np.number]).values,
                'X_test': X[80:].select_dtypes(include=[np.number]).values,
                'y_train': y[:80].values,
                'y_test': y[80:].values,
                'preprocessor': mock_preprocessor
            }
            
            result = train_tabular_model(config_dict)
            
            assert 'model' in result
            assert 'metrics' in result
            assert 'preprocessor' in result

class TestExplainability:
    """Test explainability functionality."""
    
    @pytest.mark.unit
    def test_explanation_generation(self, sample_data):
        """Test explanation generation."""
        from devethops.explainability import generate_explanation
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        features = sample_data.drop('Attrition', axis=1).iloc[:1]
        
        with patch('devethops.explainability.shap') as mock_shap:
            mock_explainer = Mock()
            mock_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, 0.4]])
            mock_shap.Explainer.return_value = mock_explainer
            
            explanation = generate_explanation(mock_model, features, 1)
            
            assert explanation is not None
            assert isinstance(explanation, dict)

class TestPipeline:
    """Test full pipeline functionality."""
    
    @pytest.mark.integration
    def test_pipeline_initialization(self, config_dict):
        """Test pipeline initialization."""
        from devethops.pipeline import EthicalMLPipeline
        
        pipeline = EthicalMLPipeline(config_dict)
        
        assert pipeline.config == config_dict
        assert hasattr(pipeline, 'run_stage')
        assert hasattr(pipeline, 'run_full_pipeline')
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_stages(self, config_dict, temp_dir):
        """Test individual pipeline stages."""
        from devethops.pipeline import EthicalMLPipeline
        
        # Update config with temp directory
        config_dict['paths']['outputs'] = str(temp_dir)
        
        pipeline = EthicalMLPipeline(config_dict)
        
        # Test each stage (with mocking to avoid actual model training)
        with patch.multiple(
            'devethops.pipeline',
            preprocess_pipeline=Mock(return_value={'X_train': np.array([[1, 2]]), 'y_train': np.array([1])}),
            train_tabular_model=Mock(return_value={'model': Mock(), 'metrics': {}}),
            evaluate_model_fairness=Mock(return_value={'disparate_impact': 0.9}),
        ):
            # Test build stage
            result = pipeline.run_stage('build')
            assert result['status'] == 'success'
            
            # Test test stage
            result = pipeline.run_stage('test')
            assert result['status'] == 'success'

if __name__ == "__main__":
    pytest.main([__file__])
