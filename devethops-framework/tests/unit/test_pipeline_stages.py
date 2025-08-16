"""
Unit tests for pipeline stages.
Tests individual pipeline stage functionality.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from src.pipeline.build_stage import BuildStage
from src.pipeline.test_stage import TestStage
from src.pipeline.deploy_stage import DeployStage
from src.pipeline.monitor_stage import MonitorStage


class TestBuildStage:
    
    @pytest.fixture
    def build_stage(self):
        """Create BuildStage instance with mock config."""
        config = {
            'bias_detection': {'enabled': True},
            'data_validation': {'enabled': True}
        }
        return BuildStage(config)
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 100),
                'feature2': np.random.normal(0, 1, 100),
                'gender': np.random.choice(['Male', 'Female'], 100),
                'label': np.random.binomial(1, 0.5, 100)
            })
            data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_load_and_validate_data(self, build_stage, sample_csv_file):
        """Test data loading and validation."""
        result = build_stage.load_and_validate_data(sample_csv_file)
        
        assert result['success'] is True
        assert 'dataset' in result
        assert len(result['dataset']) == 100
    
    @patch('src.pipeline.build_stage.logger')
    def test_detect_bias_success(self, mock_logger, build_stage, sample_csv_file):
        """Test bias detection functionality."""
        # Load data first
        load_result = build_stage.load_and_validate_data(sample_csv_file)
        dataset = load_result['dataset']
        
        result = build_stage.detect_bias(dataset)
        
        assert isinstance(result, dict)
        assert 'bias_detected' in result
        assert mock_logger.info.called
    
    def test_run_stage_success(self, build_stage, sample_csv_file):
        """Test successful build stage execution."""
        config = {'data_path': sample_csv_file}
        result = build_stage.run(config)
        
        assert result['success'] is True
        assert 'message' in result


class TestTestStage:
    
    @pytest.fixture
    def test_stage(self):
        """Create TestStage instance."""
        config = {
            'fairness_tests': True,
            'explainability_tests': True,
            'performance_tests': True
        }
        return TestStage(config)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.predict.return_value = np.random.binomial(1, 0.5, 100)
        model.predict_proba.return_value = np.random.random((100, 2))
        return model
    
    def test_run_fairness_tests(self, test_stage):
        """Test fairness testing functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 100),
                'gender': np.random.choice(['Male', 'Female'], 100),
                'label': np.random.binomial(1, 0.5, 100)
            })
            data.to_csv(f.name, index=False)
            
            result = test_stage.run_fairness_tests(f.name)
            
            assert isinstance(result, dict)
            assert 'passed' in result
            
        os.unlink(f.name)
    
    @patch('src.pipeline.test_stage.ExplainabilityAnalyzer')
    def test_run_explainability_tests(self, mock_analyzer, test_stage, mock_model):
        """Test explainability testing."""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze.return_value = {
            'explainability_score': 0.8,
            'bias_detected': False
        }
        mock_analyzer.return_value = mock_analyzer_instance
        
        X_test = np.random.random((50, 2))
        result = test_stage.run_explainability_tests(mock_model, X_test)
        
        assert isinstance(result, dict)
        assert 'passed' in result
    
    def test_run_performance_tests(self, test_stage, mock_model):
        """Test performance testing."""
        X_test = np.random.random((50, 2))
        y_test = np.random.binomial(1, 0.5, 50)
        
        result = test_stage.run_performance_tests(mock_model, X_test, y_test)
        
        assert isinstance(result, dict)
        assert 'passed' in result


class TestDeployStage:
    
    @pytest.fixture
    def deploy_stage(self):
        """Create DeployStage instance."""
        config = {
            'containerization': True,
            'kubernetes': True,
            'monitoring': True
        }
        return DeployStage(config)
    
    @pytest.fixture
    def mock_model_path(self):
        """Create mock model file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            f.write(b'mock model data')
            yield f.name
        os.unlink(f.name)
    
    @patch('src.pipeline.deploy_stage.subprocess.run')
    def test_containerize_model(self, mock_subprocess, deploy_stage, mock_model_path):
        """Test model containerization."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Successfully built"
        
        result = deploy_stage.containerize_model(mock_model_path)
        
        assert result['success'] is True
        assert mock_subprocess.called
    
    @patch('src.pipeline.deploy_stage.subprocess.run')
    def test_deploy_to_kubernetes(self, mock_subprocess, deploy_stage):
        """Test Kubernetes deployment."""
        mock_subprocess.return_value.returncode = 0
        
        result = deploy_stage.deploy_to_kubernetes('test-image:latest')
        
        assert result['success'] is True
    
    def test_run_stage(self, deploy_stage, mock_model_path):
        """Test complete deploy stage execution."""
        config = {'model_path': mock_model_path}
        
        with patch.object(deploy_stage, 'containerize_model') as mock_containerize, \
             patch.object(deploy_stage, 'deploy_to_kubernetes') as mock_deploy, \
             patch.object(deploy_stage, 'setup_monitoring') as mock_monitor:
            
            mock_containerize.return_value = {'success': True, 'image_tag': 'test:latest'}
            mock_deploy.return_value = {'success': True}
            mock_monitor.return_value = {'success': True}
            
            result = deploy_stage.run(config)
            
            assert result['success'] is True


class TestMonitorStage:
    
    @pytest.fixture
    def monitor_stage(self):
        """Create MonitorStage instance."""
        config = {
            'drift_detection': True,
            'fairness_monitoring': True,
            'performance_monitoring': True
        }
        return MonitorStage(config)
    
    def test_setup_drift_detection(self, monitor_stage):
        """Test drift detection setup."""
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        result = monitor_stage.setup_drift_detection(reference_data)
        
        assert result['success'] is True
    
    def test_monitor_fairness(self, monitor_stage):
        """Test fairness monitoring."""
        current_data = pd.DataFrame({
            'predictions': np.random.binomial(1, 0.5, 100),
            'gender': np.random.choice(['Male', 'Female'], 100)
        })
        
        result = monitor_stage.monitor_fairness(current_data)
        
        assert isinstance(result, dict)
        assert 'fairness_metrics' in result
    
    @patch('src.pipeline.monitor_stage.smtplib')
    def test_send_alert(self, mock_smtp, monitor_stage):
        """Test alert sending functionality."""
        mock_server = Mock()
        mock_smtp.SMTP.return_value.__enter__.return_value = mock_server
        
        alert_data = {
            'type': 'fairness_degradation',
            'severity': 'high',
            'message': 'Fairness score dropped below threshold'
        }
        
        result = monitor_stage.send_alert(alert_data)
        
        assert result['success'] is True
    
    def test_run_stage(self, monitor_stage):
        """Test complete monitor stage execution."""
        config = {
            'model_endpoint': 'http://test-model:8000',
            'data_source': 'test_stream'
        }
        
        with patch.object(monitor_stage, 'setup_drift_detection') as mock_drift, \
             patch.object(monitor_stage, 'monitor_fairness') as mock_fairness, \
             patch.object(monitor_stage, 'monitor_performance') as mock_performance:
            
            mock_drift.return_value = {'success': True}
            mock_fairness.return_value = {'fairness_score': 0.85}
            mock_performance.return_value = {'accuracy': 0.92}
            
            result = monitor_stage.run(config)
            
            assert result['success'] is True


class TestPipelineIntegration:
    """Integration tests for pipeline stages working together."""
    
    def test_stage_chaining(self):
        """Test that stages can be chained together."""
        # Create sample data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 100),
                'feature2': np.random.normal(0, 1, 100),
                'gender': np.random.choice(['Male', 'Female'], 100),
                'label': np.random.binomial(1, 0.5, 100)
            })
            data.to_csv(f.name, index=False)
            
            # Initialize stages
            build_stage = BuildStage({})
            test_stage = TestStage({})
            
            # Run build stage
            build_result = build_stage.run({'data_path': f.name})
            assert build_result['success'] is True
            
            # Use build output for test stage (simplified)
            test_config = {'data_path': f.name}
            test_result = test_stage.run(test_config)
            assert isinstance(test_result, dict)
            
        os.unlink(f.name)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
