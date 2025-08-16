"""
Unit tests for fairness metrics module.
Tests individual fairness metrics calculations and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from aif360.datasets import BinaryLabelDataset
from src.metrics.fairness_metrics import FairnessMetrics


class TestFairnessMetrics:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic dataset with bias
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'race': np.random.choice(['White', 'Black', 'Asian'], n_samples, p=[0.7, 0.2, 0.1])
        })
        
        # Create biased labels
        bias_factor = np.where(data['gender'] == 'Female', 0.3, 0.7)
        data['label'] = np.random.binomial(1, bias_factor)
        
        # Create biased predictions
        pred_bias = np.where(data['gender'] == 'Female', 0.2, 0.8)
        data['prediction'] = np.random.binomial(1, pred_bias)
        
        return data
    
    @pytest.fixture
    def fairness_metrics(self):
        """Create FairnessMetrics instance."""
        config = {
            'protected_attributes': ['gender', 'race'],
            'favorable_label': 1,
            'unfavorable_label': 0
        }
        return FairnessMetrics(config)
    
    def test_demographic_parity_calculation(self, fairness_metrics, sample_data):
        """Test demographic parity calculation."""
        result = fairness_metrics.demographic_parity(
            predictions=sample_data['prediction'].values,
            protected_attribute=sample_data['gender'].values
        )
        
        assert isinstance(result, float)
        assert 0 <= abs(result) <= 1
        
        # Test with perfectly fair predictions
        fair_predictions = np.ones(len(sample_data))  # All positive
        result_fair = fairness_metrics.demographic_parity(
            predictions=fair_predictions,
            protected_attribute=sample_data['gender'].values
        )
        assert abs(result_fair) < 0.01  # Should be near zero
    
    def test_disparate_impact_calculation(self, fairness_metrics, sample_data):
        """Test disparate impact calculation."""
        result = fairness_metrics.disparate_impact(
            predictions=sample_data['prediction'].values,
            protected_attribute=sample_data['gender'].values
        )
        
        assert isinstance(result, float)
        assert result >= 0
        
        # Perfect fairness should give ratio close to 1
        fair_predictions = np.random.binomial(1, 0.5, len(sample_data))
        result_fair = fairness_metrics.disparate_impact(
            predictions=fair_predictions,
            protected_attribute=sample_data['gender'].values
        )
        assert 0.8 <= result_fair <= 1.2  # Should be close to 1
    
    def test_equalized_odds_calculation(self, fairness_metrics, sample_data):
        """Test equalized odds calculation."""
        result = fairness_metrics.equalized_odds(
            y_true=sample_data['label'].values,
            predictions=sample_data['prediction'].values,
            protected_attribute=sample_data['gender'].values
        )
        
        assert isinstance(result, dict)
        assert 'tpr_difference' in result
        assert 'fpr_difference' in result
        assert isinstance(result['tpr_difference'], float)
        assert isinstance(result['fpr_difference'], float)
    
    def test_individual_fairness_calculation(self, fairness_metrics, sample_data):
        """Test individual fairness calculation."""
        # Mock similarity function
        def similarity_fn(x1, x2):
            return np.exp(-np.sum((x1 - x2) ** 2))
        
        features = sample_data[['feature1', 'feature2']].values
        predictions = sample_data['prediction'].values
        
        result = fairness_metrics.individual_fairness(
            features=features,
            predictions=predictions,
            similarity_function=similarity_fn,
            sample_size=100  # Reduced for testing
        )
        
        assert isinstance(result, float)
        assert result >= 0
    
    def test_comprehensive_evaluation(self, fairness_metrics, sample_data):
        """Test comprehensive fairness evaluation."""
        result = fairness_metrics.evaluate_comprehensive(
            y_true=sample_data['label'].values,
            predictions=sample_data['prediction'].values,
            features=sample_data[['feature1', 'feature2']].values,
            protected_attributes=sample_data[['gender', 'race']]
        )
        
        assert isinstance(result, dict)
        assert 'demographic_parity' in result
        assert 'disparate_impact' in result
        assert 'equalized_odds' in result
        assert 'individual_fairness' in result
        assert 'overall_score' in result
        
        # Check all metrics are valid
        for metric, value in result.items():
            if metric != 'equalized_odds':  # This returns a dict
                assert isinstance(value, (int, float))
    
    def test_intersectional_analysis(self, fairness_metrics, sample_data):
        """Test intersectional fairness analysis."""
        result = fairness_metrics.intersectional_analysis(
            predictions=sample_data['prediction'].values,
            protected_attributes=sample_data[['gender', 'race']]
        )
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check that intersectional groups are analyzed
        expected_groups = set()
        for gender in ['Male', 'Female']:
            for race in ['White', 'Black', 'Asian']:
                expected_groups.add(f"{gender}_{race}")
        
        # At least some intersectional groups should be present
        assert len(set(result.keys()) & expected_groups) > 0
    
    def test_edge_cases(self, fairness_metrics):
        """Test edge cases and error handling."""
        # Empty arrays
        with pytest.raises((ValueError, ZeroDivisionError)):
            fairness_metrics.demographic_parity(
                predictions=np.array([]),
                protected_attribute=np.array([])
            )
        
        # Mismatched array lengths
        with pytest.raises(ValueError):
            fairness_metrics.demographic_parity(
                predictions=np.array([1, 0, 1]),
                protected_attribute=np.array([1, 0])
            )
        
        # Single class predictions
        single_class_pred = np.zeros(100)
        single_class_attr = np.random.choice(['A', 'B'], 100)
        
        result = fairness_metrics.disparate_impact(
            predictions=single_class_pred,
            protected_attribute=single_class_attr
        )
        # Should handle gracefully (might return 0 or 1)
        assert isinstance(result, (int, float))
    
    def test_aif360_integration(self, fairness_metrics, sample_data):
        """Test integration with AIF360 library."""
        # Create AIF360 dataset
        dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=sample_data,
            label_names=['label'],
            protected_attribute_names=['gender']
        )
        
        # This should not raise an error
        assert dataset is not None
        assert len(dataset.features) > 0
    
    @patch('src.metrics.fairness_metrics.logger')
    def test_logging(self, mock_logger, fairness_metrics, sample_data):
        """Test that appropriate logging occurs."""
        fairness_metrics.evaluate_comprehensive(
            y_true=sample_data['label'].values,
            predictions=sample_data['prediction'].values,
            features=sample_data[['feature1', 'feature2']].values,
            protected_attributes=sample_data[['gender', 'race']]
        )
        
        # Verify logging calls were made
        assert mock_logger.info.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
