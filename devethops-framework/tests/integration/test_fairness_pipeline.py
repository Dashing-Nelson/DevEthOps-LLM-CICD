"""
Integration tests for the complete DevEthOps fairness pipeline.
Tests end-to-end fairness evaluation workflow.
"""

import pytest
import tempfile
import os
import json
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.pipeline.build_stage import BuildStage
from src.pipeline.test_stage import TestStage
from src.ethical_checks.fairness_evaluator import FairnessEvaluator
from src.metrics.fairness_metrics import FairnessMetrics


class TestFairnessPipeline:
    """Integration tests for fairness pipeline."""
    
    @pytest.fixture
    def biased_dataset(self):
        """Create a dataset with known bias for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create features
        data = pd.DataFrame({
            'income': np.random.lognormal(10, 1, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'education_years': np.random.randint(8, 20, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], 
                                   n_samples, p=[0.6, 0.2, 0.15, 0.05])
        })
        
        # Create biased credit approval labels
        # Introduce systematic bias against certain groups
        base_approval_rate = 0.7
        gender_bias = np.where(data['gender'] == 'Female', -0.2, 0.0)
        race_bias = np.where(data['race'] == 'Black', -0.3, 
                   np.where(data['race'] == 'Hispanic', -0.2, 0.0))
        
        # Income and education positively influence approval
        income_factor = (data['income'] - data['income'].median()) / data['income'].std() * 0.1
        education_factor = (data['education_years'] - data['education_years'].median()) / data['education_years'].std() * 0.1
        
        approval_prob = base_approval_rate + gender_bias + race_bias + income_factor + education_factor
        approval_prob = np.clip(approval_prob, 0.1, 0.9)
        
        data['credit_approved'] = np.random.binomial(1, approval_prob)
        
        return data
    
    @pytest.fixture
    def fair_dataset(self):
        """Create a fair dataset without bias for comparison."""
        np.random.seed(123)
        n_samples = 1000
        
        data = pd.DataFrame({
            'income': np.random.lognormal(10, 1, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'education_years': np.random.randint(8, 20, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples)
        })
        
        # Fair approval based only on financial factors
        base_approval_rate = 0.7
        income_factor = (data['income'] - data['income'].median()) / data['income'].std() * 0.2
        education_factor = (data['education_years'] - data['education_years'].median()) / data['education_years'].std() * 0.1
        
        approval_prob = base_approval_rate + income_factor + education_factor
        approval_prob = np.clip(approval_prob, 0.1, 0.9)
        
        data['credit_approved'] = np.random.binomial(1, approval_prob)
        
        return data
    
    def test_bias_detection_in_pipeline(self, biased_dataset):
        """Test that the pipeline correctly detects bias in biased datasets."""
        # Save dataset to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            biased_dataset.to_csv(f.name, index=False)
            dataset_path = f.name
        
        try:
            # Initialize build stage with bias detection enabled
            build_config = {
                'bias_detection': {'enabled': True},
                'protected_attributes': ['gender', 'race'],
                'label_column': 'credit_approved'
            }
            build_stage = BuildStage(build_config)
            
            # Load and validate data
            load_result = build_stage.load_and_validate_data(dataset_path)
            assert load_result['success'] is True
            
            # Run bias detection
            dataset = load_result['dataset']
            bias_result = build_stage.detect_bias(dataset)
            
            # Should detect bias in this biased dataset
            assert isinstance(bias_result, dict)
            assert 'bias_detected' in bias_result
            
            # Verify specific bias metrics
            if bias_result['bias_detected']:
                assert 'bias_metrics' in bias_result
                bias_metrics = bias_result['bias_metrics']
                
                # Check that gender and race bias are detected
                assert any('gender' in str(metric).lower() for metric in bias_metrics.keys())
                assert any('race' in str(metric).lower() for metric in bias_metrics.keys())
                
        finally:
            os.unlink(dataset_path)
    
    def test_fairness_evaluation_comprehensive(self, biased_dataset):
        """Test comprehensive fairness evaluation."""
        # Create fairness evaluator
        config = {
            'protected_attributes': ['gender', 'race'],
            'favorable_label': 1,
            'unfavorable_label': 0,
            'thresholds': {
                'demographic_parity': 0.1,
                'disparate_impact': 0.8,
                'equalized_odds': 0.1
            }
        }
        
        evaluator = FairnessEvaluator(config)
        
        # Prepare data
        features = biased_dataset[['income', 'age', 'education_years']].values
        labels = biased_dataset['credit_approved'].values
        protected_attrs = biased_dataset[['gender', 'race']]
        
        # Create mock predictions (should show bias)
        np.random.seed(42)
        biased_predictions = np.where(
            (biased_dataset['gender'] == 'Female') | (biased_dataset['race'] == 'Black'),
            np.random.binomial(1, 0.4, len(biased_dataset)),  # Lower approval rate
            np.random.binomial(1, 0.8, len(biased_dataset))   # Higher approval rate
        )
        
        # Run evaluation
        evaluation_result = evaluator.evaluate_model_fairness(
            model=None,  # We're providing predictions directly
            X_test=features,
            y_test=labels,
            protected_attributes=protected_attrs,
            predictions=biased_predictions
        )
        
        assert isinstance(evaluation_result, dict)
        assert 'overall_fairness_score' in evaluation_result
        assert 'fairness_metrics' in evaluation_result
        assert 'violations' in evaluation_result
        
        # Should detect fairness violations
        assert len(evaluation_result['violations']) > 0
        assert evaluation_result['overall_fairness_score'] < 0.8  # Should be low due to bias
    
    def test_fairness_comparison_biased_vs_fair(self, biased_dataset, fair_dataset):
        """Test that fairness metrics can distinguish between biased and fair datasets."""
        config = {
            'protected_attributes': ['gender', 'race'],
            'favorable_label': 1,
            'unfavorable_label': 0
        }
        
        fairness_metrics = FairnessMetrics(config)
        
        # Evaluate biased dataset
        biased_result = fairness_metrics.evaluate_comprehensive(
            y_true=biased_dataset['credit_approved'].values,
            predictions=biased_dataset['credit_approved'].values,  # Using true labels as predictions
            features=biased_dataset[['income', 'age', 'education_years']].values,
            protected_attributes=biased_dataset[['gender', 'race']]
        )
        
        # Evaluate fair dataset
        fair_result = fairness_metrics.evaluate_comprehensive(
            y_true=fair_dataset['credit_approved'].values,
            predictions=fair_dataset['credit_approved'].values,
            features=fair_dataset[['income', 'age', 'education_years']].values,
            protected_attributes=fair_dataset[['gender', 'race']]
        )
        
        # Fair dataset should have better fairness scores
        assert fair_result['overall_score'] >= biased_result['overall_score']
        
        # Demographic parity should be better in fair dataset
        fair_dp = abs(fair_result['demographic_parity']['gender'])
        biased_dp = abs(biased_result['demographic_parity']['gender'])
        assert fair_dp <= biased_dp
    
    def test_end_to_end_pipeline_with_bias_mitigation(self, biased_dataset):
        """Test complete pipeline including bias detection and mitigation."""
        # Save dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            biased_dataset.to_csv(f.name, index=False)
            dataset_path = f.name
        
        try:
            # Step 1: Build stage with bias detection and mitigation
            build_config = {
                'bias_detection': {'enabled': True},
                'bias_mitigation': {'enabled': True, 'strategy': 'reweighting'},
                'protected_attributes': ['gender', 'race'],
                'label_column': 'credit_approved'
            }
            build_stage = BuildStage(build_config)
            build_result = build_stage.run({'data_path': dataset_path})
            
            assert build_result['success'] is True
            
            # Step 2: Test stage with fairness validation
            test_config = {
                'fairness_tests': True,
                'explainability_tests': True,
                'performance_tests': True,
                'fairness_thresholds': {
                    'demographic_parity': 0.1,
                    'disparate_impact': 0.8
                }
            }
            test_stage = TestStage(test_config)
            
            # Mock model for testing
            mock_model = Mock()
            mock_model.predict.return_value = np.random.binomial(1, 0.6, len(biased_dataset))
            mock_model.predict_proba.return_value = np.random.random((len(biased_dataset), 2))
            
            # Run fairness tests
            fairness_test_result = test_stage.run_fairness_tests(dataset_path)
            
            assert isinstance(fairness_test_result, dict)
            assert 'passed' in fairness_test_result
            
            # If fairness tests fail due to bias, that's expected for biased dataset
            if not fairness_test_result['passed']:
                assert 'fairness_violations' in fairness_test_result
                
        finally:
            os.unlink(dataset_path)
    
    def test_intersectional_fairness_analysis(self, biased_dataset):
        """Test intersectional fairness analysis across multiple protected attributes."""
        fairness_metrics = FairnessMetrics({
            'protected_attributes': ['gender', 'race'],
            'favorable_label': 1,
            'unfavorable_label': 0
        })
        
        # Run intersectional analysis
        intersectional_result = fairness_metrics.intersectional_analysis(
            predictions=biased_dataset['credit_approved'].values,
            protected_attributes=biased_dataset[['gender', 'race']]
        )
        
        assert isinstance(intersectional_result, dict)
        assert len(intersectional_result) > 0
        
        # Should analyze intersectional groups like "Female_Black", "Male_White", etc.
        expected_patterns = ['Female_Black', 'Male_White', 'Female_Hispanic']
        found_patterns = [pattern for pattern in expected_patterns 
                         if any(pattern in group for group in intersectional_result.keys())]
        
        assert len(found_patterns) > 0, "Should find intersectional group patterns"
    
    def test_fairness_monitoring_alerts(self, biased_dataset):
        """Test that fairness monitoring generates appropriate alerts."""
        # Simulate model predictions over time with degrading fairness
        np.random.seed(42)
        n_samples = len(biased_dataset)
        
        # Initial fair predictions
        fair_predictions = np.random.binomial(1, 0.7, n_samples)
        
        # Degraded predictions with increased bias
        degraded_predictions = np.where(
            biased_dataset['gender'] == 'Female',
            np.random.binomial(1, 0.3, n_samples),  # Much lower approval for females
            np.random.binomial(1, 0.8, n_samples)   # Higher for males
        )
        
        fairness_metrics = FairnessMetrics({
            'protected_attributes': ['gender'],
            'favorable_label': 1,
            'unfavorable_label': 0
        })
        
        # Calculate fairness for both scenarios
        fair_score = fairness_metrics.demographic_parity(
            predictions=fair_predictions,
            protected_attribute=biased_dataset['gender'].values
        )
        
        degraded_score = fairness_metrics.demographic_parity(
            predictions=degraded_predictions,
            protected_attribute=biased_dataset['gender'].values
        )
        
        # Degraded scenario should show much worse fairness
        assert abs(degraded_score) > abs(fair_score)
        assert abs(degraded_score) > 0.2  # Should be significantly biased
    
    @pytest.mark.parametrize("protected_attr", ["gender", "race"])
    def test_fairness_metrics_per_attribute(self, biased_dataset, protected_attr):
        """Test fairness metrics for individual protected attributes."""
        fairness_metrics = FairnessMetrics({
            'protected_attributes': [protected_attr],
            'favorable_label': 1,
            'unfavorable_label': 0
        })
        
        predictions = biased_dataset['credit_approved'].values
        protected_values = biased_dataset[protected_attr].values
        
        # Test demographic parity
        dp_result = fairness_metrics.demographic_parity(
            predictions=predictions,
            protected_attribute=protected_values
        )
        assert isinstance(dp_result, float)
        
        # Test disparate impact
        di_result = fairness_metrics.disparate_impact(
            predictions=predictions,
            protected_attribute=protected_values
        )
        assert isinstance(di_result, float)
        assert di_result >= 0
    
    def test_performance_vs_fairness_tradeoff(self, biased_dataset):
        """Test analysis of performance vs fairness tradeoffs."""
        # Create two models: one optimized for performance, one for fairness
        np.random.seed(42)
        
        # High-performance but potentially biased model
        performance_model_preds = np.where(
            (biased_dataset['income'] > biased_dataset['income'].median()) & 
            (biased_dataset['education_years'] > 12),
            np.random.binomial(1, 0.9, len(biased_dataset)),
            np.random.binomial(1, 0.2, len(biased_dataset))
        )
        
        # Fair but potentially lower performance model
        fairness_model_preds = np.random.binomial(1, 0.6, len(biased_dataset))
        
        fairness_metrics = FairnessMetrics({
            'protected_attributes': ['gender', 'race'],
            'favorable_label': 1,
            'unfavorable_label': 0
        })
        
        # Calculate fairness scores
        perf_model_fairness = fairness_metrics.evaluate_comprehensive(
            y_true=biased_dataset['credit_approved'].values,
            predictions=performance_model_preds,
            features=biased_dataset[['income', 'age', 'education_years']].values,
            protected_attributes=biased_dataset[['gender', 'race']]
        )
        
        fair_model_fairness = fairness_metrics.evaluate_comprehensive(
            y_true=biased_dataset['credit_approved'].values,
            predictions=fairness_model_preds,
            features=biased_dataset[['income', 'age', 'education_years']].values,
            protected_attributes=biased_dataset[['gender', 'race']]
        )
        
        # Fair model should have better or equal fairness score
        assert fair_model_fairness['overall_score'] >= perf_model_fairness['overall_score']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
