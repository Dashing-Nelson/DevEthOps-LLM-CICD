"""
Test Stage - Fairness testing, explainability analysis, and performance validation
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

# Custom imports
from ..ethical_checks.fairness_evaluator import FairnessEvaluator
from ..ethical_checks.explainability_analyzer import ExplainabilityAnalyzer
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.fairness_metrics import FairnessMetrics


class FairnessViolationError(Exception):
    """Custom exception for fairness violations"""
    pass


class PerformanceViolationError(Exception):
    """Custom exception for performance violations"""
    pass


class TestStage:
    """
    Test stage for the DevEthOps pipeline.
    Handles fairness testing, explainability analysis, and performance validation.
    """
    
    def __init__(self, model, config_path: Optional[str] = None):
        """
        Initialize the Test Stage.
        
        Args:
            model: Trained ML model
            config_path: Path to the configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.config_path = config_path or "config/pipeline_config.yaml"
        self.fairness_config_path = "config/fairness_thresholds.yaml"
        self.model_config_path = "config/model_config.yaml"
        
        # Load configurations
        self.config = self._load_config()
        self.fairness_config = self._load_fairness_config()
        self.model_config = self._load_model_config()
        
        # Initialize components
        self.fairness_evaluator = FairnessEvaluator(self.fairness_config_path)
        self.explainability_analyzer = None  # Initialized when needed
        self.performance_metrics = PerformanceMetrics(self.model_config['model_config'])
        self.fairness_metrics = FairnessMetrics(self.fairness_config)
        
        # Test results storage
        self.test_results = {
            'fairness_results': {},
            'performance_results': {},
            'explainability_results': {},
            'overall_status': 'pending'
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
            
    def _load_fairness_config(self) -> Dict[str, Any]:
        """Load fairness configuration"""
        try:
            with open(self.fairness_config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Fairness config file not found: {self.fairness_config_path}")
            raise
            
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        try:
            with open(self.model_config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Model config file not found: {self.model_config_path}")
            raise
    
    def run_fairness_tests(self, test_data: pd.DataFrame, y_true: np.ndarray, 
                          protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Run comprehensive fairness tests on the model.
        
        Args:
            test_data: Test dataset
            y_true: True labels
            protected_attributes: List of protected attribute column names
            
        Returns:
            Dict containing fairness test results
        """
        self.logger.info("Running fairness tests")
        
        try:
            # Get model predictions
            y_pred = self.model.predict(test_data)
            y_pred_proba = None
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                try:
                    y_pred_proba = self.model.predict_proba(test_data)
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
                        y_pred_proba = y_pred_proba[:, 1]  # Binary classification
                except Exception as e:
                    self.logger.warning(f"Could not get prediction probabilities: {str(e)}")
            
            # Prepare sensitive features dictionary
            sensitive_features_dict = {}
            for attr in protected_attributes:
                if attr in test_data.columns:
                    # Convert to binary if needed (assuming 0/1 encoding)
                    sensitive_features = test_data[attr].values
                    sensitive_features_dict[attr] = sensitive_features
                else:
                    self.logger.warning(f"Protected attribute '{attr}' not found in test data")
            
            # Calculate fairness metrics
            fairness_results = self.fairness_metrics.calculate_all_metrics(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features_dict=sensitive_features_dict,
                X=test_data.values
            )
            
            # Check for violations
            violations = self.fairness_metrics.check_fairness_violations(fairness_results)
            
            # Calculate demographic parity for each protected attribute
            for attr in protected_attributes:
                if attr in sensitive_features_dict:
                    dp_result = self.fairness_evaluator.evaluate_demographic_parity(
                        test_data, y_pred, sensitive_features_dict[attr]
                    )
                    fairness_results[f'{attr}_demographic_parity_detailed'] = dp_result
                    
                    # Calculate equalized odds
                    eo_result = self.fairness_evaluator.evaluate_equalized_odds(
                        y_true, y_pred, sensitive_features_dict[attr]
                    )
                    fairness_results[f'{attr}_equalized_odds_detailed'] = eo_result
                    
                    # Calculate disparate impact
                    di_result = self.fairness_evaluator.evaluate_disparate_impact(
                        test_data, y_pred, sensitive_features_dict[attr]
                    )
                    fairness_results[f'{attr}_disparate_impact_detailed'] = di_result
            
            # Generate fairness report
            fairness_report = self.fairness_metrics.generate_fairness_report(
                fairness_results, violations
            )
            
            # Store results
            self.test_results['fairness_results'] = {
                'metrics': fairness_results,
                'violations': violations,
                'report': fairness_report,
                'status': 'pass' if not any(violations.values()) else 'fail'
            }
            
            self.logger.info("Fairness tests completed")
            return self.test_results['fairness_results']
            
        except Exception as e:
            self.logger.error(f"Error running fairness tests: {str(e)}")
            self.test_results['fairness_results'] = {
                'status': 'error',
                'error': str(e)
            }
            raise
    
    def generate_explanations(self, test_samples: pd.DataFrame, 
                            num_samples: int = 100) -> Dict[str, Any]:
        """
        Generate model explanations using LIME and SHAP.
        
        Args:
            test_samples: Sample data for explanation generation
            num_samples: Number of samples to explain
            
        Returns:
            Dict containing explanation results
        """
        self.logger.info("Generating model explanations")
        
        try:
            # Initialize explainability analyzer if not already done
            if self.explainability_analyzer is None:
                self.explainability_analyzer = ExplainabilityAnalyzer(
                    self.model, 
                    test_samples.values[:1000] if len(test_samples) > 1000 else test_samples.values
                )
            
            # Select samples for explanation
            sample_indices = np.random.choice(
                len(test_samples), 
                min(num_samples, len(test_samples)), 
                replace=False
            )
            explanation_samples = test_samples.iloc[sample_indices]
            
            # Generate SHAP explanations
            shap_results = {}
            try:
                shap_values = self.explainability_analyzer.compute_shap_values(
                    explanation_samples.values
                )
                
                # Calculate feature importance
                feature_importance = np.mean(np.abs(shap_values), axis=0)
                feature_names = explanation_samples.columns.tolist()
                
                shap_results = {
                    'feature_importance': dict(zip(feature_names, feature_importance)),
                    'top_features': sorted(
                        zip(feature_names, feature_importance), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10],
                    'shap_values_sample': shap_values[:min(10, len(shap_values))].tolist()
                }
                
            except Exception as e:
                self.logger.warning(f"SHAP explanation failed: {str(e)}")
                shap_results = {'error': str(e)}
            
            # Generate LIME explanations
            lime_results = {}
            try:
                lime_explanations = []
                for i in range(min(5, len(explanation_samples))):  # Explain first 5 samples
                    sample = explanation_samples.iloc[i].values
                    explanation = self.explainability_analyzer.generate_lime_explanation(sample)
                    lime_explanations.append({
                        'sample_index': sample_indices[i],
                        'explanation': explanation
                    })
                
                lime_results = {
                    'explanations': lime_explanations,
                    'num_explained_samples': len(lime_explanations)
                }
                
            except Exception as e:
                self.logger.warning(f"LIME explanation failed: {str(e)}")
                lime_results = {'error': str(e)}
            
            # Check for biased features
            biased_features = []
            if 'feature_importance' in shap_results:
                protected_attrs = self.config.get('protected_attributes', [])
                for attr in protected_attrs:
                    if attr in shap_results['feature_importance']:
                        importance = shap_results['feature_importance'][attr]
                        # Flag if protected attribute has high importance
                        if importance > 0.1:  # Threshold for concern
                            biased_features.append({
                                'attribute': attr,
                                'importance': importance,
                                'concern_level': 'high' if importance > 0.2 else 'medium'
                            })
            
            # Store results
            self.test_results['explainability_results'] = {
                'shap_results': shap_results,
                'lime_results': lime_results,
                'biased_features': biased_features,
                'status': 'success',
                'num_samples_explained': len(explanation_samples)
            }
            
            self.logger.info("Model explanations generated successfully")
            return self.test_results['explainability_results']
            
        except Exception as e:
            self.logger.error(f"Error generating explanations: {str(e)}")
            self.test_results['explainability_results'] = {
                'status': 'error',
                'error': str(e)
            }
            raise
    
    def validate_performance(self, test_data: pd.DataFrame, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Validate model performance against configured thresholds.
        
        Args:
            test_data: Test dataset
            y_true: True labels
            
        Returns:
            Dict containing performance validation results
        """
        self.logger.info("Validating model performance")
        
        try:
            # Get model predictions
            y_pred = self.model.predict(test_data)
            y_pred_proba = None
            
            if hasattr(self.model, 'predict_proba'):
                try:
                    y_pred_proba = self.model.predict_proba(test_data)
                except Exception as e:
                    self.logger.warning(f"Could not get prediction probabilities: {str(e)}")
            
            # Calculate basic metrics
            basic_metrics = self.performance_metrics.calculate_basic_metrics(
                y_true, y_pred, y_pred_proba
            )
            
            # Calculate detailed metrics
            detailed_metrics = self.performance_metrics.calculate_detailed_metrics(
                y_true, y_pred
            )
            
            # Check thresholds
            threshold_results = self.performance_metrics.check_performance_thresholds(
                basic_metrics
            )
            
            # Calculate complexity metrics
            complexity_metrics = self.performance_metrics.calculate_model_complexity_metrics(
                self.model, test_data.values
            )
            
            # Generate performance report
            performance_report = self.performance_metrics.generate_performance_report(
                basic_metrics, detailed_metrics, threshold_results
            )
            
            # Determine overall status
            passed_thresholds = sum(threshold_results.values())
            total_thresholds = len(threshold_results)
            status = 'pass' if passed_thresholds == total_thresholds else 'fail'
            
            # Store results
            self.test_results['performance_results'] = {
                'basic_metrics': basic_metrics,
                'detailed_metrics': detailed_metrics,
                'complexity_metrics': complexity_metrics,
                'threshold_results': threshold_results,
                'report': performance_report,
                'status': status,
                'thresholds_passed': f"{passed_thresholds}/{total_thresholds}"
            }
            
            self.logger.info(f"Performance validation completed: {status}")
            return self.test_results['performance_results']
            
        except Exception as e:
            self.logger.error(f"Error validating performance: {str(e)}")
            self.test_results['performance_results'] = {
                'status': 'error',
                'error': str(e)
            }
            raise
    
    def run_comprehensive_tests(self, test_data: pd.DataFrame, y_true: np.ndarray,
                               protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Run all test stage components (fairness, performance, explainability).
        
        Args:
            test_data: Test dataset
            y_true: True labels
            protected_attributes: List of protected attributes
            
        Returns:
            Dict containing all test results
        """
        self.logger.info("Running comprehensive test suite")
        
        try:
            # Run fairness tests
            fairness_results = self.run_fairness_tests(test_data, y_true, protected_attributes)
            
            # Run performance validation
            performance_results = self.validate_performance(test_data, y_true)
            
            # Generate explanations
            explanation_results = self.generate_explanations(test_data)
            
            # Determine overall test status
            fairness_passed = fairness_results.get('status') == 'pass'
            performance_passed = performance_results.get('status') == 'pass'
            explanations_generated = explanation_results.get('status') == 'success'
            
            overall_status = 'pass' if (fairness_passed and performance_passed and explanations_generated) else 'fail'
            
            self.test_results['overall_status'] = overall_status
            
            # Generate summary report
            summary_report = self._generate_summary_report()
            self.test_results['summary_report'] = summary_report
            
            self.logger.info(f"Comprehensive tests completed: {overall_status}")
            return self.test_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive tests: {str(e)}")
            self.test_results['overall_status'] = 'error'
            self.test_results['error'] = str(e)
            raise
    
    def _generate_summary_report(self) -> str:
        """Generate a summary report of all test results"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DEVETHOPS TEST STAGE SUMMARY REPORT")
        report_lines.append("=" * 60)
        
        # Overall status
        status_emoji = "✅" if self.test_results['overall_status'] == 'pass' else "❌"
        report_lines.append(f"\nOverall Status: {status_emoji} {self.test_results['overall_status'].upper()}")
        
        # Fairness results summary
        fairness_status = self.test_results['fairness_results'].get('status', 'unknown')
        fairness_emoji = "✅" if fairness_status == 'pass' else "❌"
        report_lines.append(f"\n1. Fairness Tests: {fairness_emoji} {fairness_status.upper()}")
        
        if 'violations' in self.test_results['fairness_results']:
            violations = self.test_results['fairness_results']['violations']
            total_violations = sum(len(v) for v in violations.values())
            report_lines.append(f"   - Total violations: {total_violations}")
            for level, viols in violations.items():
                if viols:
                    report_lines.append(f"   - {level.capitalize()}: {len(viols)}")
        
        # Performance results summary
        performance_status = self.test_results['performance_results'].get('status', 'unknown')
        performance_emoji = "✅" if performance_status == 'pass' else "❌"
        report_lines.append(f"\n2. Performance Tests: {performance_emoji} {performance_status.upper()}")
        
        if 'thresholds_passed' in self.test_results['performance_results']:
            thresholds_info = self.test_results['performance_results']['thresholds_passed']
            report_lines.append(f"   - Thresholds passed: {thresholds_info}")
        
        # Explainability results summary
        explainability_status = self.test_results['explainability_results'].get('status', 'unknown')
        explainability_emoji = "✅" if explainability_status == 'success' else "❌"
        report_lines.append(f"\n3. Explainability Analysis: {explainability_emoji} {explainability_status.upper()}")
        
        if 'biased_features' in self.test_results['explainability_results']:
            biased_features = self.test_results['explainability_results']['biased_features']
            report_lines.append(f"   - Potentially biased features: {len(biased_features)}")
        
        # Recommendations
        report_lines.append("\nRecommendations:")
        report_lines.append("-" * 20)
        
        if self.test_results['overall_status'] == 'pass':
            report_lines.append("🎉 All tests passed! Model ready for deployment.")
        else:
            report_lines.append("⚠️  Issues found. Address the following before deployment:")
            
            if fairness_status != 'pass':
                report_lines.append("   - Apply bias mitigation techniques")
                report_lines.append("   - Review training data for representation issues")
            
            if performance_status != 'pass':
                report_lines.append("   - Improve model performance through tuning")
                report_lines.append("   - Consider additional training data")
            
            if explainability_status != 'success':
                report_lines.append("   - Investigate explanation generation issues")
                report_lines.append("   - Ensure model interpretability requirements are met")
        
        report_lines.append("=" * 60)
        return "\n".join(report_lines)
    
    def save_test_artifacts(self, output_dir: str = "artifacts/test") -> Dict[str, str]:
        """
        Save test stage artifacts.
        
        Args:
            output_dir: Directory to save artifacts
            
        Returns:
            Dict containing paths to saved artifacts
        """
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        artifacts = {}
        
        # Save test results
        results_path = os.path.join(output_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        artifacts['test_results'] = results_path
        
        # Save individual reports
        if 'fairness_results' in self.test_results and 'report' in self.test_results['fairness_results']:
            fairness_report_path = os.path.join(output_dir, "fairness_report.txt")
            with open(fairness_report_path, 'w') as f:
                f.write(self.test_results['fairness_results']['report'])
            artifacts['fairness_report'] = fairness_report_path
        
        if 'performance_results' in self.test_results and 'report' in self.test_results['performance_results']:
            performance_report_path = os.path.join(output_dir, "performance_report.txt")
            with open(performance_report_path, 'w') as f:
                f.write(self.test_results['performance_results']['report'])
            artifacts['performance_report'] = performance_report_path
        
        if 'summary_report' in self.test_results:
            summary_report_path = os.path.join(output_dir, "summary_report.txt")
            with open(summary_report_path, 'w') as f:
                f.write(self.test_results['summary_report'])
            artifacts['summary_report'] = summary_report_path
        
        self.logger.info(f"Test artifacts saved to {output_dir}")
        return artifacts


if __name__ == "__main__":
    # Example usage
    # test_stage = TestStage(model)
    # results = test_stage.run_comprehensive_tests(test_data, y_true, ["gender", "race"])
    pass
