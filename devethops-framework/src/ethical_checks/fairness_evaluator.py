"""
Fairness Evaluator - Comprehensive fairness assessment using AIF360
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# AIF360 imports
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, RejectOptionClassification


class FairnessEvaluator:
    """
    Comprehensive fairness evaluator using AIF360 library.
    Evaluates various fairness metrics and provides detailed analysis.
    """
    
    def __init__(self, config_path: str = "config/fairness_thresholds.yaml"):
        """
        Initialize the fairness evaluator.
        
        Args:
            config_path: Path to fairness thresholds configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_config()
        self.thresholds = self.config.get('fairness_thresholds', {})
        
    def _load_config(self) -> Dict[str, Any]:
        """Load fairness configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            # Return default thresholds
            return {
                'fairness_thresholds': {
                    'demographic_parity': {'threshold': 0.1},
                    'disparate_impact': {'threshold': 0.8},
                    'equalized_odds_tpr': {'threshold': 0.1},
                    'equalized_odds_fpr': {'threshold': 0.1}
                }
            }
    
    def evaluate_demographic_parity(self, dataset: Union[pd.DataFrame, BinaryLabelDataset], 
                                   y_pred: np.ndarray, sensitive_features: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate demographic parity (statistical parity).
        
        Args:
            dataset: Dataset or DataFrame
            y_pred: Model predictions
            sensitive_features: Binary sensitive attribute array
            
        Returns:
            Dict containing demographic parity analysis
        """
        self.logger.info("Evaluating demographic parity")
        
        try:
            # Calculate selection rates for each group
            privileged_mask = sensitive_features == 1
            unprivileged_mask = sensitive_features == 0
            
            if not np.any(privileged_mask) or not np.any(unprivileged_mask):
                self.logger.warning("One or both groups are empty")
                return {
                    'privileged_selection_rate': 0.0,
                    'unprivileged_selection_rate': 0.0,
                    'difference': float('inf'),
                    'threshold': self.thresholds.get('demographic_parity', {}).get('threshold', 0.1),
                    'passed': False,
                    'error': 'Empty groups'
                }
            
            privileged_rate = np.mean(y_pred[privileged_mask])
            unprivileged_rate = np.mean(y_pred[unprivileged_mask])
            difference = abs(privileged_rate - unprivileged_rate)
            
            threshold = self.thresholds.get('demographic_parity', {}).get('threshold', 0.1)
            passed = difference <= threshold
            
            result = {
                'privileged_selection_rate': float(privileged_rate),
                'unprivileged_selection_rate': float(unprivileged_rate),
                'difference': float(difference),
                'threshold': threshold,
                'passed': passed,
                'privileged_count': int(np.sum(privileged_mask)),
                'unprivileged_count': int(np.sum(unprivileged_mask)),
                'privileged_positive_count': int(np.sum(y_pred[privileged_mask])),
                'unprivileged_positive_count': int(np.sum(y_pred[unprivileged_mask]))
            }
            
            self.logger.info(f"Demographic parity: {difference:.4f}, Threshold: {threshold}, Passed: {passed}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating demographic parity: {str(e)}")
            return {
                'error': str(e),
                'passed': False
            }
    
    def evaluate_equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              sensitive_features: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate equalized odds (equal TPR and FPR across groups).
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_features: Binary sensitive attribute
            
        Returns:
            Dict containing equalized odds analysis
        """
        self.logger.info("Evaluating equalized odds")
        
        try:
            from sklearn.metrics import confusion_matrix
            
            privileged_mask = sensitive_features == 1
            unprivileged_mask = sensitive_features == 0
            
            if not np.any(privileged_mask) or not np.any(unprivileged_mask):
                return {
                    'error': 'Empty groups',
                    'passed': False
                }
            
            # Calculate confusion matrices
            def safe_confusion_matrix(y_true_group, y_pred_group):
                """Calculate confusion matrix with error handling"""
                try:
                    cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
                    if cm.shape == (2, 2):
                        return cm
                    else:
                        # Handle case where only one class is present
                        full_cm = np.zeros((2, 2))
                        if len(np.unique(y_true_group)) == 1 and len(np.unique(y_pred_group)) == 1:
                            if y_true_group[0] == y_pred_group[0] == 1:
                                full_cm[1, 1] = len(y_true_group)  # All true positives
                            elif y_true_group[0] == y_pred_group[0] == 0:
                                full_cm[0, 0] = len(y_true_group)  # All true negatives
                        return full_cm
                except:
                    return np.zeros((2, 2))
            
            privileged_cm = safe_confusion_matrix(y_true[privileged_mask], y_pred[privileged_mask])
            unprivileged_cm = safe_confusion_matrix(y_true[unprivileged_mask], y_pred[unprivileged_mask])
            
            # Calculate TPR and FPR
            def calculate_rates(cm):
                """Calculate TPR and FPR from confusion matrix"""
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                return tpr, fpr
            
            privileged_tpr, privileged_fpr = calculate_rates(privileged_cm)
            unprivileged_tpr, unprivileged_fpr = calculate_rates(unprivileged_cm)
            
            tpr_difference = abs(privileged_tpr - unprivileged_tpr)
            fpr_difference = abs(privileged_fpr - unprivileged_fpr)
            
            tpr_threshold = self.thresholds.get('equalized_odds_tpr', {}).get('threshold', 0.1)
            fpr_threshold = self.thresholds.get('equalized_odds_fpr', {}).get('threshold', 0.1)
            
            tpr_passed = tpr_difference <= tpr_threshold
            fpr_passed = fpr_difference <= fpr_threshold
            
            result = {
                'privileged_tpr': float(privileged_tpr),
                'unprivileged_tpr': float(unprivileged_tpr),
                'tpr_difference': float(tpr_difference),
                'tpr_threshold': tpr_threshold,
                'tpr_passed': tpr_passed,
                'privileged_fpr': float(privileged_fpr),
                'unprivileged_fpr': float(unprivileged_fpr),
                'fpr_difference': float(fpr_difference),
                'fpr_threshold': fpr_threshold,
                'fpr_passed': fpr_passed,
                'overall_passed': tpr_passed and fpr_passed,
                'privileged_confusion_matrix': privileged_cm.tolist(),
                'unprivileged_confusion_matrix': unprivileged_cm.tolist()
            }
            
            self.logger.info(f"Equalized odds - TPR diff: {tpr_difference:.4f}, FPR diff: {fpr_difference:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating equalized odds: {str(e)}")
            return {
                'error': str(e),
                'passed': False
            }
    
    def evaluate_disparate_impact(self, dataset: Union[pd.DataFrame, BinaryLabelDataset], 
                                y_pred: np.ndarray, sensitive_features: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate disparate impact ratio.
        
        Args:
            dataset: Dataset or DataFrame
            y_pred: Model predictions
            sensitive_features: Binary sensitive attribute
            
        Returns:
            Dict containing disparate impact analysis
        """
        self.logger.info("Evaluating disparate impact")
        
        try:
            privileged_mask = sensitive_features == 1
            unprivileged_mask = sensitive_features == 0
            
            if not np.any(privileged_mask) or not np.any(unprivileged_mask):
                return {
                    'error': 'Empty groups',
                    'passed': False
                }
            
            privileged_rate = np.mean(y_pred[privileged_mask])
            unprivileged_rate = np.mean(y_pred[unprivileged_mask])
            
            # Calculate disparate impact ratio
            if privileged_rate == 0:
                ratio = float('inf') if unprivileged_rate > 0 else 1.0
            else:
                ratio = unprivileged_rate / privileged_rate
            
            threshold = self.thresholds.get('disparate_impact', {}).get('threshold', 0.8)
            passed = ratio >= threshold and ratio <= (1 / threshold) if threshold > 0 else False
            
            result = {
                'privileged_selection_rate': float(privileged_rate),
                'unprivileged_selection_rate': float(unprivileged_rate),
                'disparate_impact_ratio': float(ratio),
                'threshold': threshold,
                'acceptable_range': [threshold, 1/threshold if threshold > 0 else float('inf')],
                'passed': passed,
                'privileged_count': int(np.sum(privileged_mask)),
                'unprivileged_count': int(np.sum(unprivileged_mask))
            }
            
            self.logger.info(f"Disparate impact ratio: {ratio:.4f}, Threshold: {threshold}, Passed: {passed}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating disparate impact: {str(e)}")
            return {
                'error': str(e),
                'passed': False
            }
    
    def evaluate_individual_fairness(self, X: np.ndarray, y_pred: np.ndarray, 
                                   distance_metric: str = 'euclidean', 
                                   similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate individual fairness (similar individuals should receive similar predictions).
        
        Args:
            X: Feature matrix
            y_pred: Model predictions
            distance_metric: Distance metric to use
            similarity_threshold: Threshold for considering individuals as similar
            
        Returns:
            Dict containing individual fairness analysis
        """
        self.logger.info("Evaluating individual fairness")
        
        try:
            from sklearn.metrics.pairwise import pairwise_distances
            
            # Calculate pairwise distances
            distances = pairwise_distances(X, metric=distance_metric)
            n_samples = len(X)
            
            # Find similar pairs
            similar_pairs = []
            prediction_differences = []
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if distances[i, j] <= similarity_threshold:
                        similar_pairs.append((i, j))
                        pred_diff = abs(y_pred[i] - y_pred[j])
                        prediction_differences.append(pred_diff)
            
            if not prediction_differences:
                return {
                    'similar_pairs_count': 0,
                    'average_prediction_difference': 0.0,
                    'max_prediction_difference': 0.0,
                    'individual_fairness_score': 0.0,
                    'passed': True,
                    'message': 'No similar pairs found'
                }
            
            avg_pred_diff = np.mean(prediction_differences)
            max_pred_diff = np.max(prediction_differences)
            
            # Individual fairness score (lower is better)
            fairness_score = avg_pred_diff
            
            # Use a simple threshold for now
            fairness_threshold = 0.1
            passed = fairness_score <= fairness_threshold
            
            result = {
                'similar_pairs_count': len(similar_pairs),
                'average_prediction_difference': float(avg_pred_diff),
                'max_prediction_difference': float(max_pred_diff),
                'individual_fairness_score': float(fairness_score),
                'threshold': fairness_threshold,
                'passed': passed,
                'similarity_threshold_used': similarity_threshold,
                'distance_metric': distance_metric
            }
            
            self.logger.info(f"Individual fairness score: {fairness_score:.4f}, Passed: {passed}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating individual fairness: {str(e)}")
            return {
                'error': str(e),
                'passed': False
            }
    
    def comprehensive_fairness_evaluation(self, dataset: pd.DataFrame, y_true: np.ndarray, 
                                        y_pred: np.ndarray, protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive fairness evaluation across all metrics and protected attributes.
        
        Args:
            dataset: Complete dataset
            y_true: True labels
            y_pred: Model predictions
            protected_attributes: List of protected attribute column names
            
        Returns:
            Dict containing comprehensive fairness results
        """
        self.logger.info("Performing comprehensive fairness evaluation")
        
        comprehensive_results = {
            'protected_attributes': protected_attributes,
            'results_by_attribute': {},
            'overall_fairness_score': 0.0,
            'overall_passed': False,
            'summary': {}
        }
        
        total_tests = 0
        passed_tests = 0
        
        for attr in protected_attributes:
            if attr not in dataset.columns:
                self.logger.warning(f"Protected attribute '{attr}' not found in dataset")
                continue
                
            self.logger.info(f"Evaluating fairness for attribute: {attr}")
            
            sensitive_features = dataset[attr].values
            
            # Ensure binary encoding
            unique_values = np.unique(sensitive_features)
            if len(unique_values) > 2:
                self.logger.warning(f"Protected attribute '{attr}' has more than 2 unique values, using binary encoding")
                # Convert to binary (0 for smallest value, 1 for others)
                sensitive_features = (sensitive_features != unique_values[0]).astype(int)
            
            attr_results = {}
            
            # Demographic Parity
            try:
                dp_result = self.evaluate_demographic_parity(dataset, y_pred, sensitive_features)
                attr_results['demographic_parity'] = dp_result
                total_tests += 1
                if dp_result.get('passed', False):
                    passed_tests += 1
            except Exception as e:
                self.logger.error(f"Error in demographic parity for {attr}: {str(e)}")
            
            # Equalized Odds
            try:
                eo_result = self.evaluate_equalized_odds(y_true, y_pred, sensitive_features)
                attr_results['equalized_odds'] = eo_result
                total_tests += 1
                if eo_result.get('overall_passed', False):
                    passed_tests += 1
            except Exception as e:
                self.logger.error(f"Error in equalized odds for {attr}: {str(e)}")
            
            # Disparate Impact
            try:
                di_result = self.evaluate_disparate_impact(dataset, y_pred, sensitive_features)
                attr_results['disparate_impact'] = di_result
                total_tests += 1
                if di_result.get('passed', False):
                    passed_tests += 1
            except Exception as e:
                self.logger.error(f"Error in disparate impact for {attr}: {str(e)}")
            
            comprehensive_results['results_by_attribute'][attr] = attr_results
        
        # Individual Fairness (global evaluation)
        try:
            # Use only numerical columns for individual fairness
            numerical_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                X_numerical = dataset[numerical_cols].values
                if_result = self.evaluate_individual_fairness(X_numerical, y_pred)
                comprehensive_results['individual_fairness'] = if_result
                total_tests += 1
                if if_result.get('passed', False):
                    passed_tests += 1
        except Exception as e:
            self.logger.error(f"Error in individual fairness: {str(e)}")
        
        # Calculate overall metrics
        comprehensive_results['overall_fairness_score'] = passed_tests / total_tests if total_tests > 0 else 0.0
        comprehensive_results['overall_passed'] = passed_tests == total_tests
        
        comprehensive_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': f"{passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)" if total_tests > 0 else "0/0 (0%)"
        }
        
        self.logger.info(f"Comprehensive fairness evaluation completed: {passed_tests}/{total_tests} tests passed")
        return comprehensive_results
    
    def generate_fairness_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate a detailed fairness evaluation report.
        
        Args:
            evaluation_results: Results from comprehensive_fairness_evaluation
            
        Returns:
            str: Formatted report
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("COMPREHENSIVE FAIRNESS EVALUATION REPORT")
        report_lines.append("=" * 60)
        
        # Executive Summary
        overall_passed = evaluation_results.get('overall_passed', False)
        status_emoji = "✅" if overall_passed else "❌"
        summary = evaluation_results.get('summary', {})
        
        report_lines.append(f"\nExecutive Summary: {status_emoji}")
        report_lines.append(f"Overall Status: {'PASSED' if overall_passed else 'FAILED'}")
        report_lines.append(f"Tests Passed: {summary.get('success_rate', 'Unknown')}")
        report_lines.append(f"Fairness Score: {evaluation_results.get('overall_fairness_score', 0.0):.3f}")
        
        # Results by Protected Attribute
        results_by_attr = evaluation_results.get('results_by_attribute', {})
        for attr, attr_results in results_by_attr.items():
            report_lines.append(f"\n{'-' * 40}")
            report_lines.append(f"Protected Attribute: {attr.upper()}")
            report_lines.append(f"{'-' * 40}")
            
            # Demographic Parity
            if 'demographic_parity' in attr_results:
                dp = attr_results['demographic_parity']
                dp_status = "✅ PASS" if dp.get('passed', False) else "❌ FAIL"
                report_lines.append(f"\n1. Demographic Parity: {dp_status}")
                if 'difference' in dp:
                    report_lines.append(f"   Difference: {dp['difference']:.4f} (threshold: {dp.get('threshold', 'N/A')})")
                if 'privileged_selection_rate' in dp:
                    report_lines.append(f"   Privileged rate: {dp['privileged_selection_rate']:.4f}")
                    report_lines.append(f"   Unprivileged rate: {dp['unprivileged_selection_rate']:.4f}")
            
            # Equalized Odds
            if 'equalized_odds' in attr_results:
                eo = attr_results['equalized_odds']
                eo_status = "✅ PASS" if eo.get('overall_passed', False) else "❌ FAIL"
                report_lines.append(f"\n2. Equalized Odds: {eo_status}")
                if 'tpr_difference' in eo:
                    report_lines.append(f"   TPR difference: {eo['tpr_difference']:.4f} (threshold: {eo.get('tpr_threshold', 'N/A')})")
                    report_lines.append(f"   FPR difference: {eo['fpr_difference']:.4f} (threshold: {eo.get('fpr_threshold', 'N/A')})")
            
            # Disparate Impact
            if 'disparate_impact' in attr_results:
                di = attr_results['disparate_impact']
                di_status = "✅ PASS" if di.get('passed', False) else "❌ FAIL"
                report_lines.append(f"\n3. Disparate Impact: {di_status}")
                if 'disparate_impact_ratio' in di:
                    report_lines.append(f"   Ratio: {di['disparate_impact_ratio']:.4f} (threshold: {di.get('threshold', 'N/A')})")
                    if 'acceptable_range' in di:
                        report_lines.append(f"   Acceptable range: {di['acceptable_range'][0]:.3f} - {di['acceptable_range'][1]:.3f}")
        
        # Individual Fairness
        if 'individual_fairness' in evaluation_results:
            if_result = evaluation_results['individual_fairness']
            if_status = "✅ PASS" if if_result.get('passed', False) else "❌ FAIL"
            report_lines.append(f"\n{'-' * 40}")
            report_lines.append(f"Individual Fairness: {if_status}")
            report_lines.append(f"{'-' * 40}")
            if 'individual_fairness_score' in if_result:
                report_lines.append(f"Fairness Score: {if_result['individual_fairness_score']:.4f}")
                report_lines.append(f"Similar Pairs: {if_result.get('similar_pairs_count', 0)}")
        
        # Recommendations
        report_lines.append(f"\n{'=' * 30}")
        report_lines.append("RECOMMENDATIONS")
        report_lines.append(f"{'=' * 30}")
        
        if overall_passed:
            report_lines.append("🎉 Excellent! Your model passes all fairness tests.")
            report_lines.append("   Continue monitoring fairness in production.")
        else:
            report_lines.append("⚠️  Your model has fairness issues that need attention:")
            report_lines.append("")
            
            failed_tests = summary.get('total_tests', 0) - summary.get('passed_tests', 0)
            if failed_tests > 0:
                report_lines.append("Recommended actions:")
                report_lines.append("1. 🔍 Review training data for representation bias")
                report_lines.append("2. ⚖️  Apply bias mitigation techniques:")
                report_lines.append("   - Preprocessing: Reweighing, Disparate Impact Remover")
                report_lines.append("   - In-processing: Adversarial debiasing")
                report_lines.append("   - Post-processing: Calibrated Equalized Odds")
                report_lines.append("3. 📊 Collect more balanced training data")
                report_lines.append("4. 🔄 Re-evaluate after applying mitigations")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
