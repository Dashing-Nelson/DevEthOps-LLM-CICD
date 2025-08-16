"""
Fairness Metrics Calculator
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import confusion_matrix
import logging


class FairnessMetrics:
    """
    Comprehensive fairness metrics calculator for ML models.
    Implements various fairness criteria including demographic parity,
    equalized odds, and individual fairness.
    """
    
    def __init__(self, thresholds_config: Dict[str, Any]):
        """
        Initialize fairness metrics calculator.
        
        Args:
            thresholds_config: Configuration containing fairness thresholds
        """
        self.logger = logging.getLogger(__name__)
        self.thresholds = thresholds_config.get('fairness_thresholds', {})
        
    def demographic_parity_difference(self, y_pred: np.ndarray, sensitive_features: np.ndarray) -> float:
        """
        Calculate demographic parity difference.
        
        Args:
            y_pred: Model predictions (binary)
            sensitive_features: Binary sensitive attribute (0 for unprivileged, 1 for privileged)
            
        Returns:
            float: Difference in positive prediction rates between groups
        """
        try:
            privileged_mask = sensitive_features == 1
            unprivileged_mask = sensitive_features == 0
            
            privileged_rate = np.mean(y_pred[privileged_mask])
            unprivileged_rate = np.mean(y_pred[unprivileged_mask])
            
            return abs(privileged_rate - unprivileged_rate)
            
        except Exception as e:
            self.logger.error(f"Error calculating demographic parity: {str(e)}")
            return float('inf')
    
    def disparate_impact_ratio(self, y_pred: np.ndarray, sensitive_features: np.ndarray) -> float:
        """
        Calculate disparate impact ratio.
        
        Args:
            y_pred: Model predictions (binary)
            sensitive_features: Binary sensitive attribute
            
        Returns:
            float: Ratio of positive prediction rates (unprivileged/privileged)
        """
        try:
            privileged_mask = sensitive_features == 1
            unprivileged_mask = sensitive_features == 0
            
            privileged_rate = np.mean(y_pred[privileged_mask])
            unprivileged_rate = np.mean(y_pred[unprivileged_mask])
            
            if privileged_rate == 0:
                return float('inf') if unprivileged_rate > 0 else 1.0
                
            return unprivileged_rate / privileged_rate
            
        except Exception as e:
            self.logger.error(f"Error calculating disparate impact: {str(e)}")
            return 0.0
    
    def equalized_odds_difference(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate equalized odds difference (TPR and FPR differences).
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_features: Binary sensitive attribute
            
        Returns:
            Dict containing TPR and FPR differences
        """
        try:
            privileged_mask = sensitive_features == 1
            unprivileged_mask = sensitive_features == 0
            
            # Calculate confusion matrices for each group
            privileged_cm = confusion_matrix(y_true[privileged_mask], y_pred[privileged_mask])
            unprivileged_cm = confusion_matrix(y_true[unprivileged_mask], y_pred[unprivileged_mask])
            
            # Calculate TPR and FPR for each group
            def calculate_rates(cm):
                if cm.shape != (2, 2):
                    return 0.0, 0.0  # Handle edge cases
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                return tpr, fpr
            
            privileged_tpr, privileged_fpr = calculate_rates(privileged_cm)
            unprivileged_tpr, unprivileged_fpr = calculate_rates(unprivileged_cm)
            
            return {
                'tpr_difference': abs(privileged_tpr - unprivileged_tpr),
                'fpr_difference': abs(privileged_fpr - unprivileged_fpr)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating equalized odds: {str(e)}")
            return {'tpr_difference': float('inf'), 'fpr_difference': float('inf')}
    
    def individual_fairness_score(self, X: np.ndarray, y_pred: np.ndarray, 
                                distance_threshold: float = 0.1) -> float:
        """
        Calculate individual fairness score based on similar individuals.
        
        Args:
            X: Feature matrix
            y_pred: Model predictions
            distance_threshold: Threshold for considering individuals as similar
            
        Returns:
            float: Individual fairness score (lower is better)
        """
        try:
            from sklearn.metrics.pairwise import euclidean_distances
            
            # Calculate pairwise distances
            distances = euclidean_distances(X)
            n_samples = len(X)
            
            # Find similar pairs based on distance threshold
            similar_pairs = []
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if distances[i, j] <= distance_threshold:
                        similar_pairs.append((i, j))
            
            if not similar_pairs:
                return 0.0  # No similar pairs found
            
            # Calculate prediction differences for similar pairs
            prediction_differences = []
            for i, j in similar_pairs:
                diff = abs(y_pred[i] - y_pred[j])
                prediction_differences.append(diff)
            
            # Return average difference (lower is more fair)
            return np.mean(prediction_differences)
            
        except Exception as e:
            self.logger.error(f"Error calculating individual fairness: {str(e)}")
            return float('inf')
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            sensitive_features_dict: Dict[str, np.ndarray],
                            X: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate all fairness metrics for multiple sensitive attributes.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_features_dict: Dictionary of sensitive features
            X: Feature matrix (optional, for individual fairness)
            
        Returns:
            Dict containing all calculated metrics
        """
        all_metrics = {}
        
        for attr_name, sensitive_features in sensitive_features_dict.items():
            try:
                # Demographic parity
                dp_diff = self.demographic_parity_difference(y_pred, sensitive_features)
                all_metrics[f'{attr_name}_demographic_parity_difference'] = dp_diff
                
                # Disparate impact
                di_ratio = self.disparate_impact_ratio(y_pred, sensitive_features)
                all_metrics[f'{attr_name}_disparate_impact_ratio'] = di_ratio
                
                # Equalized odds
                eo_metrics = self.equalized_odds_difference(y_true, y_pred, sensitive_features)
                all_metrics[f'{attr_name}_tpr_difference'] = eo_metrics['tpr_difference']
                all_metrics[f'{attr_name}_fpr_difference'] = eo_metrics['fpr_difference']
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {attr_name}: {str(e)}")
                continue
        
        # Individual fairness (if feature matrix provided)
        if X is not None:
            try:
                individual_fairness = self.individual_fairness_score(X, y_pred)
                all_metrics['individual_fairness_score'] = individual_fairness
            except Exception as e:
                self.logger.error(f"Error calculating individual fairness: {str(e)}")
        
        return all_metrics
    
    def check_fairness_violations(self, metrics: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Check which metrics violate configured thresholds.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Dict containing violations by severity level
        """
        violations = {
            'critical': [],
            'warning': [],
            'info': []
        }
        
        alert_levels = self.thresholds.get('alert_levels', {})
        critical_threshold = alert_levels.get('critical', 0.2)
        warning_threshold = alert_levels.get('warning', 0.15)
        info_threshold = alert_levels.get('info', 0.1)
        
        for metric_name, value in metrics.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue
                
            # Determine violation level
            if 'demographic_parity' in metric_name or 'tpr_difference' in metric_name or 'fpr_difference' in metric_name:
                # Lower is better for these metrics
                if value > critical_threshold:
                    violations['critical'].append(f"{metric_name}: {value:.4f}")
                elif value > warning_threshold:
                    violations['warning'].append(f"{metric_name}: {value:.4f}")
                elif value > info_threshold:
                    violations['info'].append(f"{metric_name}: {value:.4f}")
                    
            elif 'disparate_impact' in metric_name:
                # Should be close to 1.0
                if abs(1.0 - value) > critical_threshold:
                    violations['critical'].append(f"{metric_name}: {value:.4f}")
                elif abs(1.0 - value) > warning_threshold:
                    violations['warning'].append(f"{metric_name}: {value:.4f}")
                elif abs(1.0 - value) > info_threshold:
                    violations['info'].append(f"{metric_name}: {value:.4f}")
        
        return violations
    
    def generate_fairness_report(self, metrics: Dict[str, float], violations: Dict[str, List[str]]) -> str:
        """
        Generate a comprehensive fairness report.
        
        Args:
            metrics: Calculated fairness metrics
            violations: Detected violations by severity
            
        Returns:
            str: Formatted fairness report
        """
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("FAIRNESS EVALUATION REPORT")
        report_lines.append("=" * 50)
        
        # Summary
        total_violations = sum(len(v) for v in violations.values())
        report_lines.append(f"\nSummary: {total_violations} total violations detected")
        
        # Violations by severity
        for level, violation_list in violations.items():
            if violation_list:
                report_lines.append(f"\n{level.upper()} violations ({len(violation_list)}):")
                for violation in violation_list:
                    report_lines.append(f"  - {violation}")
        
        # Detailed metrics
        report_lines.append("\nDetailed Metrics:")
        report_lines.append("-" * 30)
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                report_lines.append(f"{metric_name}: {value:.4f}")
            else:
                report_lines.append(f"{metric_name}: {value}")
        
        # Recommendations
        report_lines.append("\nRecommendations:")
        report_lines.append("-" * 30)
        
        if violations['critical']:
            report_lines.append("⚠️  CRITICAL: Immediate action required for critical violations")
            report_lines.append("   - Consider bias mitigation techniques")
            report_lines.append("   - Re-evaluate model training approach")
            
        if violations['warning']:
            report_lines.append("⚡ WARNING: Monitor closely and consider improvements")
            report_lines.append("   - Apply post-processing fairness techniques")
            report_lines.append("   - Increase monitoring frequency")
            
        if not total_violations:
            report_lines.append("✅ All fairness metrics within acceptable thresholds")
        
        report_lines.append("=" * 50)
        
        return "\n".join(report_lines)
