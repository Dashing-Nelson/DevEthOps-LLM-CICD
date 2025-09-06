"""
Fairness assessment using AIF360 metrics.

Implements Statistical Parity Difference, Disparate Impact, Demographic Parity,
Equal Opportunity Difference, Average Odds Difference with gate logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# AIF360 imports
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    AIF360_AVAILABLE = True
except ImportError:
    logging.warning("AIF360 not available. Using manual fairness metric implementations.")
    AIF360_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FairnessMetrics:
    """Container for fairness metrics results."""
    statistical_parity_difference: float
    disparate_impact: float
    equal_opportunity_difference: float
    average_odds_difference: float
    demographic_parity: float
    accuracy: float
    precision: float
    recall: float
    protected_attribute: str
    privileged_group: Any
    unprivileged_group: Any


class FairnessEvaluator:
    """
    Comprehensive fairness evaluation using AIF360 and custom implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fairness evaluator.
        
        Args:
            config: Configuration dictionary containing fairness thresholds
        """
        self.config = config
        self.thresholds = config.get('fairness_thresholds', {})
        self.use_aif360 = AIF360_AVAILABLE and config.get('use_aif360', True)
        
        # Default thresholds if not specified
        self.default_thresholds = {
            'disparate_impact': [0.8, 1.25],  # Should be between 0.8 and 1.25
            'statistical_parity_difference': 0.1,  # Absolute value should be <= 0.1
            'equal_opportunity_difference': 0.1,
            'average_odds_difference': 0.1,
            'demographic_parity': 0.1
        }
    
    def evaluate_fairness(self, X: Union[pd.DataFrame, np.ndarray], 
                         y_true: Union[pd.Series, np.ndarray],
                         y_pred: Union[pd.Series, np.ndarray],
                         protected_attribute: str,
                         privileged_groups: Optional[List[Any]] = None) -> FairnessMetrics:
        """
        Evaluate fairness metrics for a model's predictions.
        
        Args:
            X: Feature matrix (must include protected attribute)
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Name of protected attribute column
            privileged_groups: List of privileged group values
            
        Returns:
            FairnessMetrics object with computed metrics
        """
        logger.info(f"Evaluating fairness for protected attribute: {protected_attribute}")
        
        # Convert to pandas if needed
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, 'columns'):
                X = pd.DataFrame(X, columns=X.columns)
            else:
                raise ValueError("X must be a pandas DataFrame or have column information")
        
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true, name='target')
        
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred, name='predictions')
        
        # Validate protected attribute exists
        if protected_attribute not in X.columns:
            raise ValueError(f"Protected attribute '{protected_attribute}' not found in features")
        
        # Determine privileged and unprivileged groups
        if privileged_groups is None:
            privileged_groups = self._auto_detect_privileged_groups(X[protected_attribute])
        
        privileged_group = privileged_groups[0] if privileged_groups else None
        unprivileged_groups = [g for g in X[protected_attribute].unique() if g not in privileged_groups]
        unprivileged_group = unprivileged_groups[0] if unprivileged_groups else None
        
        if self.use_aif360:
            metrics = self._compute_aif360_metrics(
                X, y_true, y_pred, protected_attribute, privileged_groups
            )
        else:
            metrics = self._compute_manual_metrics(
                X, y_true, y_pred, protected_attribute, privileged_group, unprivileged_group
            )
        
        # Add overall performance metrics
        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.precision = precision_score(y_true, y_pred, average='binary')
        metrics.recall = recall_score(y_true, y_pred, average='binary')
        metrics.protected_attribute = protected_attribute
        metrics.privileged_group = privileged_group
        metrics.unprivileged_group = unprivileged_group
        
        logger.info("Fairness evaluation complete")
        return metrics
    
    def _auto_detect_privileged_groups(self, protected_series: pd.Series) -> List[Any]:
        """
        Auto-detect privileged groups based on common patterns.
        
        Args:
            protected_series: Series containing protected attribute values
            
        Returns:
            List of privileged group values
        """
        unique_values = protected_series.unique()
        
        # Common patterns for privileged groups
        privileged_patterns = {
            # Gender
            'male': ['male', 'M', 1, 'Male'],
            'female': ['female', 'F', 0, 'Female'],
            
            # Race/Ethnicity
            'white': ['white', 'White', 'Caucasian'],
            
            # Age (assume older is privileged for employment)
            'older': lambda x: x > protected_series.median() if pd.api.types.is_numeric_dtype(protected_series) else False
        }
        
        # Try to match patterns
        for pattern_name, pattern_values in privileged_patterns.items():
            if callable(pattern_values):
                # Handle numeric patterns
                if pd.api.types.is_numeric_dtype(protected_series):
                    privileged_mask = pattern_values(protected_series)
                    if privileged_mask.any():
                        return [True]  # For boolean masks
            else:
                # Handle categorical patterns
                matches = [v for v in unique_values if v in pattern_values]
                if matches:
                    return matches
        
        # Default: use majority group as privileged
        value_counts = protected_series.value_counts()
        majority_group = value_counts.index[0]
        
        logger.info(f"Auto-detected privileged group: {majority_group}")
        return [majority_group]
    
    def _compute_aif360_metrics(self, X: pd.DataFrame, y_true: pd.Series, 
                               y_pred: pd.Series, protected_attribute: str,
                               privileged_groups: List[Any]) -> FairnessMetrics:
        """
        Compute fairness metrics using AIF360.
        
        Args:
            X: Feature matrix
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute name
            privileged_groups: Privileged group values
            
        Returns:
            FairnessMetrics object
        """
        # Prepare data for AIF360
        df = X.copy()
        df['target'] = y_true
        df['predictions'] = y_pred
        
        # Create AIF360 datasets
        favorable_classes = [1]
        unfavorable_classes = [0]
        
        # Original dataset
        dataset_true = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df,
            label_names=['target'],
            protected_attribute_names=[protected_attribute],
            privileged_classes=[privileged_groups]
        )
        
        # Predicted dataset
        dataset_pred = dataset_true.copy()
        dataset_pred.labels = y_pred.values.reshape(-1, 1)
        
        # Compute dataset metrics (for original data)
        dataset_metric = BinaryLabelDatasetMetric(
            dataset_true,
            unprivileged_groups=[{protected_attribute: v} for v in 
                               X[protected_attribute].unique() if v not in privileged_groups],
            privileged_groups=[{protected_attribute: v} for v in privileged_groups]
        )
        
        # Compute classification metrics
        classification_metric = ClassificationMetric(
            dataset_true, dataset_pred,
            unprivileged_groups=[{protected_attribute: v} for v in 
                               X[protected_attribute].unique() if v not in privileged_groups],
            privileged_groups=[{protected_attribute: v} for v in privileged_groups]
        )
        
        # Extract metrics
        return FairnessMetrics(
            statistical_parity_difference=classification_metric.statistical_parity_difference(),
            disparate_impact=classification_metric.disparate_impact(),
            equal_opportunity_difference=classification_metric.equal_opportunity_difference(),
            average_odds_difference=classification_metric.average_odds_difference(),
            demographic_parity=abs(dataset_metric.statistical_parity_difference()),
            accuracy=0,  # Will be filled later
            precision=0,  # Will be filled later
            recall=0,  # Will be filled later
            protected_attribute=protected_attribute,
            privileged_group=None,  # Will be filled later
            unprivileged_group=None  # Will be filled later
        )
    
    def _compute_manual_metrics(self, X: pd.DataFrame, y_true: pd.Series,
                               y_pred: pd.Series, protected_attribute: str,
                               privileged_group: Any, unprivileged_group: Any) -> FairnessMetrics:
        """
        Compute fairness metrics manually when AIF360 is not available.
        
        Args:
            X: Feature matrix
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute name
            privileged_group: Privileged group value
            unprivileged_group: Unprivileged group value
            
        Returns:
            FairnessMetrics object
        """
        # Split by protected attribute
        privileged_mask = X[protected_attribute] == privileged_group
        unprivileged_mask = X[protected_attribute] == unprivileged_group
        
        # Predicted positive rates
        priv_pos_rate = y_pred[privileged_mask].mean()
        unpriv_pos_rate = y_pred[unprivileged_mask].mean()
        
        # True positive rates (sensitivity/recall)
        priv_tpr = self._true_positive_rate(y_true[privileged_mask], y_pred[privileged_mask])
        unpriv_tpr = self._true_positive_rate(y_true[unprivileged_mask], y_pred[unprivileged_mask])
        
        # False positive rates
        priv_fpr = self._false_positive_rate(y_true[privileged_mask], y_pred[privileged_mask])
        unpriv_fpr = self._false_positive_rate(y_true[unprivileged_mask], y_pred[unprivileged_mask])
        
        # Compute metrics
        statistical_parity_diff = priv_pos_rate - unpriv_pos_rate
        disparate_impact = unpriv_pos_rate / priv_pos_rate if priv_pos_rate > 0 else 0
        equal_opportunity_diff = priv_tpr - unpriv_tpr
        
        # Average odds difference (average of TPR and FPR differences)
        tpr_diff = priv_tpr - unpriv_tpr
        fpr_diff = priv_fpr - unpriv_fpr
        average_odds_diff = (tpr_diff + fpr_diff) / 2
        
        return FairnessMetrics(
            statistical_parity_difference=statistical_parity_diff,
            disparate_impact=disparate_impact,
            equal_opportunity_difference=equal_opportunity_diff,
            average_odds_difference=average_odds_diff,
            demographic_parity=abs(statistical_parity_diff),
            accuracy=0,  # Will be filled later
            precision=0,  # Will be filled later
            recall=0,  # Will be filled later
            protected_attribute=protected_attribute,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group
        )
    
    def _true_positive_rate(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate true positive rate (sensitivity/recall)."""
        if len(y_true) == 0:
            return 0.0
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _false_positive_rate(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate false positive rate."""
        if len(y_true) == 0:
            return 0.0
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def check_fairness_gates(self, metrics: FairnessMetrics) -> Dict[str, Any]:
        """
        Check if fairness metrics pass defined thresholds.
        
        Args:
            metrics: FairnessMetrics object
            
        Returns:
            Dictionary with gate results
        """
        logger.info("Checking fairness gates...")
        
        results = {
            'overall_pass': True,
            'individual_results': {},
            'failed_metrics': [],
            'warnings': []
        }
        
        # Check disparate impact
        di_threshold = self.thresholds.get('disparate_impact', self.default_thresholds['disparate_impact'])
        di_pass = di_threshold[0] <= metrics.disparate_impact <= di_threshold[1]
        results['individual_results']['disparate_impact'] = {
            'value': metrics.disparate_impact,
            'threshold': di_threshold,
            'pass': di_pass
        }
        if not di_pass:
            results['overall_pass'] = False
            results['failed_metrics'].append('disparate_impact')
        
        # Check statistical parity difference
        sp_threshold = self.thresholds.get('statistical_parity_difference', 
                                         self.default_thresholds['statistical_parity_difference'])
        sp_pass = abs(metrics.statistical_parity_difference) <= sp_threshold
        results['individual_results']['statistical_parity_difference'] = {
            'value': metrics.statistical_parity_difference,
            'threshold': f"±{sp_threshold}",
            'pass': sp_pass
        }
        if not sp_pass:
            results['overall_pass'] = False
            results['failed_metrics'].append('statistical_parity_difference')
        
        # Check equal opportunity difference
        eo_threshold = self.thresholds.get('equal_opportunity_difference',
                                         self.default_thresholds['equal_opportunity_difference'])
        eo_pass = abs(metrics.equal_opportunity_difference) <= eo_threshold
        results['individual_results']['equal_opportunity_difference'] = {
            'value': metrics.equal_opportunity_difference,
            'threshold': f"±{eo_threshold}",
            'pass': eo_pass
        }
        if not eo_pass:
            results['overall_pass'] = False
            results['failed_metrics'].append('equal_opportunity_difference')
        
        # Check average odds difference
        ao_threshold = self.thresholds.get('average_odds_difference',
                                         self.default_thresholds['average_odds_difference'])
        ao_pass = abs(metrics.average_odds_difference) <= ao_threshold
        results['individual_results']['average_odds_difference'] = {
            'value': metrics.average_odds_difference,
            'threshold': f"±{ao_threshold}",
            'pass': ao_pass
        }
        if not ao_pass:
            results['overall_pass'] = False
            results['failed_metrics'].append('average_odds_difference')
        
        # Log results
        if results['overall_pass']:
            logger.info("[PASS] All fairness gates passed!")
        else:
            logger.warning(f"[FAIL] Fairness gates failed: {', '.join(results['failed_metrics'])}")
        
        return results
    
    def generate_fairness_report(self, metrics: FairnessMetrics, 
                               gate_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive fairness report.
        
        Args:
            metrics: FairnessMetrics object
            gate_results: Results from check_fairness_gates
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "FAIRNESS EVALUATION REPORT",
            "=" * 60,
            f"Protected Attribute: {metrics.protected_attribute}",
            f"Privileged Group: {metrics.privileged_group}",
            f"Unprivileged Group: {metrics.unprivileged_group}",
            "",
            "PERFORMANCE METRICS:",
            f"  Accuracy: {metrics.accuracy:.4f}",
            f"  Precision: {metrics.precision:.4f}",
            f"  Recall: {metrics.recall:.4f}",
            "",
            "FAIRNESS METRICS:",
            f"  Statistical Parity Difference: {metrics.statistical_parity_difference:.4f}",
            f"  Disparate Impact: {metrics.disparate_impact:.4f}",
            f"  Equal Opportunity Difference: {metrics.equal_opportunity_difference:.4f}",
            f"  Average Odds Difference: {metrics.average_odds_difference:.4f}",
            f"  Demographic Parity: {metrics.demographic_parity:.4f}",
            "",
            "GATE RESULTS:",
        ]
        
        for metric_name, result in gate_results['individual_results'].items():
            status = "[PASS] PASS" if result['pass'] else "[FAIL] FAIL"
            report_lines.append(f"  {metric_name}: {result['value']:.4f} (threshold: {result['threshold']}) - {status}")
        
        report_lines.extend([
            "",
            f"OVERALL RESULT: {'[PASS] PASS' if gate_results['overall_pass'] else '[FAIL] FAIL'}",
            "=" * 60
        ])
        
        return "\n".join(report_lines)


def evaluate_model_fairness(model, X_test: Union[pd.DataFrame, np.ndarray],
                          y_test: Union[pd.Series, np.ndarray],
                          protected_attributes: List[str],
                          config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate fairness for a trained model across multiple protected attributes.
    
    Args:
        model: Trained model with predict method
        X_test: Test feature matrix
        y_test: Test labels
        protected_attributes: List of protected attribute names
        config: Configuration dictionary
        
    Returns:
        Dictionary with fairness evaluation results
    """
    logger.info("Evaluating model fairness across protected attributes...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Initialize evaluator
    evaluator = FairnessEvaluator(config)
    
    results = {
        'metrics': {},
        'gate_results': {},
        'reports': {},
        'overall_pass': True
    }
    
    # Evaluate each protected attribute
    for attr in protected_attributes:
        try:
            logger.info(f"Evaluating fairness for: {attr}")
            
            metrics = evaluator.evaluate_fairness(X_test, y_test, y_pred, attr)
            gate_results = evaluator.check_fairness_gates(metrics)
            report = evaluator.generate_fairness_report(metrics, gate_results)
            
            results['metrics'][attr] = metrics
            results['gate_results'][attr] = gate_results
            results['reports'][attr] = report
            
            if not gate_results['overall_pass']:
                results['overall_pass'] = False
                
        except Exception as e:
            logger.error(f"Error evaluating fairness for {attr}: {e}")
            results['overall_pass'] = False
    
    # Log summary
    if results['overall_pass']:
        logger.info("[PASS] Model passes all fairness evaluations!")
    else:
        failed_attrs = [attr for attr, gate in results['gate_results'].items() 
                       if not gate['overall_pass']]
        logger.warning(f"[FAIL] Model fails fairness evaluation for: {', '.join(failed_attrs)}")
    
    return results
