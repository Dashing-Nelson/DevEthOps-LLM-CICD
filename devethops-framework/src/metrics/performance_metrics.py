"""
Performance Metrics Calculator
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for ML models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance metrics calculator.
        
        Args:
            config: Configuration dictionary with performance thresholds
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.thresholds = self.config.get('performance_thresholds', {
            'minimum_accuracy': 0.85,
            'minimum_f1_score': 0.80,
            'minimum_auc_roc': 0.89,
            'minimum_precision': 0.80,
            'minimum_recall': 0.80
        })
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for AUC-ROC)
            
        Returns:
            Dict containing basic metrics
        """
        try:
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # AUC-ROC if probabilities are provided
            if y_pred_proba is not None:
                try:
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                        # Multi-class case
                        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                    else:
                        # Binary case
                        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                except Exception as e:
                    self.logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
                    metrics['auc_roc'] = 0.0
            
            self.logger.info(f"Basic metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {str(e)}")
            return {}
    
    def calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate detailed classification metrics including per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict containing detailed metrics
        """
        try:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Per-class metrics
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            per_class_metrics = {}
            
            for label in unique_labels:
                label_str = str(label)
                if label_str in class_report:
                    per_class_metrics[f'class_{label}_precision'] = class_report[label_str]['precision']
                    per_class_metrics[f'class_{label}_recall'] = class_report[label_str]['recall']
                    per_class_metrics[f'class_{label}_f1'] = class_report[label_str]['f1-score']
                    per_class_metrics[f'class_{label}_support'] = class_report[label_str]['support']
            
            return {
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'per_class_metrics': per_class_metrics,
                'macro_avg_precision': class_report.get('macro avg', {}).get('precision', 0.0),
                'macro_avg_recall': class_report.get('macro avg', {}).get('recall', 0.0),
                'macro_avg_f1': class_report.get('macro avg', {}).get('f1-score', 0.0),
                'weighted_avg_precision': class_report.get('weighted avg', {}).get('precision', 0.0),
                'weighted_avg_recall': class_report.get('weighted avg', {}).get('recall', 0.0),
                'weighted_avg_f1': class_report.get('weighted avg', {}).get('f1-score', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating detailed metrics: {str(e)}")
            return {}
    
    def check_performance_thresholds(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if performance metrics meet minimum thresholds.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Dict indicating which thresholds are met
        """
        threshold_results = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            metric_name = threshold_name.replace('minimum_', '')
            
            if metric_name in metrics:
                threshold_results[threshold_name] = metrics[metric_name] >= threshold_value
            else:
                threshold_results[threshold_name] = False
                
        return threshold_results
    
    def calculate_model_complexity_metrics(self, model, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Calculate model complexity and efficiency metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            
        Returns:
            Dict containing complexity metrics
        """
        import time
        import sys
        
        try:
            metrics = {}
            
            # Model size (approximate)
            try:
                model_size = sys.getsizeof(model)
                metrics['model_size_bytes'] = model_size
            except:
                metrics['model_size_bytes'] = 0
            
            # Inference time
            start_time = time.time()
            _ = model.predict(X_test[:min(100, len(X_test))])  # Test on first 100 samples
            inference_time = time.time() - start_time
            metrics['inference_time_per_100_samples'] = inference_time
            metrics['average_inference_time_per_sample'] = inference_time / min(100, len(X_test))
            
            # Memory usage during inference
            try:
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss
                _ = model.predict(X_test[:min(1000, len(X_test))])
                memory_after = process.memory_info().rss
                metrics['memory_usage_mb'] = (memory_after - memory_before) / 1024 / 1024
            except:
                metrics['memory_usage_mb'] = 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity metrics: {str(e)}")
            return {}
    
    def generate_performance_report(self, basic_metrics: Dict[str, float], 
                                  detailed_metrics: Dict[str, Any],
                                  threshold_results: Dict[str, bool]) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            basic_metrics: Basic performance metrics
            detailed_metrics: Detailed performance metrics
            threshold_results: Threshold check results
            
        Returns:
            str: Formatted performance report
        """
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("PERFORMANCE EVALUATION REPORT")
        report_lines.append("=" * 50)
        
        # Basic metrics section
        report_lines.append("\nBasic Metrics:")
        report_lines.append("-" * 20)
        for metric, value in basic_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"{metric.capitalize()}: {value:.4f}")
            else:
                report_lines.append(f"{metric.capitalize()}: {value}")
        
        # Threshold compliance
        report_lines.append("\nThreshold Compliance:")
        report_lines.append("-" * 25)
        passed_thresholds = sum(threshold_results.values())
        total_thresholds = len(threshold_results)
        report_lines.append(f"Passed: {passed_thresholds}/{total_thresholds} thresholds")
        
        for threshold, passed in threshold_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            metric_name = threshold.replace('minimum_', '')
            threshold_value = self.thresholds[threshold]
            actual_value = basic_metrics.get(metric_name, 0.0)
            report_lines.append(f"  {metric_name}: {actual_value:.4f} >= {threshold_value:.4f} {status}")
        
        # Per-class performance
        if 'per_class_metrics' in detailed_metrics:
            report_lines.append("\nPer-Class Performance:")
            report_lines.append("-" * 25)
            per_class = detailed_metrics['per_class_metrics']
            
            # Group by class
            classes = set()
            for key in per_class.keys():
                if '_' in key:
                    class_part = key.split('_')[1]
                    classes.add(class_part)
            
            for cls in sorted(classes):
                precision = per_class.get(f'class_{cls}_precision', 0.0)
                recall = per_class.get(f'class_{cls}_recall', 0.0)
                f1 = per_class.get(f'class_{cls}_f1', 0.0)
                support = per_class.get(f'class_{cls}_support', 0)
                report_lines.append(f"  Class {cls}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, N={support}")
        
        # Recommendations
        report_lines.append("\nRecommendations:")
        report_lines.append("-" * 20)
        
        failed_thresholds = [k for k, v in threshold_results.items() if not v]
        if failed_thresholds:
            report_lines.append("⚠️  Performance improvements needed:")
            for threshold in failed_thresholds:
                metric_name = threshold.replace('minimum_', '')
                report_lines.append(f"   - Improve {metric_name}")
                
            report_lines.append("\n   Suggested actions:")
            report_lines.append("   - Collect more training data")
            report_lines.append("   - Feature engineering")
            report_lines.append("   - Model hyperparameter tuning")
            report_lines.append("   - Try different algorithms")
        else:
            report_lines.append("✅ All performance thresholds met!")
        
        # Overall assessment
        report_lines.append("\nOverall Assessment:")
        report_lines.append("-" * 20)
        
        if passed_thresholds == total_thresholds:
            report_lines.append("🎉 EXCELLENT: Model meets all performance requirements")
        elif passed_thresholds >= total_thresholds * 0.8:
            report_lines.append("👍 GOOD: Model meets most performance requirements")
        elif passed_thresholds >= total_thresholds * 0.6:
            report_lines.append("⚠️  ACCEPTABLE: Model needs improvement")
        else:
            report_lines.append("❌ POOR: Model requires significant improvement")
        
        report_lines.append("=" * 50)
        
        return "\n".join(report_lines)
