"""
Model monitoring with drift detection and fairness tracking.

Implements rolling fairness metrics, data drift detection, and alerting
with placeholders for Prometheus/Grafana integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import warnings

# Drift detection imports
try:
    from scipy import stats
    from scipy.spatial.distance import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    logging.warning("SciPy not available. Some drift detection methods will be limited.")
    SCIPY_AVAILABLE = False

# Fairness evaluation
from .fairness_checks import FairnessEvaluator

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Data structure for drift alerts."""
    timestamp: datetime
    drift_type: str  # 'data', 'prediction', 'fairness'
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    severity: str  # 'low', 'medium', 'high'
    message: str


@dataclass
class ModelMetrics:
    """Container for model monitoring metrics."""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_drift: float
    data_drift: float
    fairness_metrics: Dict[str, float]
    prediction_count: int
    average_confidence: float


class DriftDetector:
    """
    Detects data and prediction drift using statistical methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize drift detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.reference_data = None
        self.reference_predictions = None
        self.thresholds = config.get('drift_thresholds', {
            'data_drift': 0.1,
            'prediction_drift': 0.1,
            'psi_threshold': 0.2  # Population Stability Index
        })
        
    def set_reference_data(self, X_reference: pd.DataFrame,
                          y_reference: Optional[pd.Series] = None) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            X_reference: Reference feature data
            y_reference: Reference target data (optional)
        """
        self.reference_data = X_reference
        self.reference_predictions = y_reference
        logger.info(f"Reference data set with {len(X_reference)} samples")
    
    def detect_data_drift(self, X_current: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift in features.
        
        Args:
            X_current: Current feature data
            
        Returns:
            Drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
        
        logger.info("Detecting data drift...")
        
        drift_results = {
            'overall_drift_score': 0.0,
            'feature_drift_scores': {},
            'drift_detected': False,
            'drifted_features': [],
            'method': 'wasserstein_distance'
        }
        
        # Check each feature for drift
        drift_scores = []
        
        for feature in self.reference_data.columns:
            if feature in X_current.columns:
                try:
                    if pd.api.types.is_numeric_dtype(self.reference_data[feature]):
                        # Numerical feature - use Wasserstein distance
                        drift_score = self._wasserstein_drift(
                            self.reference_data[feature], X_current[feature]
                        )
                    else:
                        # Categorical feature - use PSI
                        drift_score = self._psi_drift(
                            self.reference_data[feature], X_current[feature]
                        )
                    
                    drift_results['feature_drift_scores'][feature] = drift_score
                    drift_scores.append(drift_score)
                    
                    # Check if feature has drifted
                    if drift_score > self.thresholds['data_drift']:
                        drift_results['drifted_features'].append(feature)
                
                except Exception as e:
                    logger.warning(f"Error calculating drift for feature {feature}: {e}")
                    drift_results['feature_drift_scores'][feature] = 0.0
        
        # Overall drift score (average of all features)
        if drift_scores:
            drift_results['overall_drift_score'] = np.mean(drift_scores)
            drift_results['drift_detected'] = drift_results['overall_drift_score'] > self.thresholds['data_drift']
        
        logger.info(f"Data drift analysis complete. Overall score: {drift_results['overall_drift_score']:.4f}")
        
        return drift_results
    
    def detect_prediction_drift(self, y_current: pd.Series) -> Dict[str, Any]:
        """
        Detect drift in model predictions.
        
        Args:
            y_current: Current predictions
            
        Returns:
            Prediction drift results
        """
        if self.reference_predictions is None:
            logger.warning("No reference predictions available. Skipping prediction drift detection.")
            return {'drift_detected': False, 'drift_score': 0.0}
        
        logger.info("Detecting prediction drift...")
        
        try:
            if pd.api.types.is_numeric_dtype(y_current):
                # Numerical predictions
                drift_score = self._wasserstein_drift(self.reference_predictions, y_current)
            else:
                # Categorical predictions
                drift_score = self._psi_drift(self.reference_predictions, y_current)
            
            drift_detected = drift_score > self.thresholds['prediction_drift']
            
            return {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'threshold': self.thresholds['prediction_drift']
            }
        
        except Exception as e:
            logger.error(f"Error detecting prediction drift: {e}")
            return {'drift_detected': False, 'drift_score': 0.0, 'error': str(e)}
    
    def _wasserstein_drift(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Wasserstein distance for numerical features."""
        if not SCIPY_AVAILABLE:
            # Fallback to simple statistical comparison
            return abs(reference.mean() - current.mean()) / reference.std()
        
        try:
            # Remove NaN values
            ref_clean = reference.dropna().values
            cur_clean = current.dropna().values
            
            if len(ref_clean) == 0 or len(cur_clean) == 0:
                return 0.0
            
            # Normalize to [0, 1] scale
            all_values = np.concatenate([ref_clean, cur_clean])
            min_val, max_val = all_values.min(), all_values.max()
            
            if max_val == min_val:
                return 0.0
            
            ref_norm = (ref_clean - min_val) / (max_val - min_val)
            cur_norm = (cur_clean - min_val) / (max_val - min_val)
            
            return wasserstein_distance(ref_norm, cur_norm)
        
        except Exception as e:
            logger.warning(f"Error calculating Wasserstein distance: {e}")
            return 0.0
    
    def _psi_drift(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Population Stability Index for categorical features."""
        try:
            # Get value counts and normalize
            ref_counts = reference.value_counts(normalize=True)
            cur_counts = current.value_counts(normalize=True)
            
            # Align categories
            all_categories = set(ref_counts.index) | set(cur_counts.index)
            
            psi = 0.0
            for cat in all_categories:
                ref_pct = ref_counts.get(cat, 0.001)  # Small value to avoid log(0)
                cur_pct = cur_counts.get(cat, 0.001)
                
                psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
            
            return abs(psi)
        
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0


class FairnessMonitor:
    """
    Monitors fairness metrics over time.
    """
    
    def __init__(self, config: Dict[str, Any], protected_attributes: List[str]):
        """
        Initialize fairness monitor.
        
        Args:
            config: Configuration dictionary
            protected_attributes: Protected attributes to monitor
        """
        self.config = config
        self.protected_attributes = protected_attributes
        self.fairness_evaluator = FairnessEvaluator(config)
        self.fairness_history = []
        
    def evaluate_fairness(self, X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        """
        Evaluate fairness metrics for current data.
        
        Args:
            X: Feature data with protected attributes
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Fairness evaluation results
        """
        fairness_results = {}
        
        for attr in self.protected_attributes:
            if attr in X.columns:
                try:
                    metrics = self.fairness_evaluator.evaluate_fairness(
                        X, y_true, y_pred, attr
                    )
                    gate_results = self.fairness_evaluator.check_fairness_gates(metrics)
                    
                    fairness_results[attr] = {
                        'metrics': metrics,
                        'gates_passed': gate_results['overall_pass'],
                        'failed_metrics': gate_results['failed_metrics']
                    }
                
                except Exception as e:
                    logger.error(f"Error evaluating fairness for {attr}: {e}")
                    fairness_results[attr] = {'error': str(e)}
        
        # Store in history
        self.fairness_history.append({
            'timestamp': datetime.now(),
            'results': fairness_results
        })
        
        return fairness_results
    
    def get_fairness_trends(self, window_hours: int = 24) -> Dict[str, Any]:
        """
        Get fairness trends over time window.
        
        Args:
            window_hours: Time window in hours
            
        Returns:
            Fairness trends analysis
        """
        if not self.fairness_history:
            return {'error': 'No fairness history available'}
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_history = [
            entry for entry in self.fairness_history
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_history:
            return {'error': 'No recent fairness data available'}
        
        trends = {}
        
        for attr in self.protected_attributes:
            attr_trends = {
                'statistical_parity_trend': [],
                'disparate_impact_trend': [],
                'gates_passed_trend': []
            }
            
            for entry in recent_history:
                if attr in entry['results'] and 'metrics' in entry['results'][attr]:
                    metrics = entry['results'][attr]['metrics']
                    attr_trends['statistical_parity_trend'].append(
                        metrics.statistical_parity_difference
                    )
                    attr_trends['disparate_impact_trend'].append(
                        metrics.disparate_impact
                    )
                    attr_trends['gates_passed_trend'].append(
                        entry['results'][attr]['gates_passed']
                    )
            
            trends[attr] = attr_trends
        
        return trends


class ModelMonitor:
    """
    Main model monitoring orchestrator.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.drift_detector = DriftDetector(config)
        self.fairness_monitor = None
        self.metrics_history = []
        self.alerts = []
        
        # Monitoring configuration
        self.monitoring_config = {
            'drift_check_interval': config.get('drift_check_interval', 3600),  # seconds
            'fairness_check_interval': config.get('fairness_check_interval', 3600),
            'alert_thresholds': config.get('alert_thresholds', {
                'accuracy_drop': 0.05,
                'f1_drop': 0.05,
                'fairness_violation': True
            })
        }
    
    def setup_drift_detection(self, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Set up drift detection with reference data.
        
        Args:
            reference_data: Reference data for drift comparison
            
        Returns:
            Setup results
        """
        self.drift_detector.set_reference_data(reference_data)
        
        return {
            'drift_detection_enabled': True,
            'reference_data_shape': reference_data.shape,
            'thresholds': self.drift_detector.thresholds
        }
    
    def setup_fairness_monitoring(self, model, protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Set up fairness monitoring.
        
        Args:
            model: Model to monitor
            protected_attributes: Protected attributes
            
        Returns:
            Setup results
        """
        if protected_attributes:
            self.fairness_monitor = FairnessMonitor(self.config, protected_attributes)
            
            return {
                'fairness_monitoring_enabled': True,
                'protected_attributes': protected_attributes,
                'thresholds': self.config.get('fairness_thresholds', {})
            }
        else:
            return {
                'fairness_monitoring_enabled': False,
                'reason': 'No protected attributes specified'
            }
    
    def monitor_batch(self, X: pd.DataFrame, y_true: Optional[pd.Series] = None,
                     y_pred: Optional[pd.Series] = None, model=None) -> Dict[str, Any]:
        """
        Monitor a batch of predictions.
        
        Args:
            X: Feature data
            y_true: True labels (if available)
            y_pred: Predicted labels
            model: Model for generating predictions if y_pred not provided
            
        Returns:
            Monitoring results
        """
        timestamp = datetime.now()
        logger.info(f"Monitoring batch of {len(X)} samples at {timestamp}")
        
        results = {
            'timestamp': timestamp,
            'batch_size': len(X),
            'drift_detected': False,
            'fairness_violations': [],
            'alerts': []
        }
        
        # Generate predictions if not provided
        if y_pred is None and model is not None:
            y_pred = pd.Series(model.predict(X))
        
        # 1. Data drift detection
        try:
            drift_results = self.drift_detector.detect_data_drift(X)
            results['data_drift'] = drift_results
            
            if drift_results['drift_detected']:
                results['drift_detected'] = True
                alert = DriftAlert(
                    timestamp=timestamp,
                    drift_type='data',
                    metric_name='overall_drift_score',
                    current_value=drift_results['overall_drift_score'],
                    baseline_value=0.0,
                    threshold=self.drift_detector.thresholds['data_drift'],
                    severity='medium',
                    message=f"Data drift detected. Score: {drift_results['overall_drift_score']:.4f}"
                )
                self.alerts.append(alert)
                results['alerts'].append(asdict(alert))
        
        except Exception as e:
            logger.error(f"Error in data drift detection: {e}")
            results['data_drift'] = {'error': str(e)}
        
        # 2. Prediction drift detection
        if y_pred is not None:
            try:
                pred_drift_results = self.drift_detector.detect_prediction_drift(y_pred)
                results['prediction_drift'] = pred_drift_results
                
                if pred_drift_results['drift_detected']:
                    results['drift_detected'] = True
                    alert = DriftAlert(
                        timestamp=timestamp,
                        drift_type='prediction',
                        metric_name='prediction_drift_score',
                        current_value=pred_drift_results['drift_score'],
                        baseline_value=0.0,
                        threshold=pred_drift_results['threshold'],
                        severity='medium',
                        message=f"Prediction drift detected. Score: {pred_drift_results['drift_score']:.4f}"
                    )
                    self.alerts.append(alert)
                    results['alerts'].append(asdict(alert))
            
            except Exception as e:
                logger.error(f"Error in prediction drift detection: {e}")
                results['prediction_drift'] = {'error': str(e)}
        
        # 3. Fairness monitoring
        if self.fairness_monitor is not None and y_true is not None and y_pred is not None:
            try:
                fairness_results = self.fairness_monitor.evaluate_fairness(X, y_true, y_pred)
                results['fairness_evaluation'] = fairness_results
                
                # Check for fairness violations
                for attr, attr_results in fairness_results.items():
                    if not attr_results.get('gates_passed', True):
                        results['fairness_violations'].append(attr)
                        alert = DriftAlert(
                            timestamp=timestamp,
                            drift_type='fairness',
                            metric_name=f'fairness_{attr}',
                            current_value=0.0,  # Would need specific metric value
                            baseline_value=0.0,
                            threshold=0.0,
                            severity='high',
                            message=f"Fairness violation detected for {attr}"
                        )
                        self.alerts.append(alert)
                        results['alerts'].append(asdict(alert))
            
            except Exception as e:
                logger.error(f"Error in fairness monitoring: {e}")
                results['fairness_evaluation'] = {'error': str(e)}
        
        # 4. Performance metrics (if true labels available)
        if y_true is not None and y_pred is not None:
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                performance_metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='binary'),
                    'recall': recall_score(y_true, y_pred, average='binary'),
                    'f1_score': f1_score(y_true, y_pred, average='binary')
                }
                results['performance_metrics'] = performance_metrics
                
                # Store in history
                model_metrics = ModelMetrics(
                    timestamp=timestamp,
                    accuracy=performance_metrics['accuracy'],
                    precision=performance_metrics['precision'],
                    recall=performance_metrics['recall'],
                    f1_score=performance_metrics['f1_score'],
                    prediction_drift=results.get('prediction_drift', {}).get('drift_score', 0.0),
                    data_drift=results.get('data_drift', {}).get('overall_drift_score', 0.0),
                    fairness_metrics={},  # Would extract from fairness results
                    prediction_count=len(X),
                    average_confidence=0.0  # Would calculate from prediction probabilities
                )
                self.metrics_history.append(model_metrics)
            
            except Exception as e:
                logger.error(f"Error calculating performance metrics: {e}")
                results['performance_metrics'] = {'error': str(e)}
        
        logger.info(f"Monitoring complete. Alerts: {len(results['alerts'])}")
        return results
    
    def get_monitoring_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get monitoring summary for time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Monitoring summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]
        
        # Recent metrics
        recent_metrics = [
            metrics for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
        
        summary = {
            'time_period_hours': hours,
            'total_alerts': len(recent_alerts),
            'alert_breakdown': {},
            'average_metrics': {},
            'trends': {}
        }
        
        # Alert breakdown
        for alert in recent_alerts:
            drift_type = alert.drift_type
            summary['alert_breakdown'][drift_type] = summary['alert_breakdown'].get(drift_type, 0) + 1
        
        # Average metrics
        if recent_metrics:
            summary['average_metrics'] = {
                'accuracy': np.mean([m.accuracy for m in recent_metrics]),
                'f1_score': np.mean([m.f1_score for m in recent_metrics]),
                'data_drift': np.mean([m.data_drift for m in recent_metrics]),
                'prediction_drift': np.mean([m.prediction_drift for m in recent_metrics])
            }
        
        return summary
    
    def create_monitoring_config(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Create monitoring configuration for deployment.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Monitoring configuration
        """
        config = {
            'pipeline_id': pipeline_id,
            'monitoring_enabled': True,
            'drift_detection': {
                'enabled': True,
                'check_interval_seconds': self.monitoring_config['drift_check_interval'],
                'thresholds': self.drift_detector.thresholds
            },
            'fairness_monitoring': {
                'enabled': self.fairness_monitor is not None,
                'check_interval_seconds': self.monitoring_config['fairness_check_interval'],
                'protected_attributes': getattr(self.fairness_monitor, 'protected_attributes', [])
            },
            'alerting': {
                'enabled': True,
                'thresholds': self.monitoring_config['alert_thresholds']
            },
            'metrics_retention_hours': 168,  # 1 week
            'prometheus_config': self._create_prometheus_config(pipeline_id),
            'grafana_config': self._create_grafana_config(pipeline_id)
        }
        
        return config
    
    def setup_alerts(self) -> Dict[str, Any]:
        """
        Set up alerting configuration.
        
        Returns:
            Alert configuration
        """
        # TODO: Implement integration with actual alerting systems
        return {
            'alert_channels': ['stdout', 'logs'],  # Placeholder
            'alert_rules': [
                {
                    'name': 'data_drift_high',
                    'condition': 'data_drift_score > 0.2',
                    'severity': 'warning'
                },
                {
                    'name': 'fairness_violation',
                    'condition': 'fairness_gates_failed = true',
                    'severity': 'critical'
                },
                {
                    'name': 'performance_degradation',
                    'condition': 'f1_score_drop > 0.05',
                    'severity': 'warning'
                }
            ],
            'notification_channels': {
                'stdout': {'enabled': True},
                'prometheus': {'enabled': False, 'url': 'http://prometheus:9090'},
                'grafana': {'enabled': False, 'url': 'http://grafana:3000'}
            }
        }
    
    def _create_prometheus_config(self, pipeline_id: str) -> Dict[str, Any]:
        """Create Prometheus configuration."""
        return {
            'job_name': f'devethops_model_{pipeline_id}',
            'metrics_endpoint': f'/metrics',
            'scrape_interval': '30s',
            'metrics': [
                'model_accuracy',
                'model_f1_score',
                'data_drift_score',
                'prediction_drift_score',
                'fairness_violations_total',
                'predictions_total'
            ]
        }
    
    def _create_grafana_config(self, pipeline_id: str) -> Dict[str, Any]:
        """Create Grafana dashboard configuration."""
        return {
            'dashboard_name': f'DevEthOps Model Monitoring - {pipeline_id}',
            'panels': [
                {
                    'title': 'Model Performance',
                    'metrics': ['model_accuracy', 'model_f1_score'],
                    'type': 'time_series'
                },
                {
                    'title': 'Drift Detection',
                    'metrics': ['data_drift_score', 'prediction_drift_score'],
                    'type': 'time_series'
                },
                {
                    'title': 'Fairness Monitoring',
                    'metrics': ['fairness_violations_total'],
                    'type': 'stat'
                },
                {
                    'title': 'Prediction Volume',
                    'metrics': ['predictions_total'],
                    'type': 'time_series'
                }
            ]
        }
    
    def print_alerts(self) -> None:
        """Print recent alerts to stdout."""
        if not self.alerts:
            print("[PASS] No alerts")
            return
        
        print(f"🚨 {len(self.alerts)} ALERTS:")
        for alert in self.alerts[-10:]:  # Show last 10 alerts
            severity_icon = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(alert.severity, "⚪")
            print(f"  {severity_icon} [{alert.timestamp.strftime('%H:%M:%S')}] {alert.message}")


# TODO: Implement actual Prometheus metrics export
# TODO: Implement Grafana dashboard auto-generation
# TODO: Implement real-time streaming monitoring
# TODO: Add A/B testing monitoring capabilities
