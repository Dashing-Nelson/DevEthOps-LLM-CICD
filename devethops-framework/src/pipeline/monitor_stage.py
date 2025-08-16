"""
Monitor Stage - Continuous monitoring for model drift and fairness degradation
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import deque

# Drift detection imports
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

# Monitoring imports
import psutil
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart


@dataclass
class DriftAlert:
    """Data class for drift alerts"""
    timestamp: datetime
    alert_type: str
    severity: str
    metric_name: str
    current_value: float
    threshold: float
    description: str
    recommendation: str


class DriftDetector:
    """
    Drift detection component for monitoring data and concept drift.
    """
    
    def __init__(self, reference_data: np.ndarray, config: Dict[str, Any]):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset for drift comparison
            config: Configuration for drift detection
        """
        self.logger = logging.getLogger(__name__)
        self.reference_data = reference_data
        self.config = config
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_reference_stats()
        
        # Drift history
        self.drift_history = deque(maxlen=1000)
        
    def _calculate_reference_stats(self) -> Dict[str, Any]:
        """Calculate reference statistics for drift detection"""
        try:
            stats = {
                'mean': np.mean(self.reference_data, axis=0),
                'std': np.std(self.reference_data, axis=0),
                'median': np.median(self.reference_data, axis=0),
                'min': np.min(self.reference_data, axis=0),
                'max': np.max(self.reference_data, axis=0),
                'percentiles': {
                    '25': np.percentile(self.reference_data, 25, axis=0),
                    '75': np.percentile(self.reference_data, 75, axis=0),
                    '95': np.percentile(self.reference_data, 95, axis=0)
                }
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error calculating reference stats: {str(e)}")
            return {}
    
    def detect_statistical_drift(self, current_data: np.ndarray, 
                                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect statistical drift using various statistical tests.
        
        Args:
            current_data: Current data to compare against reference
            feature_names: Names of features
            
        Returns:
            Dict containing drift detection results
        """
        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(current_data.shape[1])]
            
            drift_results = {
                'timestamp': datetime.now().isoformat(),
                'overall_drift_detected': False,
                'feature_drift': {},
                'drift_score': 0.0,
                'tests_performed': []
            }
            
            n_features = min(self.reference_data.shape[1], current_data.shape[1])
            drift_scores = []
            
            for i in range(n_features):
                feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                
                ref_feature = self.reference_data[:, i]
                curr_feature = current_data[:, i]
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = stats.ks_2samp(ref_feature, curr_feature)
                
                # Mann-Whitney U test
                try:
                    mw_stat, mw_pvalue = stats.mannwhitneyu(ref_feature, curr_feature, alternative='two-sided')
                except:
                    mw_stat, mw_pvalue = 0.0, 1.0
                
                # Population Stability Index (PSI)
                psi_score = self._calculate_psi(ref_feature, curr_feature)
                
                # Feature drift assessment
                feature_drift = {
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'mw_statistic': float(mw_stat) if not np.isnan(mw_stat) else 0.0,
                    'mw_pvalue': float(mw_pvalue) if not np.isnan(mw_pvalue) else 1.0,
                    'psi_score': float(psi_score),
                    'drift_detected': ks_pvalue < 0.05 or psi_score > 0.2
                }
                
                drift_results['feature_drift'][feature_name] = feature_drift
                
                # Calculate combined drift score
                drift_score = max(ks_stat, psi_score)
                drift_scores.append(drift_score)
            
            # Overall drift assessment
            drift_results['drift_score'] = float(np.mean(drift_scores))
            drift_results['max_drift_score'] = float(np.max(drift_scores))
            drift_results['overall_drift_detected'] = drift_results['drift_score'] > self.config.get('drift_threshold', 0.15)
            
            # Record in history
            self.drift_history.append({
                'timestamp': datetime.now(),
                'drift_score': drift_results['drift_score'],
                'drift_detected': drift_results['overall_drift_detected']
            })
            
            self.logger.info(f"Statistical drift detection completed. Score: {drift_results['drift_score']:.4f}")
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Error in statistical drift detection: {str(e)}")
            return {
                'error': str(e),
                'overall_drift_detected': False,
                'drift_score': 0.0
            }
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(reference, bins=bins)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            curr_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to frequencies (add small epsilon to avoid division by zero)
            epsilon = 1e-10
            ref_freq = (ref_hist + epsilon) / (len(reference) + bins * epsilon)
            curr_freq = (curr_hist + epsilon) / (len(current) + bins * epsilon)
            
            # Calculate PSI
            psi = np.sum((curr_freq - ref_freq) * np.log(curr_freq / ref_freq))
            
            return abs(psi)
            
        except Exception as e:
            self.logger.error(f"Error calculating PSI: {str(e)}")
            return 0.0
    
    def detect_concept_drift(self, predictions: np.ndarray, true_labels: np.ndarray,
                           window_size: int = 100) -> Dict[str, Any]:
        """
        Detect concept drift by monitoring prediction accuracy over time.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            window_size: Size of the sliding window for accuracy calculation
            
        Returns:
            Dict containing concept drift results
        """
        try:
            if len(predictions) < window_size:
                return {
                    'concept_drift_detected': False,
                    'error': 'Insufficient data for concept drift detection'
                }
            
            # Calculate accuracy over sliding windows
            accuracies = []
            for i in range(len(predictions) - window_size + 1):
                window_preds = predictions[i:i + window_size]
                window_labels = true_labels[i:i + window_size]
                accuracy = np.mean(window_preds == window_labels)
                accuracies.append(accuracy)
            
            # Detect significant drops in accuracy
            accuracy_threshold = self.config.get('accuracy_drop_threshold', 0.05)
            baseline_accuracy = np.mean(accuracies[:len(accuracies)//2]) if len(accuracies) > 10 else np.mean(accuracies)
            recent_accuracy = np.mean(accuracies[-min(10, len(accuracies)//4):])
            
            accuracy_drop = baseline_accuracy - recent_accuracy
            concept_drift_detected = accuracy_drop > accuracy_threshold
            
            return {
                'concept_drift_detected': concept_drift_detected,
                'baseline_accuracy': float(baseline_accuracy),
                'recent_accuracy': float(recent_accuracy),
                'accuracy_drop': float(accuracy_drop),
                'threshold': accuracy_threshold,
                'window_size': window_size
            }
            
        except Exception as e:
            self.logger.error(f"Error in concept drift detection: {str(e)}")
            return {
                'concept_drift_detected': False,
                'error': str(e)
            }


class FairnessMonitor:
    """
    Continuous fairness monitoring component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fairness monitor.
        
        Args:
            config: Configuration for fairness monitoring
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Fairness metrics history
        self.fairness_history = deque(maxlen=1000)
        
        # Alert thresholds
        self.thresholds = config.get('fairness_thresholds', {})
    
    def monitor_fairness_metrics(self, predictions: np.ndarray, sensitive_features: Dict[str, np.ndarray],
                               true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Monitor fairness metrics in real-time.
        
        Args:
            predictions: Model predictions
            sensitive_features: Dictionary of sensitive features
            true_labels: True labels (optional)
            
        Returns:
            Dict containing fairness monitoring results
        """
        try:
            monitoring_results = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'violations': [],
                'overall_status': 'unknown'
            }
            
            for attr_name, attr_values in sensitive_features.items():
                # Calculate demographic parity
                privileged_mask = attr_values == 1
                unprivileged_mask = attr_values == 0
                
                if np.any(privileged_mask) and np.any(unprivileged_mask):
                    privileged_rate = np.mean(predictions[privileged_mask])
                    unprivileged_rate = np.mean(predictions[unprivileged_mask])
                    dp_difference = abs(privileged_rate - unprivileged_rate)
                    
                    # Calculate disparate impact
                    di_ratio = unprivileged_rate / privileged_rate if privileged_rate > 0 else 0.0
                    
                    attr_metrics = {
                        'demographic_parity_difference': float(dp_difference),
                        'disparate_impact_ratio': float(di_ratio),
                        'privileged_selection_rate': float(privileged_rate),
                        'unprivileged_selection_rate': float(unprivileged_rate)
                    }
                    
                    # Check for violations
                    dp_threshold = self.thresholds.get('demographic_parity', {}).get('threshold', 0.1)
                    di_threshold = self.thresholds.get('disparate_impact', {}).get('threshold', 0.8)
                    
                    if dp_difference > dp_threshold:
                        monitoring_results['violations'].append({
                            'type': 'demographic_parity',
                            'attribute': attr_name,
                            'value': dp_difference,
                            'threshold': dp_threshold,
                            'severity': 'high' if dp_difference > dp_threshold * 1.5 else 'medium'
                        })
                    
                    if di_ratio < di_threshold:
                        monitoring_results['violations'].append({
                            'type': 'disparate_impact',
                            'attribute': attr_name,
                            'value': di_ratio,
                            'threshold': di_threshold,
                            'severity': 'high' if di_ratio < di_threshold * 0.8 else 'medium'
                        })
                    
                    # Add equalized odds if true labels are available
                    if true_labels is not None:
                        try:
                            from sklearn.metrics import confusion_matrix
                            
                            # Calculate TPR and FPR for each group
                            def calculate_rates(y_true_group, y_pred_group):
                                if len(np.unique(y_true_group)) < 2:
                                    return 0.0, 0.0
                                cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
                                if cm.shape == (2, 2):
                                    tn, fp, fn, tp = cm.ravel()
                                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                                    return tpr, fpr
                                return 0.0, 0.0
                            
                            priv_tpr, priv_fpr = calculate_rates(true_labels[privileged_mask], predictions[privileged_mask])
                            unpriv_tpr, unpriv_fpr = calculate_rates(true_labels[unprivileged_mask], predictions[unprivileged_mask])
                            
                            tpr_diff = abs(priv_tpr - unpriv_tpr)
                            fpr_diff = abs(priv_fpr - unpriv_fpr)
                            
                            attr_metrics['tpr_difference'] = float(tpr_diff)
                            attr_metrics['fpr_difference'] = float(fpr_diff)
                            
                        except Exception as e:
                            self.logger.warning(f"Could not calculate equalized odds: {str(e)}")
                    
                    monitoring_results['metrics'][attr_name] = attr_metrics
            
            # Determine overall status
            if not monitoring_results['violations']:
                monitoring_results['overall_status'] = 'healthy'
            elif any(v['severity'] == 'high' for v in monitoring_results['violations']):
                monitoring_results['overall_status'] = 'critical'
            else:
                monitoring_results['overall_status'] = 'warning'
            
            # Store in history
            self.fairness_history.append({
                'timestamp': datetime.now(),
                'metrics': monitoring_results['metrics'],
                'violations_count': len(monitoring_results['violations']),
                'status': monitoring_results['overall_status']
            })
            
            self.logger.info(f"Fairness monitoring completed. Status: {monitoring_results['overall_status']}")
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"Error in fairness monitoring: {str(e)}")
            return {
                'error': str(e),
                'overall_status': 'error'
            }


class MonitorStage:
    """
    Monitor stage for the DevEthOps pipeline.
    Provides continuous monitoring of model performance, fairness, and system health.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Monitor Stage.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/pipeline_config.yaml"
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.drift_detector = None
        self.fairness_monitor = FairnessMonitor(self.config)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Alert history
        self.alert_history = deque(maxlen=1000)
        
        # System metrics
        self.system_metrics = deque(maxlen=100)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            return {}
    
    def initialize_drift_detector(self, reference_data: np.ndarray):
        """Initialize drift detector with reference data"""
        try:
            self.drift_detector = DriftDetector(reference_data, self.config)
            self.logger.info("Drift detector initialized")
        except Exception as e:
            self.logger.error(f"Error initializing drift detector: {str(e)}")
    
    def detect_model_drift(self, current_data: np.ndarray, 
                          feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect model drift using the current data.
        
        Args:
            current_data: Current data to analyze
            feature_names: Names of features
            
        Returns:
            Dict containing drift analysis results
        """
        if self.drift_detector is None:
            return {
                'error': 'Drift detector not initialized',
                'overall_drift_detected': False
            }
        
        return self.drift_detector.detect_statistical_drift(current_data, feature_names)
    
    def continuous_fairness_monitoring(self, predictions_stream: List[Dict[str, Any]], 
                                     monitoring_interval: int = 60):
        """
        Start continuous fairness monitoring.
        
        Args:
            predictions_stream: Stream of prediction data with sensitive features
            monitoring_interval: Monitoring interval in seconds
        """
        self.logger.info("Starting continuous fairness monitoring")
        
        def monitoring_loop():
            """Main monitoring loop"""
            while self.monitoring_active:
                try:
                    # Process recent predictions
                    if predictions_stream:
                        recent_predictions = predictions_stream[-100:]  # Last 100 predictions
                        
                        if recent_predictions:
                            # Extract data
                            predictions = np.array([p['prediction'] for p in recent_predictions])
                            
                            # Extract sensitive features
                            sensitive_features = {}
                            protected_attrs = self.config.get('protected_attributes', [])
                            
                            for attr in protected_attrs:
                                if attr in recent_predictions[0].get('features', {}):
                                    attr_values = np.array([p['features'][attr] for p in recent_predictions])
                                    sensitive_features[attr] = attr_values
                            
                            # Monitor fairness
                            if sensitive_features:
                                monitoring_results = self.fairness_monitor.monitor_fairness_metrics(
                                    predictions, sensitive_features
                                )
                                
                                # Generate alerts if needed
                                self._process_monitoring_alerts(monitoring_results)
                    
                    # Collect system metrics
                    self._collect_system_metrics()
                    
                    time.sleep(monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(monitoring_interval)
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _process_monitoring_alerts(self, monitoring_results: Dict[str, Any]):
        """Process monitoring results and generate alerts"""
        try:
            violations = monitoring_results.get('violations', [])
            
            for violation in violations:
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    alert_type='fairness_violation',
                    severity=violation['severity'],
                    metric_name=f"{violation['type']}_{violation['attribute']}",
                    current_value=violation['value'],
                    threshold=violation['threshold'],
                    description=f"Fairness violation detected: {violation['type']} for {violation['attribute']}",
                    recommendation=self._get_fairness_recommendation(violation)
                )
                
                self.alert_history.append(alert)
                self._send_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Error processing monitoring alerts: {str(e)}")
    
    def _get_fairness_recommendation(self, violation: Dict[str, Any]) -> str:
        """Get recommendation for fairness violation"""
        violation_type = violation['type']
        
        recommendations = {
            'demographic_parity': "Consider applying bias mitigation techniques or reviewing data collection process",
            'disparate_impact': "Review model training process and consider demographic parity constraints",
            'equalized_odds': "Apply post-processing techniques to equalize TPR and FPR across groups"
        }
        
        return recommendations.get(violation_type, "Review model fairness and consider retraining")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metric = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
            self.system_metrics.append(system_metric)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
    
    def _send_alert(self, alert: DriftAlert):
        """Send alert notification"""
        try:
            # Log alert
            self.logger.warning(f"ALERT: {alert.alert_type} - {alert.description}")
            
            # Send email if configured
            if self.config.get('monitoring', {}).get('alert_email'):
                self._send_email_alert(alert)
            
            # Send Slack notification if configured
            slack_webhook = self.config.get('notifications', {}).get('slack_webhook')
            if slack_webhook:
                self._send_slack_alert(alert, slack_webhook)
                
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
    
    def _send_email_alert(self, alert: DriftAlert):
        """Send email alert"""
        try:
            smtp_server = self.config.get('notifications', {}).get('email_smtp_server', 'smtp.gmail.com')
            smtp_port = self.config.get('notifications', {}).get('email_smtp_port', 587)
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = "devethops@system.local"
            msg['To'] = self.config.get('monitoring', {}).get('alert_email')
            msg['Subject'] = f"DevEthOps Alert: {alert.alert_type}"
            
            body = f"""
            DevEthOps Alert Notification
            
            Alert Type: {alert.alert_type}
            Severity: {alert.severity}
            Metric: {alert.metric_name}
            Current Value: {alert.current_value:.4f}
            Threshold: {alert.threshold:.4f}
            
            Description: {alert.description}
            
            Recommendation: {alert.recommendation}
            
            Timestamp: {alert.timestamp}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Note: Email sending would require proper SMTP configuration
            # This is a placeholder implementation
            self.logger.info(f"Email alert prepared for: {alert.alert_type}")
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")
    
    def _send_slack_alert(self, alert: DriftAlert, webhook_url: str):
        """Send Slack alert"""
        try:
            import requests
            
            severity_emoji = {
                'low': '🟡',
                'medium': '🟠', 
                'high': '🔴',
                'critical': '🚨'
            }
            
            emoji = severity_emoji.get(alert.severity, '⚠️')
            
            payload = {
                "text": f"{emoji} DevEthOps Alert: {alert.alert_type}",
                "attachments": [
                    {
                        "color": "danger" if alert.severity in ['high', 'critical'] else "warning",
                        "fields": [
                            {"title": "Metric", "value": alert.metric_name, "short": True},
                            {"title": "Severity", "value": alert.severity.upper(), "short": True},
                            {"title": "Current Value", "value": f"{alert.current_value:.4f}", "short": True},
                            {"title": "Threshold", "value": f"{alert.threshold:.4f}", "short": True},
                            {"title": "Description", "value": alert.description, "short": False},
                            {"title": "Recommendation", "value": alert.recommendation, "short": False}
                        ],
                        "footer": "DevEthOps Monitoring",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Note: This would require actual webhook URL configuration
            self.logger.info(f"Slack alert prepared for: {alert.alert_type}")
            
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {str(e)}")
    
    def generate_monitoring_report(self, time_period_hours: int = 24) -> str:
        """
        Generate monitoring report for specified time period.
        
        Args:
            time_period_hours: Time period for report generation
            
        Returns:
            str: Formatted monitoring report
        """
        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
        
        # Filter recent alerts
        recent_alerts = [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
        
        # Filter recent fairness metrics
        recent_fairness = [entry for entry in self.fairness_monitor.fairness_history 
                         if entry['timestamp'] > cutoff_time]
        
        # Filter recent system metrics
        recent_system = [metric for metric in self.system_metrics 
                        if metric['timestamp'] > cutoff_time]
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"DEVETHOPS MONITORING REPORT ({time_period_hours}h)")
        report_lines.append("=" * 60)
        
        # Executive Summary
        report_lines.append(f"\nReport Period: {cutoff_time.strftime('%Y-%m-%d %H:%M')} - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_lines.append(f"Total Alerts: {len(recent_alerts)}")
        
        alert_severity_counts = {}
        for alert in recent_alerts:
            alert_severity_counts[alert.severity] = alert_severity_counts.get(alert.severity, 0) + 1
        
        if alert_severity_counts:
            report_lines.append("Alert Breakdown:")
            for severity, count in sorted(alert_severity_counts.items()):
                report_lines.append(f"  {severity.capitalize()}: {count}")
        
        # Fairness Status
        report_lines.append("\nFairness Monitoring:")
        if recent_fairness:
            healthy_count = sum(1 for entry in recent_fairness if entry['status'] == 'healthy')
            report_lines.append(f"  Healthy periods: {healthy_count}/{len(recent_fairness)} ({healthy_count/len(recent_fairness)*100:.1f}%)")
        else:
            report_lines.append("  No fairness data available")
        
        # System Health
        report_lines.append("\nSystem Health:")
        if recent_system:
            avg_cpu = np.mean([m['cpu_percent'] for m in recent_system])
            avg_memory = np.mean([m['memory_percent'] for m in recent_system])
            avg_disk = np.mean([m['disk_percent'] for m in recent_system])
            
            report_lines.append(f"  Average CPU: {avg_cpu:.1f}%")
            report_lines.append(f"  Average Memory: {avg_memory:.1f}%")
            report_lines.append(f"  Average Disk: {avg_disk:.1f}%")
        else:
            report_lines.append("  No system metrics available")
        
        # Recent Alerts Details
        if recent_alerts:
            report_lines.append("\nRecent Alerts:")
            report_lines.append("-" * 30)
            for alert in recent_alerts[-10:]:  # Show last 10 alerts
                report_lines.append(f"[{alert.timestamp.strftime('%H:%M')}] {alert.severity.upper()}: {alert.metric_name}")
                report_lines.append(f"    {alert.description}")
        
        # Recommendations
        report_lines.append("\nRecommendations:")
        report_lines.append("-" * 30)
        
        if not recent_alerts:
            report_lines.append("✅ System operating normally")
        else:
            high_severity_alerts = [a for a in recent_alerts if a.severity in ['high', 'critical']]
            if high_severity_alerts:
                report_lines.append("🚨 URGENT: Address high-severity alerts immediately")
                report_lines.append("   - Review model performance and fairness")
                report_lines.append("   - Consider retraining or bias mitigation")
            else:
                report_lines.append("⚠️  Monitor system closely")
                report_lines.append("   - Investigate warning-level alerts")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Monitoring stopped")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'drift_detector_initialized': self.drift_detector is not None,
            'total_alerts': len(self.alert_history),
            'recent_alerts_1h': len([a for a in self.alert_history 
                                   if a.timestamp > datetime.now() - timedelta(hours=1)]),
            'system_metrics_count': len(self.system_metrics),
            'fairness_history_count': len(self.fairness_monitor.fairness_history)
        }


if __name__ == "__main__":
    # Example usage
    monitor_stage = MonitorStage()
    # reference_data = np.random.random((1000, 10))
    # monitor_stage.initialize_drift_detector(reference_data)
    # 
    # current_data = np.random.random((100, 10))
    # drift_results = monitor_stage.detect_model_drift(current_data)
