"""
Main pipeline orchestrator for DevEthOps ethical ML pipeline.

Orchestrates Build → Test → Deploy → Monitor stages with fairness gates.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import argparse
import sys

# Import DevEthOps modules
from .config import load_config
from .data_loader import load_dataset, create_train_test_split
from .preprocess import preprocess_pipeline
from .fairness_checks import evaluate_model_fairness
from .mitigation import apply_fairness_mitigation
from .explainability import run_explainability_analysis
from .models_tabular import train_tabular_model, evaluate_tabular_models
from .models_text import train_text_model
from .monitoring import ModelMonitor
from .deploy import ModelDeployer

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


class EthicalMLPipeline:
    """
    Main orchestrator for ethical ML pipeline with automated fairness checks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.artifacts = {}
        self.metrics = {}
        self.stage_results = {}
        self.fairness_gates_passed = True
        
        # Initialize components
        self.monitor = None
        self.deployer = None
        
        logger.info(f"Initialized pipeline: {self.pipeline_id}")
    
    def run_full_pipeline(self, dataset_name: str, model_type: str,
                         protected_attributes: Optional[List[str]] = None,
                         apply_mitigation: bool = True,
                         enable_explainability: bool = True,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete ethical ML pipeline.
        
        Args:
            dataset_name: Name of dataset to use
            model_type: Type of model to train
            protected_attributes: Protected attributes for fairness
            apply_mitigation: Whether to apply bias mitigation
            enable_explainability: Whether to generate explanations
            output_dir: Directory to save outputs
            
        Returns:
            Pipeline results
        """
        start_time = time.time()
        logger.info(f"Starting full pipeline: {self.pipeline_id}")
        
        if output_dir is None:
            output_dir = f"pipeline_outputs/{self.pipeline_id}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Build Stage
            logger.info("=" * 50)
            logger.info("STAGE 1: BUILD")
            logger.info("=" * 50)
            build_results = self.build_stage(
                dataset_name, protected_attributes, apply_mitigation
            )
            self.stage_results['build'] = build_results
            
            # Test Stage
            logger.info("=" * 50)
            logger.info("STAGE 2: TEST")
            logger.info("=" * 50)
            test_results = self.test_stage(
                model_type, build_results, protected_attributes, enable_explainability
            )
            self.stage_results['test'] = test_results
            
            # Check fairness gates
            if not test_results.get('fairness_gates_passed', False):
                self.fairness_gates_passed = False
                logger.error("[FAIL] Fairness gates FAILED. Consider applying mitigation or adjusting thresholds.")
                
                if self.config.get('strict_fairness_gates', True):
                    raise PipelineError("Pipeline stopped due to fairness gate failures")
            
            # Deploy Stage
            logger.info("=" * 50)
            logger.info("STAGE 3: DEPLOY")
            logger.info("=" * 50)
            deploy_results = self.deploy_stage(test_results['best_model'], output_dir)
            self.stage_results['deploy'] = deploy_results
            
            # Monitor Stage
            logger.info("=" * 50)
            logger.info("STAGE 4: MONITOR")
            logger.info("=" * 50)
            monitor_results = self.monitor_stage(test_results['best_model'], build_results)
            self.stage_results['monitor'] = monitor_results
            
            # Compile final results
            end_time = time.time()
            execution_time = end_time - start_time
            
            final_results = {
                'pipeline_id': self.pipeline_id,
                'execution_time_seconds': execution_time,
                'fairness_gates_passed': self.fairness_gates_passed,
                'stage_results': self.stage_results,
                'artifacts': self.artifacts,
                'metrics': self.metrics,
                'output_directory': str(output_dir),
                'config': self.config
            }
            
            # Save results
            self._save_pipeline_results(final_results, output_dir)
            
            logger.info(f"[PASS] Pipeline completed successfully in {execution_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"[FAIL] Pipeline failed: {e}")
            error_results = {
                'pipeline_id': self.pipeline_id,
                'status': 'failed',
                'error': str(e),
                'stage_results': self.stage_results,
                'execution_time_seconds': time.time() - start_time
            }
            self._save_pipeline_results(error_results, output_dir)
            raise
    
    def build_stage(self, dataset_name: str, protected_attributes: Optional[List[str]] = None,
                   apply_mitigation: bool = True) -> Dict[str, Any]:
        """
        Build stage: Data loading, preprocessing, and bias mitigation.
        
        Args:
            dataset_name: Name of dataset
            protected_attributes: Protected attributes
            apply_mitigation: Whether to apply mitigation
            
        Returns:
            Build stage results
        """
        logger.info("Starting Build stage...")
        
        results = {
            'stage': 'build',
            'dataset_name': dataset_name,
            'protected_attributes': protected_attributes,
            'mitigation_applied': apply_mitigation
        }
        
        try:
            # 1. Load dataset
            logger.info(f"Loading dataset: {dataset_name}")
            X, y = load_dataset(dataset_name, self.config)
            results['dataset_shape'] = X.shape
            results['class_distribution'] = y.value_counts().to_dict()
            
            # 2. Data validation
            logger.info("Performing data validation...")
            validation_results = self._validate_data(X, y, protected_attributes)
            results['data_validation'] = validation_results
            
            if not validation_results['passed']:
                raise PipelineError(f"Data validation failed: {validation_results['issues']}")
            
            # 3. Preprocessing
            logger.info("Preprocessing data...")
            preprocessing_results = preprocess_pipeline(X, y, self.config)
            results['preprocessing'] = preprocessing_results
            
            # 4. Apply bias mitigation if requested
            if apply_mitigation and protected_attributes:
                logger.info("Applying bias mitigation...")
                mitigation_results = apply_fairness_mitigation(
                    X, y, self.config, protected_attributes
                )
                results['mitigation'] = mitigation_results
                
                # Use mitigated data for training
                preprocessing_results['sample_weights'] = mitigation_results['sample_weights']
            
            # Store artifacts
            self.artifacts['data'] = {
                'X_train': preprocessing_results['X_train'],
                'X_val': preprocessing_results['X_val'],
                'X_test': preprocessing_results['X_test'],
                'y_train': preprocessing_results['y_train'],
                'y_val': preprocessing_results['y_val'],
                'y_test': preprocessing_results['y_test'],
                'preprocessor': preprocessing_results['preprocessor'],
                'sample_weights': preprocessing_results.get('sample_weights')
            }
            
            logger.info("[PASS] Build stage completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"[FAIL] Build stage failed: {e}")
            results['error'] = str(e)
            raise
    
    def test_stage(self, model_type: str, build_results: Dict[str, Any],
                  protected_attributes: Optional[List[str]] = None,
                  enable_explainability: bool = True) -> Dict[str, Any]:
        """
        Test stage: Model training, fairness evaluation, and explainability.
        
        Args:
            model_type: Type of model to train
            build_results: Results from build stage
            protected_attributes: Protected attributes
            enable_explainability: Whether to generate explanations
            
        Returns:
            Test stage results
        """
        logger.info("Starting Test stage...")
        
        results = {
            'stage': 'test',
            'model_type': model_type,
            'protected_attributes': protected_attributes
        }
        
        try:
            # Get data from build stage
            data_artifacts = self.artifacts['data']
            X_train = data_artifacts['X_train']
            X_val = data_artifacts['X_val']
            X_test = data_artifacts['X_test']
            y_train = data_artifacts['y_train']
            y_val = data_artifacts['y_val']
            y_test = data_artifacts['y_test']
            sample_weights = data_artifacts.get('sample_weights')
            
            # 1. Model training
            logger.info(f"Training {model_type} model...")
            if model_type in ['logistic_regression', 'random_forest', 'xgboost']:
                model_results = train_tabular_model(
                    model_type, X_train, y_train, self.config, sample_weights
                )
            elif model_type == 'roberta':
                # For text models - placeholder for now
                logger.warning("Text models not fully implemented in pipeline yet")
                raise PipelineError("Text models not supported in current pipeline")
            else:
                raise PipelineError(f"Unknown model type: {model_type}")
            
            results['model_training'] = model_results
            model = model_results['trainer'].model
            
            # 2. Model evaluation
            logger.info("Evaluating model performance...")
            evaluation_metrics = model_results['trainer'].evaluate_model(X_test, y_test)
            results['performance_metrics'] = evaluation_metrics
            
            # 3. Fairness evaluation
            if protected_attributes:
                logger.info("Evaluating model fairness...")
                
                # Prepare feature matrix with protected attributes
                if hasattr(X_test, 'columns'):
                    # Find protected attributes in original feature space
                    original_features = data_artifacts['preprocessor'].original_features
                    protected_in_features = [attr for attr in protected_attributes 
                                          if attr in original_features]
                    
                    if protected_in_features:
                        # Get original test data for fairness evaluation
                        original_splits = build_results['preprocessing']['original_splits']
                        X_test_original = original_splits['X_test']
                        
                        fairness_results = evaluate_model_fairness(
                            model, X_test_original, y_test, protected_in_features, self.config
                        )
                        results['fairness_evaluation'] = fairness_results
                        results['fairness_gates_passed'] = fairness_results['overall_pass']
                    else:
                        logger.warning("Protected attributes not found in features. Skipping fairness evaluation.")
                        results['fairness_gates_passed'] = True
                else:
                    logger.warning("Cannot perform fairness evaluation without feature names")
                    results['fairness_gates_passed'] = True
            else:
                logger.info("No protected attributes specified. Skipping fairness evaluation.")
                results['fairness_gates_passed'] = True
            
            # 4. Explainability analysis
            if enable_explainability:
                logger.info("Generating model explanations...")
                try:
                    explainability_results = run_explainability_analysis(
                        model, X_train, X_test, self.config
                    )
                    results['explainability'] = explainability_results
                except Exception as e:
                    logger.warning(f"Explainability analysis failed: {e}")
                    results['explainability'] = {'error': str(e)}
            
            # Store best model
            results['best_model'] = {
                'model': model,
                'trainer': model_results['trainer'],
                'type': model_type,
                'performance': evaluation_metrics
            }
            
            # Update metrics
            self.metrics.update({
                'model_performance': evaluation_metrics,
                'fairness_passed': results['fairness_gates_passed']
            })
            
            logger.info("[PASS] Test stage completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"[FAIL] Test stage failed: {e}")
            results['error'] = str(e)
            raise
    
    def deploy_stage(self, best_model: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """
        Deploy stage: Model packaging and deployment preparation.
        
        Args:
            best_model: Best model from test stage
            output_dir: Output directory
            
        Returns:
            Deploy stage results
        """
        logger.info("Starting Deploy stage...")
        
        results = {
            'stage': 'deploy'
        }
        
        try:
            # Initialize deployer
            self.deployer = ModelDeployer(self.config)
            
            # 1. Package model
            logger.info("Packaging model...")
            model_package_path = output_dir / "model_package"
            model_package_path.mkdir(exist_ok=True)
            
            # Save model
            best_model['trainer'].save_model(str(model_package_path / "model.joblib"))
            
            # 2. Create Docker image
            logger.info("Creating Docker image...")
            docker_results = self.deployer.create_docker_image(
                str(model_package_path), str(output_dir)
            )
            results['docker'] = docker_results
            
            # 3. Generate Kubernetes manifests
            logger.info("Generating Kubernetes manifests...")
            k8s_results = self.deployer.generate_k8s_manifests(
                self.pipeline_id, str(output_dir)
            )
            results['kubernetes'] = k8s_results
            
            # 4. Health check configuration
            logger.info("Setting up health checks...")
            health_check_config = self.deployer.create_health_checks()
            results['health_checks'] = health_check_config
            
            logger.info("[PASS] Deploy stage completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"[FAIL] Deploy stage failed: {e}")
            results['error'] = str(e)
            return results
    
    def monitor_stage(self, best_model: Dict[str, Any], build_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor stage: Set up monitoring and drift detection.
        
        Args:
            best_model: Best model from test stage
            build_results: Results from build stage
            
        Returns:
            Monitor stage results
        """
        logger.info("Starting Monitor stage...")
        
        results = {
            'stage': 'monitor'
        }
        
        try:
            # Initialize monitor
            self.monitor = ModelMonitor(self.config)
            
            # 1. Set up drift detection
            logger.info("Setting up drift detection...")
            reference_data = self.artifacts['data']['X_train']
            drift_setup = self.monitor.setup_drift_detection(reference_data)
            results['drift_detection'] = drift_setup
            
            # 2. Set up fairness monitoring
            logger.info("Setting up fairness monitoring...")
            fairness_setup = self.monitor.setup_fairness_monitoring(
                best_model['model'],
                self.config.get('protected_attributes', [])
            )
            results['fairness_monitoring'] = fairness_setup
            
            # 3. Create monitoring dashboard placeholders
            logger.info("Creating monitoring configuration...")
            monitoring_config = self.monitor.create_monitoring_config(self.pipeline_id)
            results['monitoring_config'] = monitoring_config
            
            # 4. Set up alerts
            logger.info("Setting up alerts...")
            alert_config = self.monitor.setup_alerts()
            results['alerts'] = alert_config
            
            logger.info("[PASS] Monitor stage completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"[FAIL] Monitor stage failed: {e}")
            results['error'] = str(e)
            return results
    
    def _validate_data(self, X: pd.DataFrame, y: pd.Series,
                      protected_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate input data quality and structure.
        
        Args:
            X: Feature matrix
            y: Target vector
            protected_attributes: Protected attributes
            
        Returns:
            Validation results
        """
        validation_results = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Basic data validation
        if len(X) == 0:
            validation_results['passed'] = False
            validation_results['issues'].append("Empty dataset")
        
        if len(X) != len(y):
            validation_results['passed'] = False
            validation_results['issues'].append("Feature and target length mismatch")
        
        # Check for excessive missing values
        missing_percentage = X.isnull().sum() / len(X)
        high_missing_cols = missing_percentage[missing_percentage > 0.5].index.tolist()
        if high_missing_cols:
            validation_results['warnings'].append(f"High missing values in: {high_missing_cols}")
        
        # Check protected attributes
        if protected_attributes:
            missing_attrs = [attr for attr in protected_attributes if attr not in X.columns]
            if missing_attrs:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Protected attributes not found: {missing_attrs}")
        
        # Class balance check
        class_counts = y.value_counts()
        min_class_ratio = class_counts.min() / class_counts.max()
        if min_class_ratio < 0.1:
            validation_results['warnings'].append(f"Severe class imbalance (ratio: {min_class_ratio:.3f})")
        
        validation_results['stats'] = {
            'num_samples': len(X),
            'num_features': len(X.columns),
            'missing_percentage': missing_percentage.mean(),
            'class_distribution': class_counts.to_dict(),
            'min_class_ratio': min_class_ratio
        }
        
        return validation_results
    
    def _save_pipeline_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """
        Save pipeline results to disk.
        
        Args:
            results: Pipeline results
            output_dir: Output directory
        """
        try:
            # Create summary report
            report_path = output_dir / "pipeline_report.json"
            
            # Make results serializable
            serializable_results = self._make_serializable(results)
            
            with open(report_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Create human-readable summary
            summary_path = output_dir / "pipeline_summary.txt"
            summary = self._generate_pipeline_summary(results)
            
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            logger.info(f"Pipeline results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
            return "Data object (not serialized)"
        elif hasattr(obj, '__dict__'):
            return f"Object of type {type(obj).__name__}"
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def _generate_pipeline_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable pipeline summary."""
        lines = [
            "=" * 60,
            "DEVETHOPS ETHICAL ML PIPELINE SUMMARY",
            "=" * 60,
            f"Pipeline ID: {results.get('pipeline_id', 'Unknown')}",
            f"Execution Time: {results.get('execution_time_seconds', 0):.2f} seconds",
            f"Fairness Gates Passed: {'[PASS] YES' if results.get('fairness_gates_passed', False) else '[FAIL] NO'}",
            "",
            "STAGE RESULTS:",
            ""
        ]
        
        stage_results = results.get('stage_results', {})
        
        for stage_name, stage_data in stage_results.items():
            status = "[PASS] SUCCESS" if 'error' not in stage_data else "[FAIL] FAILED"
            lines.append(f"  {stage_name.upper()}: {status}")
            
            if stage_name == 'test' and 'performance_metrics' in stage_data:
                metrics = stage_data['performance_metrics']
                lines.append(f"    F1-Score: {metrics.get('f1_score', 0):.4f}")
                lines.append(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
            
            if 'error' in stage_data:
                lines.append(f"    Error: {stage_data['error']}")
            
            lines.append("")
        
        lines.extend([
            "=" * 60
        ])
        
        return "\n".join(lines)


def run_cli_pipeline() -> None:
    """
    Command-line interface for running the pipeline.
    """
    parser = argparse.ArgumentParser(description="DevEthOps Ethical ML Pipeline")
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ibm_hr', 'adult', 'mimic', 'synthetic'],
                       help='Dataset to use')
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['logistic_regression', 'random_forest', 'xgboost', 'roberta'],
                       help='Model type to train')
    
    parser.add_argument('--protected', nargs='+', type=str,
                       help='Protected attributes for fairness evaluation')
    
    parser.add_argument('--apply-mitigation', action='store_true',
                       help='Apply bias mitigation techniques')
    
    parser.add_argument('--no-explainability', action='store_true',
                       help='Skip explainability analysis')
    
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = load_config()  # Use default
        
        # Initialize pipeline
        pipeline = EthicalMLPipeline(config)
        
        # Run pipeline
        results = pipeline.run_full_pipeline(
            dataset_name=args.dataset,
            model_type=args.model,
            protected_attributes=args.protected,
            apply_mitigation=args.apply_mitigation,
            enable_explainability=not args.no_explainability,
            output_dir=args.output_dir
        )
        
        print("\n[PASS] Pipeline completed successfully!")
        print(f"Results saved to: {results['output_directory']}")
        print(f"Fairness gates passed: {'[PASS] YES' if results['fairness_gates_passed'] else '[FAIL] NO'}")
        
    except Exception as e:
        print(f"\n[FAIL] Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_cli_pipeline()
