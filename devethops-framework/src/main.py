"""
Main Pipeline Orchestrator - DevEthOps Pipeline execution and coordination
"""

import os
import sys
import json
import yaml
import logging
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Pipeline stages
from .pipeline.build_stage import BuildStage
from .pipeline.test_stage import TestStage
from .pipeline.deploy_stage import DeployStage
from .pipeline.monitor_stage import MonitorStage

# Model wrapper
from .models.llm_wrapper import EthicalLLMWrapper

# Monitoring
from .monitoring.pipeline_monitor import PipelineMonitor


class FairnessViolationError(Exception):
    """Raised when fairness criteria are not met"""
    pass


class PerformanceViolationError(Exception):
    """Raised when performance criteria are not met"""
    pass


class PipelineError(Exception):
    """General pipeline error"""
    pass


class DevEthOpsPipeline:
    """
    Main DevEthOps pipeline orchestrator.
    Coordinates all pipeline stages with ethical AI integration.
    """
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize the DevEthOps pipeline.
        
        Args:
            config_path: Path to the pipeline configuration file
        """
        self.logger = self._setup_logging()
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize pipeline components
        self.build_stage = BuildStage(config_path)
        self.test_stage = None  # Initialized after model is available
        self.deploy_stage = DeployStage(config_path)
        self.monitor_stage = MonitorStage(config_path)
        self.pipeline_monitor = PipelineMonitor()
        
        # Pipeline state
        self.current_stage = None
        self.pipeline_results = {
            'pipeline_id': self._generate_pipeline_id(),
            'start_time': None,
            'end_time': None,
            'status': 'initialized',
            'stages': {},
            'artifacts': {},
            'errors': []
        }
        
        # Model instance
        self.model = None
        
        self.logger.info(f"DevEthOps Pipeline initialized - ID: {self.pipeline_results['pipeline_id']}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('devethops_pipeline.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline ID"""
        from datetime import datetime
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"devethops_{timestamp}_{short_uuid}"
    
    def run_pipeline(self, dataset_path: str, model_path: Optional[str] = None,
                    output_dir: str = "artifacts") -> Dict[str, Any]:
        """
        Run the complete DevEthOps pipeline.
        
        Args:
            dataset_path: Path to the training/evaluation dataset
            model_path: Path to pre-trained model (optional)
            output_dir: Directory for pipeline artifacts
            
        Returns:
            Dict containing pipeline results
        """
        try:
            self.pipeline_results['start_time'] = datetime.now().isoformat()
            self.pipeline_results['status'] = 'running'
            self.logger.info(f"Starting DevEthOps pipeline: {self.pipeline_results['pipeline_id']}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Stage 1: Build Stage
            self._run_build_stage(dataset_path, output_dir)
            
            # Stage 2: Model Training/Loading
            self._run_model_stage(model_path, output_dir)
            
            # Stage 3: Test Stage
            self._run_test_stage(output_dir)
            
            # Stage 4: Deploy Stage (if tests pass)
            if self._should_deploy():
                self._run_deploy_stage(output_dir)
            else:
                self.logger.warning("Deployment skipped due to test failures")
                self.pipeline_results['stages']['deploy'] = {
                    'status': 'skipped',
                    'reason': 'Test failures detected'
                }
            
            # Stage 5: Monitor Stage Setup
            self._setup_monitoring_stage(output_dir)
            
            # Finalize pipeline
            self._finalize_pipeline(output_dir)
            
            return self.pipeline_results
            
        except Exception as e:
            self._handle_pipeline_failure(e)
            raise
    
    def _run_build_stage(self, dataset_path: str, output_dir: str):
        """Run the build stage"""
        try:
            self.current_stage = 'build'
            self.logger.info("=== Running Build Stage ===")
            self.pipeline_monitor.start_stage('build')
            
            # Load dataset
            protected_attributes = self.config.get('protected_attributes', [])
            dataset = self.build_stage.load_dataset(dataset_path, protected_attributes)
            
            # Detect bias
            bias_metrics = self.build_stage.detect_bias(dataset)
            self.logger.info(f"Bias detection completed: {bias_metrics}")
            
            # Check bias thresholds
            if not self.build_stage.check_bias_thresholds(bias_metrics):
                self.logger.warning("Bias thresholds exceeded, applying mitigation")
                dataset = self.build_stage.mitigate_bias(dataset)
                
                # Re-check after mitigation
                post_mitigation_metrics = self.build_stage.detect_bias(dataset)
                self.logger.info(f"Post-mitigation bias metrics: {post_mitigation_metrics}")
                
                if not self.build_stage.check_bias_thresholds(post_mitigation_metrics):
                    self.logger.error("Bias mitigation insufficient")
                    raise FairnessViolationError("Dataset bias exceeds acceptable thresholds even after mitigation")
            
            # Prepare training data
            features, labels = self.build_stage.prepare_training_data(dataset)
            
            # Save build artifacts
            build_artifacts = self.build_stage.save_build_artifacts(
                dataset, bias_metrics, os.path.join(output_dir, "build")
            )
            
            # Record stage results
            self.pipeline_results['stages']['build'] = {
                'status': 'completed',
                'dataset_samples': len(features),
                'dataset_features': len(features.columns),
                'bias_metrics': bias_metrics,
                'artifacts': build_artifacts,
                'duration': self.pipeline_monitor.end_stage('build')
            }
            
            # Store processed data for next stages
            self.pipeline_results['processed_data'] = {
                'features': features,
                'labels': labels,
                'protected_attributes': protected_attributes
            }
            
            self.logger.info("Build stage completed successfully")
            
        except Exception as e:
            self.logger.error(f"Build stage failed: {str(e)}")
            self.pipeline_results['stages']['build'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _run_model_stage(self, model_path: Optional[str], output_dir: str):
        """Run model training or loading"""
        try:
            self.current_stage = 'model'
            self.logger.info("=== Running Model Stage ===")
            self.pipeline_monitor.start_stage('model')
            
            processed_data = self.pipeline_results['processed_data']
            features = processed_data['features']
            labels = processed_data['labels']
            
            # Initialize model wrapper
            model_config = self.config.get('model', {})
            self.model = EthicalLLMWrapper(
                model_name=model_config.get('type', 'roberta-base'),
                config_path="config/model_config.yaml"
            )
            
            if model_path and os.path.exists(model_path):
                # Load existing model
                self.logger.info(f"Loading model from {model_path}")
                if self.model.load_model(model_path):
                    model_info = self.model.get_model_info()
                    self.logger.info("Model loaded successfully")
                else:
                    raise PipelineError(f"Failed to load model from {model_path}")
            else:
                # Train new model
                self.logger.info("Training new model")
                
                # Prepare text data (this is a simplified example)
                # In practice, you'd need proper text preprocessing
                if 'text' in features.columns:
                    train_texts = features['text'].tolist()
                else:
                    # Create dummy text data for demonstration
                    train_texts = [f"Sample text {i}" for i in range(len(features))]
                
                train_labels = labels.tolist()
                
                # Extract sensitive attributes for fair training
                sensitive_attributes = {}
                for attr in processed_data['protected_attributes']:
                    if attr in features.columns:
                        sensitive_attributes[attr] = features[attr].tolist()
                
                # Split data for training/validation
                split_idx = int(0.8 * len(train_texts))
                
                training_results = self.model.train(
                    train_texts=train_texts[:split_idx],
                    train_labels=train_labels[:split_idx],
                    val_texts=train_texts[split_idx:] if split_idx < len(train_texts) else None,
                    val_labels=train_labels[split_idx:] if split_idx < len(train_labels) else None,
                    sensitive_attributes=sensitive_attributes
                )
                
                self.logger.info(f"Model training completed: {training_results}")
                model_info = self.model.get_model_info()
            
            # Save model
            model_save_path = os.path.join(output_dir, "model")
            saved_files = self.model.save_model(model_save_path)
            
            # Record stage results
            self.pipeline_results['stages']['model'] = {
                'status': 'completed',
                'model_info': model_info,
                'saved_files': saved_files,
                'training_results': training_results if 'training_results' in locals() else None,
                'duration': self.pipeline_monitor.end_stage('model')
            }
            
            self.logger.info("Model stage completed successfully")
            
        except Exception as e:
            self.logger.error(f"Model stage failed: {str(e)}")
            self.pipeline_results['stages']['model'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _run_test_stage(self, output_dir: str):
        """Run the test stage"""
        try:
            self.current_stage = 'test'
            self.logger.info("=== Running Test Stage ===")
            self.pipeline_monitor.start_stage('test')
            
            # Initialize test stage with model
            self.test_stage = TestStage(self.model, self.config_path)
            
            # Prepare test data
            processed_data = self.pipeline_results['processed_data']
            features = processed_data['features']
            labels = processed_data['labels']
            protected_attributes = processed_data['protected_attributes']
            
            # Convert labels to numpy array
            import numpy as np
            y_true = np.array(labels.tolist())
            
            # Run comprehensive tests
            test_results = self.test_stage.run_comprehensive_tests(
                features, y_true, protected_attributes
            )
            
            # Save test artifacts
            test_artifacts = self.test_stage.save_test_artifacts(
                os.path.join(output_dir, "test")
            )
            
            # Record stage results
            self.pipeline_results['stages']['test'] = {
                'status': test_results['overall_status'],
                'fairness_status': test_results['fairness_results'].get('status', 'unknown'),
                'performance_status': test_results['performance_results'].get('status', 'unknown'),
                'explainability_status': test_results['explainability_results'].get('status', 'unknown'),
                'test_results': test_results,
                'artifacts': test_artifacts,
                'duration': self.pipeline_monitor.end_stage('test')
            }
            
            # Check if tests passed
            if test_results['overall_status'] not in ['pass']:
                self.logger.warning(f"Tests did not pass: {test_results['overall_status']}")
                # Continue but mark for attention
            
            self.logger.info("Test stage completed")
            
        except Exception as e:
            self.logger.error(f"Test stage failed: {str(e)}")
            self.pipeline_results['stages']['test'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _should_deploy(self) -> bool:
        """Determine if model should be deployed based on test results"""
        test_results = self.pipeline_results.get('stages', {}).get('test', {})
        
        # Check test status
        if test_results.get('status') != 'pass':
            return False
        
        # Check individual component status
        fairness_status = test_results.get('fairness_status')
        performance_status = test_results.get('performance_status')
        
        # Allow deployment if both fairness and performance are acceptable
        return (fairness_status == 'pass' and performance_status == 'pass')
    
    def _run_deploy_stage(self, output_dir: str):
        """Run the deploy stage"""
        try:
            self.current_stage = 'deploy'
            self.logger.info("=== Running Deploy Stage ===")
            self.pipeline_monitor.start_stage('deploy')
            
            # Get model artifacts
            model_artifacts = self.pipeline_results.get('stages', {}).get('model', {}).get('saved_files', {})
            
            if not model_artifacts:
                raise PipelineError("No model artifacts available for deployment")
            
            model_path = model_artifacts.get('model')
            if not model_path:
                raise PipelineError("Model path not found in artifacts")
            
            # Containerize model
            container_image = self.deploy_stage.containerize_model(
                model_path, model_artifacts
            )
            
            # Deploy to Kubernetes
            deployment_info = self.deploy_stage.deploy_to_kubernetes(container_image)
            
            # Setup fairness monitoring
            monitoring_info = self.deploy_stage.setup_fairness_monitoring(deployment_info)
            
            # Validate deployment
            validation_results = self.deploy_stage.validate_deployment(deployment_info)
            
            # Record stage results
            self.pipeline_results['stages']['deploy'] = {
                'status': 'completed' if validation_results['overall_status'] == 'healthy' else 'partially_completed',
                'container_image': container_image,
                'deployment_info': deployment_info,
                'monitoring_info': monitoring_info,
                'validation_results': validation_results,
                'duration': self.pipeline_monitor.end_stage('deploy')
            }
            
            self.logger.info("Deploy stage completed successfully")
            
        except Exception as e:
            self.logger.error(f"Deploy stage failed: {str(e)}")
            self.pipeline_results['stages']['deploy'] = {
                'status': 'failed',
                'error': str(e)
            }
            # Don't raise - deployment failure shouldn't stop monitoring setup
    
    def _setup_monitoring_stage(self, output_dir: str):
        """Setup continuous monitoring"""
        try:
            self.current_stage = 'monitor'
            self.logger.info("=== Setting up Monitoring Stage ===")
            self.pipeline_monitor.start_stage('monitor')
            
            # Initialize drift detector with reference data
            processed_data = self.pipeline_results['processed_data']
            features = processed_data['features']
            
            # Convert to numpy array for drift detection
            import numpy as np
            reference_data = features.select_dtypes(include=[np.number]).values
            
            if len(reference_data) > 0:
                self.monitor_stage.initialize_drift_detector(reference_data)
                self.logger.info("Drift detector initialized")
            else:
                self.logger.warning("No numerical features found for drift detection")
            
            # Start continuous monitoring (this would run in background in production)
            # For demonstration, we'll just set it up
            monitoring_status = self.monitor_stage.get_monitoring_status()
            
            # Record stage results
            self.pipeline_results['stages']['monitor'] = {
                'status': 'setup_completed',
                'monitoring_status': monitoring_status,
                'duration': self.pipeline_monitor.end_stage('monitor')
            }
            
            self.logger.info("Monitoring stage setup completed")
            
        except Exception as e:
            self.logger.error(f"Monitor stage setup failed: {str(e)}")
            self.pipeline_results['stages']['monitor'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def _finalize_pipeline(self, output_dir: str):
        """Finalize pipeline execution"""
        try:
            self.pipeline_results['end_time'] = datetime.now().isoformat()
            
            # Determine overall status
            stage_statuses = [
                self.pipeline_results['stages'].get('build', {}).get('status'),
                self.pipeline_results['stages'].get('model', {}).get('status'),
                self.pipeline_results['stages'].get('test', {}).get('status')
            ]
            
            if all(status == 'completed' for status in stage_statuses):
                if self.pipeline_results['stages'].get('deploy', {}).get('status') == 'completed':
                    self.pipeline_results['status'] = 'success'
                else:
                    self.pipeline_results['status'] = 'partial_success'
            else:
                self.pipeline_results['status'] = 'failed'
            
            # Generate final report
            final_report = self.generate_pipeline_report()
            
            # Save pipeline results
            results_path = os.path.join(output_dir, "pipeline_results.json")
            with open(results_path, 'w') as f:
                json.dump(self.pipeline_results, f, indent=2, default=str)
            
            # Save final report
            report_path = os.path.join(output_dir, "pipeline_report.txt")
            with open(report_path, 'w') as f:
                f.write(final_report)
            
            self.pipeline_results['artifacts']['pipeline_results'] = results_path
            self.pipeline_results['artifacts']['pipeline_report'] = report_path
            
            self.logger.info(f"Pipeline completed with status: {self.pipeline_results['status']}")
            
        except Exception as e:
            self.logger.error(f"Error finalizing pipeline: {str(e)}")
    
    def _handle_pipeline_failure(self, error: Exception):
        """Handle pipeline failure"""
        self.pipeline_results['status'] = 'failed'
        self.pipeline_results['end_time'] = datetime.now().isoformat()
        
        error_info = {
            'stage': self.current_stage,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.pipeline_results['errors'].append(error_info)
        self.logger.error(f"Pipeline failed in stage '{self.current_stage}': {str(error)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def generate_pipeline_report(self) -> str:
        """Generate comprehensive pipeline report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DEVETHOPS PIPELINE EXECUTION REPORT")
        report_lines.append("=" * 80)
        
        # Executive Summary
        report_lines.append(f"\nPipeline ID: {self.pipeline_results['pipeline_id']}")
        report_lines.append(f"Status: {self.pipeline_results['status'].upper()}")
        report_lines.append(f"Start Time: {self.pipeline_results['start_time']}")
        report_lines.append(f"End Time: {self.pipeline_results['end_time']}")
        
        # Calculate total duration
        try:
            from datetime import datetime
            start = datetime.fromisoformat(self.pipeline_results['start_time'])
            end = datetime.fromisoformat(self.pipeline_results['end_time'])
            duration = end - start
            report_lines.append(f"Total Duration: {duration}")
        except:
            report_lines.append("Total Duration: Unknown")
        
        # Stage Summary
        report_lines.append("\nStage Summary:")
        report_lines.append("-" * 40)
        
        stages = self.pipeline_results.get('stages', {})
        for stage_name, stage_info in stages.items():
            status = stage_info.get('status', 'unknown')
            duration = stage_info.get('duration', 'unknown')
            status_emoji = {
                'completed': '✅',
                'pass': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'partial_success': '⚠️'
            }.get(status, '❓')
            
            report_lines.append(f"{stage_name.upper()}: {status_emoji} {status} (Duration: {duration})")
        
        # Detailed Results
        for stage_name, stage_info in stages.items():
            report_lines.append(f"\n{'-' * 50}")
            report_lines.append(f"{stage_name.upper()} STAGE DETAILS")
            report_lines.append(f"{'-' * 50}")
            
            if stage_name == 'build':
                bias_metrics = stage_info.get('bias_metrics', {})
                if bias_metrics:
                    report_lines.append("Bias Metrics:")
                    for metric, value in bias_metrics.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"  {metric}: {value:.4f}")
            
            elif stage_name == 'test':
                test_results = stage_info.get('test_results', {})
                fairness_status = test_results.get('fairness_results', {}).get('status', 'unknown')
                performance_status = test_results.get('performance_results', {}).get('status', 'unknown')
                
                report_lines.append(f"Fairness Tests: {fairness_status}")
                report_lines.append(f"Performance Tests: {performance_status}")
            
            elif stage_name == 'deploy':
                deployment_info = stage_info.get('deployment_info', {})
                if deployment_info:
                    report_lines.append(f"Deployment: {deployment_info.get('deployment_name', 'unknown')}")
                    report_lines.append(f"Namespace: {deployment_info.get('namespace', 'unknown')}")
        
        # Errors (if any)
        if self.pipeline_results.get('errors'):
            report_lines.append(f"\n{'-' * 50}")
            report_lines.append("ERRORS")
            report_lines.append(f"{'-' * 50}")
            
            for error in self.pipeline_results['errors']:
                report_lines.append(f"Stage: {error.get('stage', 'unknown')}")
                report_lines.append(f"Error: {error.get('error_message', 'unknown')}")
                report_lines.append("")
        
        # Recommendations
        report_lines.append(f"\n{'-' * 50}")
        report_lines.append("RECOMMENDATIONS")
        report_lines.append(f"{'-' * 50}")
        
        if self.pipeline_results['status'] == 'success':
            report_lines.append("🎉 Pipeline completed successfully!")
            report_lines.append("✅ Model deployed and monitoring active")
            report_lines.append("📊 Continue monitoring fairness metrics in production")
        
        elif self.pipeline_results['status'] == 'partial_success':
            report_lines.append("⚠️  Pipeline partially successful")
            report_lines.append("🔍 Review deployment issues")
            report_lines.append("📈 Model passed tests but deployment needs attention")
        
        else:
            report_lines.append("❌ Pipeline failed")
            report_lines.append("🛠️  Review errors and re-run pipeline")
            report_lines.append("📋 Check data quality and model configuration")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def check_bias_thresholds(self, bias_metrics: Dict[str, float]) -> bool:
        """Check if bias metrics meet thresholds"""
        return self.build_stage.check_bias_thresholds(bias_metrics)
    
    def validate_fairness(self, fairness_results: Dict[str, Any]) -> bool:
        """Validate fairness results against thresholds"""
        # This would typically check against configured thresholds
        violations = fairness_results.get('violations', [])
        return len(violations) == 0


# CLI interface
def main():
    """Main CLI interface for DevEthOps pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DevEthOps Pipeline - Ethical AI CI/CD')
    parser.add_argument('--dataset', required=True, help='Path to dataset file')
    parser.add_argument('--model', help='Path to pre-trained model (optional)')
    parser.add_argument('--config', default='config/pipeline_config.yaml', help='Pipeline configuration file')
    parser.add_argument('--output', default='artifacts', help='Output directory for artifacts')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = DevEthOpsPipeline(args.config)
        
        # Run pipeline
        results = pipeline.run_pipeline(
            dataset_path=args.dataset,
            model_path=args.model,
            output_dir=args.output
        )
        
        # Print summary
        print(f"\nPipeline completed with status: {results['status']}")
        print(f"Results saved to: {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if results['status'] in ['success', 'partial_success'] else 1)
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
