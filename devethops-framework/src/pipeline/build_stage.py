"""
Build Stage - Dataset loading, bias detection, and mitigation
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

# AIF360 imports
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover, LFR

# Custom imports
from ..metrics.fairness_metrics import FairnessMetrics


class FairnessViolationError(Exception):
    """Custom exception for fairness violations"""
    pass


class BuildStage:
    """
    Build stage for the DevEthOps pipeline.
    Handles dataset loading, bias detection, and bias mitigation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Build Stage.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/pipeline_config.yaml"
        self.fairness_config_path = "config/fairness_thresholds.yaml"
        
        # Load configurations
        self.config = self._load_config()
        self.fairness_thresholds = self._load_fairness_config()
        
        # Initialize components
        self.fairness_metrics = FairnessMetrics(self.fairness_thresholds)
        self.bias_mitigators = self._initialize_bias_mitigators()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
            
    def _load_fairness_config(self) -> Dict[str, Any]:
        """Load fairness thresholds configuration"""
        try:
            with open(self.fairness_config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Fairness config file not found: {self.fairness_config_path}")
            raise
            
    def _initialize_bias_mitigators(self) -> Dict[str, Any]:
        """Initialize bias mitigation algorithms"""
        return {
            'reweighing': Reweighing,
            'disparate_impact_remover': DisparateImpactRemover,
            'lfr': LFR
        }
    
    def load_dataset(self, dataset_path: str, protected_attributes: List[str]) -> BinaryLabelDataset:
        """
        Load and prepare dataset for bias analysis.
        
        Args:
            dataset_path: Path to the dataset file
            protected_attributes: List of protected attribute column names
            
        Returns:
            BinaryLabelDataset: AIF360 dataset object
        """
        self.logger.info(f"Loading dataset from {dataset_path}")
        
        try:
            # Load data based on file extension
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            elif dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")
                
            # Validate protected attributes exist in dataset
            missing_attrs = [attr for attr in protected_attributes if attr not in df.columns]
            if missing_attrs:
                raise ValueError(f"Protected attributes not found in dataset: {missing_attrs}")
                
            # Identify label column (assuming 'label' or 'target')
            label_column = None
            for col in ['label', 'target', 'y', 'outcome']:
                if col in df.columns:
                    label_column = col
                    break
                    
            if label_column is None:
                raise ValueError("No label column found. Expected 'label', 'target', 'y', or 'outcome'")
                
            # Convert to AIF360 BinaryLabelDataset
            dataset = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df,
                label_names=[label_column],
                protected_attribute_names=protected_attributes
            )
            
            self.logger.info(f"Dataset loaded successfully: {len(df)} samples, {len(df.columns)} features")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def detect_bias(self, dataset: BinaryLabelDataset) -> Dict[str, float]:
        """
        Detect bias in the dataset using multiple fairness metrics.
        
        Args:
            dataset: AIF360 BinaryLabelDataset
            
        Returns:
            Dict containing bias metrics
        """
        self.logger.info("Detecting bias in dataset")
        
        try:
            # Calculate bias metrics for each protected attribute
            bias_metrics = {}
            
            for attr in dataset.protected_attribute_names:
                # Define privileged and unprivileged groups
                privileged_groups = [{attr: dataset.privileged_protected_attributes[0]}]
                unprivileged_groups = [{attr: dataset.unprivileged_protected_attributes[0]}]
                
                # Create metric calculator
                metric = BinaryLabelDatasetMetric(
                    dataset, 
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups
                )
                
                # Calculate various fairness metrics
                metrics_for_attr = {
                    f'{attr}_demographic_parity': abs(metric.statistical_parity_difference()),
                    f'{attr}_disparate_impact': metric.disparate_impact(),
                    f'{attr}_consistency': metric.consistency()[0] if hasattr(metric, 'consistency') else 0.0
                }
                
                bias_metrics.update(metrics_for_attr)
                
                self.logger.info(f"Bias metrics for {attr}: {metrics_for_attr}")
            
            # Calculate overall bias score
            bias_metrics['overall_bias_score'] = self._calculate_overall_bias_score(bias_metrics)
            
            return bias_metrics
            
        except Exception as e:
            self.logger.error(f"Error detecting bias: {str(e)}")
            raise
    
    def _calculate_overall_bias_score(self, bias_metrics: Dict[str, float]) -> float:
        """Calculate an overall bias score from individual metrics"""
        # Simple weighted average approach
        dp_scores = [v for k, v in bias_metrics.items() if 'demographic_parity' in k]
        di_scores = [1 - v for k, v in bias_metrics.items() if 'disparate_impact' in k]  # Convert to bias score
        
        all_scores = dp_scores + di_scores
        return np.mean(all_scores) if all_scores else 0.0
    
    def check_bias_thresholds(self, bias_metrics: Dict[str, float]) -> bool:
        """
        Check if bias metrics exceed configured thresholds.
        
        Args:
            bias_metrics: Dictionary of bias metrics
            
        Returns:
            bool: True if all metrics are within acceptable thresholds
        """
        violations = []
        thresholds = self.fairness_thresholds['fairness_thresholds']
        
        for metric, value in bias_metrics.items():
            if 'demographic_parity' in metric:
                if value > thresholds['demographic_parity']['threshold']:
                    violations.append(f"{metric}: {value:.4f} > {thresholds['demographic_parity']['threshold']}")
                    
            elif 'disparate_impact' in metric:
                if value < thresholds['disparate_impact']['threshold']:
                    violations.append(f"{metric}: {value:.4f} < {thresholds['disparate_impact']['threshold']}")
        
        if violations:
            self.logger.warning(f"Bias threshold violations detected: {violations}")
            return False
            
        self.logger.info("All bias metrics within acceptable thresholds")
        return True
    
    def mitigate_bias(self, dataset: BinaryLabelDataset, method: str = 'reweighing') -> BinaryLabelDataset:
        """
        Apply bias mitigation techniques to the dataset.
        
        Args:
            dataset: Input dataset with detected bias
            method: Bias mitigation method to apply
            
        Returns:
            BinaryLabelDataset: Dataset with reduced bias
        """
        self.logger.info(f"Applying bias mitigation using {method}")
        
        try:
            if method not in self.bias_mitigators:
                raise ValueError(f"Unknown bias mitigation method: {method}")
            
            # Define privileged and unprivileged groups for the first protected attribute
            protected_attr = dataset.protected_attribute_names[0]
            privileged_groups = [{protected_attr: dataset.privileged_protected_attributes[0]}]
            unprivileged_groups = [{protected_attr: dataset.unprivileged_protected_attributes[0]}]
            
            # Apply mitigation
            if method == 'reweighing':
                mitigator = Reweighing(
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups
                )
                mitigated_dataset = mitigator.fit_transform(dataset)
                
            elif method == 'disparate_impact_remover':
                mitigator = DisparateImpactRemover(repair_level=0.8)
                mitigated_dataset = mitigator.fit_transform(dataset)
                
            elif method == 'lfr':
                mitigator = LFR(
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups,
                    k=5,  # Number of prototypes
                    Ax=0.01,  # Fairness penalty parameter
                    Ay=1.0,  # Accuracy penalty parameter
                    Az=50.0  # Reconstruction penalty parameter
                )
                mitigated_dataset = mitigator.fit_transform(dataset)
                
            self.logger.info(f"Bias mitigation completed using {method}")
            return mitigated_dataset
            
        except Exception as e:
            self.logger.error(f"Error in bias mitigation: {str(e)}")
            raise
    
    def prepare_training_data(self, dataset: BinaryLabelDataset) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert AIF360 dataset to training-ready format.
        
        Args:
            dataset: AIF360 BinaryLabelDataset
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        self.logger.info("Preparing training data")
        
        try:
            # Convert to pandas
            df = dataset.convert_to_dataframe()[0]
            
            # Separate features and labels
            label_column = dataset.label_names[0]
            features = df.drop(columns=[label_column])
            labels = df[label_column]
            
            self.logger.info(f"Training data prepared: {len(features)} samples, {len(features.columns)} features")
            return features, labels
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def save_build_artifacts(self, dataset: BinaryLabelDataset, bias_metrics: Dict[str, float], 
                           output_dir: str = "artifacts/build") -> Dict[str, str]:
        """
        Save build stage artifacts for later stages.
        
        Args:
            dataset: Processed dataset
            bias_metrics: Calculated bias metrics
            output_dir: Directory to save artifacts
            
        Returns:
            Dict containing paths to saved artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        artifacts = {}
        
        # Save processed dataset
        df, _ = dataset.convert_to_dataframe()
        dataset_path = os.path.join(output_dir, "processed_dataset.csv")
        df.to_csv(dataset_path, index=False)
        artifacts['dataset'] = dataset_path
        
        # Save bias metrics
        import json
        metrics_path = os.path.join(output_dir, "bias_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(bias_metrics, f, indent=2)
        artifacts['bias_metrics'] = metrics_path
        
        self.logger.info(f"Build artifacts saved to {output_dir}")
        return artifacts


# Custom exceptions
class DatasetLoadError(Exception):
    """Raised when dataset cannot be loaded"""
    pass


class BiasDetectionError(Exception):
    """Raised when bias detection fails"""
    pass


class BiasMitigationError(Exception):
    """Raised when bias mitigation fails"""
    pass


if __name__ == "__main__":
    # Example usage
    build_stage = BuildStage()
    
    # This would be replaced with actual dataset path
    # dataset = build_stage.load_dataset("data/sample_dataset.csv", ["gender", "race"])
    # bias_metrics = build_stage.detect_bias(dataset)
    # 
    # if not build_stage.check_bias_thresholds(bias_metrics):
    #     dataset = build_stage.mitigate_bias(dataset)
    # 
    # features, labels = build_stage.prepare_training_data(dataset)
