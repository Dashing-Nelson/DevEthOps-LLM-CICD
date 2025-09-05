"""
Configuration management for DevEthOps pipeline.

Handles loading of YAML configurations for datasets, fairness thresholds,
model parameters, and pipeline settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default settings.yaml
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If config file not found or invalid
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "settings.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load fairness thresholds if separate file exists
        fairness_config_path = config_path.parent / "fairness_thresholds.yaml"
        if fairness_config_path.exists():
            with open(fairness_config_path, 'r') as f:
                fairness_config = yaml.safe_load(f)
            config['fairness_thresholds'] = fairness_config.get('thresholds', {})
        
        # Apply environment variable overrides
        config = _apply_env_overrides(config)
        
        # Validate configuration
        _validate_config(config)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading config: {e}")


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.
    
    Environment variables should be prefixed with DEVETHOPS_
    """
    # TODO: Implement environment variable override logic
    # Example: DEVETHOPS_DATASET_NAME -> config['dataset']['name']
    
    env_overrides = {
        'DEVETHOPS_DATASET_NAME': ['dataset', 'name'],
        'DEVETHOPS_MODEL_TYPE': ['model', 'type'],
        'DEVETHOPS_DATA_PATH': ['paths', 'data'],
        'DEVETHOPS_CUDA_AVAILABLE': ['compute', 'cuda_available']
    }
    
    for env_var, config_path in env_overrides.items():
        value = os.getenv(env_var)
        if value:
            # Navigate to nested config location
            current = config
            for key in config_path[:-1]:
                current = current.setdefault(key, {})
            
            # Convert boolean strings
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            
            current[config_path[-1]] = value
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigError: If configuration is invalid
    """
    required_sections = ['dataset', 'model', 'fairness_thresholds', 'paths']
    
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Required configuration section missing: {section}")
    
    # Validate dataset configuration
    dataset_config = config['dataset']
    if 'name' not in dataset_config:
        raise ConfigError("Dataset name not specified in configuration")
    
    if 'protected_attributes' not in dataset_config:
        raise ConfigError("Protected attributes not specified in configuration")
    
    # Validate fairness thresholds
    fairness_config = config['fairness_thresholds']
    required_metrics = ['disparate_impact', 'statistical_parity_difference']
    
    for metric in required_metrics:
        if metric not in fairness_config:
            logger.warning(f"Fairness threshold for {metric} not specified, using defaults")
    
    logger.info("Configuration validation passed")


def get_dataset_config(dataset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get configuration specific to a dataset.
    
    Args:
        dataset_name: Name of the dataset
        config: Main configuration dictionary
        
    Returns:
        Dataset-specific configuration
    """
    dataset_configs = {
        'ibm_hr': {
            'file_path': 'data/raw/ibm_hr.csv',
            'target_column': 'Attrition',
            'protected_attributes': ['Gender', 'Age'],
            'categorical_features': ['BusinessTravel', 'Department', 'EducationField', 
                                   'Gender', 'JobRole', 'MaritalStatus', 'OverTime'],
            'drop_columns': ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
        },
        'adult': {
            'file_path': 'data/raw/adult.csv',
            'target_column': 'income',
            'protected_attributes': ['sex', 'race'],
            'categorical_features': ['workclass', 'education', 'marital-status', 
                                   'occupation', 'relationship', 'race', 'sex', 'native-country'],
            'drop_columns': ['fnlwgt']
        },
        'mimic': {
            'file_path': 'data/raw/mimic_subset.csv',
            'target_column': 'hospital_expire_flag',
            'protected_attributes': ['gender', 'insurance'],
            'categorical_features': ['gender', 'insurance'],
            'drop_columns': ['subject_id', 'hadm_id']
        }
    }
    
    base_config = config.get('dataset', {})
    specific_config = dataset_configs.get(dataset_name, {})
    
    # Merge configurations with specific taking precedence
    merged_config = {**base_config, **specific_config}
    merged_config['name'] = dataset_name
    
    return merged_config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path to save configuration file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {output_path}")
