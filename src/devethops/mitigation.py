"""
Bias mitigation techniques using AIF360 and custom implementations.

Implements reweighing, optimized pre-processing, and other fairness-aware
techniques to reduce bias in datasets and models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import compute_sample_weight
from sklearn.model_selection import GridSearchCV
import warnings

# AIF360 imports
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing, OptimPreproc
    from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
    AIF360_AVAILABLE = True
except ImportError:
    logging.warning("AIF360 not available. Using manual bias mitigation implementations.")
    AIF360_AVAILABLE = False

logger = logging.getLogger(__name__)


class BiasError(Exception):
    """Custom exception for bias mitigation errors."""
    pass


class BiasReweigher(BaseEstimator, TransformerMixin):
    """
    Implements reweighing technique for bias mitigation.
    
    Adjusts sample weights to ensure fairness across protected groups.
    """
    
    def __init__(self, protected_attribute: str, privileged_groups: Optional[List[Any]] = None):
        """
        Initialize bias reweigher.
        
        Args:
            protected_attribute: Name of protected attribute column
            privileged_groups: List of privileged group values
        """
        self.protected_attribute = protected_attribute
        self.privileged_groups = privileged_groups
        self.weights_ = None
        self.group_weights_ = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BiasReweigher':
        """
        Fit the reweigher to compute sample weights.
        
        Args:
            X: Feature matrix containing protected attribute
            y: Target vector
            
        Returns:
            Self
        """
        logger.info(f"Fitting bias reweigher for attribute: {self.protected_attribute}")
        
        if self.protected_attribute not in X.columns:
            raise BiasError(f"Protected attribute '{self.protected_attribute}' not found in features")
        
        # Auto-detect privileged groups if not provided
        if self.privileged_groups is None:
            self.privileged_groups = self._auto_detect_privileged_groups(X[self.protected_attribute])
        
        # Compute reweighing factors
        self.weights_, self.group_weights_ = self._compute_weights(X, y)
        
        logger.info(f"Reweighing complete. Group weights: {self.group_weights_}")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform returns sample weights for the given data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Sample weights array
        """
        if self.weights_ is None:
            raise BiasError("Reweigher not fitted. Call fit first.")
        
        weights = np.ones(len(X))
        
        for group_key, group_weight in self.group_weights_.items():
            protected_val, target_val = group_key
            mask = (X[self.protected_attribute] == protected_val)
            weights[mask] = group_weight
        
        return weights
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Fit reweigher and return sample weights.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Sample weights array
        """
        return self.fit(X, y).transform(X)
    
    def _auto_detect_privileged_groups(self, protected_series: pd.Series) -> List[Any]:
        """Auto-detect privileged groups."""
        value_counts = protected_series.value_counts()
        majority_group = value_counts.index[0]
        return [majority_group]
    
    def _compute_weights(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, Dict]:
        """
        Compute reweighing factors for fairness.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (sample_weights, group_weights_dict)
        """
        # Get unique values
        protected_values = X[self.protected_attribute].unique()
        target_values = y.unique()
        
        # Compute probabilities for each group
        total_size = len(X)
        group_probs = {}
        group_weights = {}
        
        for prot_val in protected_values:
            for target_val in target_values:
                mask = (X[self.protected_attribute] == prot_val) & (y == target_val)
                observed_prob = mask.sum() / total_size
                
                # Expected probability (assuming independence)
                prot_prob = (X[self.protected_attribute] == prot_val).sum() / total_size
                target_prob = (y == target_val).sum() / total_size
                expected_prob = prot_prob * target_prob
                
                # Reweighing factor
                if observed_prob > 0:
                    weight = expected_prob / observed_prob
                else:
                    weight = 1.0
                
                group_key = (prot_val, target_val)
                group_probs[group_key] = observed_prob
                group_weights[group_key] = weight
        
        # Apply weights to samples
        sample_weights = np.ones(len(X))
        for group_key, group_weight in group_weights.items():
            prot_val, target_val = group_key
            mask = (X[self.protected_attribute] == prot_val) & (y == target_val)
            sample_weights[mask] = group_weight
        
        return sample_weights, group_weights


class FairnessMitigator:
    """
    Main class for applying various bias mitigation techniques.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fairness mitigator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.use_aif360 = AIF360_AVAILABLE and config.get('use_aif360', True)
        self.mitigation_method = config.get('mitigation_method', 'reweighing')
        self.fitted_mitigators = {}
        
    def apply_mitigation(self, X: pd.DataFrame, y: pd.Series,
                        protected_attributes: List[str],
                        method: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Apply bias mitigation to dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            protected_attributes: List of protected attribute names
            method: Mitigation method ('reweighing', 'optimized_preprocessing', 'manual')
            
        Returns:
            Tuple of (mitigated_X, mitigated_y, mitigation_info)
        """
        if method is None:
            method = self.mitigation_method
        
        logger.info(f"Applying bias mitigation method: {method}")
        
        mitigation_info = {
            'method': method,
            'protected_attributes': protected_attributes,
            'original_shape': X.shape,
            'sample_weights': None,
            'reweighing_factors': {}
        }
        
        if method == 'reweighing':
            X_mitigated, y_mitigated, weights_info = self._apply_reweighing(
                X, y, protected_attributes
            )
            mitigation_info.update(weights_info)
            
        elif method == 'optimized_preprocessing' and self.use_aif360:
            X_mitigated, y_mitigated, optim_info = self._apply_optimized_preprocessing(
                X, y, protected_attributes
            )
            mitigation_info.update(optim_info)
            
        elif method == 'sampling':
            X_mitigated, y_mitigated, sampling_info = self._apply_sampling_mitigation(
                X, y, protected_attributes
            )
            mitigation_info.update(sampling_info)
            
        else:
            logger.warning(f"Unknown mitigation method: {method}. Returning original data.")
            X_mitigated, y_mitigated = X, y
        
        mitigation_info['mitigated_shape'] = X_mitigated.shape
        logger.info(f"Mitigation complete. Shape: {X.shape} -> {X_mitigated.shape}")
        
        return X_mitigated, y_mitigated, mitigation_info
    
    def _apply_reweighing(self, X: pd.DataFrame, y: pd.Series,
                         protected_attributes: List[str]) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Apply reweighing mitigation.
        
        Args:
            X: Feature matrix
            y: Target vector
            protected_attributes: Protected attributes
            
        Returns:
            Tuple of (X, y, reweighing_info)
        """
        logger.info("Applying reweighing mitigation...")
        
        all_weights = np.ones(len(X))
        reweighing_factors = {}
        
        # Apply reweighing for each protected attribute
        for attr in protected_attributes:
            if attr not in X.columns:
                logger.warning(f"Protected attribute '{attr}' not found in features. Skipping.")
                continue
            
            # Fit reweigher
            reweigher = BiasReweigher(protected_attribute=attr)
            weights = reweigher.fit_transform(X, y)
            
            # Store fitted reweigher
            self.fitted_mitigators[f'reweigher_{attr}'] = reweigher
            
            # Combine weights (multiply)
            all_weights *= weights
            reweighing_factors[attr] = reweigher.group_weights_
        
        # Normalize weights
        all_weights = all_weights / all_weights.mean()
        
        return X, y, {
            'sample_weights': all_weights,
            'reweighing_factors': reweighing_factors,
            'fitted_reweighers': list(self.fitted_mitigators.keys())
        }
    
    def _apply_optimized_preprocessing(self, X: pd.DataFrame, y: pd.Series,
                                     protected_attributes: List[str]) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Apply AIF360 optimized preprocessing.
        
        Args:
            X: Feature matrix
            y: Target vector
            protected_attributes: Protected attributes
            
        Returns:
            Tuple of (mitigated_X, mitigated_y, optim_info)
        """
        logger.info("Applying AIF360 optimized preprocessing...")
        
        if not self.use_aif360:
            logger.warning("AIF360 not available. Falling back to reweighing.")
            return self._apply_reweighing(X, y, protected_attributes)
        
        try:
            # Prepare data for AIF360
            df = X.copy()
            df['target'] = y
            
            # For each protected attribute, apply optimized preprocessing
            mitigated_datasets = []
            
            for attr in protected_attributes:
                if attr not in X.columns:
                    continue
                
                # Create AIF360 dataset
                privileged_groups = self._get_privileged_groups(X[attr])
                unprivileged_groups = [v for v in X[attr].unique() if v not in privileged_groups]
                
                dataset = BinaryLabelDataset(
                    favorable_label=1,
                    unfavorable_label=0,
                    df=df,
                    label_names=['target'],
                    protected_attribute_names=[attr],
                    privileged_classes=[privileged_groups]
                )
                
                # Apply optimized preprocessing
                optim_preproc = OptimPreproc(
                    OptimPreproc.DI,  # Disparate impact objective
                    seed=self.config.get('random_state', 42)
                )
                
                mitigated_dataset = optim_preproc.fit_transform(dataset)
                mitigated_datasets.append(mitigated_dataset)
            
            # Use the first mitigated dataset (can be extended for multiple attributes)
            if mitigated_datasets:
                mitigated_df = mitigated_datasets[0].convert_to_dataframe()[0]
                X_mitigated = mitigated_df.drop(columns=['target'])
                y_mitigated = mitigated_df['target']
            else:
                X_mitigated, y_mitigated = X, y
            
            return X_mitigated, y_mitigated, {
                'method_details': 'AIF360 OptimPreproc with DI objective',
                'num_attributes_processed': len([a for a in protected_attributes if a in X.columns])
            }
            
        except Exception as e:
            logger.error(f"AIF360 optimized preprocessing failed: {e}")
            logger.info("Falling back to reweighing method...")
            return self._apply_reweighing(X, y, protected_attributes)
    
    def _apply_sampling_mitigation(self, X: pd.DataFrame, y: pd.Series,
                                 protected_attributes: List[str]) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Apply sampling-based mitigation (balancing groups).
        
        Args:
            X: Feature matrix
            y: Target vector
            protected_attributes: Protected attributes
            
        Returns:
            Tuple of (mitigated_X, mitigated_y, sampling_info)
        """
        logger.info("Applying sampling-based mitigation...")
        
        # For simplicity, balance the first protected attribute
        if not protected_attributes or protected_attributes[0] not in X.columns:
            return X, y, {'method_details': 'No sampling applied - protected attribute not found'}
        
        attr = protected_attributes[0]
        
        # Find minority group size
        group_sizes = X.groupby([attr, y]).size()
        min_group_size = group_sizes.min()
        
        # Sample each group to minimum size
        sampled_indices = []
        
        for (attr_val, target_val), group_size in group_sizes.items():
            mask = (X[attr] == attr_val) & (y == target_val)
            group_indices = X[mask].index.tolist()
            
            if len(group_indices) > min_group_size:
                # Undersample
                sampled_group_indices = np.random.choice(
                    group_indices, min_group_size, replace=False
                )
            else:
                # Oversample
                sampled_group_indices = np.random.choice(
                    group_indices, min_group_size, replace=True
                )
            
            sampled_indices.extend(sampled_group_indices)
        
        # Create balanced dataset
        X_mitigated = X.loc[sampled_indices].reset_index(drop=True)
        y_mitigated = y.loc[sampled_indices].reset_index(drop=True)
        
        return X_mitigated, y_mitigated, {
            'method_details': f'Balanced sampling on {attr}',
            'min_group_size': min_group_size,
            'original_size': len(X),
            'mitigated_size': len(X_mitigated)
        }
    
    def _get_privileged_groups(self, protected_series: pd.Series) -> List[Any]:
        """Get privileged groups for a protected attribute."""
        # Simple heuristic: majority group is privileged
        value_counts = protected_series.value_counts()
        return [value_counts.index[0]]
    
    def get_sample_weights(self, X: pd.DataFrame, y: pd.Series,
                          protected_attributes: List[str]) -> np.ndarray:
        """
        Get sample weights for fair training without modifying the dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            protected_attributes: Protected attributes
            
        Returns:
            Sample weights array
        """
        logger.info("Computing sample weights for fair training...")
        
        all_weights = np.ones(len(X))
        
        for attr in protected_attributes:
            if attr not in X.columns:
                continue
            
            reweigher = BiasReweigher(protected_attribute=attr)
            weights = reweigher.fit_transform(X, y)
            all_weights *= weights
        
        # Normalize weights
        all_weights = all_weights / all_weights.mean()
        
        return all_weights


def apply_fairness_mitigation(X: pd.DataFrame, y: pd.Series,
                            config: Dict[str, Any],
                            protected_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Apply fairness mitigation with configuration.
    
    Args:
        X: Feature matrix
        y: Target vector
        config: Configuration dictionary
        protected_attributes: Protected attributes (from config if None)
        
    Returns:
        Dictionary with mitigation results
    """
    if protected_attributes is None:
        protected_attributes = config.get('protected_attributes', [])
    
    # Initialize mitigator
    mitigator = FairnessMitigator(config)
    
    # Apply mitigation
    X_mitigated, y_mitigated, mitigation_info = mitigator.apply_mitigation(
        X, y, protected_attributes
    )
    
    # Get sample weights (useful for training)
    sample_weights = mitigator.get_sample_weights(X, y, protected_attributes)
    
    return {
        'X_mitigated': X_mitigated,
        'y_mitigated': y_mitigated,
        'sample_weights': sample_weights,
        'mitigation_info': mitigation_info,
        'mitigator': mitigator
    }


def create_fair_training_weights(X: pd.DataFrame, y: pd.Series,
                               protected_attributes: List[str],
                               method: str = 'reweighing') -> np.ndarray:
    """
    Create sample weights for fair model training.
    
    Args:
        X: Feature matrix
        y: Target vector
        protected_attributes: Protected attributes
        method: Weighting method
        
    Returns:
        Sample weights array
    """
    logger.info(f"Creating fair training weights using {method}...")
    
    if method == 'reweighing':
        all_weights = np.ones(len(X))
        
        for attr in protected_attributes:
            if attr not in X.columns:
                continue
            
            reweigher = BiasReweigher(protected_attribute=attr)
            weights = reweigher.fit_transform(X, y)
            all_weights *= weights
        
        # Normalize
        all_weights = all_weights / all_weights.mean()
        return all_weights
        
    elif method == 'sklearn_balanced':
        # Use sklearn's built-in class balancing
        return compute_sample_weight('balanced', y)
        
    else:
        logger.warning(f"Unknown weighting method: {method}. Using uniform weights.")
        return np.ones(len(X))


# TODO: Implement additional mitigation techniques
# - Adversarial debiasing
# - Fair representation learning  
# - Post-processing calibration
# - Multi-task fairness
