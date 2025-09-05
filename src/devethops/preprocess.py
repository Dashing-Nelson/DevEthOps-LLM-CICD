"""
Data preprocessing utilities for DevEthOps pipeline.

Handles cleaning, one-hot encoding, scaling, train/val/test splits,
and SMOTE for class balancing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class PreprocessorError(Exception):
    """Custom exception for preprocessing errors."""
    pass


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for fairness-aware ML.
    
    Handles categorical encoding, numerical scaling, and maintains
    fairness considerations throughout preprocessing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.preprocessor = None
        self.label_encoders = {}
        self.feature_names = None
        self.protected_attributes = config.get('protected_attributes', [])
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit preprocessor and transform features.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Tuple of (transformed_features, preprocessing_info)
        """
        logger.info("Fitting and transforming data...")
        
        # Store original feature names
        self.original_features = list(X.columns)
        
        # Identify feature types
        categorical_features, numerical_features = self._identify_feature_types(X)
        
        # Create preprocessing pipeline
        self.preprocessor = self._create_preprocessing_pipeline(
            categorical_features, numerical_features
        )
        
        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        self.feature_names = self._get_feature_names(categorical_features, numerical_features)
        
        # Create preprocessing info
        preprocessing_info = {
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'feature_names': self.feature_names,
            'protected_attributes': self.protected_attributes,
            'original_shape': X.shape,
            'transformed_shape': X_transformed.shape
        }
        
        logger.info(f"Preprocessing complete. Shape: {X.shape} -> {X_transformed.shape}")
        
        return X_transformed, preprocessing_info
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
        """
        if self.preprocessor is None:
            raise PreprocessorError("Preprocessor not fitted. Call fit_transform first.")
        
        return self.preprocessor.transform(X)
    
    def _identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify categorical and numerical features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            elif pd.api.types.is_bool_dtype(X[col]):
                categorical_features.append(col)
            else:
                # Check if it's actually categorical (low cardinality)
                unique_values = X[col].nunique()
                if unique_values <= 10 and unique_values < len(X) * 0.05:
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
        
        logger.info(f"Identified {len(categorical_features)} categorical and "
                   f"{len(numerical_features)} numerical features")
        
        return categorical_features, numerical_features
    
    def _create_preprocessing_pipeline(self, categorical_features: List[str], 
                                     numerical_features: List[str]) -> ColumnTransformer:
        """
        Create sklearn preprocessing pipeline.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            
        Returns:
            ColumnTransformer pipeline
        """
        transformers = []
        
        # Numerical features: scaling
        if numerical_features:
            numerical_transformer = Pipeline([
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, numerical_features))
        
        # Categorical features: one-hot encoding
        if categorical_features:
            categorical_transformer = Pipeline([
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        return ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    def _get_feature_names(self, categorical_features: List[str], 
                          numerical_features: List[str]) -> List[str]:
        """
        Get feature names after transformation.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            
        Returns:
            List of transformed feature names
        """
        feature_names = []
        
        # Numerical features keep their names
        feature_names.extend(numerical_features)
        
        # Categorical features get expanded names
        if categorical_features:
            cat_transformer = self.preprocessor.named_transformers_['cat']
            onehot_encoder = cat_transformer.named_steps['onehot']
            
            cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
            feature_names.extend(cat_feature_names)
        
        return feature_names
    
    def save(self, filepath: str) -> None:
        """
        Save fitted preprocessor to disk.
        
        Args:
            filepath: Path to save preprocessor
        """
        if self.preprocessor is None:
            raise PreprocessorError("No fitted preprocessor to save")
        
        save_data = {
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'original_features': self.original_features,
            'protected_attributes': self.protected_attributes,
            'config': self.config
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """
        Load fitted preprocessor from disk.
        
        Args:
            filepath: Path to load preprocessor from
            
        Returns:
            Loaded DataPreprocessor instance
        """
        save_data = joblib.load(filepath)
        
        instance = cls(save_data['config'])
        instance.preprocessor = save_data['preprocessor']
        instance.feature_names = save_data['feature_names']
        instance.original_features = save_data['original_features']
        instance.protected_attributes = save_data['protected_attributes']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return instance


class DataSplitter:
    """
    Handles train/validation/test splits with fairness considerations.
    """
    
    @staticmethod
    def create_splits(X: pd.DataFrame, y: pd.Series, 
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     random_state: int = 42,
                     stratify_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create train/validation/test splits with optional stratification.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            stratify_cols: Columns to stratify on (including target)
            
        Returns:
            Dictionary with split data
        """
        logger.info("Creating train/validation/test splits...")
        
        # Prepare stratification
        stratify_data = None
        if stratify_cols:
            # Combine target and protected attributes for stratification
            stratify_features = []
            if 'target' in stratify_cols or len(stratify_cols) == 0:
                stratify_features.append(y)
            
            for col in stratify_cols:
                if col != 'target' and col in X.columns:
                    stratify_features.append(X[col])
            
            if stratify_features:
                stratify_data = pd.concat(stratify_features, axis=1)
                # Create combined stratification key
                stratify_data = stratify_data.apply(lambda x: '_'.join(x.astype(str)), axis=1)
                
                # Check if stratification is possible
                value_counts = stratify_data.value_counts()
                min_count = value_counts.min()
                required_count = max(2, int(1 / min(test_size, val_size)))
                
                if min_count < required_count:
                    logger.warning("Some stratification groups too small, using target only")
                    stratify_data = y
        else:
            stratify_data = y
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_data if stratify_data is not None else None
        )
        
        # Update stratification data for second split
        if stratify_data is not None:
            stratify_temp = stratify_data.loc[X_temp.index] if hasattr(stratify_data, 'loc') else y_temp
        else:
            stratify_temp = None
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify_temp
        )
        
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        # Log split statistics
        for split_name, split_data in splits.items():
            logger.info(f"{split_name}: {len(split_data)} samples")
        
        return splits


class ClassBalancer:
    """
    Handles class imbalance using SMOTE and other techniques.
    """
    
    def __init__(self, method: str = 'smote', random_state: int = 42):
        """
        Initialize class balancer.
        
        Args:
            method: Balancing method ('smote', 'random_oversample', 'none')
            random_state: Random seed
        """
        self.method = method
        self.random_state = random_state
        self.balancer = None
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply class balancing to training data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Balanced (X, y)
        """
        if self.method == 'none':
            return X, y
        
        logger.info(f"Applying class balancing: {self.method}")
        
        # Check class distribution before balancing
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution before balancing: {dict(zip(unique, counts))}")
        
        if self.method == 'smote':
            self.balancer = SMOTE(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown balancing method: {self.method}")
        
        try:
            X_balanced, y_balanced = self.balancer.fit_resample(X, y)
            
            # Check class distribution after balancing
            unique, counts = np.unique(y_balanced, return_counts=True)
            logger.info(f"Class distribution after balancing: {dict(zip(unique, counts))}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.warning(f"Class balancing failed: {e}. Returning original data.")
            return X, y


def preprocess_pipeline(X: pd.DataFrame, y: pd.Series, 
                       config: Dict[str, Any],
                       apply_balancing: bool = True) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline.
    
    Args:
        X: Feature matrix
        y: Target vector
        config: Configuration dictionary
        apply_balancing: Whether to apply class balancing
        
    Returns:
        Dictionary with preprocessed data and metadata
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Create splits
    splits = DataSplitter.create_splits(
        X, y,
        test_size=config.get('test_size', 0.2),
        val_size=config.get('val_size', 0.1),
        random_state=config.get('random_state', 42),
        stratify_cols=config.get('stratify_cols', ['target'])
    )
    
    # Fit preprocessor on training data
    X_train_processed, preprocessing_info = preprocessor.fit_transform(splits['X_train'])
    
    # Transform validation and test sets
    X_val_processed = preprocessor.transform(splits['X_val'])
    X_test_processed = preprocessor.transform(splits['X_test'])
    
    # Apply class balancing to training data if requested
    if apply_balancing and config.get('balancing_method', 'smote') != 'none':
        balancer = ClassBalancer(
            method=config.get('balancing_method', 'smote'),
            random_state=config.get('random_state', 42)
        )
        X_train_processed, y_train_balanced = balancer.fit_resample(
            X_train_processed, splits['y_train'].values
        )
        y_train_balanced = pd.Series(y_train_balanced, name=splits['y_train'].name)
    else:
        y_train_balanced = splits['y_train']
    
    # Compile results
    result = {
        'X_train': X_train_processed,
        'X_val': X_val_processed,
        'X_test': X_test_processed,
        'y_train': y_train_balanced,
        'y_val': splits['y_val'],
        'y_test': splits['y_test'],
        'preprocessor': preprocessor,
        'preprocessing_info': preprocessing_info,
        'original_splits': splits
    }
    
    logger.info("Preprocessing pipeline complete")
    return result


def save_preprocessing_artifacts(preprocessing_result: Dict[str, Any], 
                               output_dir: str) -> None:
    """
    Save preprocessing artifacts to disk.
    
    Args:
        preprocessing_result: Result from preprocess_pipeline
        output_dir: Directory to save artifacts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessor
    preprocessor_path = output_dir / "preprocessor.joblib"
    preprocessing_result['preprocessor'].save(str(preprocessor_path))
    
    # Save processed data
    data_path = output_dir / "processed_data.joblib"
    save_data = {
        'X_train': preprocessing_result['X_train'],
        'X_val': preprocessing_result['X_val'],
        'X_test': preprocessing_result['X_test'],
        'y_train': preprocessing_result['y_train'],
        'y_val': preprocessing_result['y_val'],
        'y_test': preprocessing_result['y_test'],
        'preprocessing_info': preprocessing_result['preprocessing_info']
    }
    joblib.dump(save_data, data_path)
    
    logger.info(f"Preprocessing artifacts saved to {output_dir}")


def load_preprocessing_artifacts(input_dir: str) -> Dict[str, Any]:
    """
    Load preprocessing artifacts from disk.
    
    Args:
        input_dir: Directory containing artifacts
        
    Returns:
        Dictionary with loaded preprocessing results
    """
    input_dir = Path(input_dir)
    
    # Load preprocessor
    preprocessor_path = input_dir / "preprocessor.joblib"
    preprocessor = DataPreprocessor.load(str(preprocessor_path))
    
    # Load processed data
    data_path = input_dir / "processed_data.joblib"
    data = joblib.load(data_path)
    
    result = {**data, 'preprocessor': preprocessor}
    
    logger.info(f"Preprocessing artifacts loaded from {input_dir}")
    return result
