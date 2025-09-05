"""
Data loading utilities for DevEthOps pipeline.

Supports loading IBM HR, Adult Census, MIMIC-III datasets with caching
and synthetic bias injection utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import logging
import pickle
import hashlib
from sklearn.model_selection import train_test_split
import warnings

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass


class BaseDataLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, cache_dir: str = "data/cache", use_cache: bool = True):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory for caching processed data
            use_cache: Whether to use cached data if available
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
    
    def _get_cache_path(self, data_path: str, processing_params: Dict) -> Path:
        """Generate cache file path based on data path and processing parameters."""
        # Create hash of file path and processing parameters
        param_str = str(sorted(processing_params.items()))
        cache_key = hashlib.md5(f"{data_path}_{param_str}".encode()).hexdigest()
        return self.cache_dir / f"cached_data_{cache_key}.pkl"
    
    def _save_to_cache(self, data: Tuple, cache_path: Path) -> None:
        """Save processed data to cache."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Tuple]:
        """Load processed data from cache."""
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded from cache: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None


class IBMHRLoader(BaseDataLoader):
    """Loader for IBM HR Analytics Employee Attrition dataset."""
    
    def load_data(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess IBM HR dataset.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Tuple of (features_df, target_series)
        """
        cache_path = self._get_cache_path(file_path, kwargs)
        
        if self.use_cache and cache_path.exists():
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        try:
            # Load raw data
            df = pd.read_csv(file_path)
            logger.info(f"Loaded IBM HR data: {df.shape}")
            
            # Basic validation
            if 'Attrition' not in df.columns:
                raise DataLoaderError("Target column 'Attrition' not found")
            
            # Drop unnecessary columns
            drop_cols = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
            drop_cols = [col for col in drop_cols if col in df.columns]
            df = df.drop(columns=drop_cols)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Create age buckets for fairness analysis
            df['AgeBucket'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                                   labels=['<30', '30-40', '40-50', '50+'])
            
            # Encode target variable
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
            
            # Split features and target
            target = df['Attrition']
            features = df.drop(columns=['Attrition'])
            
            # Cache processed data
            data = (features, target)
            if self.use_cache:
                self._save_to_cache(data, cache_path)
            
            return data
            
        except Exception as e:
            raise DataLoaderError(f"Error loading IBM HR data: {e}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df


class AdultCensusLoader(BaseDataLoader):
    """Loader for Adult Census Income dataset."""
    
    def load_data(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess Adult Census dataset.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Tuple of (features_df, target_series)
        """
        cache_path = self._get_cache_path(file_path, kwargs)
        
        if self.use_cache and cache_path.exists():
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        try:
            # Load raw data with proper column names
            column_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
            ]
            
            df = pd.read_csv(file_path, names=column_names, skipinitialspace=True)
            logger.info(f"Loaded Adult Census data: {df.shape}")
            
            # Handle missing values (marked as '?')
            df = df.replace('?', np.nan)
            df = self._handle_missing_values(df)
            
            # Drop fnlwgt column as it's not needed for prediction
            if 'fnlwgt' in df.columns:
                df = df.drop(columns=['fnlwgt'])
            
            # Encode target variable
            df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})
            
            # Split features and target
            target = df['income']
            features = df.drop(columns=['income'])
            
            # Cache processed data
            data = (features, target)
            if self.use_cache:
                self._save_to_cache(data, cache_path)
            
            return data
            
        except Exception as e:
            raise DataLoaderError(f"Error loading Adult Census data: {e}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df


class MIMICLoader(BaseDataLoader):
    """Loader for MIMIC-III subset dataset."""
    
    def load_data(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess MIMIC-III subset.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Tuple of (features_df, target_series)
            
        Note:
            Requires credentialed access to PhysioNet MIMIC-III dataset.
            See docs/README.md for access instructions.
        """
        cache_path = self._get_cache_path(file_path, kwargs)
        
        if self.use_cache and cache_path.exists():
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        try:
            # Check if file exists
            if not Path(file_path).exists():
                logger.warning(f"MIMIC-III file not found: {file_path}")
                logger.warning("This dataset requires credentialed access from PhysioNet")
                raise DataLoaderError(f"MIMIC-III dataset not available: {file_path}")
            
            # Load raw data
            df = pd.read_csv(file_path)
            logger.info(f"Loaded MIMIC-III data: {df.shape}")
            
            # Basic validation
            if 'hospital_expire_flag' not in df.columns:
                raise DataLoaderError("Target column 'hospital_expire_flag' not found")
            
            # Drop ID columns
            drop_cols = ['subject_id', 'hadm_id']
            drop_cols = [col for col in drop_cols if col in df.columns]
            df = df.drop(columns=drop_cols)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Split features and target
            target = df['hospital_expire_flag']
            features = df.drop(columns=['hospital_expire_flag'])
            
            # Cache processed data
            data = (features, target)
            if self.use_cache:
                self._save_to_cache(data, cache_path)
            
            return data
            
        except Exception as e:
            raise DataLoaderError(f"Error loading MIMIC-III data: {e}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with medical data considerations."""
        # For medical data, be more conservative with missing value imputation
        
        # Drop rows with too many missing values (>30%)
        missing_threshold = 0.3
        df = df.dropna(thresh=int((1 - missing_threshold) * len(df.columns)))
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if len(df[col].mode()) > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        return df


class SyntheticBiasInjector:
    """Utility for injecting controlled bias into datasets for testing."""
    
    @staticmethod
    def inject_bias(X: pd.DataFrame, y: pd.Series, 
                   protected_attr: str, 
                   bias_ratio: Tuple[float, float] = (0.6, 0.4),
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Inject bias by resampling to achieve specified majority/minority split.
        
        Args:
            X: Feature matrix
            y: Target vector
            protected_attr: Name of protected attribute column
            bias_ratio: Tuple of (majority_ratio, minority_ratio)
            random_state: Random seed for reproducibility
            
        Returns:
            Biased dataset (X, y)
        """
        if protected_attr not in X.columns:
            raise ValueError(f"Protected attribute '{protected_attr}' not found in features")
        
        np.random.seed(random_state)
        
        # Identify protected groups
        protected_values = X[protected_attr].unique()
        if len(protected_values) != 2:
            # For multi-class, create binary groups
            majority_group = X[protected_attr].mode()[0]
            X_binary = X.copy()
            X_binary[f'{protected_attr}_binary'] = (X[protected_attr] == majority_group).astype(int)
            protected_attr = f'{protected_attr}_binary'
            protected_values = [0, 1]
        
        # Calculate target sizes
        total_size = len(X)
        majority_size = int(total_size * bias_ratio[0])
        minority_size = int(total_size * bias_ratio[1])
        
        # Split by protected attribute
        majority_mask = X[protected_attr] == protected_values[0]
        minority_mask = X[protected_attr] == protected_values[1]
        
        X_majority = X[majority_mask]
        y_majority = y[majority_mask]
        X_minority = X[minority_mask]
        y_minority = y[minority_mask]
        
        # Resample to achieve bias ratio
        if len(X_majority) > majority_size:
            sample_idx = np.random.choice(len(X_majority), majority_size, replace=False)
            X_majority = X_majority.iloc[sample_idx]
            y_majority = y_majority.iloc[sample_idx]
        elif len(X_majority) < majority_size:
            sample_idx = np.random.choice(len(X_majority), majority_size, replace=True)
            X_majority = X_majority.iloc[sample_idx]
            y_majority = y_majority.iloc[sample_idx]
        
        if len(X_minority) > minority_size:
            sample_idx = np.random.choice(len(X_minority), minority_size, replace=False)
            X_minority = X_minority.iloc[sample_idx]
            y_minority = y_minority.iloc[sample_idx]
        elif len(X_minority) < minority_size:
            sample_idx = np.random.choice(len(X_minority), minority_size, replace=True)
            X_minority = X_minority.iloc[sample_idx]
            y_minority = y_minority.iloc[sample_idx]
        
        # Combine and shuffle
        X_biased = pd.concat([X_majority, X_minority], ignore_index=True)
        y_biased = pd.concat([y_majority, y_minority], ignore_index=True)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_biased))
        X_biased = X_biased.iloc[shuffle_idx].reset_index(drop=True)
        y_biased = y_biased.iloc[shuffle_idx].reset_index(drop=True)
        
        logger.info(f"Injected bias: {bias_ratio[0]:.1%}/{bias_ratio[1]:.1%} split on {protected_attr}")
        
        return X_biased, y_biased


def load_dataset(dataset_name: str, config: Dict[str, Any], 
                apply_bias: bool = False, 
                bias_ratio: Tuple[float, float] = (0.6, 0.4)) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Universal dataset loader.
    
    Args:
        dataset_name: Name of dataset ('ibm_hr', 'adult', 'mimic', 'synthetic')
        config: Configuration dictionary
        apply_bias: Whether to apply synthetic bias injection
        bias_ratio: Bias ratio for synthetic injection
        
    Returns:
        Tuple of (features, target)
    """
    loaders = {
        'ibm_hr': IBMHRLoader(),
        'adult': AdultCensusLoader(),
        'mimic': MIMICLoader()
    }
    
    if dataset_name not in loaders:
        raise DataLoaderError(f"Unknown dataset: {dataset_name}")
    
    loader = loaders[dataset_name]
    dataset_config = config.get('dataset', {})
    file_path = dataset_config.get('file_path', f'data/raw/{dataset_name}.csv')
    
    # Load data
    X, y = loader.load_data(file_path)
    
    # Apply synthetic bias if requested
    if apply_bias and 'protected_attributes' in dataset_config:
        protected_attr = dataset_config['protected_attributes'][0]  # Use first protected attribute
        X, y = SyntheticBiasInjector.inject_bias(X, y, protected_attr, bias_ratio)
    
    logger.info(f"Dataset loaded: {dataset_name}, shape: {X.shape}")
    return X, y


def create_train_test_split(X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2, 
                          val_size: float = 0.1,
                          random_state: int = 42,
                          stratify: bool = True) -> Dict[str, Any]:
    """
    Create train/validation/test splits.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        Dictionary with train/val/test splits
    """
    stratify_y = y if stratify else None
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    stratify_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=stratify_temp
    )
    
    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
