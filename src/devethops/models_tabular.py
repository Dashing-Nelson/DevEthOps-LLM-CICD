"""
Tabular machine learning models for DevEthOps pipeline.

Supports LogisticRegression, RandomForest, XGBoost with fairness-aware training.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
import joblib
import json

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.calibration import calibration_curve
import warnings

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logging.warning("XGBoost not available. XGBoost models will not be supported.")
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Custom exception for model training errors."""
    pass


class TabularModelTrainer:
    """
    Trainer for tabular machine learning models with fairness considerations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.model_type = None
        self.training_history = {}
        self.feature_names = None
        
    def train_logistic_regression(self, X_train: Union[pd.DataFrame, np.ndarray],
                                 y_train: Union[pd.Series, np.ndarray],
                                 sample_weight: Optional[np.ndarray] = None,
                                 hyperparameter_search: bool = True) -> Dict[str, Any]:
        """
        Train logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            sample_weight: Sample weights for fairness
            hyperparameter_search: Whether to perform hyperparameter tuning
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Logistic Regression model...")
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        
        # Configure model
        base_params = {
            'random_state': self.config.get('random_state', 42),
            'max_iter': self.config.get('max_iter', 1000),
            'solver': 'liblinear'  # Good for small datasets
        }
        
        if hyperparameter_search:
            # Hyperparameter search
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(**base_params),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train, sample_weight=sample_weight)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
        else:
            # Use default parameters
            self.model = LogisticRegression(**base_params)
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            best_params = base_params
        
        self.model_type = 'logistic_regression'
        
        # Training results
        results = {
            'model_type': 'logistic_regression',
            'best_params': best_params,
            'feature_names': self.feature_names,
            'model_object': self.model
        }
        
        # Add feature importance (coefficients)
        if self.feature_names:
            coef = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coef,
                'abs_coefficient': np.abs(coef)
            }).sort_values('abs_coefficient', ascending=False)
            results['feature_importance'] = feature_importance
        
        logger.info("Logistic Regression training complete")
        return results
    
    def train_random_forest(self, X_train: Union[pd.DataFrame, np.ndarray],
                           y_train: Union[pd.Series, np.ndarray],
                           sample_weight: Optional[np.ndarray] = None,
                           hyperparameter_search: bool = True) -> Dict[str, Any]:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            sample_weight: Sample weights for fairness
            hyperparameter_search: Whether to perform hyperparameter tuning
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Random Forest model...")
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        
        # Configure model
        base_params = {
            'random_state': self.config.get('random_state', 42),
            'n_jobs': -1
        }
        
        if hyperparameter_search:
            # Hyperparameter search
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(**base_params),
                param_grid,
                cv=3,  # Reduced CV for faster training
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train, sample_weight=sample_weight)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
        else:
            # Use reasonable default parameters
            default_params = {
                **base_params,
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
            self.model = RandomForestClassifier(**default_params)
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            best_params = default_params
        
        self.model_type = 'random_forest'
        
        # Training results
        results = {
            'model_type': 'random_forest',
            'best_params': best_params,
            'feature_names': self.feature_names,
            'model_object': self.model
        }
        
        # Add feature importance
        if self.feature_names:
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance
        
        logger.info("Random Forest training complete")
        return results
    
    def train_xgboost(self, X_train: Union[pd.DataFrame, np.ndarray],
                     y_train: Union[pd.Series, np.ndarray],
                     sample_weight: Optional[np.ndarray] = None,
                     hyperparameter_search: bool = True) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            sample_weight: Sample weights for fairness
            hyperparameter_search: Whether to perform hyperparameter tuning
            
        Returns:
            Training results dictionary
        """
        if not XGBOOST_AVAILABLE:
            raise ModelError("XGBoost not available. Please install xgboost package.")
        
        logger.info("Training XGBoost model...")
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        
        # Configure model
        base_params = {
            'random_state': self.config.get('random_state', 42),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        
        if hyperparameter_search:
            # Hyperparameter search
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            grid_search = GridSearchCV(
                xgb.XGBClassifier(**base_params),
                param_grid,
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train, sample_weight=sample_weight)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
        else:
            # Use reasonable default parameters
            default_params = {
                **base_params,
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.9
            }
            self.model = xgb.XGBClassifier(**default_params)
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            best_params = default_params
        
        self.model_type = 'xgboost'
        
        # Training results
        results = {
            'model_type': 'xgboost',
            'best_params': best_params,
            'feature_names': self.feature_names,
            'model_object': self.model
        }
        
        # Add feature importance
        if self.feature_names:
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance
        
        logger.info("XGBoost training complete")
        return results
    
    def evaluate_model(self, X_test: Union[pd.DataFrame, np.ndarray],
                      y_test: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ModelError("No model trained. Train a model first.")
        
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Model evaluation complete. F1-score: {metrics['f1_score']:.4f}")
        return metrics
    
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        if self.model is None:
            raise ModelError("No model trained. Train a model first.")
        
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='f1')
        
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_folds': cv_folds
        }
        
        logger.info(f"Cross-validation complete. Mean F1: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        return results
    
    def save_model(self, filepath: str, include_metadata: bool = True) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
            include_metadata: Whether to save metadata alongside model
        """
        if self.model is None:
            raise ModelError("No model to save. Train a model first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, filepath)
        
        if include_metadata:
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'config': self.config,
                'training_timestamp': pd.Timestamp.now().isoformat()
            }
            
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        filepath = Path(filepath)
        
        # Load model
        self.model = joblib.load(filepath)
        
        # Load metadata if available
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.model_type = metadata.get('model_type')
            self.feature_names = metadata.get('feature_names')
        
        logger.info(f"Model loaded from {filepath}")


def train_tabular_model(model_type: str, X_train: Union[pd.DataFrame, np.ndarray],
                       y_train: Union[pd.Series, np.ndarray],
                       config: Dict[str, Any],
                       sample_weight: Optional[np.ndarray] = None,
                       hyperparameter_search: bool = True) -> Dict[str, Any]:
    """
    Train a tabular model with specified configuration.
    
    Args:
        model_type: Type of model ('logistic_regression', 'random_forest', 'xgboost')
        X_train: Training features
        y_train: Training targets
        config: Configuration dictionary
        sample_weight: Sample weights for fairness
        hyperparameter_search: Whether to perform hyperparameter tuning
        
    Returns:
        Training results
    """
    trainer = TabularModelTrainer(config)
    
    if model_type == 'logistic_regression':
        results = trainer.train_logistic_regression(
            X_train, y_train, sample_weight, hyperparameter_search
        )
    elif model_type == 'random_forest':
        results = trainer.train_random_forest(
            X_train, y_train, sample_weight, hyperparameter_search
        )
    elif model_type == 'xgboost':
        results = trainer.train_xgboost(
            X_train, y_train, sample_weight, hyperparameter_search
        )
    else:
        raise ModelError(f"Unknown model type: {model_type}")
    
    # Add trainer to results for further use
    results['trainer'] = trainer
    
    return results


def evaluate_tabular_models(models: Dict[str, Any], X_test: Union[pd.DataFrame, np.ndarray],
                           y_test: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Evaluate multiple trained models and compare performance.
    
    Args:
        models: Dictionary of model_name -> training_results
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Comparison results
    """
    logger.info("Evaluating and comparing multiple models...")
    
    results = {
        'individual_results': {},
        'comparison': None
    }
    
    evaluation_results = []
    
    for model_name, model_info in models.items():
        if 'trainer' in model_info:
            trainer = model_info['trainer']
            metrics = trainer.evaluate_model(X_test, y_test)
            metrics['model_name'] = model_name
            metrics['model_type'] = model_info['model_type']
            
            results['individual_results'][model_name] = metrics
            evaluation_results.append(metrics)
    
    if evaluation_results:
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(evaluation_results)
        comparison_df = comparison_df.set_index('model_name')
        results['comparison'] = comparison_df
        
        # Find best model
        best_model_idx = comparison_df['f1_score'].idxmax()
        results['best_model'] = best_model_idx
        results['best_f1_score'] = comparison_df.loc[best_model_idx, 'f1_score']
        
        logger.info(f"Best model: {best_model_idx} (F1: {results['best_f1_score']:.4f})")
    
    return results


def create_model_ensemble(models: Dict[str, Any], method: str = 'voting') -> Dict[str, Any]:
    """
    Create ensemble of trained models.
    
    Args:
        models: Dictionary of trained models
        method: Ensemble method ('voting', 'stacking')
        
    Returns:
        Ensemble model information
    """
    # TODO: Implement model ensemble functionality
    logger.info(f"Model ensemble creation ({method}) not implemented yet")
    
    return {
        'ensemble_method': method,
        'base_models': list(models.keys()),
        'status': 'not_implemented'
    }
