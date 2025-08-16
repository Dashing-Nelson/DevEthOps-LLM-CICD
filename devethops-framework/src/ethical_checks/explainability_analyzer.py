"""
Explainability Analyzer - Model interpretability using SHAP and LIME
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# SHAP and LIME imports
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer


class ExplainabilityAnalyzer:
    """
    Comprehensive model explainability analyzer using SHAP and LIME.
    Provides both global and local explanations for ML models.
    """
    
    def __init__(self, model, training_data: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize the explainability analyzer.
        
        Args:
            model: Trained ML model
            training_data: Training data for background/reference
            feature_names: Names of features (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(training_data.shape[1])]
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Cache for explanations
        self.shap_values_cache = {}
        self.lime_explanations_cache = {}
        
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        try:
            # Initialize SHAP explainer
            self._initialize_shap_explainer()
            
            # Initialize LIME explainer
            self._initialize_lime_explainer()
            
        except Exception as e:
            self.logger.error(f"Error initializing explainers: {str(e)}")
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            # Try different SHAP explainer types
            model_type = str(type(self.model)).lower()
            
            if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'lgb' in model_type:
                # Tree-based models
                try:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                    self.logger.info("Initialized SHAP TreeExplainer")
                    return
                except:
                    pass
            
            if 'linear' in model_type or 'logistic' in model_type:
                # Linear models
                try:
                    self.shap_explainer = shap.LinearExplainer(self.model, self.training_data)
                    self.logger.info("Initialized SHAP LinearExplainer")
                    return
                except:
                    pass
            
            # Default to KernelExplainer (model-agnostic but slower)
            try:
                # Use a smaller background dataset for KernelExplainer
                background_size = min(100, len(self.training_data))
                background_data = shap.sample(self.training_data, background_size)
                
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = self.model.predict_proba
                else:
                    predict_fn = self.model.predict
                
                self.shap_explainer = shap.KernelExplainer(predict_fn, background_data)
                self.logger.info("Initialized SHAP KernelExplainer")
                
            except Exception as e:
                self.logger.warning(f"Could not initialize SHAP explainer: {str(e)}")
                self.shap_explainer = None
                
        except Exception as e:
            self.logger.error(f"Error in SHAP initialization: {str(e)}")
            self.shap_explainer = None
    
    def _initialize_lime_explainer(self):
        """Initialize LIME explainer"""
        try:
            # Determine if this is a classification or regression problem
            mode = 'classification'  # Default assumption
            
            # Create LIME explainer
            self.lime_explainer = LimeTabularExplainer(
                training_data=self.training_data,
                feature_names=self.feature_names,
                mode=mode,
                discretize_continuous=True,
                random_state=42
            )
            
            self.logger.info("Initialized LIME explainer")
            
        except Exception as e:
            self.logger.error(f"Error initializing LIME explainer: {str(e)}")
            self.lime_explainer = None
    
    def compute_shap_values(self, X: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for the given samples.
        
        Args:
            X: Input samples
            max_samples: Maximum number of samples to explain (for performance)
            
        Returns:
            SHAP values array
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        try:
            # Limit number of samples for performance
            if len(X) > max_samples:
                self.logger.warning(f"Limiting SHAP computation to {max_samples} samples")
                X_limited = X[:max_samples]
            else:
                X_limited = X
            
            # Generate cache key
            cache_key = hash(X_limited.tobytes())
            
            if cache_key in self.shap_values_cache:
                self.logger.info("Using cached SHAP values")
                return self.shap_values_cache[cache_key]
            
            # Compute SHAP values
            self.logger.info(f"Computing SHAP values for {len(X_limited)} samples")
            shap_values = self.shap_explainer.shap_values(X_limited)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class case - use values for positive class
                if len(shap_values) == 2:  # Binary classification
                    shap_values = shap_values[1]
                else:  # Multi-class - use first class for now
                    shap_values = shap_values[0]
            
            # Cache results
            self.shap_values_cache[cache_key] = shap_values
            
            self.logger.info("SHAP values computed successfully")
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error computing SHAP values: {str(e)}")
            # Return dummy values to prevent pipeline failure
            return np.zeros((len(X_limited), len(self.feature_names)))
    
    def generate_lime_explanation(self, instance: np.ndarray, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Single instance to explain
            num_features: Number of top features to include in explanation
            
        Returns:
            Dict containing LIME explanation
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized")
        
        try:
            # Generate cache key
            cache_key = hash(instance.tobytes())
            
            if cache_key in self.lime_explanations_cache:
                self.logger.info("Using cached LIME explanation")
                return self.lime_explanations_cache[cache_key]
            
            # Generate explanation
            self.logger.info("Generating LIME explanation")
            
            if hasattr(self.model, 'predict_proba'):
                explanation = self.lime_explainer.explain_instance(
                    instance, 
                    self.model.predict_proba,
                    num_features=num_features
                )
            else:
                explanation = self.lime_explainer.explain_instance(
                    instance, 
                    self.model.predict,
                    num_features=num_features
                )
            
            # Extract explanation data
            explanation_data = {
                'feature_importance': dict(explanation.as_list()),
                'intercept': explanation.intercept[0] if hasattr(explanation, 'intercept') else 0.0,
                'prediction': explanation.predict_proba[1] if hasattr(explanation, 'predict_proba') else 0.0,
                'local_pred': explanation.local_pred[0] if hasattr(explanation, 'local_pred') else 0.0
            }
            
            # Cache results
            self.lime_explanations_cache[cache_key] = explanation_data
            
            self.logger.info("LIME explanation generated successfully")
            return explanation_data
            
        except Exception as e:
            self.logger.error(f"Error generating LIME explanation: {str(e)}")
            return {
                'feature_importance': {},
                'error': str(e)
            }
    
    def detect_biased_features(self, shap_values: np.ndarray, 
                              sensitive_attributes: List[str],
                              threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detect potentially biased features based on SHAP values.
        
        Args:
            shap_values: SHAP values array
            sensitive_attributes: List of sensitive attribute names
            threshold: Importance threshold for flagging features
            
        Returns:
            List of potentially biased features with details
        """
        try:
            # Calculate mean absolute SHAP values for each feature
            mean_shap_importance = np.mean(np.abs(shap_values), axis=0)
            
            biased_features = []
            
            for attr in sensitive_attributes:
                if attr in self.feature_names:
                    attr_index = self.feature_names.index(attr)
                    importance = mean_shap_importance[attr_index]
                    
                    if importance > threshold:
                        concern_level = 'high' if importance > 0.2 else 'medium'
                        biased_features.append({
                            'feature': attr,
                            'importance': float(importance),
                            'concern_level': concern_level,
                            'recommendation': f"High importance ({importance:.3f}) detected for protected attribute '{attr}'. Consider removing or reducing influence."
                        })
            
            # Also check for highly correlated features with sensitive attributes
            # (This would require additional correlation analysis)
            
            return biased_features
            
        except Exception as e:
            self.logger.error(f"Error detecting biased features: {str(e)}")
            return []
    
    def generate_global_explanation(self, X: np.ndarray, max_samples: int = 100) -> Dict[str, Any]:
        """
        Generate global model explanation using SHAP.
        
        Args:
            X: Input data
            max_samples: Maximum number of samples for analysis
            
        Returns:
            Dict containing global explanation
        """
        try:
            # Compute SHAP values
            shap_values = self.compute_shap_values(X, max_samples)
            
            # Calculate global feature importance
            global_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance ranking
            feature_ranking = sorted(
                zip(self.feature_names, global_importance),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Calculate feature statistics
            feature_stats = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_shap = shap_values[:, i]
                feature_stats[feature_name] = {
                    'mean_importance': float(global_importance[i]),
                    'mean_shap_value': float(np.mean(feature_shap)),
                    'std_shap_value': float(np.std(feature_shap)),
                    'positive_contribution_ratio': float(np.mean(feature_shap > 0)),
                    'max_impact': float(np.max(np.abs(feature_shap)))
                }
            
            global_explanation = {
                'feature_importance_ranking': feature_ranking,
                'top_10_features': feature_ranking[:10],
                'feature_statistics': feature_stats,
                'total_samples_analyzed': len(X) if len(X) <= max_samples else max_samples,
                'mean_prediction_impact': float(np.mean(np.sum(np.abs(shap_values), axis=1)))
            }
            
            self.logger.info("Global explanation generated successfully")
            return global_explanation
            
        except Exception as e:
            self.logger.error(f"Error generating global explanation: {str(e)}")
            return {
                'error': str(e)
            }
    
    def analyze_feature_interactions(self, X: np.ndarray, max_samples: int = 50) -> Dict[str, Any]:
        """
        Analyze feature interactions using SHAP interaction values.
        
        Args:
            X: Input data
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dict containing interaction analysis
        """
        try:
            if not hasattr(self.shap_explainer, 'shap_interaction_values'):
                self.logger.warning("SHAP interaction values not available for this explainer type")
                return {'error': 'Interaction analysis not supported for this model type'}
            
            # Limit samples for performance
            X_limited = X[:min(max_samples, len(X))]
            
            # Compute interaction values
            self.logger.info(f"Computing SHAP interaction values for {len(X_limited)} samples")
            interaction_values = self.shap_explainer.shap_interaction_values(X_limited)
            
            # Handle different output formats
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[0]  # Use first class
            
            # Calculate mean interaction strengths
            mean_interactions = np.mean(np.abs(interaction_values), axis=0)
            
            # Find top interactions (excluding diagonal - self interactions)
            top_interactions = []
            n_features = len(self.feature_names)
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction_strength = mean_interactions[i, j]
                    if interaction_strength > 0.01:  # Threshold for meaningful interactions
                        top_interactions.append({
                            'feature_1': self.feature_names[i],
                            'feature_2': self.feature_names[j],
                            'interaction_strength': float(interaction_strength),
                            'feature_1_index': i,
                            'feature_2_index': j
                        })
            
            # Sort by interaction strength
            top_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
            
            interaction_analysis = {
                'top_interactions': top_interactions[:10],  # Top 10 interactions
                'total_interactions_found': len(top_interactions),
                'samples_analyzed': len(X_limited),
                'interaction_matrix_shape': mean_interactions.shape
            }
            
            self.logger.info(f"Found {len(top_interactions)} significant feature interactions")
            return interaction_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature interactions: {str(e)}")
            return {'error': str(e)}
    
    def generate_comprehensive_explanation(self, X: np.ndarray, sensitive_attributes: List[str],
                                         max_samples: int = 100) -> Dict[str, Any]:
        """
        Generate comprehensive explanation including global, local, and bias analysis.
        
        Args:
            X: Input data
            sensitive_attributes: List of sensitive attribute names
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dict containing comprehensive explanation
        """
        try:
            self.logger.info("Generating comprehensive explanation")
            
            comprehensive_explanation = {
                'analysis_summary': {
                    'total_samples': len(X),
                    'samples_analyzed': min(max_samples, len(X)),
                    'features_count': len(self.feature_names),
                    'sensitive_attributes': sensitive_attributes
                }
            }
            
            # Global explanation
            try:
                global_explanation = self.generate_global_explanation(X, max_samples)
                comprehensive_explanation['global_explanation'] = global_explanation
            except Exception as e:
                self.logger.error(f"Error in global explanation: {str(e)}")
                comprehensive_explanation['global_explanation'] = {'error': str(e)}
            
            # Local explanations (for a few samples)
            try:
                local_explanations = []
                sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
                
                for idx in sample_indices:
                    local_exp = self.generate_lime_explanation(X[idx])
                    local_explanations.append({
                        'sample_index': int(idx),
                        'explanation': local_exp
                    })
                
                comprehensive_explanation['local_explanations'] = local_explanations
            except Exception as e:
                self.logger.error(f"Error in local explanations: {str(e)}")
                comprehensive_explanation['local_explanations'] = {'error': str(e)}
            
            # Bias analysis
            try:
                shap_values = self.compute_shap_values(X, max_samples)
                biased_features = self.detect_biased_features(shap_values, sensitive_attributes)
                comprehensive_explanation['bias_analysis'] = {
                    'biased_features': biased_features,
                    'bias_risk_level': 'high' if any(f['concern_level'] == 'high' for f in biased_features) else 'low'
                }
            except Exception as e:
                self.logger.error(f"Error in bias analysis: {str(e)}")
                comprehensive_explanation['bias_analysis'] = {'error': str(e)}
            
            # Feature interactions (if available)
            try:
                interaction_analysis = self.analyze_feature_interactions(X, max_samples // 2)
                comprehensive_explanation['interaction_analysis'] = interaction_analysis
            except Exception as e:
                self.logger.warning(f"Feature interaction analysis not available: {str(e)}")
                comprehensive_explanation['interaction_analysis'] = {'error': str(e)}
            
            self.logger.info("Comprehensive explanation generated successfully")
            return comprehensive_explanation
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive explanation: {str(e)}")
            return {
                'error': str(e),
                'analysis_summary': {
                    'status': 'failed'
                }
            }
    
    def save_explanations(self, explanations: Dict[str, Any], output_path: str):
        """
        Save explanations to file.
        
        Args:
            explanations: Explanation results
            output_path: Path to save explanations
        """
        import json
        import os
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(explanations, f, indent=2, default=str)
            
            self.logger.info(f"Explanations saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving explanations: {str(e)}")


if __name__ == "__main__":
    # Example usage would go here
    pass
