"""
Explainability analysis using LIME and SHAP for tabular and text data.

Provides local (LIME) and global (SHAP) explanations for model predictions
with support for fairness-aware interpretability.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logging.warning("SHAP not available. SHAP explanations will not be generated.")
    SHAP_AVAILABLE = False

# LIME imports
try:
    import lime
    import lime.lime_tabular
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    logging.warning("LIME not available. LIME explanations will not be generated.")
    LIME_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExplainabilityError(Exception):
    """Custom exception for explainability errors."""
    pass


class TabularExplainer:
    """
    Explainer for tabular data using SHAP and LIME.
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Initialize tabular explainer.
        
        Args:
            model: Trained model with predict/predict_proba methods
            X_train: Training data for baseline/background
            feature_names: Names of features
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or (list(X_train.columns) if hasattr(X_train, 'columns') else None)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        if SHAP_AVAILABLE:
            self._init_shap_explainer()
        
        if LIME_AVAILABLE:
            self._init_lime_explainer()
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer."""
        try:
            # Try different SHAP explainers based on model type
            if hasattr(self.model, 'predict_proba'):
                # For tree-based models
                if hasattr(self.model, 'feature_importances_'):
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # For general models, use Kernel explainer with background data
                    background = shap.sample(self.X_train, 100)  # Sample for efficiency
                    self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
            else:
                # For regression or models without predict_proba
                background = shap.sample(self.X_train, 100)
                self.shap_explainer = shap.KernelExplainer(self.model.predict, background)
            
            logger.info(f"SHAP explainer initialized: {type(self.shap_explainer).__name__}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _init_lime_explainer(self):
        """Initialize LIME explainer."""
        try:
            # Convert training data to numpy if needed
            if hasattr(self.X_train, 'values'):
                training_data = self.X_train.values
            else:
                training_data = self.X_train
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                class_names=['Class 0', 'Class 1'] if hasattr(self.model, 'predict_proba') else None,
                mode='classification' if hasattr(self.model, 'predict_proba') else 'regression',
                discretize_continuous=True
            )
            
            logger.info("LIME explainer initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize LIME explainer: {e}")
            self.lime_explainer = None
    
    def explain_global(self, X_test: pd.DataFrame, max_display: int = 20) -> Dict[str, Any]:
        """
        Generate global explanations using SHAP.
        
        Args:
            X_test: Test data to explain
            max_display: Maximum number of features to display
            
        Returns:
            Dictionary with global explanation results
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            logger.warning("SHAP not available for global explanations")
            return {'error': 'SHAP not available'}
        
        logger.info("Generating global SHAP explanations...")
        
        try:
            # Compute SHAP values
            if hasattr(self.X_train, 'values'):
                X_test_values = X_test.values
            else:
                X_test_values = X_test
            
            # Handle different explainer types
            if isinstance(self.shap_explainer, shap.TreeExplainer):
                shap_values = self.shap_explainer.shap_values(X_test_values)
            else:
                # For KernelExplainer, limit samples for efficiency
                sample_size = min(100, len(X_test))
                X_sample = X_test_values[:sample_size]
                shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            if self.feature_names:
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
            else:
                importance_df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(feature_importance))],
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
            
            return {
                'shap_values': shap_values,
                'feature_importance': importance_df,
                'explainer_type': type(self.shap_explainer).__name__,
                'num_samples_explained': shap_values.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Error generating global explanations: {e}")
            return {'error': str(e)}
    
    def explain_local(self, instance: Union[pd.Series, np.ndarray], 
                     instance_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate local explanation for a single instance.
        
        Args:
            instance: Single instance to explain
            instance_idx: Index of instance (for SHAP)
            
        Returns:
            Dictionary with local explanation results
        """
        results = {}
        
        # SHAP local explanation
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                if hasattr(instance, 'values'):
                    instance_values = instance.values.reshape(1, -1)
                else:
                    instance_values = np.array(instance).reshape(1, -1)
                
                if isinstance(self.shap_explainer, shap.TreeExplainer):
                    shap_values = self.shap_explainer.shap_values(instance_values)
                else:
                    shap_values = self.shap_explainer.shap_values(instance_values)
                
                # Handle multi-class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class
                
                shap_values = shap_values[0]  # Get first (and only) instance
                
                if self.feature_names:
                    shap_explanation = pd.DataFrame({
                        'feature': self.feature_names,
                        'shap_value': shap_values,
                        'feature_value': instance_values[0]
                    }).sort_values('shap_value', key=abs, ascending=False)
                else:
                    shap_explanation = pd.DataFrame({
                        'feature': [f'feature_{i}' for i in range(len(shap_values))],
                        'shap_value': shap_values,
                        'feature_value': instance_values[0]
                    }).sort_values('shap_value', key=abs, ascending=False)
                
                results['shap'] = {
                    'explanation': shap_explanation,
                    'base_value': getattr(self.shap_explainer, 'expected_value', 0),
                    'prediction_impact': shap_values.sum()
                }
                
            except Exception as e:
                logger.error(f"Error generating SHAP local explanation: {e}")
                results['shap'] = {'error': str(e)}
        
        # LIME local explanation
        if LIME_AVAILABLE and self.lime_explainer is not None:
            try:
                if hasattr(instance, 'values'):
                    instance_values = instance.values
                else:
                    instance_values = np.array(instance)
                
                # Get prediction function
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = self.model.predict_proba
                else:
                    predict_fn = self.model.predict
                
                explanation = self.lime_explainer.explain_instance(
                    instance_values, 
                    predict_fn,
                    num_features=min(10, len(instance_values))
                )
                
                # Extract explanation data
                lime_data = explanation.as_list()
                lime_df = pd.DataFrame(lime_data, columns=['feature', 'importance'])
                
                results['lime'] = {
                    'explanation': lime_df,
                    'explanation_object': explanation
                }
                
            except Exception as e:
                logger.error(f"Error generating LIME local explanation: {e}")
                results['lime'] = {'error': str(e)}
        
        return results
    
    def plot_global_importance(self, global_explanation: Dict[str, Any], 
                             output_path: Optional[str] = None,
                             top_k: int = 20) -> plt.Figure:
        """
        Plot global feature importance.
        
        Args:
            global_explanation: Results from explain_global
            output_path: Path to save plot
            top_k: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if 'feature_importance' not in global_explanation:
            raise ExplainabilityError("No feature importance data in global explanation")
        
        importance_df = global_explanation['feature_importance'].head(top_k)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', orient='h')
        plt.title(f'Global Feature Importance (Top {top_k})')
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Global importance plot saved to {output_path}")
        
        return plt.gcf()
    
    def plot_shap_summary(self, global_explanation: Dict[str, Any],
                         X_test: pd.DataFrame,
                         output_path: Optional[str] = None) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            global_explanation: Results from explain_global
            X_test: Test data
            output_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            raise ExplainabilityError("SHAP not available")
        
        if 'shap_values' not in global_explanation:
            raise ExplainabilityError("No SHAP values in global explanation")
        
        shap_values = global_explanation['shap_values']
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        
        if hasattr(X_test, 'values'):
            X_test_values = X_test.values
        else:
            X_test_values = X_test
        
        # Limit to same number of samples as SHAP values
        X_display = X_test_values[:shap_values.shape[0]]
        
        shap.summary_plot(
            shap_values, 
            X_display, 
            feature_names=self.feature_names,
            show=False
        )
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {output_path}")
        
        return plt.gcf()


class TextExplainer:
    """
    Explainer for text data using LIME.
    """
    
    def __init__(self, model, class_names: Optional[List[str]] = None):
        """
        Initialize text explainer.
        
        Args:
            model: Trained text model
            class_names: Names of classes
        """
        self.model = model
        self.class_names = class_names or ['Negative', 'Positive']
        self.lime_explainer = None
        
        if LIME_AVAILABLE:
            self._init_lime_explainer()
    
    def _init_lime_explainer(self):
        """Initialize LIME text explainer."""
        try:
            self.lime_explainer = lime.lime_text.LimeTextExplainer(
                class_names=self.class_names,
                mode='classification'
            )
            logger.info("LIME text explainer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LIME text explainer: {e}")
            self.lime_explainer = None
    
    def explain_text(self, text: str, predict_fn: Callable,
                    num_features: int = 10) -> Dict[str, Any]:
        """
        Explain a single text instance.
        
        Args:
            text: Text to explain
            predict_fn: Function that takes text and returns predictions
            num_features: Number of features to show
            
        Returns:
            Dictionary with explanation results
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {'error': 'LIME not available for text explanation'}
        
        try:
            explanation = self.lime_explainer.explain_instance(
                text, predict_fn, num_features=num_features
            )
            
            # Extract explanation data
            explanation_data = explanation.as_list()
            explanation_df = pd.DataFrame(explanation_data, columns=['word', 'importance'])
            
            return {
                'explanation': explanation_df,
                'explanation_object': explanation,
                'prediction_proba': explanation.predict_proba
            }
            
        except Exception as e:
            logger.error(f"Error explaining text: {e}")
            return {'error': str(e)}


class ExplainabilityAnalyzer:
    """
    Main class for comprehensive explainability analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize explainability analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.explainers = {}
        self.explanations = {}
    
    def analyze_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                     model_type: str = 'tabular', 
                     protected_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive explainability analysis for a model.
        
        Args:
            model: Trained model
            X_train: Training data
            X_test: Test data
            model_type: Type of model ('tabular' or 'text')
            protected_attributes: Protected attributes for fairness analysis
            
        Returns:
            Dictionary with explainability results
        """
        logger.info(f"Starting explainability analysis for {model_type} model...")
        
        results = {
            'model_type': model_type,
            'global_explanation': None,
            'sample_local_explanations': [],
            'fairness_explanations': {},
            'plots': []
        }
        
        if model_type == 'tabular':
            results.update(self._analyze_tabular_model(
                model, X_train, X_test, protected_attributes
            ))
        elif model_type == 'text':
            results.update(self._analyze_text_model(model, X_test))
        
        logger.info("Explainability analysis complete")
        return results
    
    def _analyze_tabular_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              protected_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze tabular model explainability."""
        results = {}
        
        # Initialize explainer
        explainer = TabularExplainer(model, X_train)
        self.explainers['tabular'] = explainer
        
        # Global explanations
        logger.info("Generating global explanations...")
        global_explanation = explainer.explain_global(X_test)
        results['global_explanation'] = global_explanation
        
        # Sample local explanations
        logger.info("Generating sample local explanations...")
        sample_size = min(5, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        
        local_explanations = []
        for idx in sample_indices:
            instance = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
            local_exp = explainer.explain_local(instance, idx)
            local_exp['instance_index'] = idx
            local_explanations.append(local_exp)
        
        results['sample_local_explanations'] = local_explanations
        
        # Fairness-aware explanations
        if protected_attributes:
            logger.info("Generating fairness-aware explanations...")
            fairness_explanations = self._analyze_fairness_explanations(
                explainer, X_test, protected_attributes
            )
            results['fairness_explanations'] = fairness_explanations
        
        return results
    
    def _analyze_text_model(self, model, X_test) -> Dict[str, Any]:
        """Analyze text model explainability."""
        # TODO: Implement text model explainability
        logger.info("Text model explainability analysis not implemented yet")
        return {
            'global_explanation': None,
            'sample_local_explanations': [],
            'note': 'Text explainability analysis to be implemented'
        }
    
    def _analyze_fairness_explanations(self, explainer: TabularExplainer,
                                     X_test: pd.DataFrame,
                                     protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Analyze explanations from a fairness perspective.
        
        Args:
            explainer: Fitted explainer
            X_test: Test data
            protected_attributes: Protected attributes
            
        Returns:
            Dictionary with fairness explanation analysis
        """
        fairness_results = {}
        
        for attr in protected_attributes:
            if attr not in X_test.columns:
                continue
            
            logger.info(f"Analyzing fairness explanations for: {attr}")
            
            # Get unique groups
            groups = X_test[attr].unique()
            group_explanations = {}
            
            for group in groups:
                # Sample instances from this group
                group_mask = X_test[attr] == group
                group_data = X_test[group_mask]
                
                if len(group_data) == 0:
                    continue
                
                # Get sample explanations for this group
                sample_size = min(3, len(group_data))
                sample_indices = np.random.choice(len(group_data), sample_size, replace=False)
                
                group_local_explanations = []
                for idx in sample_indices:
                    instance = group_data.iloc[idx]
                    local_exp = explainer.explain_local(instance)
                    group_local_explanations.append(local_exp)
                
                group_explanations[str(group)] = {
                    'local_explanations': group_local_explanations,
                    'group_size': len(group_data)
                }
            
            fairness_results[attr] = group_explanations
        
        return fairness_results
    
    def generate_explainability_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive explainability report.
        
        Args:
            analysis_results: Results from analyze_model
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "EXPLAINABILITY ANALYSIS REPORT",
            "=" * 60,
            f"Model Type: {analysis_results['model_type']}",
            ""
        ]
        
        # Global explanation summary
        global_exp = analysis_results.get('global_explanation')
        if global_exp and 'feature_importance' in global_exp:
            importance_df = global_exp['feature_importance']
            report_lines.extend([
                "TOP GLOBAL FEATURE IMPORTANCE:",
                ""
            ])
            
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                report_lines.append(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
            report_lines.append("")
        
        # Local explanation summary
        local_exps = analysis_results.get('sample_local_explanations', [])
        if local_exps:
            report_lines.extend([
                f"LOCAL EXPLANATIONS ({len(local_exps)} samples):",
                ""
            ])
            
            for i, local_exp in enumerate(local_exps):
                if 'shap' in local_exp and 'explanation' in local_exp['shap']:
                    shap_exp = local_exp['shap']['explanation']
                    top_feature = shap_exp.iloc[0]
                    report_lines.append(
                        f"  Sample {i+1}: Top feature = {top_feature['feature']} "
                        f"(SHAP: {top_feature['shap_value']:.4f})"
                    )
            
            report_lines.append("")
        
        # Fairness explanation summary
        fairness_exps = analysis_results.get('fairness_explanations', {})
        if fairness_exps:
            report_lines.extend([
                "FAIRNESS EXPLANATION ANALYSIS:",
                ""
            ])
            
            for attr, groups in fairness_exps.items():
                report_lines.append(f"  Protected Attribute: {attr}")
                for group, group_data in groups.items():
                    group_size = group_data['group_size']
                    num_explanations = len(group_data['local_explanations'])
                    report_lines.append(f"    Group {group}: {group_size} instances, {num_explanations} explanations")
                report_lines.append("")
        
        report_lines.extend([
            "=" * 60
        ])
        
        return "\n".join(report_lines)
    
    def save_plots(self, analysis_results: Dict[str, Any], output_dir: str) -> List[str]:
        """
        Save explainability plots.
        
        Args:
            analysis_results: Results from analyze_model
            output_dir: Directory to save plots
            
        Returns:
            List of saved plot paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_plots = []
        
        # Check if we have tabular explainer and global explanation
        if ('tabular' in self.explainers and 
            analysis_results.get('global_explanation') and
            'feature_importance' in analysis_results['global_explanation']):
            
            explainer = self.explainers['tabular']
            global_exp = analysis_results['global_explanation']
            
            # Global importance plot
            try:
                importance_plot_path = output_dir / "global_feature_importance.png"
                explainer.plot_global_importance(global_exp, str(importance_plot_path))
                saved_plots.append(str(importance_plot_path))
            except Exception as e:
                logger.error(f"Error saving global importance plot: {e}")
            
            # SHAP summary plot (if available)
            if SHAP_AVAILABLE and 'shap_values' in global_exp:
                try:
                    # We need the test data for SHAP summary plot
                    # This is a limitation - we'd need to pass X_test to this method
                    logger.info("SHAP summary plot requires test data - skipping for now")
                except Exception as e:
                    logger.error(f"Error saving SHAP summary plot: {e}")
        
        logger.info(f"Saved {len(saved_plots)} explainability plots to {output_dir}")
        return saved_plots


def run_explainability_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               config: Dict[str, Any],
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run comprehensive explainability analysis.
    
    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        config: Configuration dictionary
        output_dir: Directory to save results
        
    Returns:
        Dictionary with analysis results
    """
    # Initialize analyzer
    analyzer = ExplainabilityAnalyzer(config)
    
    # Run analysis
    protected_attributes = config.get('protected_attributes', [])
    results = analyzer.analyze_model(
        model, X_train, X_test, 
        model_type='tabular',
        protected_attributes=protected_attributes
    )
    
    # Generate report
    report = analyzer.generate_explainability_report(results)
    results['report'] = report
    
    # Save plots if output directory specified
    if output_dir:
        saved_plots = analyzer.save_plots(results, output_dir)
        results['saved_plots'] = saved_plots
        
        # Save report
        report_path = Path(output_dir) / "explainability_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Explainability report saved to {report_path}")
    
    return results
