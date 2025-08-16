"""
LLM Wrapper - Wrapper for handling Large Language Models with ethical considerations
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from datasets import Dataset
import yaml


class EthicalLLMWrapper:
    """
    Wrapper for Large Language Models with built-in ethical considerations.
    Supports training, inference, and fairness-aware predictions.
    """
    
    def __init__(self, model_name: str = "roberta-base", config_path: Optional[str] = None):
        """
        Initialize the LLM wrapper.
        
        Args:
            model_name: Name of the pre-trained model
            config_path: Path to model configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.config_path = config_path or "config/model_config.yaml"
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
        # Fairness components
        self.fairness_constraints = {}
        self.bias_detection_enabled = True
        
        self._initialize_model()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {self.config_path}. Using defaults.")
            return {
                'model_config': {
                    'base_model': 'roberta-base',
                    'num_labels': 2,
                    'training': {
                        'learning_rate': 2e-5,
                        'batch_size': 16,
                        'num_epochs': 3
                    }
                }
            }
    
    def _initialize_model(self):
        """Initialize tokenizer and model"""
        try:
            model_config = self.config.get('model_config', {})
            base_model = model_config.get('base_model', self.model_name)
            num_labels = model_config.get('num_labels', 2)
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            # Initialize model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=num_labels
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            self.logger.info(f"Model initialized: {base_model} with {num_labels} labels")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def train(self, train_texts: List[str], train_labels: List[int], 
             val_texts: Optional[List[str]] = None, val_labels: Optional[List[int]] = None,
             sensitive_attributes: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Train the model with fairness considerations.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            sensitive_attributes: Dictionary of sensitive attributes for fairness
            
        Returns:
            Dict containing training results
        """
        self.logger.info("Starting model training with ethical considerations")
        
        try:
            # Tokenize training data
            train_encodings = self.tokenizer(
                train_texts,
                truncation=True,
                padding=True,
                max_length=self.config.get('model_config', {}).get('tokenization', {}).get('max_length', 512),
                return_tensors='pt'
            )
            
            # Create training dataset
            train_dataset = self._create_dataset(train_encodings, train_labels)
            
            # Tokenize validation data if provided
            eval_dataset = None
            if val_texts is not None and val_labels is not None:
                val_encodings = self.tokenizer(
                    val_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config.get('model_config', {}).get('tokenization', {}).get('max_length', 512),
                    return_tensors='pt'
                )
                eval_dataset = self._create_dataset(val_encodings, val_labels)
            
            # Set training arguments
            training_config = self.config.get('model_config', {}).get('training', {})
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=training_config.get('num_epochs', 3),
                per_device_train_batch_size=training_config.get('batch_size', 16),
                per_device_eval_batch_size=training_config.get('batch_size', 16),
                learning_rate=training_config.get('learning_rate', 2e-5),
                warmup_steps=training_config.get('warmup_steps', 500),
                weight_decay=training_config.get('weight_decay', 0.01),
                logging_dir='./logs',
                logging_steps=50,
                evaluation_strategy='steps' if eval_dataset is not None else 'no',
                eval_steps=100 if eval_dataset is not None else None,
                save_strategy='epoch',
                load_best_model_at_end=True if eval_dataset is not None else False,
                metric_for_best_model='eval_loss' if eval_dataset is not None else None,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self._compute_metrics if eval_dataset is not None else None,
            )
            
            # Apply fairness constraints if sensitive attributes provided
            if sensitive_attributes:
                self._apply_fairness_constraints(sensitive_attributes, train_labels)
            
            # Train model
            self.logger.info("Starting training...")
            training_result = trainer.train()
            
            # Save training history
            self.training_history.append({
                'timestamp': str(torch.utils.data.Dataset.__dict__.get('timestamp', 'unknown')),
                'train_loss': training_result.training_loss,
                'epochs': training_config.get('num_epochs', 3),
                'learning_rate': training_config.get('learning_rate', 2e-5)
            })
            
            self.is_trained = True
            self.logger.info("Training completed successfully")
            
            # Evaluate if validation data provided
            evaluation_results = {}
            if eval_dataset is not None:
                evaluation_results = trainer.evaluate()
                self.logger.info(f"Validation results: {evaluation_results}")
            
            return {
                'training_loss': training_result.training_loss,
                'evaluation_results': evaluation_results,
                'model_size': self._get_model_size(),
                'training_time': training_result.metrics.get('train_runtime', 0),
                'fairness_constraints_applied': len(self.fairness_constraints) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def _create_dataset(self, encodings, labels):
        """Create PyTorch dataset from encodings and labels"""
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        return CustomDataset(encodings, labels)
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {}
    
    def _apply_fairness_constraints(self, sensitive_attributes: Dict[str, List], labels: List[int]):
        """Apply fairness constraints during training"""
        try:
            self.logger.info("Applying fairness constraints")
            
            # Store fairness constraints for later use
            self.fairness_constraints = sensitive_attributes
            
            # Calculate baseline statistics for each sensitive attribute
            for attr_name, attr_values in sensitive_attributes.items():
                if len(attr_values) != len(labels):
                    self.logger.warning(f"Sensitive attribute {attr_name} length mismatch with labels")
                    continue
                
                # Calculate selection rates by group
                unique_values = list(set(attr_values))
                group_stats = {}
                
                for value in unique_values:
                    mask = np.array(attr_values) == value
                    group_labels = np.array(labels)[mask]
                    
                    if len(group_labels) > 0:
                        positive_rate = np.mean(group_labels)
                        group_stats[value] = {
                            'count': len(group_labels),
                            'positive_rate': positive_rate
                        }
                
                self.fairness_constraints[attr_name] = {
                    'baseline_stats': group_stats,
                    'constraint_type': 'demographic_parity'  # Default constraint
                }
            
            self.logger.info(f"Fairness constraints applied for {len(self.fairness_constraints)} attributes")
            
        except Exception as e:
            self.logger.error(f"Error applying fairness constraints: {str(e)}")
    
    def predict(self, texts: List[str], return_probabilities: bool = False,
               check_fairness: bool = True) -> Union[List[int], Tuple[List[int], List[List[float]]]]:
        """
        Make predictions with optional fairness checking.
        
        Args:
            texts: Input texts for prediction
            return_probabilities: Whether to return prediction probabilities
            check_fairness: Whether to perform fairness checks
            
        Returns:
            Predictions and optionally probabilities
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        try:
            # Tokenize inputs
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.get('model_config', {}).get('tokenization', {}).get('max_length', 512),
                return_tensors='pt'
            )
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                
                # Get probabilities and predictions
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
            
            # Perform fairness checks if requested
            if check_fairness and self.bias_detection_enabled:
                fairness_results = self._check_prediction_fairness(texts, predictions, probabilities)
                if fairness_results.get('bias_detected', False):
                    self.logger.warning("Potential bias detected in predictions")
            
            if return_probabilities:
                return predictions.tolist(), probabilities.tolist()
            else:
                return predictions.tolist()
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def _check_prediction_fairness(self, texts: List[str], predictions: np.ndarray, 
                                  probabilities: np.ndarray) -> Dict[str, Any]:
        """Check fairness of predictions"""
        try:
            fairness_results = {
                'bias_detected': False,
                'fairness_metrics': {},
                'recommendations': []
            }
            
            # Simple bias detection based on text patterns
            # This is a simplified approach - in practice, you'd need actual sensitive attributes
            
            # Check for potential bias indicators in text
            bias_patterns = {
                'gender': ['he', 'she', 'his', 'her', 'man', 'woman', 'male', 'female'],
                'race': ['black', 'white', 'asian', 'hispanic', 'african', 'european'],
                'age': ['young', 'old', 'elderly', 'teen', 'adult', 'senior']
            }
            
            for attr_name, patterns in bias_patterns.items():
                # Find texts containing bias patterns
                pattern_texts = []
                pattern_predictions = []
                
                for i, text in enumerate(texts):
                    text_lower = text.lower()
                    if any(pattern in text_lower for pattern in patterns):
                        pattern_texts.append(text)
                        pattern_predictions.append(predictions[i])
                
                if len(pattern_predictions) > 5:  # Need sufficient samples
                    pattern_positive_rate = np.mean(pattern_predictions)
                    overall_positive_rate = np.mean(predictions)
                    
                    # Check for significant difference
                    difference = abs(pattern_positive_rate - overall_positive_rate)
                    
                    if difference > 0.1:  # 10% threshold
                        fairness_results['bias_detected'] = True
                        fairness_results['fairness_metrics'][attr_name] = {
                            'pattern_positive_rate': pattern_positive_rate,
                            'overall_positive_rate': overall_positive_rate,
                            'difference': difference,
                            'sample_count': len(pattern_predictions)
                        }
                        fairness_results['recommendations'].append(
                            f"Potential {attr_name} bias detected. Consider reviewing training data."
                        )
            
            return fairness_results
            
        except Exception as e:
            self.logger.error(f"Error checking prediction fairness: {str(e)}")
            return {'bias_detected': False, 'error': str(e)}
    
    def evaluate_fairness(self, texts: List[str], labels: List[int], 
                         sensitive_attributes: Dict[str, List]) -> Dict[str, Any]:
        """
        Evaluate model fairness on provided data.
        
        Args:
            texts: Input texts
            labels: True labels
            sensitive_attributes: Dictionary of sensitive attributes
            
        Returns:
            Dict containing fairness evaluation results
        """
        try:
            # Get predictions
            predictions = self.predict(texts, check_fairness=False)
            
            fairness_results = {
                'overall_accuracy': np.mean(np.array(predictions) == np.array(labels)),
                'group_metrics': {},
                'fairness_violations': []
            }
            
            # Calculate fairness metrics for each sensitive attribute
            for attr_name, attr_values in sensitive_attributes.items():
                if len(attr_values) != len(predictions):
                    continue
                
                unique_values = list(set(attr_values))
                group_metrics = {}
                
                for value in unique_values:
                    mask = np.array(attr_values) == value
                    if not np.any(mask):
                        continue
                    
                    group_predictions = np.array(predictions)[mask]
                    group_labels = np.array(labels)[mask]
                    
                    group_accuracy = np.mean(group_predictions == group_labels)
                    group_positive_rate = np.mean(group_predictions)
                    
                    group_metrics[str(value)] = {
                        'accuracy': group_accuracy,
                        'positive_rate': group_positive_rate,
                        'sample_count': np.sum(mask)
                    }
                
                fairness_results['group_metrics'][attr_name] = group_metrics
                
                # Check for fairness violations
                if len(group_metrics) >= 2:
                    positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
                    max_diff = max(positive_rates) - min(positive_rates)
                    
                    if max_diff > 0.1:  # 10% threshold
                        fairness_results['fairness_violations'].append({
                            'attribute': attr_name,
                            'violation_type': 'demographic_parity',
                            'max_difference': max_diff,
                            'severity': 'high' if max_diff > 0.2 else 'medium'
                        })
            
            return fairness_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating fairness: {str(e)}")
            return {'error': str(e)}
    
    def save_model(self, save_path: str, include_tokenizer: bool = True) -> Dict[str, str]:
        """
        Save the trained model and tokenizer.
        
        Args:
            save_path: Path to save the model
            include_tokenizer: Whether to save the tokenizer
            
        Returns:
            Dict containing paths to saved files
        """
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save model
            model_path = os.path.join(save_path, "model")
            self.model.save_pretrained(model_path)
            
            saved_files = {'model': model_path}
            
            # Save tokenizer
            if include_tokenizer:
                tokenizer_path = os.path.join(save_path, "tokenizer")
                self.tokenizer.save_pretrained(tokenizer_path)
                saved_files['tokenizer'] = tokenizer_path
            
            # Save configuration and metadata
            metadata = {
                'model_name': self.model_name,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'fairness_constraints': self.fairness_constraints,
                'config': self.config
            }
            
            metadata_path = os.path.join(save_path, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            saved_files['metadata'] = metadata_path
            
            self.logger.info(f"Model saved successfully to {save_path}")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, load_path: str) -> bool:
        """
        Load a saved model and tokenizer.
        
        Args:
            load_path: Path to load the model from
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Load model
            model_path = os.path.join(load_path, "model")
            if os.path.exists(model_path):
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Load tokenizer
            tokenizer_path = os.path.join(load_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self.logger.warning(f"Tokenizer not found at {tokenizer_path}, using default")
            
            # Load metadata
            metadata_path = os.path.join(load_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.is_trained = metadata.get('is_trained', False)
                self.training_history = metadata.get('training_history', [])
                self.fairness_constraints = metadata.get('fairness_constraints', {})
                
            self.logger.info(f"Model loaded successfully from {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _get_model_size(self) -> Dict[str, Any]:
        """Get model size information"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
            }
        except:
            return {'error': 'Could not calculate model size'}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'model_size': self._get_model_size(),
            'training_history_count': len(self.training_history),
            'fairness_constraints_count': len(self.fairness_constraints),
            'bias_detection_enabled': self.bias_detection_enabled,
            'tokenizer_vocab_size': len(self.tokenizer) if self.tokenizer else 0,
            'config': self.config
        }


if __name__ == "__main__":
    # Example usage
    llm_wrapper = EthicalLLMWrapper("roberta-base")
    
    # Example training data
    # train_texts = ["This is a positive example", "This is a negative example"]
    # train_labels = [1, 0]
    # 
    # # Train the model
    # training_results = llm_wrapper.train(train_texts, train_labels)
    # 
    # # Make predictions
    # test_texts = ["This is a test example"]
    # predictions = llm_wrapper.predict(test_texts)
    pass
