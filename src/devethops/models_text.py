"""
Text classification models using RoBERTa-base and Hugging Face Transformers.

Supports fine-tuning for classification tasks with fairness considerations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import torch
from dataclasses import dataclass

# Hugging Face imports
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
        pipeline
    )
    from datasets import Dataset, DatasetDict
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("Transformers library not available. Text models will not be supported.")
    TRANSFORMERS_AVAILABLE = False

# Scikit-learn for metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

logger = logging.getLogger(__name__)


class TextModelError(Exception):
    """Custom exception for text model errors."""
    pass


@dataclass
class TextModelConfig:
    """Configuration for text model training."""
    model_name: str = "roberta-base"
    num_labels: int = 2
    max_length: int = 512
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    fp16: bool = False  # Set to True if GPU supports it
    gradient_accumulation_steps: int = 1


class TextDataProcessor:
    """
    Processor for text data to prepare for model training.
    """
    
    def __init__(self, tokenizer_name: str = "roberta-base", max_length: int = 512):
        """
        Initialize text data processor.
        
        Args:
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
        """
        if not TRANSFORMERS_AVAILABLE:
            raise TextModelError("Transformers library not available")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with tokenized inputs
        """
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def create_dataset(self, texts: List[str], labels: Optional[List[int]] = None) -> Dataset:
        """
        Create Hugging Face Dataset from texts and labels.
        
        Args:
            texts: List of text strings
            labels: List of labels (optional for inference)
            
        Returns:
            Hugging Face Dataset
        """
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Prepare dataset dict
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        if labels is not None:
            dataset_dict['labels'] = torch.tensor(labels)
        
        return Dataset.from_dict(dataset_dict)


class TextModelTrainer:
    """
    Trainer for text classification models using RoBERTa and similar.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text model trainer.
        
        Args:
            config: Configuration dictionary
        """
        if not TRANSFORMERS_AVAILABLE:
            raise TextModelError("Transformers library not available")
        
        self.config = config
        self.model_config = TextModelConfig(**config.get('text_model', {}))
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.data_processor = None
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Enable FP16 if GPU is available and supports it
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.model_config.fp16 = True
    
    def prepare_model(self, num_labels: int = 2) -> None:
        """
        Prepare model and tokenizer.
        
        Args:
            num_labels: Number of classification labels
        """
        logger.info(f"Loading model: {self.model_config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name,
            num_labels=num_labels
        )
        
        # Initialize data processor
        self.data_processor = TextDataProcessor(
            self.model_config.model_name,
            self.model_config.max_length
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def train_model(self, train_texts: List[str], train_labels: List[int],
                   val_texts: Optional[List[str]] = None,
                   val_labels: Optional[List[int]] = None,
                   output_dir: str = "text_model_output") -> Dict[str, Any]:
        """
        Train text classification model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            output_dir: Directory to save model outputs
            
        Returns:
            Training results
        """
        logger.info("Starting text model training...")
        
        if self.model is None:
            self.prepare_model(num_labels=len(set(train_labels)))
        
        # Create datasets
        train_dataset = self.data_processor.create_dataset(train_texts, train_labels)
        
        eval_dataset = None
        if val_texts is not None and val_labels is not None:
            eval_dataset = self.data_processor.create_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.model_config.num_epochs,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            learning_rate=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
            warmup_steps=self.model_config.warmup_steps,
            logging_steps=self.model_config.logging_steps,
            evaluation_strategy=self.model_config.evaluation_strategy,
            save_strategy=self.model_config.save_strategy,
            save_total_limit=self.model_config.save_total_limit,
            load_best_model_at_end=self.model_config.load_best_model_at_end,
            metric_for_best_model=self.model_config.metric_for_best_model,
            greater_is_better=self.model_config.greater_is_better,
            fp16=self.model_config.fp16,
            gradient_accumulation_steps=self.model_config.gradient_accumulation_steps,
            dataloader_pin_memory=False,  # May help with some GPU issues
            report_to=None  # Disable wandb/tensorboard logging
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.model_config.early_stopping_patience)]
        )
        
        # Train model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Training results
        results = {
            'train_result': train_result,
            'model_path': output_dir,
            'training_args': training_args.to_dict(),
            'model_config': self.model_config.__dict__
        }
        
        logger.info("Text model training complete")
        return results
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute metrics for evaluation.
        
        Args:
            eval_pred: Predictions and labels from trainer
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def evaluate_model(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, Any]:
        """
        Evaluate trained model on test data.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Evaluation results
        """
        if self.trainer is None:
            raise TextModelError("Model not trained. Train model first.")
        
        logger.info("Evaluating text model...")
        
        # Create test dataset
        test_dataset = self.data_processor.create_dataset(test_texts, test_labels)
        
        # Evaluate
        eval_results = self.trainer.evaluate(test_dataset)
        
        # Get predictions for additional metrics
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
        
        # Additional metrics
        roc_auc = roc_auc_score(test_labels, y_pred_proba)
        
        results = {
            'eval_results': eval_results,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'roc_auc': roc_auc,
            'test_labels': test_labels
        }
        
        logger.info(f"Text model evaluation complete. F1: {eval_results.get('eval_f1', 0):.4f}")
        return results
    
    def predict_texts(self, texts: List[str]) -> Dict[str, Any]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            Predictions and probabilities
        """
        if self.trainer is None:
            raise TextModelError("Model not trained. Train model first.")
        
        # Create dataset without labels
        dataset = self.data_processor.create_dataset(texts)
        
        # Make predictions
        predictions = self.trainer.predict(dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confidence_scores': np.max(y_pred_proba, axis=1)
        }
    
    def save_model(self, output_dir: str) -> None:
        """
        Save trained model and tokenizer.
        
        Args:
            output_dir: Directory to save model
        """
        if self.trainer is None:
            raise TextModelError("Model not trained. Train model first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Save configuration
        config_path = output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.model_config.__dict__, f, indent=2)
        
        logger.info(f"Text model saved to {output_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """
        Load pre-trained model.
        
        Args:
            model_dir: Directory containing saved model
        """
        model_dir = Path(model_dir)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        
        # Load configuration if available
        config_path = model_dir / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.model_config = TextModelConfig(**config_dict)
        
        # Recreate data processor
        self.data_processor = TextDataProcessor(
            str(model_dir),
            self.model_config.max_length
        )
        
        logger.info(f"Text model loaded from {model_dir}")


class TextClassificationPipeline:
    """
    High-level pipeline for text classification with fairness considerations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text classification pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.trainer = TextModelTrainer(config)
        self.pipeline = None
    
    def train_pipeline(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None,
                      text_column: str = 'text', label_column: str = 'label',
                      output_dir: str = "text_model_output") -> Dict[str, Any]:
        """
        Train complete text classification pipeline.
        
        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame (optional)
            text_column: Name of text column
            label_column: Name of label column
            output_dir: Output directory for model
            
        Returns:
            Training results
        """
        logger.info("Training text classification pipeline...")
        
        # Prepare training data
        train_texts = train_data[text_column].tolist()
        train_labels = train_data[label_column].tolist()
        
        val_texts = None
        val_labels = None
        if val_data is not None:
            val_texts = val_data[text_column].tolist()
            val_labels = val_data[label_column].tolist()
        
        # Train model
        results = self.trainer.train_model(
            train_texts, train_labels, val_texts, val_labels, output_dir
        )
        
        # Create inference pipeline
        self.pipeline = pipeline(
            "text-classification",
            model=self.trainer.model,
            tokenizer=self.trainer.tokenizer,
            return_all_scores=True
        )
        
        return results
    
    def evaluate_pipeline(self, test_data: pd.DataFrame,
                         text_column: str = 'text', label_column: str = 'label') -> Dict[str, Any]:
        """
        Evaluate pipeline on test data.
        
        Args:
            test_data: Test data DataFrame
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Evaluation results
        """
        test_texts = test_data[text_column].tolist()
        test_labels = test_data[label_column].tolist()
        
        return self.trainer.evaluate_model(test_texts, test_labels)
    
    def predict_pipeline(self, texts: List[str]) -> Dict[str, Any]:
        """
        Make predictions using pipeline.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Predictions
        """
        if self.pipeline is None:
            return self.trainer.predict_texts(texts)
        
        # Use Hugging Face pipeline for inference
        results = self.pipeline(texts)
        
        # Extract predictions and probabilities
        predictions = []
        probabilities = []
        
        for result in results:
            # Assuming binary classification
            pos_score = next((r['score'] for r in result if r['label'] == 'LABEL_1'), 0.5)
            neg_score = next((r['score'] for r in result if r['label'] == 'LABEL_0'), 0.5)
            
            predictions.append(1 if pos_score > neg_score else 0)
            probabilities.append([neg_score, pos_score])
        
        return {
            'predictions': np.array(predictions),
            'probabilities': np.array(probabilities),
            'confidence_scores': np.max(probabilities, axis=1)
        }


def train_text_model(train_data: pd.DataFrame, config: Dict[str, Any],
                    val_data: Optional[pd.DataFrame] = None,
                    text_column: str = 'text', label_column: str = 'label',
                    output_dir: str = "text_model_output") -> Dict[str, Any]:
    """
    High-level function to train a text classification model.
    
    Args:
        train_data: Training data
        config: Configuration dictionary
        val_data: Validation data (optional)
        text_column: Name of text column
        label_column: Name of label column
        output_dir: Output directory
        
    Returns:
        Training results
    """
    if not TRANSFORMERS_AVAILABLE:
        raise TextModelError("Transformers library not available for text models")
    
    pipeline = TextClassificationPipeline(config)
    results = pipeline.train_pipeline(
        train_data, val_data, text_column, label_column, output_dir
    )
    
    # Add pipeline to results for further use
    results['pipeline'] = pipeline
    
    return results


def create_text_explainer(model_path: str, texts: List[str]) -> Dict[str, Any]:
    """
    Create explainer for text model predictions.
    
    Args:
        model_path: Path to trained model
        texts: Sample texts for explanation
        
    Returns:
        Text explanations (placeholder for now)
    """
    # TODO: Implement text model explainability
    logger.info("Text model explainability not implemented yet")
    
    return {
        'explainer_type': 'text_lime',
        'sample_explanations': [],
        'status': 'not_implemented'
    }
