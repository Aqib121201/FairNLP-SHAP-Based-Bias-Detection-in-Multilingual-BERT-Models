"""
Model training module for FairNLP project.

This module handles training of BERT models for sentiment analysis and
translation tasks with integrated fairness metrics.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM, TrainingArguments, Trainer,
    get_linear_schedule_with_warmup, AdamW
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .config import Config

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        """Initialize the dataset."""
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get text and label
        text = str(row['text'])
        label = int(row['label'])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TranslationDataset(Dataset):
    """Dataset class for translation tasks."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        """Initialize the dataset."""
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get source and target texts
        source_text = str(row['source_text'])
        target_text = str(row['target_text'])
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

class FairSentimentModel(nn.Module):
    """Fair sentiment analysis model with demographic awareness."""
    
    def __init__(self, model_name: str, num_labels: int, num_demographic_features: int = 0):
        """Initialize the model."""
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Demographic feature processing
        if num_demographic_features > 0:
            self.demographic_encoder = nn.Linear(num_demographic_features, 64)
            self.combined_classifier = nn.Linear(self.bert.config.hidden_size + 64, num_labels)
        else:
            self.demographic_encoder = None
            self.combined_classifier = None
    
    def forward(self, input_ids, attention_mask, demographic_features=None):
        """Forward pass."""
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Handle demographic features if provided
        if demographic_features is not None and self.demographic_encoder is not None:
            demo_encoded = self.demographic_encoder(demographic_features)
            demo_encoded = F.relu(demo_encoded)
            combined = torch.cat([pooled_output, demo_encoded], dim=1)
            logits = self.combined_classifier(combined)
        else:
            logits = self.classifier(pooled_output)
        
        return logits

class FairTranslationModel(nn.Module):
    """Fair translation model with demographic awareness."""
    
    def __init__(self, model_name: str, num_demographic_features: int = 0):
        """Initialize the model."""
        super().__init__()
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Demographic feature processing
        if num_demographic_features > 0:
            self.demographic_encoder = nn.Linear(num_demographic_features, 64)
            self.demographic_projection = nn.Linear(64, self.transformer.config.hidden_size)
        else:
            self.demographic_encoder = None
            self.demographic_projection = None
    
    def forward(self, input_ids, attention_mask, labels=None, demographic_features=None):
        """Forward pass."""
        # Standard transformer forward pass
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Integrate demographic features if provided
        if demographic_features is not None and self.demographic_encoder is not None:
            demo_encoded = self.demographic_encoder(demographic_features)
            demo_encoded = F.relu(demo_encoded)
            demo_projected = self.demographic_projection(demo_encoded)
            
            # Add demographic information to encoder outputs
            if hasattr(outputs, 'encoder_last_hidden_state'):
                outputs.encoder_last_hidden_state += demo_projected.unsqueeze(1)
        
        return outputs

class FairnessAwareTrainer(Trainer):
    """Custom trainer with fairness-aware training."""
    
    def __init__(self, *args, fairness_weight: float = 0.1, **kwargs):
        """Initialize the trainer."""
        super().__init__(*args, **kwargs)
        self.fairness_weight = fairness_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with fairness penalty."""
        # Standard loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add fairness penalty if demographic features are available
        if 'demographic_features' in inputs:
            demographic_features = inputs['demographic_features']
            fairness_penalty = self.compute_fairness_penalty(outputs.logits, demographic_features)
            loss += self.fairness_weight * fairness_penalty
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_fairness_penalty(self, logits, demographic_features):
        """Compute fairness penalty based on demographic parity."""
        # Group predictions by demographic features
        predictions = torch.argmax(logits, dim=1)
        
        # Compute demographic parity violation
        penalty = 0.0
        for i in range(demographic_features.shape[1]):
            group_mask = demographic_features[:, i] == 1
            if group_mask.sum() > 0:
                group_pred = predictions[group_mask].float().mean()
                overall_pred = predictions.float().mean()
                penalty += (group_pred - overall_pred) ** 2
        
        return penalty

class ModelTrainer:
    """Main model trainer class."""
    
    def __init__(self, config: Config):
        """Initialize the trainer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models and tokenizers
        self.models = {}
        self.tokenizers = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models and tokenizers for each language."""
        model_configs = {
            'mbert': self.config.model.mbert_model,
            'english': self.config.model.english_bert,
            'german': self.config.model.german_bert,
            'hindi': self.config.model.hindi_bert
        }
        
        for name, model_name in model_configs.items():
            try:
                # Initialize tokenizer
                self.tokenizers[name] = AutoTokenizer.from_pretrained(model_name)
                
                # Initialize model
                if name == 'mbert':
                    # Use multilingual model for all languages
                    self.models[name] = AutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=3  # Positive, Negative, Neutral
                    )
                else:
                    # Language-specific models
                    self.models[name] = AutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=3
                    )
                
                self.logger.info(f"Initialized {name} model: {model_name}")
                
            except Exception as e:
                self.logger.warning(f"Could not initialize {name} model: {e}")
    
    def prepare_sentiment_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, SentimentDataset]:
        """Prepare sentiment analysis datasets."""
        datasets = {}
        
        for split_name, df in data.items():
            # Use mBERT tokenizer for all languages
            tokenizer = self.tokenizers.get('mbert', self.tokenizers[list(self.tokenizers.keys())[0]])
            
            dataset = SentimentDataset(
                df, 
                tokenizer, 
                max_length=self.config.model.max_length
            )
            datasets[split_name] = dataset
        
        return datasets
    
    def prepare_translation_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, TranslationDataset]:
        """Prepare translation datasets."""
        datasets = {}
        
        for split_name, df in data.items():
            # Use mBERT tokenizer for all languages
            tokenizer = self.tokenizers.get('mbert', self.tokenizers[list(self.tokenizers.keys())[0]])
            
            dataset = TranslationDataset(
                df, 
                tokenizer, 
                max_length=self.config.model.max_length
            )
            datasets[split_name] = dataset
        
        return datasets
    
    def train_sentiment_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                            model_name: str = 'mbert') -> Dict:
        """Train sentiment analysis model."""
        self.logger.info(f"Training sentiment model: {model_name}")
        
        # Prepare datasets
        datasets = self.prepare_sentiment_data({
            'train': train_data,
            'validation': val_data
        })
        
        # Get model and tokenizer
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"models/sentiment_{model_name}",
            num_train_epochs=self.config.model.num_epochs,
            per_device_train_batch_size=self.config.model.batch_size,
            per_device_eval_batch_size=self.config.model.batch_size,
            warmup_steps=self.config.model.warmup_steps,
            weight_decay=self.config.model.weight_decay,
            logging_dir=f"logs/sentiment_{model_name}",
            logging_steps=self.config.model.logging_steps,
            evaluation_strategy=self.config.model.evaluation_strategy,
            eval_steps=self.config.model.eval_steps,
            save_steps=self.config.model.save_steps,
            save_total_limit=self.config.model.save_total_limit,
            load_best_model_at_end=self.config.model.load_best_model_at_end,
            metric_for_best_model=self.config.model.metric_for_best_model,
            greater_is_better=False,  # For loss
            dataloader_num_workers=4,
            remove_unused_columns=False
        )
        
        # Initialize trainer
        trainer = FairnessAwareTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            tokenizer=tokenizer,
            compute_metrics=self.compute_sentiment_metrics,
            fairness_weight=0.1
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model_path = self.config.get_model_path(f"sentiment_{model_name}")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        
        self.logger.info(f"Training completed. Validation results: {eval_results}")
        
        return {
            'model_path': model_path,
            'eval_results': eval_results,
            'trainer': trainer
        }
    
    def train_translation_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                              model_name: str = 'mbert') -> Dict:
        """Train translation model."""
        self.logger.info(f"Training translation model: {model_name}")
        
        # Prepare datasets
        datasets = self.prepare_translation_data({
            'train': train_data,
            'validation': val_data
        })
        
        # Get model and tokenizer
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"models/translation_{model_name}",
            num_train_epochs=self.config.model.num_epochs,
            per_device_train_batch_size=self.config.model.batch_size,
            per_device_eval_batch_size=self.config.model.batch_size,
            warmup_steps=self.config.model.warmup_steps,
            weight_decay=self.config.model.weight_decay,
            logging_dir=f"logs/translation_{model_name}",
            logging_steps=self.config.model.logging_steps,
            evaluation_strategy=self.config.model.evaluation_strategy,
            eval_steps=self.config.model.eval_steps,
            save_steps=self.config.model.save_steps,
            save_total_limit=self.config.model.save_total_limit,
            load_best_model_at_end=self.config.model.load_best_model_at_end,
            metric_for_best_model=self.config.model.metric_for_best_model,
            greater_is_better=False,  # For loss
            dataloader_num_workers=4,
            remove_unused_columns=False,
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=4
        )
        
        # Initialize trainer
        trainer = FairnessAwareTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            tokenizer=tokenizer,
            compute_metrics=self.compute_translation_metrics,
            fairness_weight=0.1
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model_path = self.config.get_model_path(f"translation_{model_name}")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        
        self.logger.info(f"Training completed. Validation results: {eval_results}")
        
        return {
            'model_path': model_path,
            'eval_results': eval_results,
            'trainer': trainer
        }
    
    def compute_sentiment_metrics(self, eval_pred):
        """Compute metrics for sentiment analysis."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        # Calculate AUC for binary classification (positive vs others)
        if len(np.unique(labels)) == 2:
            auc = roc_auc_score(labels, predictions)
        else:
            auc = roc_auc_score(labels, predictions, multi_class='ovr')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def compute_translation_metrics(self, eval_pred):
        """Compute metrics for translation."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        tokenizer = self.tokenizers.get('mbert', self.tokenizers[list(self.tokenizers.keys())[0]])
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate BLEU score (simplified)
        bleu_score = self.calculate_bleu_score(decoded_preds, decoded_labels)
        
        return {
            'bleu': bleu_score
        }
    
    def calculate_bleu_score(self, predictions, references):
        """Calculate simplified BLEU score."""
        # This is a simplified BLEU calculation
        # In practice, you would use nltk.translate.bleu_score
        total_score = 0
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            
            # Calculate n-gram overlap
            matches = sum(1 for word in pred_words if word in ref_words)
            if len(pred_words) > 0:
                precision = matches / len(pred_words)
                total_score += precision
        
        return total_score / len(predictions) if predictions else 0
    
    def evaluate_model(self, model_path: str, test_data: pd.DataFrame, task: str = 'sentiment') -> Dict:
        """Evaluate trained model on test set."""
        self.logger.info(f"Evaluating {task} model: {model_path}")
        
        # Load model and tokenizer
        if task == 'sentiment':
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        
        # Prepare test dataset
        if task == 'sentiment':
            test_dataset = SentimentDataset(test_data, tokenizer, self.config.model.max_length)
        else:
            test_dataset = TranslationDataset(test_data, tokenizer, self.config.model.max_length)
        
        test_loader = DataLoader(test_dataset, batch_size=self.config.model.batch_size, shuffle=False)
        
        # Evaluate
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                
                if task == 'sentiment':
                    preds = torch.argmax(outputs.logits, dim=1)
                    predictions.extend(preds.cpu().numpy())
                    labels.extend(batch['labels'].cpu().numpy())
                else:
                    # For translation, generate predictions
                    generated_ids = model.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_length=128,
                        num_beams=4,
                        early_stopping=True
                    )
                    predictions.extend(generated_ids.cpu().numpy())
                    labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        if task == 'sentiment':
            metrics = self.compute_sentiment_metrics((predictions, labels))
        else:
            metrics = self.compute_translation_metrics((predictions, labels))
        
        self.logger.info(f"Test results: {metrics}")
        
        return {
            'predictions': predictions,
            'labels': labels,
            'metrics': metrics
        }
    
    def save_training_results(self, results: Dict, task: str, model_name: str):
        """Save training results to disk."""
        results_path = self.config.get_model_path(f"{task}_{model_name}_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved training results to {results_path}")

def main():
    """Main function for model training pipeline."""
    config = Config()
    trainer = ModelTrainer(config)
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format=config.logging.log_format,
        handlers=[
            logging.FileHandler(config.get_log_path()),
            logging.StreamHandler()
        ]
    )
    
    # Load processed data
    from .data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor(config)
    
    # Train sentiment models
    sentiment_data, _ = preprocessor.load_processed_data('sentiment')
    if sentiment_data:
        for model_name in ['mbert', 'english', 'german', 'hindi']:
            if model_name in trainer.models:
                results = trainer.train_sentiment_model(
                    sentiment_data['train'],
                    sentiment_data['validation'],
                    model_name
                )
                trainer.save_training_results(results, 'sentiment', model_name)
                
                # Evaluate on test set
                test_results = trainer.evaluate_model(
                    results['model_path'],
                    sentiment_data['test'],
                    'sentiment'
                )
    
    # Train translation models
    translation_data, _ = preprocessor.load_processed_data('translation')
    if translation_data:
        for model_name in ['mbert']:  # Only mBERT for translation
            if model_name in trainer.models:
                results = trainer.train_translation_model(
                    translation_data['train'],
                    translation_data['validation'],
                    model_name
                )
                trainer.save_training_results(results, 'translation', model_name)
                
                # Evaluate on test set
                test_results = trainer.evaluate_model(
                    results['model_path'],
                    translation_data['test'],
                    'translation'
                )

if __name__ == "__main__":
    main() 