"""
Data preprocessing module for FairNLP project.

This module handles data loading, cleaning, tokenization, and preparation
for multilingual bias detection analysis.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

from .config import Config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataPreprocessor:
    """Data preprocessing class for multilingual bias detection."""
    
    def __init__(self, config: Config):
        """Initialize the data preprocessor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizers for each language
        self.tokenizers = self._initialize_tokenizers()
        
        # Label encoders for categorical variables
        self.label_encoders = {}
        
        # Language mapping
        self.language_mapping = {
            'en': 'english',
            'de': 'german', 
            'hi': 'hindi'
        }
        
        # Demographic attribute mappings
        self.demographic_mappings = {
            'gender': ['male', 'female', 'other'],
            'age_group': ['18-25', '26-35', '36-45', '46-55', '55+'],
            'region': ['north_america', 'europe', 'asia', 'other'],
            'education': ['high_school', 'bachelor', 'master', 'phd', 'other']
        }
    
    def _initialize_tokenizers(self) -> Dict[str, AutoTokenizer]:
        """Initialize BERT tokenizers for each language."""
        tokenizers = {}
        
        model_mapping = {
            'en': self.config.model.english_bert,
            'de': self.config.model.german_bert,
            'hi': self.config.model.hindi_bert
        }
        
        for lang, model_name in model_mapping.items():
            try:
                tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
                self.logger.info(f"Initialized tokenizer for {lang}: {model_name}")
            except Exception as e:
                self.logger.warning(f"Could not load tokenizer for {lang}: {e}")
                # Fallback to mBERT
                tokenizers[lang] = AutoTokenizer.from_pretrained(self.config.model.mbert_model)
        
        return tokenizers
    
    def load_sentiment_data(self, file_path: str) -> pd.DataFrame:
        """Load sentiment analysis dataset."""
        self.logger.info(f"Loading sentiment data from {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Validate required columns
        required_columns = ['text', 'label', 'language']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.logger.info(f"Loaded {len(df)} samples")
        return df
    
    def load_translation_data(self, file_path: str) -> pd.DataFrame:
        """Load translation dataset."""
        self.logger.info(f"Loading translation data from {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Validate required columns
        required_columns = ['source_text', 'target_text', 'source_lang', 'target_lang']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.logger.info(f"Loaded {len(df)} translation pairs")
        return df
    
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return 'unknown'
    
    def clean_text(self, text: str, language: str = 'en') -> str:
        """Clean and normalize text."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove special characters but keep language-specific characters
        if language == 'hi':
            # Keep Devanagari script characters
            text = re.sub(r'[^\u0900-\u097F\u0020-\u007E]', '', text)
        elif language == 'de':
            # Keep German umlauts and special characters
            text = re.sub(r'[^\u0020-\u007E\u00A0-\u00FF]', '', text)
        else:
            # English: keep basic ASCII and common punctuation
            text = re.sub(r'[^\u0020-\u007E]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text
    
    def extract_demographic_features(self, text: str, metadata: Dict = None) -> Dict:
        """Extract demographic features from text and metadata."""
        features = {}
        
        # Extract from metadata if available
        if metadata:
            for attr in self.config.data.demographic_attributes:
                if attr in metadata:
                    features[attr] = metadata[attr]
        
        # Extract gender indicators from text
        gender_indicators = {
            'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'boys', 'father', 'son'],
            'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'girls', 'mother', 'daughter']
        }
        
        text_lower = text.lower()
        male_count = sum(1 for word in gender_indicators['male'] if word in text_lower)
        female_count = sum(1 for word in gender_indicators['female'] if word in text_lower)
        
        if male_count > female_count:
            features['inferred_gender'] = 'male'
        elif female_count > male_count:
            features['inferred_gender'] = 'female'
        else:
            features['inferred_gender'] = 'neutral'
        
        # Extract cultural indicators
        cultural_indicators = {
            'western': ['christmas', 'halloween', 'thanksgiving', 'easter', 'starbucks', 'mcdonalds'],
            'indian': ['diwali', 'holi', 'ramadan', 'taj', 'bollywood', 'curry'],
            'german': ['oktoberfest', 'bier', 'wurst', 'autobahn', 'berlin', 'munich']
        }
        
        for culture, indicators in cultural_indicators.items():
            count = sum(1 for word in indicators if word.lower() in text_lower)
            features[f'{culture}_cultural_indicators'] = count
        
        return features
    
    def tokenize_text(self, text: str, language: str = 'en') -> Dict:
        """Tokenize text using appropriate BERT tokenizer."""
        if language not in self.tokenizers:
            # Fallback to mBERT
            tokenizer = self.tokenizers.get('en', self.tokenizers[list(self.tokenizers.keys())[0]])
        else:
            tokenizer = self.tokenizers[language]
        
        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.model.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding.get('token_type_ids', None)
        }
    
    def prepare_sentiment_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Prepare sentiment analysis dataset."""
        self.logger.info("Preparing sentiment dataset")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            # Clean text
            cleaned_text = self.clean_text(row['text'], row['language'])
            
            if len(cleaned_text) < self.config.data.min_text_length:
                continue
            
            # Extract demographic features
            metadata = row.to_dict() if hasattr(row, 'to_dict') else {}
            demographic_features = self.extract_demographic_features(cleaned_text, metadata)
            
            # Tokenize
            tokenized = self.tokenize_text(cleaned_text, row['language'])
            
            processed_row = {
                'id': idx,
                'text': cleaned_text,
                'language': row['language'],
                'label': row['label'],
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                **demographic_features
            }
            
            processed_data.append(processed_row)
        
        processed_df = pd.DataFrame(processed_data)
        
        # Encode categorical variables
        encoders = {}
        for attr in self.config.data.demographic_attributes:
            if attr in processed_df.columns:
                le = LabelEncoder()
                processed_df[f'{attr}_encoded'] = le.fit_transform(processed_df[attr].fillna('unknown'))
                encoders[attr] = le
        
        self.label_encoders.update(encoders)
        
        self.logger.info(f"Prepared {len(processed_df)} samples")
        return processed_df, encoders
    
    def prepare_translation_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Prepare translation dataset."""
        self.logger.info("Preparing translation dataset")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            # Clean source and target texts
            source_cleaned = self.clean_text(row['source_text'], row['source_lang'])
            target_cleaned = self.clean_text(row['target_text'], row['target_lang'])
            
            if (len(source_cleaned) < self.config.data.min_text_length or 
                len(target_cleaned) < self.config.data.min_text_length):
                continue
            
            # Tokenize source and target
            source_tokenized = self.tokenize_text(source_cleaned, row['source_lang'])
            target_tokenized = self.tokenize_text(target_cleaned, row['target_lang'])
            
            # Extract demographic features from source text
            demographic_features = self.extract_demographic_features(source_cleaned)
            
            processed_row = {
                'id': idx,
                'source_text': source_cleaned,
                'target_text': target_cleaned,
                'source_lang': row['source_lang'],
                'target_lang': row['target_lang'],
                'source_input_ids': source_tokenized['input_ids'],
                'source_attention_mask': source_tokenized['attention_mask'],
                'target_input_ids': target_tokenized['input_ids'],
                'target_attention_mask': target_tokenized['attention_mask'],
                **demographic_features
            }
            
            processed_data.append(processed_row)
        
        processed_df = pd.DataFrame(processed_data)
        
        # Encode categorical variables
        encoders = {}
        for attr in self.config.data.demographic_attributes:
            if attr in processed_df.columns:
                le = LabelEncoder()
                processed_df[f'{attr}_encoded'] = le.fit_transform(processed_df[attr].fillna('unknown'))
                encoders[attr] = le
        
        self.label_encoders.update(encoders)
        
        self.logger.info(f"Prepared {len(processed_df)} translation pairs")
        return processed_df, encoders
    
    def split_dataset(self, df: pd.DataFrame, task: str = 'sentiment') -> Dict[str, pd.DataFrame]:
        """Split dataset into train/validation/test sets."""
        self.logger.info(f"Splitting {task} dataset")
        
        # Stratify by label and language
        if task == 'sentiment':
            stratify_cols = ['label', 'language']
        else:  # translation
            stratify_cols = ['source_lang', 'target_lang']
        
        # Create stratification column
        df['stratify_col'] = df[stratify_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        
        # Split into train and temp
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - self.config.data.train_split),
            stratify=df['stratify_col'],
            random_state=self.config.experiment.random_seed
        )
        
        # Split temp into validation and test
        val_size = self.config.data.val_split / (self.config.data.val_split + self.config.data.test_split)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['stratify_col'],
            random_state=self.config.experiment.random_seed
        )
        
        # Remove stratify column
        for df_split in [train_df, val_df, test_df]:
            df_split.drop('stratify_col', axis=1, inplace=True)
        
        splits = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        self.logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return splits
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame], task: str) -> None:
        """Save processed data to disk."""
        output_dir = Path(self.config.data.processed_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, df in data.items():
            output_path = output_dir / f"{task}_{split_name}.pkl"
            df.to_pickle(output_path)
            self.logger.info(f"Saved {split_name} data to {output_path}")
        
        # Save label encoders
        encoder_path = output_dir / f"{task}_encoders.pkl"
        import pickle
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        self.logger.info(f"Saved label encoders to {encoder_path}")
    
    def load_processed_data(self, task: str) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """Load processed data from disk."""
        input_dir = Path(self.config.data.processed_data_dir)
        
        data = {}
        for split in ['train', 'validation', 'test']:
            input_path = input_dir / f"{task}_{split}.pkl"
            if input_path.exists():
                data[split] = pd.read_pickle(input_path)
                self.logger.info(f"Loaded {split} data from {input_path}")
        
        # Load label encoders
        encoder_path = input_dir / f"{task}_encoders.pkl"
        encoders = {}
        if encoder_path.exists():
            import pickle
            with open(encoder_path, 'rb') as f:
                encoders = pickle.load(f)
            self.logger.info(f"Loaded label encoders from {encoder_path}")
        
        return data, encoders
    
    def get_dataset_statistics(self, df: pd.DataFrame, task: str = 'sentiment') -> Dict:
        """Get comprehensive dataset statistics."""
        stats = {
            'total_samples': len(df),
            'languages': df['language'].value_counts().to_dict() if 'language' in df.columns else {},
            'labels': df['label'].value_counts().to_dict() if 'label' in df.columns else {}
        }
        
        # Text length statistics
        if task == 'sentiment':
            text_col = 'text'
        else:
            text_col = 'source_text'
        
        if text_col in df.columns:
            text_lengths = df[text_col].str.len()
            stats['text_length'] = {
                'mean': text_lengths.mean(),
                'std': text_lengths.std(),
                'min': text_lengths.min(),
                'max': text_lengths.max(),
                'median': text_lengths.median()
            }
        
        # Demographic statistics
        for attr in self.config.data.demographic_attributes:
            if attr in df.columns:
                stats[f'{attr}_distribution'] = df[attr].value_counts().to_dict()
        
        return stats

def main():
    """Main function for data preprocessing pipeline."""
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format=config.logging.log_format,
        handlers=[
            logging.FileHandler(config.get_log_path()),
            logging.StreamHandler()
        ]
    )
    
    # Process sentiment data
    sentiment_file = config.get_data_path('raw', 'sentiment_data.csv')
    if os.path.exists(sentiment_file):
        sentiment_df = preprocessor.load_sentiment_data(sentiment_file)
        processed_sentiment, encoders = preprocessor.prepare_sentiment_dataset(sentiment_df)
        sentiment_splits = preprocessor.split_dataset(processed_sentiment, 'sentiment')
        preprocessor.save_processed_data(sentiment_splits, 'sentiment')
        
        # Print statistics
        stats = preprocessor.get_dataset_statistics(processed_sentiment, 'sentiment')
        print("Sentiment Dataset Statistics:")
        print(json.dumps(stats, indent=2))
    
    # Process translation data
    translation_file = config.get_data_path('raw', 'translation_data.csv')
    if os.path.exists(translation_file):
        translation_df = preprocessor.load_translation_data(translation_file)
        processed_translation, encoders = preprocessor.prepare_translation_dataset(translation_df)
        translation_splits = preprocessor.split_dataset(processed_translation, 'translation')
        preprocessor.save_processed_data(translation_splits, 'translation')
        
        # Print statistics
        stats = preprocessor.get_dataset_statistics(processed_translation, 'translation')
        print("Translation Dataset Statistics:")
        print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main() 