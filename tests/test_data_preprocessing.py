"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.config import Config
from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor:
    """Test class for DataPreprocessor."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def preprocessor(self, config):
        """Create test preprocessor."""
        return DataPreprocessor(config)
    
    @pytest.fixture
    def sample_sentiment_data(self):
        """Create sample sentiment data."""
        return pd.DataFrame({
            'text': [
                'This is a great product!',
                'I hate this service.',
                'The quality is okay.',
                'Amazing experience!',
                'Terrible customer service.'
            ],
            'label': [2, 0, 1, 2, 0],  # 0=negative, 1=neutral, 2=positive
            'language': ['en', 'en', 'en', 'en', 'en']
        })
    
    @pytest.fixture
    def sample_translation_data(self):
        """Create sample translation data."""
        return pd.DataFrame({
            'source_text': [
                'Hello world',
                'Good morning',
                'How are you?',
                'Thank you',
                'Goodbye'
            ],
            'target_text': [
                'Hallo Welt',
                'Guten Morgen',
                'Wie geht es dir?',
                'Danke',
                'Auf Wiedersehen'
            ],
            'source_lang': ['en', 'en', 'en', 'en', 'en'],
            'target_lang': ['de', 'de', 'de', 'de', 'de']
        })
    
    def test_init(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.config is not None
        assert hasattr(preprocessor, 'tokenizers')
        assert hasattr(preprocessor, 'label_encoders')
    
    def test_clean_text(self, preprocessor):
        """Test text cleaning functionality."""
        # Test basic cleaning
        text = "Hello, world! This is a test. http://example.com"
        cleaned = preprocessor.clean_text(text, 'en')
        assert 'http://example.com' not in cleaned
        assert 'Hello, world!' in cleaned
        
        # Test empty text
        assert preprocessor.clean_text('', 'en') == ''
        assert preprocessor.clean_text(None, 'en') == ''
        
        # Test German text
        german_text = "Hallo Welt! Das ist ein Test."
        cleaned_german = preprocessor.clean_text(german_text, 'de')
        assert 'Hallo' in cleaned_german
        assert 'Welt' in cleaned_german
    
    def test_detect_language(self, preprocessor):
        """Test language detection."""
        # Test English
        assert preprocessor.detect_language("Hello world") == 'en'
        
        # Test German
        assert preprocessor.detect_language("Hallo Welt") == 'de'
        
        # Test empty text
        assert preprocessor.detect_language("") == 'unknown'
    
    def test_extract_demographic_features(self, preprocessor):
        """Test demographic feature extraction."""
        text = "She is a great doctor and he is a nurse."
        features = preprocessor.extract_demographic_features(text)
        
        assert 'inferred_gender' in features
        assert features['inferred_gender'] in ['male', 'female', 'neutral']
        
        # Test with metadata
        metadata = {'gender': 'female', 'age': '25-35'}
        features_with_meta = preprocessor.extract_demographic_features(text, metadata)
        assert features_with_meta['gender'] == 'female'
        assert features_with_meta['age'] == '25-35'
    
    def test_load_sentiment_data(self, preprocessor, sample_sentiment_data):
        """Test sentiment data loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_sentiment_data.to_csv(f.name, index=False)
            
            try:
                loaded_data = preprocessor.load_sentiment_data(f.name)
                assert len(loaded_data) == len(sample_sentiment_data)
                assert all(col in loaded_data.columns for col in ['text', 'label', 'language'])
            finally:
                os.unlink(f.name)
    
    def test_load_translation_data(self, preprocessor, sample_translation_data):
        """Test translation data loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_translation_data.to_csv(f.name, index=False)
            
            try:
                loaded_data = preprocessor.load_translation_data(f.name)
                assert len(loaded_data) == len(sample_translation_data)
                assert all(col in loaded_data.columns for col in ['source_text', 'target_text', 'source_lang', 'target_lang'])
            finally:
                os.unlink(f.name)
    
    def test_prepare_sentiment_dataset(self, preprocessor, sample_sentiment_data):
        """Test sentiment dataset preparation."""
        processed_df, encoders = preprocessor.prepare_sentiment_dataset(sample_sentiment_data)
        
        assert len(processed_df) > 0
        assert 'input_ids' in processed_df.columns
        assert 'attention_mask' in processed_df.columns
        assert len(encoders) >= 0  # May be empty if no demographic attributes
    
    def test_prepare_translation_dataset(self, preprocessor, sample_translation_data):
        """Test translation dataset preparation."""
        processed_df, encoders = preprocessor.prepare_translation_dataset(sample_translation_data)
        
        assert len(processed_df) > 0
        assert 'source_input_ids' in processed_df.columns
        assert 'target_input_ids' in processed_df.columns
        assert len(encoders) >= 0
    
    def test_split_dataset(self, preprocessor, sample_sentiment_data):
        """Test dataset splitting."""
        splits = preprocessor.split_dataset(sample_sentiment_data, 'sentiment')
        
        assert 'train' in splits
        assert 'validation' in splits
        assert 'test' in splits
        
        # Check that splits are not empty
        assert len(splits['train']) > 0
        assert len(splits['validation']) > 0
        assert len(splits['test']) > 0
        
        # Check that all data is accounted for
        total_samples = len(splits['train']) + len(splits['validation']) + len(splits['test'])
        assert total_samples == len(sample_sentiment_data)
    
    def test_get_dataset_statistics(self, preprocessor, sample_sentiment_data):
        """Test dataset statistics calculation."""
        stats = preprocessor.get_dataset_statistics(sample_sentiment_data, 'sentiment')
        
        assert 'total_samples' in stats
        assert 'languages' in stats
        assert 'labels' in stats
        assert 'text_length' in stats
        
        assert stats['total_samples'] == len(sample_sentiment_data)
        assert 'en' in stats['languages']
        assert len(stats['labels']) > 0
    
    @patch('src.data_preprocessing.AutoTokenizer.from_pretrained')
    def test_initialize_tokenizers(self, mock_tokenizer, preprocessor):
        """Test tokenizer initialization."""
        # Mock tokenizer
        mock_tokenizer.return_value = Mock()
        
        # Test initialization
        tokenizers = preprocessor._initialize_tokenizers()
        
        assert 'en' in tokenizers
        assert 'de' in tokenizers
        assert 'hi' in tokenizers
    
    def test_tokenize_text(self, preprocessor):
        """Test text tokenization."""
        text = "Hello world"
        tokenized = preprocessor.tokenize_text(text, 'en')
        
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert tokenized['input_ids'].shape[0] > 0
    
    def test_save_and_load_processed_data(self, preprocessor, sample_sentiment_data):
        """Test saving and loading processed data."""
        # Prepare data
        processed_df, encoders = preprocessor.prepare_sentiment_dataset(sample_sentiment_data)
        splits = preprocessor.split_dataset(processed_df, 'sentiment')
        
        # Save data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the processed data directory
            original_dir = preprocessor.config.data.processed_data_dir
            preprocessor.config.data.processed_data_dir = temp_dir
            
            try:
                preprocessor.save_processed_data(splits, 'sentiment')
                
                # Load data
                loaded_data, loaded_encoders = preprocessor.load_processed_data('sentiment')
                
                assert 'train' in loaded_data
                assert 'validation' in loaded_data
                assert 'test' in loaded_data
                assert len(loaded_encoders) >= 0
            finally:
                preprocessor.config.data.processed_data_dir = original_dir

if __name__ == "__main__":
    pytest.main([__file__]) 