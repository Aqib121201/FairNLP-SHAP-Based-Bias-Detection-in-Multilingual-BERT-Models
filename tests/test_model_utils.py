"""
Unit tests for model utilities module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.config import Config
from src.model_utils import ModelUtils

class TestModelUtils:
    """Test class for ModelUtils."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def utils(self, config):
        """Create test model utils."""
        return ModelUtils(config)
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        return np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        return np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    
    @pytest.fixture
    def sample_protected_attributes(self):
        """Create sample protected attributes."""
        return {
            'gender': np.array(['male', 'female', 'male', 'female', 'male', 
                               'female', 'male', 'female', 'male', 'female']),
            'age_group': np.array(['18-25', '26-35', '18-25', '26-35', '18-25',
                                  '26-35', '18-25', '26-35', '18-25', '26-35'])
        }
    
    def test_init(self, utils):
        """Test model utils initialization."""
        assert utils.config is not None
        assert hasattr(utils, 'logger')
    
    def test_calculate_demographic_parity(self, utils, sample_predictions, sample_protected_attributes):
        """Test demographic parity calculation."""
        gender_attr = sample_protected_attributes['gender']
        dp = utils._calculate_demographic_parity(sample_predictions, gender_attr)
        
        assert isinstance(dp, float)
        assert dp >= 0.0
        assert dp <= 1.0
    
    def test_calculate_equalized_odds(self, utils, sample_predictions, sample_labels, sample_protected_attributes):
        """Test equalized odds calculation."""
        gender_attr = sample_protected_attributes['gender']
        eo = utils._calculate_equalized_odds(sample_predictions, sample_labels, gender_attr)
        
        assert isinstance(eo, dict)
        assert 'tpr_diff' in eo
        assert 'fpr_diff' in eo
        assert isinstance(eo['tpr_diff'], float)
        assert isinstance(eo['fpr_diff'], float)
    
    def test_calculate_equal_opportunity(self, utils, sample_predictions, sample_labels, sample_protected_attributes):
        """Test equal opportunity calculation."""
        gender_attr = sample_protected_attributes['gender']
        eo = utils._calculate_equal_opportunity(sample_predictions, sample_labels, gender_attr)
        
        assert isinstance(eo, float)
        assert eo >= 0.0
        assert eo <= 1.0
    
    def test_calculate_statistical_parity_difference(self, utils, sample_predictions, sample_protected_attributes):
        """Test statistical parity difference calculation."""
        gender_attr = sample_protected_attributes['gender']
        spd = utils._calculate_statistical_parity_difference(sample_predictions, gender_attr)
        
        assert isinstance(spd, float)
        assert spd >= 0.0
        assert spd <= 1.0
    
    def test_calculate_disparate_impact(self, utils, sample_predictions, sample_protected_attributes):
        """Test disparate impact calculation."""
        gender_attr = sample_protected_attributes['gender']
        di = utils._calculate_disparate_impact(sample_predictions, gender_attr)
        
        assert isinstance(di, float)
        assert di >= 0.0
        assert di <= 1.0
    
    def test_calculate_fairness_metrics(self, utils, sample_predictions, sample_labels, sample_protected_attributes):
        """Test comprehensive fairness metrics calculation."""
        fairness_metrics = utils.calculate_fairness_metrics(
            sample_predictions, sample_labels, sample_protected_attributes
        )
        
        assert isinstance(fairness_metrics, dict)
        assert 'gender' in fairness_metrics
        assert 'age_group' in fairness_metrics
        assert 'overall' in fairness_metrics
        
        # Check gender metrics
        gender_metrics = fairness_metrics['gender']
        assert 'demographic_parity' in gender_metrics
        assert 'equalized_odds' in gender_metrics
        assert 'equal_opportunity' in gender_metrics
        assert 'statistical_parity_difference' in gender_metrics
        assert 'disparate_impact' in gender_metrics
        assert 'group_metrics' in gender_metrics
    
    def test_calculate_kl_divergence(self, utils, sample_predictions, sample_labels, sample_protected_attributes):
        """Test KL divergence calculation."""
        kl_divergences = utils.calculate_kl_divergence(
            sample_predictions, sample_labels, sample_protected_attributes
        )
        
        assert isinstance(kl_divergences, dict)
        assert 'gender' in kl_divergences
        assert 'age_group' in kl_divergences
        
        for attr, kl_div in kl_divergences.items():
            assert isinstance(kl_div, float)
            assert kl_div >= 0.0
    
    def test_calculate_sentiment_polarity_bias(self, utils, sample_predictions, sample_protected_attributes):
        """Test sentiment polarity bias calculation."""
        polarity_bias = utils.calculate_sentiment_polarity_bias(
            sample_predictions, sample_protected_attributes
        )
        
        assert isinstance(polarity_bias, dict)
        assert 'gender' in polarity_bias
        assert 'age_group' in polarity_bias
        
        for attr, bias in polarity_bias.items():
            assert isinstance(bias, float)
    
    def test_perform_statistical_tests(self, utils, sample_predictions, sample_labels, sample_protected_attributes):
        """Test statistical tests performance."""
        statistical_tests = utils.perform_statistical_tests(
            sample_predictions, sample_labels, sample_protected_attributes
        )
        
        assert isinstance(statistical_tests, dict)
        assert 'gender' in statistical_tests
        assert 'age_group' in statistical_tests
        
        # Check gender tests
        gender_tests = statistical_tests['gender']
        assert 'chi_square' in gender_tests
        assert 'mann_whitney_u' in gender_tests
        assert 't_test' in gender_tests
        
        for test_name, test_result in gender_tests.items():
            assert 'statistic' in test_result
            assert 'p_value' in test_result
            assert 'significant' in test_result
            assert isinstance(test_result['statistic'], float)
            assert isinstance(test_result['p_value'], float)
            assert isinstance(test_result['significant'], bool)
    
    def test_evaluate_model_performance(self, utils, sample_predictions, sample_labels):
        """Test model performance evaluation."""
        performance_metrics = utils.evaluate_model_performance(
            sample_predictions, sample_labels, 'sentiment'
        )
        
        assert isinstance(performance_metrics, dict)
        assert 'accuracy' in performance_metrics
        assert 'precision' in performance_metrics
        assert 'recall' in performance_metrics
        assert 'f1_score' in performance_metrics
        assert 'auc' in performance_metrics
        assert 'confusion_matrix' in performance_metrics
        assert 'classification_report' in performance_metrics
        assert 'per_class_metrics' in performance_metrics
        
        # Check metric types and ranges
        assert isinstance(performance_metrics['accuracy'], float)
        assert 0.0 <= performance_metrics['accuracy'] <= 1.0
        assert isinstance(performance_metrics['precision'], float)
        assert 0.0 <= performance_metrics['precision'] <= 1.0
        assert isinstance(performance_metrics['recall'], float)
        assert 0.0 <= performance_metrics['recall'] <= 1.0
        assert isinstance(performance_metrics['f1_score'], float)
        assert 0.0 <= performance_metrics['f1_score'] <= 1.0
    
    def test_compute_kl_divergence(self, utils):
        """Test KL divergence computation."""
        # Test with simple distributions
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        
        kl_div = utils._compute_kl_divergence(p, q)
        
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        
        # Test with identical distributions
        kl_div_identical = utils._compute_kl_divergence(p, p)
        assert kl_div_identical == 0.0
    
    def test_apply_bias_mitigation(self, utils):
        """Test bias mitigation application."""
        # Create sample data
        data = pd.DataFrame({
            'text': ['He is good', 'She is bad', 'They are okay'],
            'gender': ['male', 'female', 'other'],
            'predictions': [0.8, 0.3, 0.6]
        })
        
        # Test reweighing
        reweighed_data = utils.apply_bias_mitigation(data, 'reweighing')
        assert 'sample_weight' in reweighed_data.columns
        
        # Test preprocessing
        preprocessed_data = utils.apply_bias_mitigation(data, 'preprocessing')
        assert 'text_processed' in preprocessed_data.columns
        
        # Test postprocessing
        postprocessed_data = utils.apply_bias_mitigation(data, 'postprocessing')
        assert 'predictions_adjusted' in postprocessed_data.columns
        
        # Test unknown strategy
        unchanged_data = utils.apply_bias_mitigation(data, 'unknown_strategy')
        assert len(unchanged_data) == len(data)
    
    def test_edge_cases(self, utils):
        """Test edge cases and error handling."""
        # Test with empty predictions
        empty_predictions = np.array([])
        empty_labels = np.array([])
        empty_attributes = {'gender': np.array([])}
        
        # These should handle empty arrays gracefully
        fairness_metrics = utils.calculate_fairness_metrics(
            empty_predictions, empty_labels, empty_attributes
        )
        assert isinstance(fairness_metrics, dict)
        
        # Test with single group
        single_group_predictions = np.array([0, 1, 0, 1])
        single_group_labels = np.array([0, 1, 0, 1])
        single_group_attributes = {'gender': np.array(['male', 'male', 'male', 'male'])}
        
        fairness_metrics_single = utils.calculate_fairness_metrics(
            single_group_predictions, single_group_labels, single_group_attributes
        )
        assert isinstance(fairness_metrics_single, dict)
    
    def test_save_and_load_evaluation_results(self, utils, sample_predictions, sample_labels):
        """Test saving and loading evaluation results."""
        import tempfile
        import os
        
        # Generate evaluation results
        performance_metrics = utils.evaluate_model_performance(
            sample_predictions, sample_labels, 'sentiment'
        )
        
        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the reports directory
            original_dir = utils.config.reports_dir
            utils.config.reports_dir = temp_dir
            
            try:
                utils.save_evaluation_results(performance_metrics, 'sentiment', 'test_model')
                
                # Check if file was created
                expected_file = os.path.join(temp_dir, 'sentiment_test_model_evaluation.json')
                assert os.path.exists(expected_file)
                
                # Test loading
                loaded_results = utils.load_evaluation_results('sentiment', 'test_model')
                assert isinstance(loaded_results, dict)
                assert 'accuracy' in loaded_results
                
            finally:
                utils.config.reports_dir = original_dir
    
    def test_overall_fairness_calculation(self, utils):
        """Test overall fairness calculation."""
        # Create mock fairness metrics
        mock_fairness_metrics = {
            'gender': {
                'demographic_parity': 0.15,  # Above threshold
                'equalized_odds': {'tpr_diff': 0.05, 'fpr_diff': 0.03},  # Below threshold
                'equal_opportunity': 0.08,  # Below threshold
                'statistical_parity_difference': 0.12,  # Above threshold
                'disparate_impact': 0.75  # Below 0.8 threshold
            },
            'age_group': {
                'demographic_parity': 0.05,  # Below threshold
                'equalized_odds': {'tpr_diff': 0.12, 'fpr_diff': 0.08},  # Above threshold
                'equal_opportunity': 0.15,  # Above threshold
                'statistical_parity_difference': 0.03,  # Below threshold
                'disparate_impact': 0.85  # Above 0.8 threshold
            }
        }
        
        overall = utils._calculate_overall_fairness(mock_fairness_metrics)
        
        assert isinstance(overall, dict)
        assert 'demographic_parity_violations' in overall
        assert 'equalized_odds_violations' in overall
        assert 'equal_opportunity_violations' in overall
        assert 'statistical_parity_violations' in overall
        assert 'disparate_impact_violations' in overall
        
        # Check that violations are counted correctly
        assert overall['demographic_parity_violations'] == 1  # gender only
        assert overall['equalized_odds_violations'] == 1  # age_group only
        assert overall['equal_opportunity_violations'] == 1  # age_group only
        assert overall['statistical_parity_violations'] == 1  # gender only
        assert overall['disparate_impact_violations'] == 1  # gender only

if __name__ == "__main__":
    pytest.main([__file__]) 