"""
Model utilities for FairNLP project.

This module provides utility functions for model evaluation, fairness metrics,
bias mitigation, and other model-related operations.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

from .config import Config
from pathlib import Path

class ModelUtils:
    """Utility class for model operations and fairness analysis."""
    
    def __init__(self, config: Config):
        """Initialize the model utilities."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_fairness_metrics(self, predictions: np.ndarray, labels: np.ndarray,
                                 protected_attributes: Dict[str, np.ndarray]) -> Dict:
        """Calculate comprehensive fairness metrics."""
        self.logger.info("Calculating fairness metrics")
        
        fairness_metrics = {}
        
        for attr_name, attr_values in protected_attributes.items():
            attr_metrics = self._calculate_attribute_fairness(
                predictions, labels, attr_values, attr_name
            )
            fairness_metrics[attr_name] = attr_metrics
        
        # Calculate overall fairness metrics
        fairness_metrics['overall'] = self._calculate_overall_fairness(fairness_metrics)
        
        return fairness_metrics
    
    def _calculate_attribute_fairness(self, predictions: np.ndarray, labels: np.ndarray,
                                    attr_values: np.ndarray, attr_name: str) -> Dict:
        """Calculate fairness metrics for a specific protected attribute."""
        metrics = {}
        
        # Get unique attribute values
        unique_values = np.unique(attr_values)
        
        # Calculate demographic parity
        metrics['demographic_parity'] = self._calculate_demographic_parity(
            predictions, attr_values
        )
        
        # Calculate equalized odds
        metrics['equalized_odds'] = self._calculate_equalized_odds(
            predictions, labels, attr_values
        )
        
        # Calculate equal opportunity
        metrics['equal_opportunity'] = self._calculate_equal_opportunity(
            predictions, labels, attr_values
        )
        
        # Calculate statistical parity difference
        metrics['statistical_parity_difference'] = self._calculate_statistical_parity_difference(
            predictions, attr_values
        )
        
        # Calculate disparate impact
        metrics['disparate_impact'] = self._calculate_disparate_impact(
            predictions, attr_values
        )
        
        # Calculate group-specific metrics
        group_metrics = {}
        for value in unique_values:
            mask = attr_values == value
            group_predictions = predictions[mask]
            group_labels = labels[mask]
            
            if len(group_predictions) > 0:
                group_metrics[str(value)] = {
                    'accuracy': accuracy_score(group_labels, group_predictions),
                    'precision': precision_recall_fscore_support(
                        group_labels, group_predictions, average='weighted'
                    )[0],
                    'recall': precision_recall_fscore_support(
                        group_labels, group_predictions, average='weighted'
                    )[1],
                    'f1': precision_recall_fscore_support(
                        group_labels, group_predictions, average='weighted'
                    )[2],
                    'positive_rate': np.mean(group_predictions == 1),
                    'sample_size': len(group_predictions)
                }
        
        metrics['group_metrics'] = group_metrics
        
        return metrics
    
    def _calculate_demographic_parity(self, predictions: np.ndarray, 
                                    attr_values: np.ndarray) -> float:
        """Calculate demographic parity violation."""
        unique_values = np.unique(attr_values)
        
        if len(unique_values) < 2:
            return 0.0
        
        # Calculate positive prediction rates for each group
        positive_rates = []
        for value in unique_values:
            mask = attr_values == value
            positive_rate = np.mean(predictions[mask] == 1)
            positive_rates.append(positive_rate)
        
        # Calculate maximum difference
        max_diff = max(positive_rates) - min(positive_rates)
        
        return float(max_diff)
    
    def _calculate_equalized_odds(self, predictions: np.ndarray, labels: np.ndarray,
                                attr_values: np.ndarray) -> Dict:
        """Calculate equalized odds violation."""
        unique_values = np.unique(attr_values)
        
        if len(unique_values) < 2:
            return {'tpr_diff': 0.0, 'fpr_diff': 0.0}
        
        # Calculate TPR and FPR for each group
        tpr_values = []
        fpr_values = []
        
        for value in unique_values:
            mask = attr_values == value
            group_predictions = predictions[mask]
            group_labels = labels[mask]
            
            if len(group_predictions) > 0:
                # True Positive Rate
                tpr = np.sum((group_predictions == 1) & (group_labels == 1)) / np.sum(group_labels == 1)
                tpr_values.append(tpr)
                
                # False Positive Rate
                fpr = np.sum((group_predictions == 1) & (group_labels == 0)) / np.sum(group_labels == 0)
                fpr_values.append(fpr)
        
        # Calculate differences
        tpr_diff = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0.0
        fpr_diff = max(fpr_values) - min(fpr_values) if len(fpr_values) > 1 else 0.0
        
        return {
            'tpr_diff': float(tpr_diff),
            'fpr_diff': float(fpr_diff)
        }
    
    def _calculate_equal_opportunity(self, predictions: np.ndarray, labels: np.ndarray,
                                   attr_values: np.ndarray) -> float:
        """Calculate equal opportunity violation."""
        unique_values = np.unique(attr_values)
        
        if len(unique_values) < 2:
            return 0.0
        
        # Calculate TPR for each group
        tpr_values = []
        
        for value in unique_values:
            mask = attr_values == value
            group_predictions = predictions[mask]
            group_labels = labels[mask]
            
            if len(group_predictions) > 0:
                tpr = np.sum((group_predictions == 1) & (group_labels == 1)) / np.sum(group_labels == 1)
                tpr_values.append(tpr)
        
        # Calculate maximum difference
        max_diff = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0.0
        
        return float(max_diff)
    
    def _calculate_statistical_parity_difference(self, predictions: np.ndarray,
                                               attr_values: np.ndarray) -> float:
        """Calculate statistical parity difference."""
        unique_values = np.unique(attr_values)
        
        if len(unique_values) < 2:
            return 0.0
        
        # Calculate positive prediction rates
        positive_rates = []
        for value in unique_values:
            mask = attr_values == value
            positive_rate = np.mean(predictions[mask] == 1)
            positive_rates.append(positive_rate)
        
        # Calculate difference between maximum and minimum rates
        diff = max(positive_rates) - min(positive_rates)
        
        return float(diff)
    
    def _calculate_disparate_impact(self, predictions: np.ndarray,
                                  attr_values: np.ndarray) -> float:
        """Calculate disparate impact ratio."""
        unique_values = np.unique(attr_values)
        
        if len(unique_values) < 2:
            return 1.0
        
        # Calculate positive prediction rates
        positive_rates = []
        for value in unique_values:
            mask = attr_values == value
            positive_rate = np.mean(predictions[mask] == 1)
            positive_rates.append(positive_rate)
        
        # Calculate ratio of minimum to maximum rates
        min_rate = min(positive_rates)
        max_rate = max(positive_rates)
        
        ratio = min_rate / max_rate if max_rate > 0 else 1.0
        
        return float(ratio)
    
    def _calculate_overall_fairness(self, fairness_metrics: Dict) -> Dict:
        """Calculate overall fairness metrics."""
        overall = {
            'demographic_parity_violations': 0,
            'equalized_odds_violations': 0,
            'equal_opportunity_violations': 0,
            'statistical_parity_violations': 0,
            'disparate_impact_violations': 0
        }
        
        for attr_name, attr_metrics in fairness_metrics.items():
            if attr_name == 'overall':
                continue
            
            # Count violations
            if attr_metrics['demographic_parity'] > self.config.fairness.bias_threshold:
                overall['demographic_parity_violations'] += 1
            
            if (attr_metrics['equalized_odds']['tpr_diff'] > self.config.fairness.bias_threshold or
                attr_metrics['equalized_odds']['fpr_diff'] > self.config.fairness.bias_threshold):
                overall['equalized_odds_violations'] += 1
            
            if attr_metrics['equal_opportunity'] > self.config.fairness.bias_threshold:
                overall['equal_opportunity_violations'] += 1
            
            if attr_metrics['statistical_parity_difference'] > self.config.fairness.bias_threshold:
                overall['statistical_parity_violations'] += 1
            
            if attr_metrics['disparate_impact'] < 0.8:  # 80% rule
                overall['disparate_impact_violations'] += 1
        
        return overall
    
    def calculate_kl_divergence(self, predictions: np.ndarray, labels: np.ndarray,
                               protected_attributes: Dict[str, np.ndarray]) -> Dict:
        """Calculate KL divergence between demographic groups."""
        kl_divergences = {}
        
        for attr_name, attr_values in protected_attributes.items():
            unique_values = np.unique(attr_values)
            
            if len(unique_values) < 2:
                kl_divergences[attr_name] = 0.0
                continue
            
            # Calculate prediction distributions for each group
            distributions = []
            for value in unique_values:
                mask = attr_values == value
                group_predictions = predictions[mask]
                
                # Create histogram of predictions
                hist, _ = np.histogram(group_predictions, bins=10, range=(0, 1), density=True)
                distributions.append(hist)
            
            # Calculate KL divergence between all pairs
            kl_values = []
            for i in range(len(distributions)):
                for j in range(i + 1, len(distributions)):
                    kl_div = self._compute_kl_divergence(distributions[i], distributions[j])
                    kl_values.append(kl_div)
            
            # Take the maximum KL divergence
            kl_divergences[attr_name] = float(max(kl_values)) if kl_values else 0.0
        
        return kl_divergences
    
    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two distributions."""
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL divergence
        kl_div = np.sum(p * np.log(p / q))
        
        return float(kl_div)
    
    def calculate_sentiment_polarity_bias(self, predictions: np.ndarray, 
                                        protected_attributes: Dict[str, np.ndarray]) -> Dict:
        """Calculate sentiment polarity bias across demographic groups."""
        polarity_bias = {}
        
        for attr_name, attr_values in protected_attributes.items():
            unique_values = np.unique(attr_values)
            
            if len(unique_values) < 2:
                polarity_bias[attr_name] = 0.0
                continue
            
            # Calculate average sentiment scores for each group
            group_sentiments = []
            for value in unique_values:
                mask = attr_values == value
                group_predictions = predictions[mask]
                
                # Convert predictions to sentiment scores (assuming 0=negative, 1=neutral, 2=positive)
                sentiment_scores = (group_predictions - 1)  # Convert to [-1, 0, 1]
                avg_sentiment = np.mean(sentiment_scores)
                group_sentiments.append(avg_sentiment)
            
            # Calculate bias as the difference between maximum and minimum average sentiments
            bias = max(group_sentiments) - min(group_sentiments)
            polarity_bias[attr_name] = float(bias)
        
        return polarity_bias
    
    def perform_statistical_tests(self, predictions: np.ndarray, labels: np.ndarray,
                                protected_attributes: Dict[str, np.ndarray]) -> Dict:
        """Perform statistical tests for bias detection."""
        statistical_tests = {}
        
        for attr_name, attr_values in protected_attributes.items():
            unique_values = np.unique(attr_values)
            
            if len(unique_values) < 2:
                statistical_tests[attr_name] = {}
                continue
            
            attr_tests = {}
            
            # Chi-square test for independence
            try:
                contingency_table = pd.crosstab(predictions, attr_values)
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                attr_tests['chi_square'] = {
                    'statistic': float(chi2),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except Exception as e:
                self.logger.warning(f"Chi-square test failed for {attr_name}: {e}")
                attr_tests['chi_square'] = {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
            
            # Mann-Whitney U test
            try:
                group1_mask = attr_values == unique_values[0]
                group2_mask = attr_values == unique_values[1]
                
                group1_predictions = predictions[group1_mask]
                group2_predictions = predictions[group2_mask]
                
                if len(group1_predictions) > 0 and len(group2_predictions) > 0:
                    statistic, p_value = mannwhitneyu(group1_predictions, group2_predictions, alternative='two-sided')
                    attr_tests['mann_whitney_u'] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
                else:
                    attr_tests['mann_whitney_u'] = {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
            except Exception as e:
                self.logger.warning(f"Mann-Whitney U test failed for {attr_name}: {e}")
                attr_tests['mann_whitney_u'] = {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
            
            # T-test
            try:
                if len(group1_predictions) > 0 and len(group2_predictions) > 0:
                    statistic, p_value = ttest_ind(group1_predictions, group2_predictions)
                    attr_tests['t_test'] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
                else:
                    attr_tests['t_test'] = {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
            except Exception as e:
                self.logger.warning(f"T-test failed for {attr_name}: {e}")
                attr_tests['t_test'] = {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
            
            statistical_tests[attr_name] = attr_tests
        
        return statistical_tests
    
    def evaluate_model_performance(self, predictions: np.ndarray, labels: np.ndarray,
                                 task: str = 'sentiment') -> Dict:
        """Evaluate model performance with comprehensive metrics."""
        self.logger.info(f"Evaluating {task} model performance")
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # AUC calculation
        try:
            if len(np.unique(labels)) == 2:
                auc = roc_auc_score(labels, predictions)
            else:
                auc = roc_auc_score(labels, predictions, multi_class='ovr')
        except Exception as e:
            self.logger.warning(f"AUC calculation failed: {e}")
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(labels, predictions, output_dict=True)
        
        # Per-class metrics
        per_class_metrics = {}
        for class_label in np.unique(labels):
            class_mask = labels == class_label
            class_predictions = predictions[class_mask]
            class_accuracy = accuracy_score(labels[class_mask], class_predictions)
            
            per_class_metrics[f"class_{class_label}"] = {
                'accuracy': float(class_accuracy),
                'support': int(np.sum(class_mask))
            }
        
        performance_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'per_class_metrics': per_class_metrics
        }
        
        return performance_metrics
    
    def apply_bias_mitigation(self, model, data: pd.DataFrame, 
                            mitigation_strategy: str = 'reweighing') -> pd.DataFrame:
        """Apply bias mitigation techniques to the dataset."""
        self.logger.info(f"Applying {mitigation_strategy} bias mitigation")
        
        if mitigation_strategy == 'reweighing':
            return self._apply_reweighing(data)
        elif mitigation_strategy == 'preprocessing':
            return self._apply_preprocessing(data)
        elif mitigation_strategy == 'postprocessing':
            return self._apply_postprocessing(data)
        else:
            self.logger.warning(f"Unknown mitigation strategy: {mitigation_strategy}")
            return data
    
    def _apply_reweighing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply reweighing technique for bias mitigation."""
        # This is a simplified implementation
        # In practice, you would use fairlearn's Reweighing algorithm
        
        # Calculate sample weights based on demographic parity
        weights = np.ones(len(data))
        
        # Example: balance gender representation
        if 'gender' in data.columns:
            gender_counts = data['gender'].value_counts()
            total_samples = len(data)
            
            for gender in gender_counts.index:
                gender_mask = data['gender'] == gender
                target_weight = total_samples / (len(gender_counts) * gender_counts[gender])
                weights[gender_mask] = target_weight
        
        data['sample_weight'] = weights
        return data
    
    def _apply_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing technique for bias mitigation."""
        # This is a simplified implementation
        # In practice, you would use techniques like disparate impact remover
        
        # Example: remove gender-specific terms
        if 'text' in data.columns:
            gender_terms = ['he', 'she', 'him', 'her', 'his', 'hers', 'man', 'woman']
            
            def remove_gender_terms(text):
                text_lower = text.lower()
                for term in gender_terms:
                    text_lower = text_lower.replace(term, '[GENDER]')
                return text_lower
            
            data['text_processed'] = data['text'].apply(remove_gender_terms)
        
        return data
    
    def _apply_postprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply postprocessing technique for bias mitigation."""
        # This is a simplified implementation
        # In practice, you would use techniques like equalized odds postprocessing
        
        # Example: adjust predictions based on demographic parity
        if 'predictions' in data.columns and 'gender' in data.columns:
            # Calculate adjustment factors
            gender_predictions = data.groupby('gender')['predictions'].mean()
            overall_mean = data['predictions'].mean()
            
            # Apply adjustments
            def adjust_prediction(row):
                gender = row['gender']
                prediction = row['predictions']
                adjustment = overall_mean - gender_predictions[gender]
                return max(0, min(1, prediction + adjustment))
            
            data['predictions_adjusted'] = data.apply(adjust_prediction, axis=1)
        
        return data
    
    def save_evaluation_results(self, results: Dict, task: str, model_name: str) -> None:
        """Save evaluation results to disk."""
        output_dir = Path(self.config.reports_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{task}_{model_name}_evaluation.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {output_path}")
    
    def load_evaluation_results(self, task: str, model_name: str) -> Dict:
        """Load evaluation results from disk."""
        input_path = Path(self.config.reports_dir) / f"{task}_{model_name}_evaluation.json"
        
        if input_path.exists():
            with open(input_path, 'r') as f:
                results = json.load(f)
            return results
        else:
            self.logger.warning(f"Evaluation results not found: {input_path}")
            return {}

def main():
    """Main function for model utilities testing."""
    config = Config()
    utils = ModelUtils(config)
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format=config.logging.log_format,
        handlers=[
            logging.FileHandler(config.get_log_path()),
            logging.StreamHandler()
        ]
    )
    
    # Example usage
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic predictions and labels
    predictions = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Generate synthetic protected attributes
    gender = np.random.choice(['male', 'female'], size=n_samples, p=[0.5, 0.5])
    age_group = np.random.choice(['18-25', '26-35', '36-45'], size=n_samples, p=[0.4, 0.4, 0.2])
    
    protected_attributes = {
        'gender': gender,
        'age_group': age_group
    }
    
    # Calculate fairness metrics
    fairness_metrics = utils.calculate_fairness_metrics(
        predictions, labels, protected_attributes
    )
    
    print("Fairness Metrics:")
    print(json.dumps(fairness_metrics, indent=2))
    
    # Calculate KL divergence
    kl_divergences = utils.calculate_kl_divergence(
        predictions, labels, protected_attributes
    )
    
    print("\nKL Divergences:")
    print(json.dumps(kl_divergences, indent=2))
    
    # Calculate sentiment polarity bias
    polarity_bias = utils.calculate_sentiment_polarity_bias(
        predictions, protected_attributes
    )
    
    print("\nSentiment Polarity Bias:")
    print(json.dumps(polarity_bias, indent=2))
    
    # Perform statistical tests
    statistical_tests = utils.perform_statistical_tests(
        predictions, labels, protected_attributes
    )
    
    print("\nStatistical Tests:")
    print(json.dumps(statistical_tests, indent=2))
    
    # Evaluate model performance
    performance_metrics = utils.evaluate_model_performance(
        predictions, labels, 'sentiment'
    )
    
    print("\nPerformance Metrics:")
    print(json.dumps(performance_metrics, indent=2))

if __name__ == "__main__":
    main() 