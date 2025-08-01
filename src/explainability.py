"""
Explainability module for FairNLP project.

This module implements SHAP-based analysis for bias detection in
multilingual BERT models, providing both local and global explanations.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import shap
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

from .config import Config

class SHAPExplainer:
    """SHAP-based explainer for bias detection in multilingual models."""
    
    def __init__(self, config: Config):
        """Initialize the SHAP explainer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SHAP explainers
        self.explainers = {}
        self.background_data = {}
        
        # Language mapping
        self.language_mapping = {
            'en': 'English',
            'de': 'German',
            'hi': 'Hindi'
        }
    
    def load_model(self, model_path: str, task: str = 'sentiment') -> Tuple[Any, Any]:
        """Load trained model and tokenizer."""
        self.logger.info(f"Loading model from {model_path}")
        
        try:
            if task == 'sentiment':
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                model = AutoModel.from_pretrained(model_path)
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            model.to(self.device)
            model.eval()
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_background_data(self, data: pd.DataFrame, tokenizer, 
                              task: str = 'sentiment', sample_size: int = 100) -> np.ndarray:
        """Prepare background data for SHAP analysis."""
        self.logger.info(f"Preparing background data for {task} task")
        
        # Sample background data
        background_df = data.sample(n=min(sample_size, len(data)), random_state=42)
        
        background_texts = []
        for _, row in background_df.iterrows():
            if task == 'sentiment':
                text = str(row['text'])
            else:
                text = str(row['source_text'])
            background_texts.append(text)
        
        # Tokenize background texts
        background_encodings = tokenizer(
            background_texts,
            truncation=True,
            padding=True,
            max_length=self.config.model.max_length,
            return_tensors='pt'
        )
        
        return background_encodings
    
    def create_model_wrapper(self, model, tokenizer, task: str = 'sentiment'):
        """Create a wrapper function for the model to work with SHAP."""
        
        def model_wrapper(texts):
            """Wrapper function that takes text and returns predictions."""
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize
            encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.model.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**encodings)
                
                if task == 'sentiment':
                    # For sentiment analysis, return probabilities
                    probs = torch.softmax(outputs.logits, dim=1)
                    return probs.cpu().numpy()
                else:
                    # For translation, return hidden states
                    return outputs.last_hidden_state.cpu().numpy()
        
        return model_wrapper
    
    def analyze_sentiment_bias(self, model_path: str, data: pd.DataFrame, 
                             demographic_attr: str = 'gender') -> Dict:
        """Analyze bias in sentiment analysis using SHAP."""
        self.logger.info(f"Analyzing sentiment bias for {demographic_attr}")
        
        # Load model
        model, tokenizer = self.load_model(model_path, 'sentiment')
        
        # Prepare background data
        background_encodings = self.prepare_background_data(
            data, tokenizer, 'sentiment', self.config.shap.background_samples
        )
        
        # Create model wrapper
        model_wrapper = self.create_model_wrapper(model, tokenizer, 'sentiment')
        
        # Initialize SHAP explainer
        explainer = shap.Explainer(
            model_wrapper,
            background_encodings,
            nsamples=self.config.shap.nsamples
        )
        
        # Analyze bias across demographic groups
        bias_results = {}
        
        for group in data[demographic_attr].unique():
            self.logger.info(f"Analyzing group: {group}")
            
            # Filter data for this group
            group_data = data[data[demographic_attr] == group]
            
            if len(group_data) == 0:
                continue
            
            # Sample texts for analysis
            sample_texts = group_data['text'].sample(
                n=min(50, len(group_data)), 
                random_state=42
            ).tolist()
            
            # Get SHAP values
            shap_values = explainer(sample_texts)
            
            # Analyze SHAP values
            group_analysis = self.analyze_shap_values(shap_values, sample_texts, tokenizer)
            
            bias_results[group] = {
                'shap_values': shap_values,
                'texts': sample_texts,
                'analysis': group_analysis
            }
        
        return bias_results
    
    def analyze_translation_bias(self, model_path: str, data: pd.DataFrame,
                               demographic_attr: str = 'gender') -> Dict:
        """Analyze bias in translation using SHAP."""
        self.logger.info(f"Analyzing translation bias for {demographic_attr}")
        
        # Load model
        model, tokenizer = self.load_model(model_path, 'translation')
        
        # Prepare background data
        background_encodings = self.prepare_background_data(
            data, tokenizer, 'translation', self.config.shap.background_samples
        )
        
        # Create model wrapper
        model_wrapper = self.create_model_wrapper(model, tokenizer, 'translation')
        
        # Initialize SHAP explainer
        explainer = shap.Explainer(
            model_wrapper,
            background_encodings,
            nsamples=self.config.shap.nsamples
        )
        
        # Analyze bias across demographic groups
        bias_results = {}
        
        for group in data[demographic_attr].unique():
            self.logger.info(f"Analyzing group: {group}")
            
            # Filter data for this group
            group_data = data[data[demographic_attr] == group]
            
            if len(group_data) == 0:
                continue
            
            # Sample texts for analysis
            sample_texts = group_data['source_text'].sample(
                n=min(50, len(group_data)), 
                random_state=42
            ).tolist()
            
            # Get SHAP values
            shap_values = explainer(sample_texts)
            
            # Analyze SHAP values
            group_analysis = self.analyze_shap_values(shap_values, sample_texts, tokenizer)
            
            bias_results[group] = {
                'shap_values': shap_values,
                'texts': sample_texts,
                'analysis': group_analysis
            }
        
        return bias_results
    
    def analyze_shap_values(self, shap_values, texts: List[str], tokenizer) -> Dict:
        """Analyze SHAP values to identify bias patterns."""
        analysis = {
            'feature_importance': {},
            'bias_indicators': [],
            'token_analysis': {}
        }
        
        # Get feature importance
        if hasattr(shap_values, 'values'):
            # For newer SHAP versions
            values = shap_values.values
        else:
            # For older SHAP versions
            values = shap_values
        
        # Analyze token-level importance
        for i, text in enumerate(texts):
            # Tokenize text
            tokens = tokenizer.tokenize(text)
            
            if i < len(values):
                text_values = values[i]
                
                # Get top contributing tokens
                if len(tokens) > 0 and len(text_values) > 0:
                    # Align tokens with SHAP values
                    token_importance = self.align_tokens_with_shap(tokens, text_values)
                    
                    # Identify bias indicators
                    bias_tokens = self.identify_bias_tokens(token_importance)
                    analysis['bias_indicators'].extend(bias_tokens)
                    
                    # Store token analysis
                    analysis['token_analysis'][text] = token_importance
        
        # Aggregate feature importance across all texts
        analysis['feature_importance'] = self.aggregate_feature_importance(values)
        
        return analysis
    
    def align_tokens_with_shap(self, tokens: List[str], shap_values: np.ndarray) -> Dict:
        """Align tokens with their corresponding SHAP values."""
        token_importance = {}
        
        # Handle different SHAP value formats
        if len(shap_values.shape) > 1:
            # Multi-dimensional SHAP values
            shap_values = np.mean(shap_values, axis=0)
        
        # Align tokens with values
        for i, token in enumerate(tokens):
            if i < len(shap_values):
                token_importance[token] = float(shap_values[i])
        
        return token_importance
    
    def identify_bias_tokens(self, token_importance: Dict) -> List[Dict]:
        """Identify tokens that might indicate bias."""
        bias_tokens = []
        
        # Define bias indicators
        gender_indicators = {
            'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'father', 'son'],
            'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'mother', 'daughter']
        }
        
        cultural_indicators = {
            'western': ['christmas', 'halloween', 'thanksgiving', 'easter', 'starbucks'],
            'indian': ['diwali', 'holi', 'ramadan', 'taj', 'bollywood'],
            'german': ['oktoberfest', 'bier', 'wurst', 'autobahn', 'berlin']
        }
        
        # Check for gender bias
        for token, importance in token_importance.items():
            token_lower = token.lower()
            
            # Check gender indicators
            for gender, indicators in gender_indicators.items():
                if token_lower in indicators:
                    bias_tokens.append({
                        'token': token,
                        'importance': importance,
                        'bias_type': 'gender',
                        'category': gender
                    })
            
            # Check cultural indicators
            for culture, indicators in cultural_indicators.items():
                if token_lower in indicators:
                    bias_tokens.append({
                        'token': token,
                        'importance': importance,
                        'bias_type': 'cultural',
                        'category': culture
                    })
        
        return bias_tokens
    
    def aggregate_feature_importance(self, shap_values: np.ndarray) -> Dict:
        """Aggregate feature importance across all samples."""
        if len(shap_values.shape) > 1:
            # Take mean across samples
            aggregated = np.mean(np.abs(shap_values), axis=0)
        else:
            aggregated = np.abs(shap_values)
        
        return {
            'mean_importance': float(np.mean(aggregated)),
            'std_importance': float(np.std(aggregated)),
            'max_importance': float(np.max(aggregated)),
            'min_importance': float(np.min(aggregated))
        }
    
    def create_bias_visualizations(self, bias_results: Dict, task: str, 
                                 demographic_attr: str) -> Dict:
        """Create visualizations for bias analysis."""
        self.logger.info(f"Creating bias visualizations for {task}")
        
        visualizations = {}
        
        # 1. SHAP Summary Plot
        fig_summary = self.create_shap_summary_plot(bias_results, task, demographic_attr)
        visualizations['summary_plot'] = fig_summary
        
        # 2. Bias Comparison Plot
        fig_comparison = self.create_bias_comparison_plot(bias_results, task, demographic_attr)
        visualizations['comparison_plot'] = fig_comparison
        
        # 3. Token Importance Plot
        fig_tokens = self.create_token_importance_plot(bias_results, task, demographic_attr)
        visualizations['token_plot'] = fig_tokens
        
        # 4. Demographic Parity Plot
        fig_parity = self.create_demographic_parity_plot(bias_results, task, demographic_attr)
        visualizations['parity_plot'] = fig_parity
        
        return visualizations
    
    def create_shap_summary_plot(self, bias_results: Dict, task: str, 
                               demographic_attr: str) -> go.Figure:
        """Create SHAP summary plot."""
        fig = make_subplots(
            rows=len(bias_results), cols=1,
            subplot_titles=[f"{self.language_mapping.get(group, group)}" 
                          for group in bias_results.keys()],
            vertical_spacing=0.1
        )
        
        for i, (group, results) in enumerate(bias_results.items()):
            if 'shap_values' in results:
                shap_values = results['shap_values']
                
                # Get SHAP values for plotting
                if hasattr(shap_values, 'values'):
                    values = shap_values.values
                else:
                    values = shap_values
                
                # Flatten values for plotting
                if len(values.shape) > 2:
                    values = values.reshape(values.shape[0], -1)
                
                # Create heatmap
                heatmap = go.Heatmap(
                    z=values,
                    colorscale='RdBu',
                    showscale=True,
                    name=group
                )
                
                fig.add_trace(heatmap, row=i+1, col=1)
        
        fig.update_layout(
            title=f"SHAP Summary Plot - {task.title()} Task",
            height=300 * len(bias_results),
            width=800
        )
        
        return fig
    
    def create_bias_comparison_plot(self, bias_results: Dict, task: str,
                                  demographic_attr: str) -> go.Figure:
        """Create bias comparison plot across demographic groups."""
        groups = list(bias_results.keys())
        bias_scores = []
        
        for group in groups:
            if 'analysis' in bias_results[group]:
                analysis = bias_results[group]['analysis']
                if 'feature_importance' in analysis:
                    bias_scores.append(analysis['feature_importance']['mean_importance'])
                else:
                    bias_scores.append(0)
            else:
                bias_scores.append(0)
        
        fig = go.Figure(data=[
            go.Bar(
                x=groups,
                y=bias_scores,
                text=[f"{score:.4f}" for score in bias_scores],
                textposition='auto',
                marker_color=['red' if score > 0.1 else 'green' for score in bias_scores]
            )
        ])
        
        fig.update_layout(
            title=f"Bias Comparison Across {demographic_attr.title()} Groups - {task.title()}",
            xaxis_title=demographic_attr.title(),
            yaxis_title="Bias Score",
            height=500,
            width=700
        )
        
        return fig
    
    def create_token_importance_plot(self, bias_results: Dict, task: str,
                                   demographic_attr: str) -> go.Figure:
        """Create token importance plot."""
        all_tokens = {}
        
        # Collect all bias tokens
        for group, results in bias_results.items():
            if 'analysis' in results and 'bias_indicators' in results['analysis']:
                for token_info in results['analysis']['bias_indicators']:
                    token = token_info['token']
                    if token not in all_tokens:
                        all_tokens[token] = {
                            'groups': [],
                            'importance': [],
                            'bias_type': token_info['bias_type'],
                            'category': token_info['category']
                        }
                    
                    all_tokens[token]['groups'].append(group)
                    all_tokens[token]['importance'].append(token_info['importance'])
        
        # Create plot
        tokens = list(all_tokens.keys())
        importance_means = [np.mean(all_tokens[t]['importance']) for t in tokens]
        bias_types = [all_tokens[t]['bias_type'] for t in tokens]
        
        fig = go.Figure(data=[
            go.Bar(
                x=tokens,
                y=importance_means,
                text=[f"{imp:.4f}" for imp in importance_means],
                textposition='auto',
                marker_color=['red' if bt == 'gender' else 'blue' for bt in bias_types]
            )
        ])
        
        fig.update_layout(
            title=f"Token Importance Analysis - {task.title()}",
            xaxis_title="Tokens",
            yaxis_title="Mean SHAP Importance",
            height=600,
            width=800
        )
        
        return fig
    
    def create_demographic_parity_plot(self, bias_results: Dict, task: str,
                                     demographic_attr: str) -> go.Figure:
        """Create demographic parity plot."""
        groups = list(bias_results.keys())
        parity_scores = []
        
        # Calculate demographic parity scores
        for group in groups:
            if 'analysis' in bias_results[group]:
                analysis = bias_results[group]['analysis']
                if 'feature_importance' in analysis:
                    # Use feature importance as proxy for parity
                    parity_scores.append(analysis['feature_importance']['mean_importance'])
                else:
                    parity_scores.append(0)
            else:
                parity_scores.append(0)
        
        # Calculate overall mean
        overall_mean = np.mean(parity_scores)
        
        # Calculate parity violations
        parity_violations = [abs(score - overall_mean) for score in parity_scores]
        
        fig = go.Figure(data=[
            go.Bar(
                x=groups,
                y=parity_violations,
                text=[f"{violation:.4f}" for violation in parity_violations],
                textposition='auto',
                marker_color=['red' if violation > 0.05 else 'green' for violation in parity_violations]
            )
        ])
        
        fig.update_layout(
            title=f"Demographic Parity Violations - {task.title()}",
            xaxis_title=demographic_attr.title(),
            yaxis_title="Parity Violation Score",
            height=500,
            width=700
        )
        
        return fig
    
    def save_visualizations(self, visualizations: Dict, task: str, 
                          demographic_attr: str) -> None:
        """Save visualizations to disk."""
        output_dir = Path(self.config.visualizations_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for viz_name, fig in visualizations.items():
            output_path = output_dir / f"{task}_{demographic_attr}_{viz_name}.html"
            fig.write_html(str(output_path))
            
            # Also save as PNG
            png_path = output_dir / f"{task}_{demographic_attr}_{viz_name}.png"
            fig.write_image(str(png_path))
            
            self.logger.info(f"Saved {viz_name} to {output_path}")
    
    def generate_bias_report(self, bias_results: Dict, task: str, 
                           demographic_attr: str) -> Dict:
        """Generate comprehensive bias report."""
        self.logger.info(f"Generating bias report for {task}")
        
        report = {
            'task': task,
            'demographic_attribute': demographic_attr,
            'summary': {},
            'detailed_analysis': {},
            'recommendations': []
        }
        
        # Summary statistics
        groups = list(bias_results.keys())
        bias_scores = []
        
        for group in groups:
            if 'analysis' in bias_results[group]:
                analysis = bias_results[group]['analysis']
                if 'feature_importance' in analysis:
                    bias_scores.append(analysis['feature_importance']['mean_importance'])
                else:
                    bias_scores.append(0)
            else:
                bias_scores.append(0)
        
        report['summary'] = {
            'total_groups': len(groups),
            'mean_bias_score': float(np.mean(bias_scores)),
            'std_bias_score': float(np.std(bias_scores)),
            'max_bias_score': float(np.max(bias_scores)),
            'min_bias_score': float(np.min(bias_scores)),
            'bias_threshold_exceeded': sum(1 for score in bias_scores if score > self.config.fairness.bias_threshold)
        }
        
        # Detailed analysis
        for group in groups:
            if 'analysis' in bias_results[group]:
                analysis = bias_results[group]['analysis']
                report['detailed_analysis'][group] = {
                    'bias_score': bias_scores[groups.index(group)],
                    'bias_indicators_count': len(analysis.get('bias_indicators', [])),
                    'feature_importance': analysis.get('feature_importance', {}),
                    'bias_types': list(set([ind['bias_type'] for ind in analysis.get('bias_indicators', [])]))
                }
        
        # Generate recommendations
        report['recommendations'] = self.generate_recommendations(bias_results, task)
        
        return report
    
    def generate_recommendations(self, bias_results: Dict, task: str) -> List[str]:
        """Generate recommendations based on bias analysis."""
        recommendations = []
        
        # Analyze bias patterns
        high_bias_groups = []
        bias_types = set()
        
        for group, results in bias_results.items():
            if 'analysis' in results:
                analysis = results['analysis']
                if 'feature_importance' in analysis:
                    bias_score = analysis['feature_importance']['mean_importance']
                    if bias_score > self.config.fairness.bias_threshold:
                        high_bias_groups.append(group)
                
                # Collect bias types
                for indicator in analysis.get('bias_indicators', []):
                    bias_types.add(indicator['bias_type'])
        
        # Generate specific recommendations
        if high_bias_groups:
            recommendations.append(
                f"High bias detected in groups: {', '.join(high_bias_groups)}. "
                "Consider data augmentation or debiasing techniques."
            )
        
        if 'gender' in bias_types:
            recommendations.append(
                "Gender bias detected. Consider gender-neutral language training "
                "or balanced gender representation in training data."
            )
        
        if 'cultural' in bias_types:
            recommendations.append(
                "Cultural bias detected. Consider diverse cultural contexts "
                "in training data and evaluation."
            )
        
        if task == 'sentiment':
            recommendations.append(
                "For sentiment analysis, ensure balanced sentiment distribution "
                "across demographic groups."
            )
        elif task == 'translation':
            recommendations.append(
                "For translation tasks, consider language-specific cultural "
                "nuances and context."
            )
        
        return recommendations
    
    def save_bias_report(self, report: Dict, task: str, demographic_attr: str) -> None:
        """Save bias report to disk."""
        output_dir = Path(self.config.reports_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{task}_{demographic_attr}_bias_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Saved bias report to {output_path}")

def main():
    """Main function for SHAP analysis pipeline."""
    config = Config()
    explainer = SHAPExplainer(config)
    
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
    
    # Analyze sentiment bias
    sentiment_data, _ = preprocessor.load_processed_data('sentiment')
    if sentiment_data:
        for demographic_attr in ['gender', 'age_group', 'region']:
            if demographic_attr in sentiment_data['test'].columns:
                # Analyze bias for each model
                for model_name in ['mbert', 'english', 'german', 'hindi']:
                    model_path = config.get_model_path(f"sentiment_{model_name}")
                    if os.path.exists(model_path):
                        bias_results = explainer.analyze_sentiment_bias(
                            model_path, sentiment_data['test'], demographic_attr
                        )
                        
                        # Create visualizations
                        visualizations = explainer.create_bias_visualizations(
                            bias_results, 'sentiment', demographic_attr
                        )
                        
                        # Save visualizations
                        explainer.save_visualizations(visualizations, 'sentiment', demographic_attr)
                        
                        # Generate and save report
                        report = explainer.generate_bias_report(
                            bias_results, 'sentiment', demographic_attr
                        )
                        explainer.save_bias_report(report, 'sentiment', demographic_attr)
    
    # Analyze translation bias
    translation_data, _ = preprocessor.load_processed_data('translation')
    if translation_data:
        for demographic_attr in ['gender', 'age_group', 'region']:
            if demographic_attr in translation_data['test'].columns:
                model_path = config.get_model_path("translation_mbert")
                if os.path.exists(model_path):
                    bias_results = explainer.analyze_translation_bias(
                        model_path, translation_data['test'], demographic_attr
                    )
                    
                    # Create visualizations
                    visualizations = explainer.create_bias_visualizations(
                        bias_results, 'translation', demographic_attr
                    )
                    
                    # Save visualizations
                    explainer.save_visualizations(visualizations, 'translation', demographic_attr)
                    
                    # Generate and save report
                    report = explainer.generate_bias_report(
                        bias_results, 'translation', demographic_attr
                    )
                    explainer.save_bias_report(report, 'translation', demographic_attr)

if __name__ == "__main__":
    main() 