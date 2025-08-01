#!/usr/bin/env python3
"""
FairNLP Pipeline Orchestrator

This script orchestrates the complete FairNLP pipeline for SHAP-based bias detection
in multilingual BERT models. It coordinates data preprocessing, model training,
fairness analysis, and report generation.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.explainability import SHAPExplainer
from src.model_utils import ModelUtils

def setup_logging(config: Config) -> None:
    """Set up logging configuration."""
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format=config.logging.log_format,
        handlers=[
            logging.FileHandler(config.get_log_path()),
            logging.StreamHandler()
        ]
    )

def run_data_preprocessing(config: Config) -> Dict:
    """Run data preprocessing pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing pipeline")
    
    preprocessor = DataPreprocessor(config)
    results = {}
    
    # Process sentiment data
    sentiment_file = config.get_data_path('raw', 'sentiment_data.csv')
    if os.path.exists(sentiment_file):
        logger.info("Processing sentiment data")
        sentiment_df = preprocessor.load_sentiment_data(sentiment_file)
        processed_sentiment, encoders = preprocessor.prepare_sentiment_dataset(sentiment_df)
        sentiment_splits = preprocessor.split_dataset(processed_sentiment, 'sentiment')
        preprocessor.save_processed_data(sentiment_splits, 'sentiment')
        
        # Get statistics
        stats = preprocessor.get_dataset_statistics(processed_sentiment, 'sentiment')
        results['sentiment'] = {
            'processed_samples': len(processed_sentiment),
            'statistics': stats
        }
        logger.info(f"Processed {len(processed_sentiment)} sentiment samples")
    else:
        logger.warning("Sentiment data file not found")
    
    # Process translation data
    translation_file = config.get_data_path('raw', 'translation_data.csv')
    if os.path.exists(translation_file):
        logger.info("Processing translation data")
        translation_df = preprocessor.load_translation_data(translation_file)
        processed_translation, encoders = preprocessor.prepare_translation_dataset(translation_df)
        translation_splits = preprocessor.split_dataset(processed_translation, 'translation')
        preprocessor.save_processed_data(translation_splits, 'translation')
        
        # Get statistics
        stats = preprocessor.get_dataset_statistics(processed_translation, 'translation')
        results['translation'] = {
            'processed_samples': len(processed_translation),
            'statistics': stats
        }
        logger.info(f"Processed {len(processed_translation)} translation samples")
    else:
        logger.warning("Translation data file not found")
    
    logger.info("Data preprocessing completed")
    return results

def run_model_training(config: Config) -> Dict:
    """Run model training pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training pipeline")
    
    trainer = ModelTrainer(config)
    preprocessor = DataPreprocessor(config)
    results = {}
    
    # Train sentiment models
    sentiment_data, _ = preprocessor.load_processed_data('sentiment')
    if sentiment_data:
        logger.info("Training sentiment models")
        sentiment_results = {}
        
        for model_name in ['mbert', 'english', 'german', 'hindi']:
            if model_name in trainer.models:
                logger.info(f"Training {model_name} sentiment model")
                try:
                    model_result = trainer.train_sentiment_model(
                        sentiment_data['train'],
                        sentiment_data['validation'],
                        model_name
                    )
                    sentiment_results[model_name] = model_result
                    
                    # Evaluate on test set
                    test_results = trainer.evaluate_model(
                        model_result['model_path'],
                        sentiment_data['test'],
                        'sentiment'
                    )
                    sentiment_results[model_name]['test_results'] = test_results
                    
                    logger.info(f"Completed training for {model_name}")
                except Exception as e:
                    logger.error(f"Error training {model_name} model: {e}")
                    sentiment_results[model_name] = {'error': str(e)}
        
        results['sentiment'] = sentiment_results
    else:
        logger.warning("No sentiment data available for training")
    
    # Train translation models
    translation_data, _ = preprocessor.load_processed_data('translation')
    if translation_data:
        logger.info("Training translation models")
        translation_results = {}
        
        for model_name in ['mbert']:  # Only mBERT for translation
            if model_name in trainer.models:
                logger.info(f"Training {model_name} translation model")
                try:
                    model_result = trainer.train_translation_model(
                        translation_data['train'],
                        translation_data['validation'],
                        model_name
                    )
                    translation_results[model_name] = model_result
                    
                    # Evaluate on test set
                    test_results = trainer.evaluate_model(
                        model_result['model_path'],
                        translation_data['test'],
                        'translation'
                    )
                    translation_results[model_name]['test_results'] = test_results
                    
                    logger.info(f"Completed training for {model_name}")
                except Exception as e:
                    logger.error(f"Error training {model_name} model: {e}")
                    translation_results[model_name] = {'error': str(e)}
        
        results['translation'] = translation_results
    else:
        logger.warning("No translation data available for training")
    
    logger.info("Model training completed")
    return results

def run_fairness_analysis(config: Config) -> Dict:
    """Run fairness analysis pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting fairness analysis pipeline")
    
    explainer = SHAPExplainer(config)
    utils = ModelUtils(config)
    preprocessor = DataPreprocessor(config)
    results = {}
    
    # Analyze sentiment bias
    sentiment_data, _ = preprocessor.load_processed_data('sentiment')
    if sentiment_data:
        logger.info("Analyzing sentiment bias")
        sentiment_results = {}
        
        for demographic_attr in ['gender', 'age_group', 'region']:
            if demographic_attr in sentiment_data['test'].columns:
                logger.info(f"Analyzing {demographic_attr} bias in sentiment")
                attr_results = {}
                
                for model_name in ['mbert', 'english', 'german', 'hindi']:
                    model_path = config.get_model_path(f"sentiment_{model_name}")
                    if os.path.exists(model_path):
                        try:
                            # SHAP analysis
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
                            
                            attr_results[model_name] = {
                                'bias_results': bias_results,
                                'report': report
                            }
                            
                            logger.info(f"Completed {demographic_attr} analysis for {model_name}")
                        except Exception as e:
                            logger.error(f"Error analyzing {demographic_attr} bias for {model_name}: {e}")
                            attr_results[model_name] = {'error': str(e)}
                
                sentiment_results[demographic_attr] = attr_results
        
        results['sentiment'] = sentiment_results
    else:
        logger.warning("No sentiment data available for bias analysis")
    
    # Analyze translation bias
    translation_data, _ = preprocessor.load_processed_data('translation')
    if translation_data:
        logger.info("Analyzing translation bias")
        translation_results = {}
        
        for demographic_attr in ['gender', 'age_group', 'region']:
            if demographic_attr in translation_data['test'].columns:
                logger.info(f"Analyzing {demographic_attr} bias in translation")
                attr_results = {}
                
                model_name = 'mbert'
                model_path = config.get_model_path(f"translation_{model_name}")
                if os.path.exists(model_path):
                    try:
                        # SHAP analysis
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
                        
                        attr_results[model_name] = {
                            'bias_results': bias_results,
                            'report': report
                        }
                        
                        logger.info(f"Completed {demographic_attr} analysis for {model_name}")
                    except Exception as e:
                        logger.error(f"Error analyzing {demographic_attr} bias for {model_name}: {e}")
                        attr_results[model_name] = {'error': str(e)}
                
                translation_results[demographic_attr] = attr_results
    else:
        logger.warning("No translation data available for bias analysis")
    
    results['translation'] = translation_results
    
    logger.info("Fairness analysis completed")
    return results

def generate_final_report(config: Config, preprocessing_results: Dict, 
                         training_results: Dict, fairness_results: Dict) -> None:
    """Generate final comprehensive report."""
    logger = logging.getLogger(__name__)
    logger.info("Generating final report")
    
    report = {
        'project_info': {
            'name': 'FairNLP: SHAP-Based Bias Detection in Multilingual BERT Models',
            'version': '1.0.0',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'model': config.model.__dict__,
                'data': config.data.__dict__,
                'fairness': config.fairness.__dict__,
                'shap': config.shap.__dict__
            }
        },
        'preprocessing_results': preprocessing_results,
        'training_results': training_results,
        'fairness_results': fairness_results,
        'summary': {}
    }
    
    # Generate summary statistics
    summary = {
        'total_models_trained': 0,
        'total_bias_analyses': 0,
        'high_bias_models': 0,
        'recommendations': []
    }
    
    # Count models and analyses
    if 'sentiment' in training_results:
        summary['total_models_trained'] += len(training_results['sentiment'])
    if 'translation' in training_results:
        summary['total_models_trained'] += len(training_results['translation'])
    
    if 'sentiment' in fairness_results:
        for attr in fairness_results['sentiment']:
            summary['total_bias_analyses'] += len(fairness_results['sentiment'][attr])
    if 'translation' in fairness_results:
        for attr in fairness_results['translation']:
            summary['total_bias_analyses'] += len(fairness_results['translation'][attr])
    
    # Identify high bias models
    for task in ['sentiment', 'translation']:
        if task in fairness_results:
            for attr in fairness_results[task]:
                for model_name, results in fairness_results[task][attr].items():
                    if 'report' in results and 'summary' in results['report']:
                        bias_score = results['report']['summary'].get('mean_bias_score', 0)
                        if bias_score > config.fairness.bias_threshold:
                            summary['high_bias_models'] += 1
                            summary['recommendations'].append(
                                f"High bias detected in {task} {model_name} model for {attr}"
                            )
    
    report['summary'] = summary
    
    # Save final report
    output_path = Path(config.reports_dir) / "final_report.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Final report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("FAIRNLP PIPELINE COMPLETED")
    print("="*80)
    print(f"Total models trained: {summary['total_models_trained']}")
    print(f"Total bias analyses: {summary['total_bias_analyses']}")
    print(f"High bias models detected: {summary['high_bias_models']}")
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  - {rec}")
    print("\nReports and visualizations saved to:")
    print(f"  - Reports: {config.reports_dir}")
    print(f"  - Visualizations: {config.visualizations_dir}")
    print(f"  - Models: {config.models_dir}")
    print("="*80)

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="FairNLP Pipeline Orchestrator")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing step'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip fairness analysis step'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['sentiment', 'translation', 'both'],
        default='both',
        help='Which task to run (sentiment, translation, or both)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting FairNLP pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Task: {args.task}")
    
    # Initialize results
    preprocessing_results = {}
    training_results = {}
    fairness_results = {}
    
    try:
        # Step 1: Data Preprocessing
        if not args.skip_preprocessing:
            logger.info("Step 1: Data Preprocessing")
            preprocessing_results = run_data_preprocessing(config)
        else:
            logger.info("Skipping data preprocessing")
        
        # Step 2: Model Training
        if not args.skip_training:
            logger.info("Step 2: Model Training")
            training_results = run_model_training(config)
        else:
            logger.info("Skipping model training")
        
        # Step 3: Fairness Analysis
        if not args.skip_analysis:
            logger.info("Step 3: Fairness Analysis")
            fairness_results = run_fairness_analysis(config)
        else:
            logger.info("Skipping fairness analysis")
        
        # Step 4: Generate Final Report
        logger.info("Step 4: Generating Final Report")
        generate_final_report(config, preprocessing_results, training_results, fairness_results)
        
        logger.info("FairNLP pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 