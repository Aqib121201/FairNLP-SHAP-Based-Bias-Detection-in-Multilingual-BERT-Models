"""
FairNLP: SHAP-Based Bias Detection in Multilingual BERT Models

A comprehensive framework for detecting and analyzing bias in multilingual
language models using SHAP values and fairness metrics.
"""

__version__ = "1.0.0"
__author__ = "FairNLP Research Team"
__email__ = "fairnlp@research.institution.edu"

# Main imports
from .config import Config
from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .model_utils import ModelUtils
from .explainability import SHAPExplainer

__all__ = [
    "Config",
    "DataPreprocessor", 
    "ModelTrainer",
    "ModelUtils",
    "SHAPExplainer"
] 