"""
Configuration management for FairNLP project.

This module contains all configuration parameters, paths, and settings
for the bias detection pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for BERT models."""
    
    # Model names and paths
    mbert_model: str = "bert-base-multilingual-cased"
    english_bert: str = "bert-base-uncased"
    german_bert: str = "bert-base-german-cased"
    hindi_bert: str = "bert-base-hindi"
    
    # Model parameters
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Training parameters
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Model saving
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    external_data_dir: str = "data/external"
    
    # Dataset parameters
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Text processing
    min_text_length: int = 10
    max_text_length: int = 500
    remove_special_chars: bool = True
    lowercase: bool = False  # Keep case for BERT
    
    # Languages
    supported_languages: List[str] = field(default_factory=lambda: ["en", "de", "hi"])
    
    # Demographic attributes
    demographic_attributes: List[str] = field(default_factory=lambda: [
        "gender", "age", "region", "education"
    ])

@dataclass
class FairnessConfig:
    """Configuration for fairness metrics."""
    
    # Bias detection parameters
    bias_threshold: float = 0.1
    confidence_level: float = 0.95
    
    # Metrics to compute
    fairness_metrics: List[str] = field(default_factory=lambda: [
        "demographic_parity",
        "equalized_odds", 
        "equal_opportunity",
        "kl_divergence",
        "sentiment_polarity_bias"
    ])
    
    # Protected attributes
    protected_attributes: List[str] = field(default_factory=lambda: [
        "gender", "age_group", "region"
    ])
    
    # Bias mitigation strategies
    mitigation_strategies: List[str] = field(default_factory=lambda: [
        "adversarial_debiasing",
        "reweighing",
        "preprocessing",
        "postprocessing"
    ])

@dataclass
class SHAPConfig:
    """Configuration for SHAP analysis."""
    
    # SHAP parameters
    background_samples: int = 100
    nsamples: int = 100
    l1_reg: str = "auto"
    
    # Explanation types
    explanation_types: List[str] = field(default_factory=lambda: [
        "summary_plot",
        "waterfall_plot",
        "force_plot",
        "dependence_plot"
    ])
    
    # Visualization settings
    plot_height: int = 400
    plot_width: int = 600
    save_format: str = "png"
    dpi: int = 300

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    
    # Random seeds
    random_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"
    
    # Evaluation metrics
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "auc"
    ])
    
    # Statistical testing
    statistical_tests: List[str] = field(default_factory=lambda: [
        "mann_whitney_u",
        "chi_square",
        "t_test"
    ])

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    # Logging levels
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Log files
    log_dir: str = "logs"
    log_file: str = "fairnlp.log"
    
    # Logging components
    log_components: List[str] = field(default_factory=lambda: [
        "data_preprocessing",
        "model_training", 
        "fairness_analysis",
        "shap_analysis"
    ])

@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    # Base paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    visualizations_dir: Path = base_dir / "visualizations"
    reports_dir: Path = base_dir / "report"
    notebooks_dir: Path = base_dir / "notebooks"
    tests_dir: Path = base_dir / "tests"
    app_dir: Path = base_dir / "app"
    
    # Create directories if they don't exist
    def __post_init__(self):
        for path in [self.data_dir, self.models_dir, self.visualizations_dir, 
                    self.reports_dir, self.notebooks_dir, self.tests_dir, self.app_dir]:
            path.mkdir(parents=True, exist_ok=True)

class Config:
    """Main configuration class that combines all sub-configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.model = ModelConfig()
        self.data = DataConfig()
        self.fairness = FairnessConfig()
        self.shap = SHAPConfig()
        self.experiment = ExperimentConfig()
        self.logging = LoggingConfig()
        self.paths = PathConfig()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'fairness': self.fairness.__dict__,
            'shap': self.shap.__dict__,
            'experiment': self.experiment.__dict__,
            'logging': self.logging.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_model_path(self, model_name: str) -> str:
        """Get path for a specific model."""
        return str(self.paths.models_dir / f"{model_name}")
    
    def get_data_path(self, data_type: str, filename: str) -> str:
        """Get path for data files."""
        if data_type == "raw":
            return str(self.paths.data_dir / "raw" / filename)
        elif data_type == "processed":
            return str(self.paths.data_dir / "processed" / filename)
        elif data_type == "external":
            return str(self.paths.data_dir / "external" / filename)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def get_visualization_path(self, filename: str) -> str:
        """Get path for visualization files."""
        return str(self.paths.visualizations_dir / filename)
    
    def get_log_path(self) -> str:
        """Get path for log file."""
        return str(self.paths.base_dir / self.logging.log_dir / self.logging.log_file)

# Default configuration instance
config = Config()

# Environment-specific configurations
def get_config(environment: str = "default") -> Config:
    """Get configuration for specific environment."""
    config_path = f"configs/{environment}.yaml"
    if os.path.exists(config_path):
        return Config(config_path)
    else:
        return Config()

if __name__ == "__main__":
    # Save default configuration
    config.save_to_file("configs/default.yaml")
    print("Default configuration saved to configs/default.yaml") 