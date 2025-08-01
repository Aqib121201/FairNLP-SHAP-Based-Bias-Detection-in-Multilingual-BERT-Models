# FairNLP Makefile
# Common commands for project management

.PHONY: help install setup test clean run-pipeline run-dashboard build-docker run-docker

# Default target
help:
	@echo "FairNLP: SHAP-Based Bias Detection in Multilingual BERT Models"
	@echo ""
	@echo "Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  setup          - Setup project environment"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean generated files"
	@echo "  run-pipeline   - Run complete pipeline"
	@echo "  run-dashboard  - Start Streamlit dashboard"
	@echo "  build-docker   - Build Docker image"
	@echo "  run-docker     - Run Docker container"
	@echo "  format         - Format code with black"
	@echo "  lint           - Lint code with flake8"
	@echo "  docs           - Generate documentation"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Setup project environment
setup:
	@echo "Setting up project environment..."
	mkdir -p data/{raw,processed,external}
	mkdir -p models visualizations reports logs configs
	python src/config.py
	@echo "Project setup complete!"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Test coverage report generated in htmlcov/"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ htmlcov/
	rm -rf logs/*.log
	@echo "Cleanup complete!"

# Run complete pipeline
run-pipeline:
	@echo "Running FairNLP pipeline..."
	python run_pipeline.py

# Run pipeline with specific task
run-pipeline-sentiment:
	@echo "Running sentiment analysis pipeline..."
	python run_pipeline.py --task sentiment

run-pipeline-translation:
	@echo "Running translation pipeline..."
	python run_pipeline.py --task translation

# Start Streamlit dashboard
run-dashboard:
	@echo "Starting Streamlit dashboard..."
	streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0

# Build Docker image
build-docker:
	@echo "Building Docker image..."
	docker build -t fairnlp:latest -f docker/Dockerfile .

# Build development Docker image
build-docker-dev:
	@echo "Building development Docker image..."
	docker build -t fairnlp:dev --target development -f docker/Dockerfile .

# Run Docker container
run-docker:
	@echo "Running Docker container..."
	docker run -p 8501:8501 -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models fairnlp:latest

# Run Docker container in pipeline mode
run-docker-pipeline:
	@echo "Running Docker container in pipeline mode..."
	docker run -e FAIRNLP_ACTION=pipeline -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models fairnlp:latest

# Format code
format:
	@echo "Formatting code with black..."
	black src/ tests/ app/ --line-length 88

# Lint code
lint:
	@echo "Linting code with flake8..."
	flake8 src/ tests/ app/ --max-line-length 88 --ignore E203,W503

# Generate documentation
docs:
	@echo "Generating documentation..."
	pdoc --html src/ --output-dir docs/
	@echo "Documentation generated in docs/"

# Data preprocessing only
preprocess:
	@echo "Running data preprocessing..."
	python run_pipeline.py --skip-training --skip-analysis

# Model training only
train:
	@echo "Running model training..."
	python run_pipeline.py --skip-preprocessing --skip-analysis

# Bias analysis only
analyze:
	@echo "Running bias analysis..."
	python run_pipeline.py --skip-preprocessing --skip-training

# Run individual components
run-preprocessing:
	@echo "Running data preprocessing..."
	python src/data_preprocessing.py

run-training:
	@echo "Running model training..."
	python src/model_training.py

run-explainability:
	@echo "Running SHAP analysis..."
	python src/explainability.py

# Development commands
dev-install:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install black flake8 mypy pytest pytest-cov

dev-setup: dev-install setup
	@echo "Development environment setup complete!"

# Quick start
quick-start: setup install run-dashboard
	@echo "Quick start complete! Dashboard should be available at http://localhost:8501"

# Full pipeline with data generation
full-pipeline:
	@echo "Running full pipeline with sample data generation..."
	python -c "
import pandas as pd
import numpy as np
import os

# Create sample sentiment data
sentiment_data = pd.DataFrame({
    'text': [
        'This product is amazing!',
        'I hate this service.',
        'The quality is okay.',
        'Great experience overall.',
        'Terrible customer support.',
        'She is a great doctor.',
        'He is a terrible nurse.',
        'They are excellent teachers.'
    ],
    'label': [2, 0, 1, 2, 0, 2, 0, 2],
    'language': ['en', 'en', 'en', 'en', 'en', 'en', 'en', 'en']
})

# Create sample translation data
translation_data = pd.DataFrame({
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

# Save sample data
os.makedirs('data/raw', exist_ok=True)
sentiment_data.to_csv('data/raw/sentiment_data.csv', index=False)
translation_data.to_csv('data/raw/translation_data.csv', index=False)
print('Sample data generated successfully!')
"
	python run_pipeline.py
	@echo "Full pipeline completed!"

# Show project status
status:
	@echo "=== FairNLP Project Status ==="
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "Installed packages:"
	@pip list | grep -E "(torch|transformers|shap|streamlit)"
	@echo ""
	@echo "Directory structure:"
	@ls -la
	@echo ""
	@echo "Data files:"
	@ls -la data/raw/ 2>/dev/null || echo "No raw data files found"
	@echo ""
	@echo "Models:"
	@ls -la models/ 2>/dev/null || echo "No models found"
	@echo ""
	@echo "Reports:"
	@ls -la reports/ 2>/dev/null || echo "No reports found" 