#!/bin/bash

# FairNLP Docker Entrypoint Script

set -e

echo "🚀 Starting FairNLP Application..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for dependencies
wait_for_dependencies() {
    echo "⏳ Checking dependencies..."
    
    # Check if Python is available
    if ! command_exists python; then
        echo "❌ Python not found"
        exit 1
    fi
    
    # Check if required packages are installed
    python -c "import torch, transformers, shap, streamlit" 2>/dev/null || {
        echo "❌ Required Python packages not found"
        exit 1
    }
    
    echo "✅ Dependencies check passed"
}

# Function to initialize directories
init_directories() {
    echo "📁 Initializing directories..."
    
    mkdir -p /app/data/{raw,processed,external}
    mkdir -p /app/models
    mkdir -p /app/visualizations
    mkdir -p /app/reports
    mkdir -p /app/logs
    mkdir -p /app/configs
    
    echo "✅ Directories initialized"
}

# Function to check configuration
check_config() {
    echo "⚙️ Checking configuration..."
    
    if [ ! -f "/app/configs/default.yaml" ]; then
        echo "📝 Generating default configuration..."
        cd /app && python -c "from src.config import Config; Config().save_to_file('configs/default.yaml')"
    fi
    
    echo "✅ Configuration check passed"
}

# Function to run data preprocessing (if data exists)
run_preprocessing() {
    if [ -f "/app/data/raw/sentiment_data.csv" ] || [ -f "/app/data/raw/translation_data.csv" ]; then
        echo "🔄 Running data preprocessing..."
        cd /app && python run_pipeline.py --skip-training --skip-analysis
    else
        echo "ℹ️ No raw data found, skipping preprocessing"
    fi
}

# Function to start Streamlit
start_streamlit() {
    echo "🌐 Starting Streamlit dashboard..."
    
    # Set Streamlit configuration
    export STREAMLIT_SERVER_PORT=8501
    export STREAMLIT_SERVER_ADDRESS=0.0.0.0
    export STREAMLIT_SERVER_HEADLESS=true
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    
    # Start Streamlit
    cd /app && streamlit run app/app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false
}

# Function to run pipeline
run_pipeline() {
    echo "🔧 Running FairNLP pipeline..."
    cd /app && python run_pipeline.py
}

# Main execution
main() {
    echo "🧠 FairNLP Container Starting..."
    echo "=================================="
    
    # Wait for dependencies
    wait_for_dependencies
    
    # Initialize directories
    init_directories
    
    # Check configuration
    check_config
    
    # Determine action based on environment variable
    if [ "$FAIRNLP_ACTION" = "pipeline" ]; then
        echo "🔧 Running pipeline mode..."
        run_preprocessing
        run_pipeline
    elif [ "$FAIRNLP_ACTION" = "preprocessing" ]; then
        echo "🔄 Running preprocessing mode..."
        run_preprocessing
    elif [ "$FAIRNLP_ACTION" = "dashboard" ]; then
        echo "🌐 Running dashboard mode..."
        start_streamlit
    else
        # Default: start dashboard
        echo "🌐 Starting dashboard (default mode)..."
        start_streamlit
    fi
}

# Handle signals
trap 'echo "🛑 Received signal, shutting down..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@" 