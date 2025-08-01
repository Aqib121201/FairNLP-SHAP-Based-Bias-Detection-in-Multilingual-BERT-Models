# FairNLP Dockerfile
# Multi-stage build for optimized container

# Base stage with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install additional development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external \
    models visualizations reports logs configs

# Set permissions
RUN chmod +x run_pipeline.py

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external \
    models visualizations reports logs configs

# Set permissions
RUN chmod +x run_pipeline.py

# Create non-root user
RUN useradd -m -u 1000 fairnlp && \
    chown -R fairnlp:fairnlp /app

USER fairnlp

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 