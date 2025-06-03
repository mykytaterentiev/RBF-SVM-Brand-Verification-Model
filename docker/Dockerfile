FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY setup.py .
COPY pyproject.toml .

# Install package
RUN pip install -e .

# Create data and models directories
RUN mkdir -p data models logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV DATA_PATH=/app/data
ENV MODEL_OUTPUT_DIR=/app/models

# Default command
CMD ["python", "scripts/train_model.py", "--help"]
