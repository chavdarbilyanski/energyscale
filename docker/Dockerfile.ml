# Use a Python base image with ML dependencies
FROM python:3.10-slim

# Install system dependencies for ML (e.g., for TensorFlow, PyTorch)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy ML requirements
COPY requirements-ml.txt .

# Install ML dependencies
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy ML code
COPY src/ml/ .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Command for ML inference (modify based on your needs)
CMD ["python", "inference/api.py"]