FROM python:3.10-slim

# Install system dependencies for TensorFlow and Edge Impulse
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory and copy code
WORKDIR /app
COPY . /app

# Entry point for Edge Impulse
ENTRYPOINT ["python3", "model_train.py"]
