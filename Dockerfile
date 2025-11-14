FROM python:3.10-slim

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory and copy code
WORKDIR /app
COPY . /app

# Set entrypoint for Edge Impulse
ENTRYPOINT ["python3", "model_train.py"]
