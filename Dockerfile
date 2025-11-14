FROM python:3.10-slim
RUN pip install --no-cache-dir tensorflow==2.13.0 numpy pandas edgeimpulse-learning-blocks


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app

ENTRYPOINT ["python3", "model_train.py"]
