FROM tensorflow/tensorflow:2.13.0-lite

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app

ENTRYPOINT ["python3", "model_train.py"]
