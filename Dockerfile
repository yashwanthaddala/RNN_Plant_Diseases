FROM tensorflow/tensorflow:2.13.0-lite


RUN pip install --no-cache-dir numpy pillow edgeimpulse-learning-blocks

WORKDIR /app
COPY . /app

ENTRYPOINT ["python3", "model_training.py"]
