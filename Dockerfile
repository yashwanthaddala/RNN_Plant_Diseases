FROM tensorflow/tensorflow:2.12.0

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "train.py"]
