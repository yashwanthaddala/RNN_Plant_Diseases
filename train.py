import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import numpy as np

# === Required Edge Impulse arguments ===
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "/data")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "/output")
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "model.h5")

# Load training and testing data from Edge Impulse
train_features = np.load(os.path.join(DATA_DIRECTORY, "X_train.npy"))
train_labels = np.load(os.path.join(DATA_DIRECTORY, "y_train.npy"))
test_features = np.load(os.path.join(DATA_DIRECTORY, "X_test.npy"))
test_labels = np.load(os.path.join(DATA_DIRECTORY, "y_test.npy"))

# Normalize data
train_features = train_features / 255.0
test_features = test_features / 255.0

# Reshape for RNN input: (samples, timesteps, features)
# For simplicity, we treat image rows as time steps
n_samples, img_height, img_width, channels = train_features.shape
train_features = train_features.reshape((n_samples, img_height, img_width * channels))
test_features = test_features.reshape((test_features.shape[0], img_height, img_width * channels))

num_classes = len(np.unique(train_labels))

# === Model Definition ===
model = Sequential([
    LSTM(64, input_shape=(img_height, img_width * channels), return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Training ===
history = model.fit(
    train_features, train_labels,
    validation_data=(test_features, test_labels),
    epochs=10,
    batch_size=32
)

# === Save model ===
model.save(MODEL_FILE)

# === Save metadata ===
metadata = {
    "accuracy": float(history.history["val_accuracy"][-1]),
    "loss": float(history.history["val_loss"][-1]),
}
with open(os.path.join(MODEL_OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f)
