import tensorflow as tf
from tensorflow.keras import layers, models
import ei_tensorflow.training

def load_model(input_shape, num_classes, **kwargs):
    """
    Builds and returns an RNN-based model for image classification.
    Edge Impulse automatically calls this function.
    """

    # Convert 2D image into a sequence for the RNN
    # Flatten image into sequence of rows
    model = models.Sequential()

    # Reshape image: (H, W, C) -> (H, W*C)
    model.add(layers.Reshape((input_shape[0], input_shape[1] * input_shape[2]),
                             input_shape=input_shape))

    # First LSTM layer
    model.add(layers.LSTM(128, return_sequences=True))

    # Second LSTM layer
    model.add(layers.LSTM(64))

    # Dense classifier
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train(model, dataset, **kwargs):
    """
    Training loop used by Edge Impulse.
    """

    X_train, X_test, y_train, y_test = dataset

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=kwargs.get("epochs", 20),
        batch_size=kwargs.get("batch_size", 32),
        verbose=2
    )

    return history


def evaluate(model, dataset, **kwargs):
    X_train, X_test, y_train, y_test = dataset
    return model.evaluate(X_test, y_test, verbose=0)
