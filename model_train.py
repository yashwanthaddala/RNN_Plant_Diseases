import tensorflow as tf
from tensorflow.keras import layers, models
from ei_tensorflow.training import train_model

def build_model(input_shape, num_classes, **kwargs):
    """
    Build an RNN (LSTM) model that treats the image as a sequence of rows.
    """
    model = models.Sequential([
        layers.Reshape(
            (input_shape[0], input_shape[1] * input_shape[2]),
            input_shape=input_shape
        ),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # This is all Edge Impulse needs
    train_model(build_model)
