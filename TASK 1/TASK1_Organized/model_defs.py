"""
model_defs.py
-------------
Shared model architecture definitions.

Why this file exists: saving/loading a FULL Keras model (architecture +
weights together, e.g. via model.save('x.h5') / load_model('x.h5')) can
break across different TensorFlow/Keras versions (e.g. training on Kaggle's
newer TF/Keras vs. running locally on an older one causes errors like
"Unrecognized keyword arguments: ['batch_shape', 'optional']").

The fix: define the model architecture ONCE here, share it between the
training scripts and core.py, and only save/load the WEIGHTS (a plain
numeric file that has no version-specific serialization format). As long as
the architecture defined here matches on both ends, this works reliably
across TensorFlow versions.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 96


def build_age_gender_model(img_size=IMG_SIZE):
    inputs = layers.Input(shape=(img_size, img_size, 3))

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    age_output = layers.Dense(1, activation="linear", name="age")(x)
    gender_output = layers.Dense(1, activation="sigmoid", name="gender")(x)

    model = models.Model(inputs=inputs, outputs=[age_output, gender_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"age": "mse", "gender": "binary_crossentropy"},
        loss_weights={"age": 1.0, "gender": 5.0},
        metrics={"age": "mae", "gender": "accuracy"},
    )
    return model


def build_hair_length_model(img_size=IMG_SIZE):
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=None
    )
    # The complete trained backbone is restored from hair_length_weights.weights.h5
    # in core.py, so inference does not need an ImageNet download.
    base.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="hair_length")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
