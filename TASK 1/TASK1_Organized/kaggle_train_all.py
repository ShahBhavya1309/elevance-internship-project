# ============================================================
# COMPLETE TRAINING SCRIPT - paste this whole block in ONE cell
# ============================================================

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

IMG_SIZE = 96

# ---------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------

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
    base = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
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


# ---------------------------------------------------
# PART 1: TRAIN AGE + GENDER MODEL (UTKFace)
# ---------------------------------------------------

AGE_GENDER_DATASET_DIR = "/kaggle/input/datasets/jangedoo/utkface-new/UTKFace"
AGE_EPOCHS = 12
AGE_BATCH_SIZE = 64
AGE_MAX_IMAGES = 12000

def load_age_gender_data(dataset_dir, img_size=IMG_SIZE, max_images=None):
    images, ages, genders = [], [], []
    files = sorted(os.listdir(dataset_dir))
    if max_images:
        files = files[:max_images]
    for fname in files:
        parts = fname.split("_")
        if len(parts) < 3:
            continue
        try:
            age = int(parts[0])
            gender = int(parts[1])
        except ValueError:
            continue
        img_path = os.path.join(dataset_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        ages.append(age)
        genders.append(gender)
    images = np.array(images, dtype="float32") / 255.0
    ages = np.array(ages, dtype="float32")
    genders = np.array(genders, dtype="float32")
    print(f"Loaded {len(images)} images for age/gender model.")
    return images, ages, genders


print("=== Training Age/Gender Model ===")
images, ages, genders = load_age_gender_data(AGE_GENDER_DATASET_DIR, max_images=AGE_MAX_IMAGES)

X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val = train_test_split(
    images, ages, genders, test_size=0.15, random_state=42
)

age_gender_model = build_age_gender_model()
age_gender_model.summary()

age_gender_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        "age_gender_weights.weights.h5", save_best_only=True, monitor="val_loss", save_weights_only=True
    ),
]

age_gender_model.fit(
    X_train,
    {"age": y_age_train, "gender": y_gender_train},
    validation_data=(X_val, {"age": y_age_val, "gender": y_gender_val}),
    epochs=AGE_EPOCHS,
    batch_size=AGE_BATCH_SIZE,
    callbacks=age_gender_callbacks,
)

age_gender_model.save_weights("age_gender_weights.weights.h5")
print("Saved age_gender_weights.weights.h5")

# free memory before next training
del images, ages, genders, X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val
import gc; gc.collect()


# ---------------------------------------------------
# PART 2: TRAIN HAIR-LENGTH MODEL (Gender Classification Dataset as proxy)
# ---------------------------------------------------

HAIR_DATASET_DIR = "/kaggle/input/datasets/cashutosh/gender-classification-dataset/Training"
HAIR_CLASS_NAMES = ["male", "female"]   # male->0 (Short proxy), female->1 (Long proxy)
HAIR_BATCH_SIZE = 64
HAIR_EPOCHS = 6

print("\n=== Training Hair-Length Model ===")

hair_train_ds = tf.keras.utils.image_dataset_from_directory(
    HAIR_DATASET_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=HAIR_CLASS_NAMES,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=HAIR_BATCH_SIZE,
    validation_split=0.15,
    subset="training",
    seed=42,
)
hair_val_ds = tf.keras.utils.image_dataset_from_directory(
    HAIR_DATASET_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=HAIR_CLASS_NAMES,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=HAIR_BATCH_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=42,
)

AUTOTUNE = tf.data.AUTOTUNE
hair_train_ds = hair_train_ds.prefetch(AUTOTUNE)
hair_val_ds = hair_val_ds.prefetch(AUTOTUNE)

hair_model = build_hair_length_model()
hair_model.summary()

hair_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]

hair_model.fit(hair_train_ds, validation_data=hair_val_ds, epochs=HAIR_EPOCHS, callbacks=hair_callbacks)

hair_model.save_weights("hair_length_weights.weights.h5")
print("Saved hair_length_weights.weights.h5")

print("\n=== DONE. Download age_gender_weights.weights.h5 and hair_length_weights.weights.h5 from Output. ===")
