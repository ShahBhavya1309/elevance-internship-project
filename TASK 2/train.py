"""
train.py  -  Train a multi-task CNN (age regression + gender classification) on UTKFace.

Usage:
    python train.py --data data/UTKFace
    python train.py --data data/UTKFace --head-epochs 8 --tune-epochs 20

UTKFace file names look like:  [age]_[gender]_[race]_[datetime].jpg   (gender: 0=male, 1=female)

Outputs:
    models/age_gender_model.keras   final model used by app.py
    models/config.json              calibrated senior threshold (chosen on validation set)
    results/metrics.json            MAE, gender accuracy/F1, senior precision/recall/F1
    results/training_curves.png     loss / age MAE / gender accuracy curves
    results/confusion_matrix.png    gender + senior confusion matrices
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 160
BATCH = 64
SEED = 42
SENIOR_AGE = 60          # senior if age > 60
MODEL_DIR = "models"
RESULT_DIR = "results"


# ----------------------------------------------------------------------------- data
def load_labels(data_dir):
    paths, ages, genders = [], [], []
    for p in glob.glob(os.path.join(data_dir, "*.jpg")):
        parts = os.path.basename(p).split("_")
        try:
            age, gender = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue                                  # skip badly named files
        if gender not in (0, 1) or not (1 <= age <= 100):
            continue
        paths.append(p)
        ages.append(age)
        genders.append(gender)
    if not paths:
        raise SystemExit(f"No valid UTKFace images found in '{data_dir}'.")
    print(f"Loaded {len(paths)} images.")
    return np.array(paths), np.array(ages, "float32"), np.array(genders, "float32")


def split_data(ages):
    """70/15/15 split, stratified by decade of age so every split has seniors."""
    idx = np.arange(len(ages))
    bins = np.minimum(ages // 10, 8).astype(int)
    train, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=bins)
    val, test = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=bins[tmp])
    return train, val, test


def make_ds(paths, ages, genders, training):
    def _load(path, age, gender):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))          # float32, 0..255
        labels = {"age": tf.reshape(age / 100.0, (1,)),          # age scaled to 0..1
                  "gender": tf.reshape(gender, (1,))}
        return img, labels

    ds = tf.data.Dataset.from_tensor_slices((paths, ages, genders))
    if training:
        ds = ds.shuffle(min(len(paths), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------- model
def build_model():
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    aug = keras.Sequential([                      # only active while training
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.25, value_range=(0, 255)),
        layers.RandomContrast(0.25),
    ], name="augmentation")
    x = aug(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]

    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                          input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    base.trainable = False
    x = base(x, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    a = layers.Dense(64, activation="relu")(x)
    age = layers.Dense(1, activation="sigmoid", name="age")(a)         # age / 100
    g = layers.Dense(64, activation="relu")(x)
    gender = layers.Dense(1, activation="sigmoid", name="gender")(g)   # P(female)

    model = keras.Model(inp, {"age": age, "gender": gender}, name="age_gender_net")
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss={"age": "mean_absolute_error", "gender": "binary_crossentropy"},
        loss_weights={"age": 3.0, "gender": 1.0},
        metrics={"age": ["mae"], "gender": ["accuracy"]},
    )


def make_callbacks():
    os.makedirs(MODEL_DIR, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]


# ----------------------------------------------------------------------------- evaluation
def predict_arrays(model, ds):
    out = model.predict(ds, verbose=0)
    if isinstance(out, dict):
        age, gender = out["age"], out["gender"]
    else:
        age, gender = out[0], out[1]
    return age.ravel() * 100.0, gender.ravel()


def calibrate_threshold(true_age, pred_age):
    """Pick the age cut-off that gives the best F1 for 'true age > 60' on the VALIDATION set.
    Regression models under-predict old ages, so the best cut-off is often a bit below 60."""
    true_sen = true_age > SENIOR_AGE
    best_t, best_f1 = float(SENIOR_AGE), -1.0
    for t in np.arange(45, 66.5, 0.5):
        f1 = f1_score(true_sen, pred_age > t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def plot_cm(ax, cm, labels, title):
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")


def evaluate(model, ds, true_age, true_gender, history, cal_threshold):
    os.makedirs(RESULT_DIR, exist_ok=True)
    pred_age, pred_gprob = predict_arrays(model, ds)
    pred_g = (pred_gprob > 0.5).astype(int)
    true_g = true_gender.astype(int)

    mae = float(np.mean(np.abs(pred_age - true_age)))
    print(f"\nAge MAE: {mae:.2f} years")
    print("\nGender report:\n", classification_report(true_g, pred_g,
                                                      target_names=["Male", "Female"], digits=3))

    metrics = {"age_mae_years": mae,
               "gender_accuracy": float((pred_g == true_g).mean()),
               "gender_f1_macro": float(f1_score(true_g, pred_g, average="macro"))}

    true_sen = true_age > SENIOR_AGE
    results = {}
    for name, thr in [("threshold_60", SENIOR_AGE), ("threshold_calibrated", cal_threshold)]:
        ps = pred_age > thr
        results[name] = {"threshold": thr,
                         "precision": float(precision_score(true_sen, ps, zero_division=0)),
                         "recall": float(recall_score(true_sen, ps, zero_division=0)),
                         "f1": float(f1_score(true_sen, ps, zero_division=0))}
        print(f"Senior (>60) @ age>{thr}:  P={results[name]['precision']:.3f}  "
              f"R={results[name]['recall']:.3f}  F1={results[name]['f1']:.3f}")
    metrics["senior"] = results

    with open(os.path.join(RESULT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_cm(axes[0], confusion_matrix(true_g, pred_g), ["Male", "Female"], "Gender")
    plot_cm(axes[1], confusion_matrix(true_sen, pred_age > SENIOR_AGE), ["Not senior", "Senior"],
            "Senior (age > 60)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)

    # training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (k, title, scale) in zip(axes, [("loss", "Loss", 1), ("age_mae", "Age MAE (years)", 100),
                                            ("gender_accuracy", "Gender accuracy", 1)]):
        ax.plot(np.array(history.get(k, [])) * scale, label="train")
        ax.plot(np.array(history.get("val_" + k, [])) * scale, label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "training_curves.png"), dpi=130)
    plt.close(fig)
    print(f"\nSaved metrics and plots to '{RESULT_DIR}/'.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/UTKFace")
    ap.add_argument("--head-epochs", type=int, default=8, help="epochs with frozen backbone")
    ap.add_argument("--tune-epochs", type=int, default=20, help="epochs fine-tuning backbone")
    ap.add_argument("--limit", type=int, default=0, help="use only N images (quick smoke test)")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    paths, ages, genders = load_labels(args.data)
    if args.limit:
        sel = np.random.RandomState(SEED).permutation(len(paths))[:args.limit]
        paths, ages, genders = paths[sel], ages[sel], genders[sel]

    tr, va, te = split_data(ages)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "test_files.txt"), "w") as f:   # demo uses ONLY unseen images
        f.write("\n".join(str(p) for p in paths[te]))

    # class imbalance: seniors are rare -> repeat images of age >= 55 three times in TRAIN only
    seniors = tr[ages[tr] >= 55]
    tr = np.concatenate([tr, seniors, seniors])
    print(f"train={len(tr)} (after oversampling)  val={len(va)}  test={len(te)}")

    train_ds = make_ds(paths[tr], ages[tr], genders[tr], training=True)
    val_ds = make_ds(paths[va], ages[va], genders[va], training=False)
    test_ds = make_ds(paths[te], ages[te], genders[te], training=False)

    model, base = build_model()

    # phase 1: train the new heads only
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=make_callbacks())

    # phase 2: fine-tune the top 50 backbone layers (BatchNorm stays frozen)
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-4)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.tune_epochs, callbacks=make_callbacks())

    history = {k: h1.history.get(k, []) + h2.history.get(k, [])
               for k in set(h1.history) | set(h2.history)}

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "age_gender_model.keras"))

    # choose a senior cut-off on validation data (never on test data)
    val_pred_age, _ = predict_arrays(model, val_ds)
    cal_t, cal_f1 = calibrate_threshold(ages[va], val_pred_age)
    print(f"Calibrated senior threshold (validation): {cal_t}  (F1={cal_f1:.3f})")
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump({"img_size": IMG_SIZE, "senior_threshold_calibrated": cal_t}, f, indent=2)

    evaluate(model, test_ds, ages[te], genders[te], history, cal_t)


if __name__ == "__main__":
    main()
