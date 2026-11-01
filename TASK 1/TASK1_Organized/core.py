"""
core.py
-------
Inference pipeline using YOUR OWN trained models (produced by
kaggle_train_all.py on Kaggle):
    - age_gender_weights.weights.h5
    - hair_length_weights.weights.h5

NOTE: we load WEIGHTS into a freshly rebuilt architecture (from
model_defs.py) rather than loading a full saved model file. This avoids
"Unrecognized keyword arguments: batch_shape / optional" errors that happen
when a model is trained on one TensorFlow/Keras version (e.g. Kaggle's) and
loaded on a different one (e.g. your local install). The ".weights.h5"
extension (not just ".h5") is required by newer Keras when saving
weights-only files.

Pipeline:
  1. Detect the face (OpenCV Haar Cascade).
  2. Predict AGE and REAL GENDER from the face crop.
  3. Predict HAIR LENGTH (Short/Long) from a wider head+shoulders crop.
  4. Apply the assignment's OVERRIDE RULE:
        - age in [20, 30]  -> final gender decided PURELY by hair length
              long hair  -> "Female"
              short hair -> "Male"
          (even if it contradicts the real model's gender prediction)
        - age < 20 or > 30 -> final gender = the real model's prediction
          (hair length is ignored)
"""

import os
import cv2
import numpy as np
from model_defs import build_age_gender_model, build_hair_length_model, IMG_SIZE

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AGE_GENDER_WEIGHTS_PATH = os.path.join(PROJECT_DIR, "age_gender_weights.weights.h5")
HAIR_WEIGHTS_PATH = os.path.join(PROJECT_DIR, "hair_length_weights.weights.h5")

_age_gender_model = None
_hair_model = None


class DetectionError(Exception):
    """Raised when no face can be found in the image."""
    pass


class ModelNotFoundError(Exception):
    """Raised when the trained .weights.h5 files are missing."""
    pass


def _load_models():
    global _age_gender_model, _hair_model
    if not os.path.exists(AGE_GENDER_WEIGHTS_PATH):
        raise ModelNotFoundError(
            f"'{AGE_GENDER_WEIGHTS_PATH}' not found. Train it on Kaggle using "
            "kaggle_train_all.py and download that file next to core.py."
        )
    if not os.path.exists(HAIR_WEIGHTS_PATH):
        raise ModelNotFoundError(
            f"'{HAIR_WEIGHTS_PATH}' not found. Train it on Kaggle using "
            "kaggle_train_all.py and download that file next to core.py."
        )
    if _age_gender_model is None:
        _age_gender_model = build_age_gender_model()
        _age_gender_model.load_weights(AGE_GENDER_WEIGHTS_PATH)
    if _hair_model is None:
        _hair_model = build_hair_length_model()
        _hair_model.load_weights(HAIR_WEIGHTS_PATH)


def detect_face_box(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        raise DetectionError("No face detected in the image.")
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]


def _crop(bgr_image, x1, y1, x2, y2):
    H, W = bgr_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return bgr_image[y1:y2, x1:x2]


def estimate_age_and_real_gender(bgr_image, face_box):
    """Uses age_gender_model.h5 on a tight face crop."""
    _load_models()
    x, y, w, h = face_box
    pad_w, pad_h = int(0.25 * w), int(0.25 * h)
    face = _crop(bgr_image, x - pad_w, y - pad_h, x + w + pad_w, y + h + pad_h)

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    batch = np.expand_dims(face_resized, axis=0)

    age_pred, gender_pred = _age_gender_model.predict(batch, verbose=0)
    age = max(0, int(round(float(age_pred[0][0]))))
    gender_prob = float(gender_pred[0][0])  # 0 = male .. 1 = female (UTKFace convention)
    real_gender = "Female" if gender_prob >= 0.5 else "Male"
    confidence = round((gender_prob if real_gender == "Female" else 1 - gender_prob) * 100, 1)
    return age, real_gender, confidence


def estimate_hair_length(bgr_image, face_box):
    """Uses hair_length_model.h5 on a wider head+shoulders crop."""
    _load_models()
    x, y, w, h = face_box
    x1 = x - int(0.4 * w)
    y1 = y - int(0.3 * h)
    x2 = x + w + int(0.4 * w)
    y2 = y + int(1.8 * h)
    roi = _crop(bgr_image, x1, y1, x2, y2)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE)).astype("float32")
    batch = np.expand_dims(roi_resized, axis=0)

    prob = float(_hair_model.predict(batch, verbose=0)[0][0])  # 0 = short .. 1 = long
    label = "Long" if prob >= 0.5 else "Short"
    return label, round(prob, 3)


def predict(bgr_image):
    """
    Full pipeline. Returns a dict with all intermediate values plus the
    final displayed gender, so the GUI can show the "why" behind the answer.
    """
    face_box = detect_face_box(bgr_image)
    age, real_gender, gender_conf = estimate_age_and_real_gender(bgr_image, face_box)
    hair_label, hair_score = estimate_hair_length(bgr_image, face_box)

    in_target_age_range = 20 <= age <= 30

    if in_target_age_range:
        # OVERRIDE RULE: ignore the real detected gender, decide purely by hair
        final_gender = "Female" if hair_label == "Long" else "Male"
        rule_used = "hair-length override (age 20-30)"
    else:
        final_gender = real_gender
        rule_used = "real model prediction (age outside 20-30)"

    return {
        "face_box": face_box,
        "age": age,
        "real_gender": real_gender,
        "gender_confidence": gender_conf,
        "hair_label": hair_label,
        "hair_score": hair_score,
        "in_target_age_range": in_target_age_range,
        "final_gender": final_gender,
        "rule_used": rule_used,
    }
