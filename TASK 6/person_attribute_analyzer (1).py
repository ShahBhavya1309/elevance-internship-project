"""
PERSON ATTRIBUTE ANALYZER  -  emotion / age / dress colour with nationality-based output
=========================================================================================
IMPORTANT DESIGN NOTE
  A person's nationality CANNOT be determined from a face image (looks do not show
  citizenship; a "face -> nationality" model would just guess ethnicity from
  appearance, which is unreliable and discriminatory).  So in this app the
  NATIONALITY IS CHOSEN BY THE USER in the GUI, and the model predicts only things that
  can really be seen in a picture: EMOTION, AGE and DRESS COLOUR.

OUTPUT RULES (driven by the selected nationality)
  India                       -> Age + Dress colour + Emotion
  United States               -> Age + Emotion
  Africa (any African country)-> Emotion + Dress colour
  Any other country           -> Nationality (the country chosen) + Emotion

NO DATASET DOWNLOAD
  * 5 emotions, age and 11 dress colours are learned from portraits that this script
    DRAWS itself (random skin tone, hair, wrinkles, expression, clothing).
    (skin tone is random and unrelated to any label)
  * Models: RandomForest classifier (emotion), RandomForest regressor (age),
            RandomForest classifier (dress colour)  - scikit-learn.
  * The built-in emotion / age models understand the generated portraits.
  * For REAL photos: face + dress colour work with plain OpenCV; for real-photo
    emotion + age install the optional DeepFace package  (pip install deepface)
    - it downloads its pretrained weights on first use.

INSTALL (Python 3.9+):
    pip install opencv-python scikit-learn pillow numpy
    (optional, real-photo emotion/age)  pip install deepface

RUN:
    python person_attribute_analyzer.py
"""

import os
import math
import pickle
import importlib.util

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MODEL_FILE = "person_model_cache.pkl"
MODEL_VERSION = 1
PW, PH = 360, 440                     # size of generated portraits
PANEL_W, PANEL_H = 360, 440
THUMB_W, THUMB_H = 72, 88
N_SAMPLES = 10
HAS_DEEPFACE = importlib.util.find_spec("deepface") is not None

EMOTIONS = ["Happy", "Sad", "Angry", "Surprised", "Neutral"]
AFRICAN_COUNTRIES = [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon",
    "Central African Republic", "Chad", "Comoros", "Congo (DRC)", "Congo (Republic)", "Cote d'Ivoire",
    "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia",
    "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi",
    "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
    "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa",
    "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"]
OTHER_COUNTRIES = [
    "Afghanistan", "Albania", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahrain",
    "Bangladesh", "Belarus", "Belgium", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Brazil", "Brunei",
    "Bulgaria", "Cambodia", "Canada", "Chile", "China", "Colombia", "Costa Rica", "Croatia", "Cuba",
    "Cyprus", "Czech Republic", "Denmark", "Dominican Republic", "Ecuador", "El Salvador", "Estonia",
    "Fiji", "Finland", "France", "Georgia", "Germany", "Greece", "Guatemala", "Haiti", "Honduras",
    "Hong Kong", "Hungary", "Iceland", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
    "Lithuania", "Luxembourg", "Malaysia", "Maldives", "Malta", "Mexico", "Moldova", "Mongolia",
    "Montenegro", "Myanmar", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palestine", "Panama", "Paraguay", "Peru",
    "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Saudi Arabia", "Serbia",
    "Singapore", "Slovakia", "Slovenia", "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland",
    "Syria", "Taiwan", "Tajikistan", "Thailand", "Trinidad and Tobago", "Turkey", "Turkmenistan",
    "Ukraine", "United Arab Emirates", "United Kingdom", "Uruguay", "Uzbekistan", "Venezuela",
    "Vietnam", "Yemen"]
NATIONALITIES = sorted(["India", "United States"] + AFRICAN_COUNTRIES + OTHER_COUNTRIES)
_AFRICAN_LOWER = {c.lower() for c in AFRICAN_COUNTRIES} | {"africa", "african"}
_US_NAMES = {"united states", "united states of america", "usa", "u.s.a.", "us", "u.s.", "american"}


def resolve_group(nationality):
    """Map the chosen nationality to one of: India / United States / Africa / Other."""
    t = nationality.strip().lower()
    if t in ("india", "indian"):
        return "India"
    if t in _US_NAMES:
        return "United States"
    if t in _AFRICAN_LOWER:
        return "Africa"
    return "Other"

# OpenCV HSV ranges (H 0-179) for dress colours
DRESS_RANGES = {
    "Red":    ((-6, 6),    (160, 255), (110, 230)),
    "Blue":   ((100, 125), (150, 255), (80, 230)),
    "Green":  ((40, 80),   (100, 230), (70, 190)),
    "Yellow": ((22, 32),   (150, 255), (190, 255)),
    "Orange": ((8, 18),    (190, 255), (190, 255)),
    "Purple": ((130, 155), (100, 255), (80, 230)),
    "Pink":   ((160, 175), (50, 130),  (200, 255)),
    "Brown":  ((8, 20),    (120, 220), (50, 120)),
    "White":  ((0, 179),   (0, 25),    (205, 255)),
    "Black":  ((0, 179),   (0, 60),    (5, 55)),
    "Grey":   ((0, 179),   (0, 30),    (80, 190)),
}
DRESS_NAMES = list(DRESS_RANGES.keys())

# random skin tones used by the portrait generator (BGR) - NOT linked to any label
SKIN_TONES = [(190, 210, 245), (170, 200, 235), (150, 185, 225), (125, 165, 205),
              (100, 140, 185), (80, 115, 160), (60, 90, 130), (45, 65, 100)]
HAIR_BASES = [(25, 25, 30), (30, 45, 70), (40, 70, 110), (110, 170, 215), (40, 60, 150)]


def sample_bgr(ranges, name, rng):
    (h0, h1), (s0, s1), (v0, v1) = ranges[name]
    hsv = np.uint8([[[rng.uniform(h0, h1) % 180, rng.uniform(s0, s1), rng.uniform(v0, v1)]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def darker(c, f):
    return tuple(int(np.clip(v * f, 0, 255)) for v in c)


# ----------------------------------------------------------------------------
# BUILT-IN PORTRAIT GENERATOR (replaces a downloaded dataset)
# ----------------------------------------------------------------------------
FACE_ROI = (100, 100, 160, 195)      # x, y, w, h  in a generated portrait
DRESS_ROI = (120, 350, 120, 80)


def generate_portrait(rng, emotion=None, age=None, dress=None):
    emotion = emotion or str(rng.choice(EMOTIONS))
    age = float(age) if age is not None else float(rng.uniform(4, 85))
    dress = dress or str(rng.choice(DRESS_NAMES))

    img = np.empty((PH, PW, 3), np.uint8)
    img[:] = tuple(int(v) for v in rng.integers(190, 245, 3))
    skin = SKIN_TONES[int(rng.integers(len(SKIN_TONES)))]
    skin_d = darker(skin, 0.78)
    dress_col = sample_bgr(DRESS_RANGES, dress, rng)

    k = 0.84 + 0.16 * float(np.clip((age - 4) / 16.0, 0, 1))
    rx, ry = int(66 * k), int(84 * k)
    cx, cy = 180, int(195 + (1 - k) * 60)

    grey = float(np.clip((age - 38) / 32.0 + rng.normal(0, 0.1), 0, 1))
    base = HAIR_BASES[int(rng.integers(len(HAIR_BASES)))]
    hair = tuple(int(base[i] * (1 - grey) + 205 * grey) for i in range(3))

    # neck, dress, v-neck
    cv2.rectangle(img, (cx - 24, cy + int(0.8 * ry)), (cx + 24, 330), skin_d, -1)
    cv2.ellipse(img, (cx, 470), (150, 170), 0, 180, 360, dress_col, -1)
    cv2.fillPoly(img, [np.array([(cx - 28, 304), (cx + 28, 304), (cx, 345)], np.int32)], skin_d)
    cv2.ellipse(img, (cx, 470), (150, 170), 0, 180, 360, darker(dress_col, 0.8), 3)

    # hair behind head
    thin = age > 60 and rng.random() < 0.4
    if not thin:
        cv2.ellipse(img, (cx, cy - int(0.1 * ry)), (rx + 8, ry + 6), 0, 0, 360, hair, -1)
        if 10 < age < 60 and rng.random() < 0.35:
            cv2.ellipse(img, (cx, cy + int(0.3 * ry)), (rx + 14, int(ry * 1.05)), 0, 0, 360, hair, -1)
    # ears + head
    for sx in (-1, 1):
        cv2.ellipse(img, (cx + sx * rx, cy), (8, 16), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, skin, -1)
    # fringe / hairline
    fh = rng.uniform(0.72, 0.78) if age < 60 else rng.uniform(0.75, 0.95)
    m = np.zeros((PH, PW), np.uint8)
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    m[int(cy - ry * fh):, :] = 0
    img[m > 0] = hair

    # eyes
    ex, ey = int(0.48 * rx), cy - int(0.06 * ry)
    child = age < 13
    erx = int(0.2 * rx * (1.15 if child else 1.0))
    ery = int(0.1 * ry * (1.4 if child else (1.0 if age < 60 else 0.75)))
    if emotion == "Surprised":
        ery = int(ery * 1.7)
    elif emotion == "Angry":
        ery = max(3, int(ery * 0.7))
    for sx in (-1, 1):
        cv2.ellipse(img, (cx + sx * ex, ey), (erx, ery), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(img, (cx + sx * ex, ey), max(2, int(min(erx, ery) * 0.65)), (30, 40, 55), -1)

    # eyebrows
    yb = cy - int(0.30 * ry)
    oy, iy, arch = {"Neutral": (0, 0, 0), "Happy": (-2, -2, -3), "Sad": (8, -8, 0),
                    "Angry": (-6, 8, 0), "Surprised": (-16, -16, -6)}[emotion]
    brow_c = darker(hair, 0.7) if grey < 0.6 else (150, 150, 150)
    for sx in (-1, 1):
        outer = (cx + sx * (ex + int(0.22 * rx)), yb + oy)
        inner = (cx + sx * (ex - int(0.22 * rx)), yb + iy)
        mid = ((outer[0] + inner[0]) // 2, (outer[1] + inner[1]) // 2 + arch)
        cv2.polylines(img, [np.array([outer, mid, inner], np.int32)], False, brow_c,
                      3 if age > 60 else 4, cv2.LINE_AA)

    # nose
    ny = cy + int(0.3 * ry)
    cv2.line(img, (cx, cy + int(0.05 * ry)), (cx - 3, ny), skin_d, 2, cv2.LINE_AA)
    cv2.circle(img, (cx - 7, ny + 4), 2, skin_d, -1)
    cv2.circle(img, (cx + 7, ny + 4), 2, skin_d, -1)

    # mouth
    lip = (70, 70, 170)
    my, mw = cy + int(0.52 * ry), int(0.42 * rx)
    if emotion == "Happy":
        c0, ax = (cx, my - int(0.1 * ry)), (mw, int(0.28 * ry))
        if rng.random() < 0.6:
            cv2.ellipse(img, c0, (mw - 3, ax[1] - 2), 0, 15, 165, (245, 245, 245), -1)
        cv2.ellipse(img, c0, ax, 0, 5, 175, lip, 3, cv2.LINE_AA)
    elif emotion == "Sad":
        cv2.ellipse(img, (cx, my + int(0.2 * ry)), (mw, int(0.22 * ry)), 0, 200, 340, lip, 3, cv2.LINE_AA)
    elif emotion == "Angry":
        if rng.random() < 0.5:
            cv2.rectangle(img, (cx - int(mw * 0.8), my - 5), (cx + int(mw * 0.8), my + 7), (245, 245, 245), -1)
        cv2.line(img, (cx - mw, my + 2), (cx + mw, my + 2), lip, 4, cv2.LINE_AA)
    elif emotion == "Surprised":
        cv2.ellipse(img, (cx, my + 4), (int(0.2 * rx), int(0.22 * ry)), 0, 0, 360, (40, 40, 90), -1)
        cv2.ellipse(img, (cx, my + 4), (int(0.2 * rx), int(0.22 * ry)), 0, 0, 360, lip, 2, cv2.LINE_AA)
    else:
        cv2.line(img, (cx - mw, my), (cx + mw, my), lip, 3, cv2.LINE_AA)

    # age wrinkles
    nw = int(np.clip((age - 35) / 10.0, 0, 5))
    for i in range(min(nw, 3)):
        y = cy - int(0.68 * ry) + i * 7
        cv2.polylines(img, [np.array([(cx - int(0.45 * rx), y + 1), (cx, y - 1), (cx + int(0.45 * rx), y + 1)], np.int32)],
                      False, skin_d, 1, cv2.LINE_AA)
    if nw >= 2:
        for sx in (-1, 1):
            cv2.line(img, (cx + sx * int(0.22 * rx), cy + int(0.25 * ry)),
                     (cx + sx * int(0.36 * rx), cy + int(0.5 * ry)), skin_d, 1, cv2.LINE_AA)
    if nw >= 3:
        for sx in (-1, 1):
            x0 = cx + sx * (ex + erx + 2)
            cv2.line(img, (x0, ey), (x0 + sx * 9, ey - 5), skin_d, 1, cv2.LINE_AA)
            cv2.line(img, (x0, ey + 2), (x0 + sx * 9, ey + 6), skin_d, 1, cv2.LINE_AA)
    if nw >= 4:
        for sx in (-1, 1):
            cv2.ellipse(img, (cx + sx * ex, ey + ery + 6), (erx, 5), 0, 20, 160, skin_d, 1, cv2.LINE_AA)

    # camera effects
    M = cv2.getRotationMatrix2D((PW / 2, PH / 2), rng.uniform(-4, 4), 1.0)
    img = cv2.warpAffine(img, M, (PW, PH), borderMode=cv2.BORDER_REPLICATE)
    img = np.clip(img.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, {"emotion": emotion, "age": age, "dress": dress}


# ----------------------------------------------------------------------------
# FEATURES
# ----------------------------------------------------------------------------
_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def face_features(face_bgr):
    g = cv2.cvtColor(cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    hog = _HOG.compute(cv2.equalizeHist(g)).ravel()
    tiny = cv2.resize(g, (12, 12), interpolation=cv2.INTER_AREA).ravel() / 255.0
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F))
    tex = [lap[r0:r1, c0:c1].mean() / 255.0 for r0, r1, c0, c1 in
           ((12, 22, 12, 52), (22, 30, 8, 24), (22, 30, 40, 56), (36, 50, 10, 24), (36, 50, 40, 54))]
    return np.concatenate([hog, tiny, tex, [g[:12].mean() / 255.0, g[:12].std() / 255.0, g.mean() / 255.0]])


def colour_features(patch_bgr):
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].ravel().astype(np.float32)
    s = hsv[..., 1].ravel().astype(np.float32)
    v = hsv[..., 2].ravel().astype(np.float32)
    n = max(1, h.size)
    chroma = (s > 40) & (v > 45)
    hist, _ = np.histogram(h[chroma], bins=18, range=(0, 180))
    f = list(hist / n)
    f += [np.mean(v < 50), np.mean((s < 40) & (v >= 185)), np.mean((s < 40) & (v >= 50) & (v < 185)),
          s.mean() / 255.0, v.mean() / 255.0]
    if chroma.any():
        ang = h[chroma] * (2 * math.pi / 180.0)
        f += [float(np.cos(ang).mean()), float(np.sin(ang).mean()),
              float(s[chroma].mean() / 255.0), float(v[chroma].mean() / 255.0)]
    else:
        f += [0.0, 0.0, 0.0, 0.0]
    return np.array(f, dtype=np.float32)


def render_dress_patch(name, rng):
    p = np.empty((50, 70, 3), np.float32)
    p[:] = sample_bgr(DRESS_RANGES, name, rng)
    p *= np.linspace(rng.uniform(0.8, 1.05), rng.uniform(0.8, 1.05), 70)[None, :, None]
    for _ in range(int(rng.integers(0, 3))):               # folds
        x = int(rng.integers(0, 70))
        p[:, max(0, x - 2): x + 2] *= rng.uniform(0.8, 0.95)
    if rng.random() < 0.4:                                  # a bit of skin / neck
        sk = SKIN_TONES[int(rng.integers(len(SKIN_TONES)))]
        a, b = int(rng.integers(3, 12)), int(rng.integers(8, 25))
        p[:a, 35 - b // 2: 35 + b // 2] = sk
    p = p * rng.uniform(0.85, 1.08) + rng.normal(0, 4, p.shape)
    return np.clip(p, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------
class Models:
    def __init__(self):
        self.emo = self.age = self.dress = None
        self.metrics = {}

    def load_or_train(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as fh:
                    d = pickle.load(fh)
                if d.get("version") == MODEL_VERSION:
                    self.emo, self.age, self.dress, self.metrics = d["emo"], d["age"], d["dress"], d["metrics"]
                    return
            except Exception:
                pass
        self.train()

    def train(self, n_faces=2400, n_dress=260, seed=11):
        rng = np.random.default_rng(seed)
        X, ye, ya = [], [], []
        for _ in range(n_faces):
            img, t = generate_portrait(rng)
            x, y, w, h = FACE_ROI
            X.append(face_features(img[y:y + h, x:x + w]))
            ye.append(t["emotion"]); ya.append(t["age"])
        X, ye, ya = np.array(X), np.array(ye), np.array(ya)
        Xtr, Xva, ye_tr, ye_va, ya_tr, ya_va = train_test_split(X, ye, ya, test_size=0.2, random_state=0)
        p1 = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0).fit(Xtr, ye_tr)
        p2 = RandomForestRegressor(n_estimators=150, max_features="sqrt", n_jobs=-1, random_state=0).fit(Xtr, ya_tr)
        self.metrics["emotion_acc"] = float(p1.score(Xva, ye_va))
        self.metrics["age_mae"] = float(np.mean(np.abs(p2.predict(Xva) - ya_va)))
        self.emo = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(X, ye)
        self.age = RandomForestRegressor(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=0).fit(X, ya)

        Xd, yd = [], []
        for name in DRESS_NAMES:
            for _ in range(n_dress):
                Xd.append(colour_features(render_dress_patch(name, rng))); yd.append(name)
        Xd, yd = np.array(Xd), np.array(yd)
        a, b, c, d = train_test_split(Xd, yd, test_size=0.2, random_state=0, stratify=yd)
        pd_ = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(a, c)
        self.metrics["dress_acc"] = float(pd_.score(b, d))
        self.dress = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Xd, yd)
        with open(MODEL_FILE, "wb") as fh:
            pickle.dump({"version": MODEL_VERSION, "emo": self.emo, "age": self.age,
                         "dress": self.dress, "metrics": self.metrics}, fh)

    def predict_face(self, face_bgr):
        f = face_features(face_bgr)
        p = self.emo.predict_proba([f])[0]
        i = int(np.argmax(p))
        return str(self.emo.classes_[i]), float(p[i]), float(self.age.predict([f])[0])

    def predict_dress(self, patch_bgr):
        p = self.dress.predict_proba([colour_features(patch_bgr)])[0]
        i = int(np.argmax(p))
        return str(self.dress.classes_[i]), float(p[i])


# ----------------------------------------------------------------------------
# IMAGE ANALYSIS
# ----------------------------------------------------------------------------
_HAAR = None


def find_face_haar(img):
    global _HAAR
    if _HAAR is None:
        _HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = _HAAR.detectMultiScale(g, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def dress_roi_from_face(face, shape):
    x, y, w, h = face
    H, W = shape[:2]
    y0, y1 = y + int(1.3 * h), min(H, y + int(2.3 * h))
    x0, x1 = max(0, x + int(0.05 * w)), min(W, x + int(0.95 * w))
    if y1 - y0 < 12 or x1 - x0 < 12:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def analyse_image(img, models, use_deepface):
    """Returns dict: face, dress_box, emotion, age, dress, notes."""
    H, W = img.shape[:2]
    res = {"face": None, "dress_box": None, "emotion": None, "age": None,
           "dress": None, "dress_bgr": None, "notes": []}

    if (H, W) == (PH, PW):                                   # generated portrait
        res["face"], res["dress_box"] = FACE_ROI, DRESS_ROI
        x, y, w, h = FACE_ROI
        emo, _, age = models.predict_face(img[y:y + h, x:x + w])
        res["emotion"], res["age"] = emo, age
    else:                                                    # any other picture
        face = None
        if use_deepface:
            try:
                from deepface import DeepFace
                r = DeepFace.analyze(img, actions=["emotion", "age"], enforce_detection=False,
                                     detector_backend="opencv", silent=True)
                r = r[0] if isinstance(r, list) else r
                reg = r["region"]
                if reg["w"] < 0.95 * W or reg["h"] < 0.95 * H:
                    face = (int(reg["x"]), int(reg["y"]), int(reg["w"]), int(reg["h"]))
                    res["emotion"] = str(r["dominant_emotion"]).capitalize()
                    res["age"] = float(r["age"])
            except Exception as e:
                res["notes"].append(f"DeepFace error: {e}")
        if face is None:
            face = find_face_haar(img)
        if face is None:
            res["notes"].append("No face found in the image.")
            return res
        res["face"] = face
        if res["emotion"] is None:
            res["notes"].append("Emotion/age for real photos need DeepFace (pip install deepface).")
        res["dress_box"] = dress_roi_from_face(face, img.shape)
        if res["dress_box"] is None:
            res["notes"].append("Clothing is not visible below the face.")

    if res["dress_box"] is not None:
        x, y, w, h = res["dress_box"]
        patch = img[y:y + h, x:x + w]
        res["dress"], _ = models.predict_dress(patch)
        res["dress_bgr"] = tuple(int(v) for v in patch.reshape(-1, 3).mean(axis=0))
    return res


def build_output(nationality, r):
    """Apply the task rules: which fields are shown for which nationality."""
    na = "N/A"
    group = resolve_group(nationality)
    age = f"~{r['age']:.0f} years" if r["age"] is not None else na
    emo = r["emotion"] or na
    dress = r["dress"] or na
    if group == "India":
        return [("Age", age), ("Dress colour", dress), ("Emotion", emo)]
    if group == "United States":
        return [("Age", age), ("Emotion", emo)]
    if group == "Africa":
        return [("Emotion", emo), ("Dress colour", dress)]
    # any other nationality -> nationality + emotion only
    return [("Nationality", nationality.strip() or "Not selected - choose one above"), ("Emotion", emo)]


def annotate(img, r):
    out = img.copy()
    if r["face"]:
        x, y, w, h = r["face"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        if r["emotion"]:
            cv2.putText(out, r["emotion"], (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
    if r["dress_box"]:
        x, y, w, h = r["dress_box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 120, 0), 2)
        if r["dress"]:
            cv2.putText(out, r["dress"], (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
def fit_photo(bgr, w, h):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ih, iw = rgb.shape[:2]
    s = min(w / iw, h / ih)
    rgb = cv2.resize(rgb, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    canvas = Image.new("RGB", (w, h), (20, 20, 24))
    im = Image.fromarray(rgb)
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return ImageTk.PhotoImage(canvas)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Person Attribute Analyzer - Emotion / Age / Dress colour")
        root.configure(bg="#1e1f26")
        self.rng = np.random.default_rng()
        self.models = Models()
        self.ready = False
        self.samples = []
        self.thumb_labels = []
        self.cur = None
        self.build_ui()
        root.after(200, self.init_models)

    # ---------- UI ----------
    def build_ui(self):
        bg, panel, fg = "#1e1f26", "#2a2c36", "#f0f0f0"
        top = tk.Frame(self.root, bg=bg)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 4))
        ttk.Button(top, text="Upload Image", command=self.upload).pack(side="left", padx=3)
        ttk.Button(top, text="New Sample Portraits", command=self.new_samples).pack(side="left", padx=3)
        tk.Label(top, text="   Nationality:", bg=bg, fg=fg).pack(side="left")
        self.nat_var = tk.StringVar(value="India")
        self.nat_box = ttk.Combobox(top, textvariable=self.nat_var, values=NATIONALITIES, width=26)
        self.nat_box.pack(side="left", padx=3)
        self.nat_box.bind("<<ComboboxSelected>>", lambda e: self.analyse())
        self.nat_box.bind("<Return>", lambda e: self.analyse())
        self.nat_box.bind("<KeyRelease>", self.filter_nationalities)
        ttk.Button(top, text="Analyse", command=self.analyse).pack(side="left", padx=8)
        self.df_var = tk.BooleanVar(value=HAS_DEEPFACE)
        tk.Checkbutton(top, text="Use DeepFace for real photos", variable=self.df_var, bg=bg, fg=fg,
                       selectcolor="#444", activebackground=bg, activeforeground=fg,
                       state="normal" if HAS_DEEPFACE else "disabled").pack(side="left", padx=6)

        def make_panel(col, title):
            f = tk.Frame(self.root, bg=panel, padx=6, pady=6)
            f.grid(row=1, column=col, padx=(10 if col == 0 else 4, 4), pady=4, sticky="n")
            tk.Label(f, text=title, bg=panel, fg="#6cc4ff", font=("Segoe UI", 11, "bold")).pack(anchor="w")
            lab = tk.Label(f, bg="black")
            lab.pack()
            blank = ImageTk.PhotoImage(Image.new("RGB", (PANEL_W, PANEL_H), (20, 20, 24)))
            lab.configure(image=blank)
            lab.image = blank
            return lab

        self.preview_lbl = make_panel(0, "INPUT PREVIEW")
        self.result_lbl = make_panel(1, "ANALYSED IMAGE")

        self.out_frame = tk.Frame(self.root, bg=panel, padx=14, pady=10, width=320)
        self.out_frame.grid(row=1, column=2, padx=(4, 10), pady=4, sticky="nsew")
        self.out_frame.grid_propagate(False)
        tk.Label(self.out_frame, text="OUTPUT", bg=panel, fg="#6cc4ff", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.rows_frame = tk.Frame(self.out_frame, bg=panel)
        self.rows_frame.pack(fill="x", pady=6)
        self.note_var = tk.StringVar()
        tk.Label(self.out_frame, textvariable=self.note_var, bg=panel, fg="#ffb347", wraplength=290,
                 justify="left", font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        self.truth_var = tk.StringVar()
        tk.Label(self.out_frame, textvariable=self.truth_var, bg=panel, fg="#aaaaaa", wraplength=290,
                 justify="left", font=("Segoe UI", 9)).pack(anchor="w", pady=(10, 0))

        tf = tk.Frame(self.root, bg=bg)
        tf.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 4))
        tk.Label(tf, text="Built-in sample portraits (click to preview; the nationality label of each "
                          "sample is assigned at random, just for the demo):",
                 bg=bg, fg=fg, font=("Segoe UI", 9)).pack(anchor="w")
        self.thumb_frame = tk.Frame(tf, bg=bg)
        self.thumb_frame.pack(anchor="w")

        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(self.root, textvariable=self.status_var, bg=bg, fg="#aaaaaa", anchor="w",
                 font=("Segoe UI", 9)).grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 8))

    # ---------- startup ----------
    def init_models(self):
        self.status_var.set("Training models on built-in generated portraits (first run only, ~30-90 s)...")
        self.root.update()
        try:
            self.models.load_or_train()
        except Exception as e:
            messagebox.showerror("Error", f"Could not prepare the models:\n{e}")
            return
        self.ready = True
        m = self.models.metrics
        self.status_var.set(f"Ready.  Validation: emotion {m['emotion_acc'] * 100:.1f}%  |  "
                            f"age error ~{m['age_mae']:.1f} years  |  dress colour {m['dress_acc'] * 100:.1f}%"
                            f"   (on generated portraits)")
        self.new_samples()

    def new_samples(self):
        if not self.ready:
            return
        self.samples = []
        for _ in range(N_SAMPLES):
            img, t = generate_portrait(self.rng)
            self.samples.append((img, t, str(self.rng.choice(self.sample_nationalities()))))
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self.thumb_labels = []
        for i, (img, t, g) in enumerate(self.samples):
            lab = tk.Label(self.thumb_frame, bd=3, relief="flat", bg="#1e1f26", cursor="hand2")
            ph = fit_photo(img, THUMB_W, THUMB_H)
            lab.configure(image=ph)
            lab.image = ph
            lab.grid(row=0, column=i, padx=3, pady=3)
            lab.bind("<Button-1>", lambda e, k=i: self.select_sample(k))
            self.thumb_labels.append(lab)
        self.select_sample(0)

    def select_sample(self, i):
        for j, lab in enumerate(self.thumb_labels):
            lab.configure(relief="solid" if j == i else "flat", bg="#6cc4ff" if j == i else "#1e1f26")
        img, t, g = self.samples[i]
        self.cur = {"img": img, "truth": t, "name": f"sample portrait {i + 1}"}
        self.nat_var.set(g)
        self.show_preview()
        self.analyse()

    # ---------- nationality helpers ----------
    @staticmethod
    def sample_nationalities():
        # demo picks: all four rule types appear (India / US / an African country / another country)
        return ["India", "United States", "Nigeria", "Kenya", "Brazil", "Japan", "France", "Germany"]

    def filter_nationalities(self, event):
        if event.keysym in ("Return", "Up", "Down", "Escape", "Tab"):
            return
        typed = self.nat_var.get().strip().lower()
        self.nat_box["values"] = [c for c in NATIONALITIES if typed in c.lower()] if typed else NATIONALITIES

    # ---------- actions ----------
    def show_preview(self):
        ph = fit_photo(self.cur["img"], PANEL_W, PANEL_H)
        self.preview_lbl.configure(image=ph)
        self.preview_lbl.image = ph

    def upload(self):
        if not self.ready:
            return
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read that image.")
            return
        if max(img.shape[:2]) > 1400:
            s = 1400.0 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self.cur = {"img": img, "truth": None, "name": os.path.basename(path)}
        self.show_preview()
        self.analyse()

    def analyse(self):
        if not self.ready or self.cur is None:
            return
        r = analyse_image(self.cur["img"], self.models, self.df_var.get() and HAS_DEEPFACE)
        ph = fit_photo(annotate(self.cur["img"], r), PANEL_W, PANEL_H)
        self.result_lbl.configure(image=ph)
        self.result_lbl.image = ph

        for w in self.rows_frame.winfo_children():
            w.destroy()
        nat = self.nat_var.get().strip()
        tk.Label(self.rows_frame, text=f"Selected nationality: {nat or '-'}   (rule: {resolve_group(nat)})",
                 bg="#2a2c36", fg="#bbbbbb", font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 6))
        for label, value in build_output(nat, r):
            tk.Label(self.rows_frame, text=label.upper(), bg="#2a2c36", fg="#8aa0b8",
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            row = tk.Frame(self.rows_frame, bg="#2a2c36")
            row.pack(anchor="w", pady=(0, 8))
            if label == "Dress colour" and r["dress_bgr"]:
                b, g, rr = r["dress_bgr"]
                tk.Label(row, bg=f"#{rr:02x}{g:02x}{b:02x}", width=3, height=1).pack(side="left", padx=(0, 8))
            tk.Label(row, text=value, bg="#2a2c36", fg="#7CFC9A",
                     font=("Segoe UI", 18, "bold")).pack(side="left")
        self.note_var.set("  ".join(r["notes"]))
        t = self.cur["truth"]
        self.truth_var.set(f"(Generated portrait - actual: {t['emotion']}, age {t['age']:.0f}, "
                           f"dress {t['dress']})" if t else "")
        self.status_var.set(f"{self.cur['name']} analysed.")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
