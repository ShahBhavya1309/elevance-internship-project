"""
CAR COLOUR DETECTION + CAR / PEOPLE COUNTER AT A TRAFFIC SIGNAL
===============================================================
* NO dataset download: traffic-signal scenes (road, signal, cars of many colours,
  pedestrians) are generated inside this script, and the colour-classifier is
  trained on data that the script also generates.
* ML model : RandomForest (scikit-learn) that predicts the car colour
             (Blue, Red, Green, Yellow, Orange, White, Black, Silver)
             from HSV colour features of the car image.
* Output   : RED rectangle  -> BLUE cars
             BLUE rectangle -> cars of every other colour
             + number of cars, blue cars, other cars and PEOPLE at the signal.
* GUI      : Tkinter - input preview, detection output, sample-scene thumbnails,
             upload your own image, save result.

Detectors
  1) "Built-in (offline)"  : static-camera background subtraction + shape rules.
                             Works on the built-in generated scenes (no internet).
  2) "YOLOv8 (real photos)": OPTIONAL. Needs  pip install ultralytics  and downloads
                             the small yolov8n.pt weights (~6 MB) on first use.
                             Use it for real photographs of traffic.

INSTALL (Python 3.9+):
    pip install opencv-python scikit-learn pillow numpy
    (optional, for real photos)  pip install ultralytics

RUN:
    python car_colour_detector.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MODEL_FILE = "car_colour_model_cache.pkl"
MODEL_VERSION = 3
SCENE_W, SCENE_H = 800, 600
PANEL_W, PANEL_H = 480, 360
THUMB_W, THUMB_H = 112, 84
N_SAMPLES = 8

RED_BOX = (0, 0, 255)      # BGR  -> used for BLUE cars
BLUE_BOX = (255, 0, 0)     # BGR  -> used for all OTHER cars
PERSON_BOX = (0, 170, 0)   # BGR  -> green, people

HAS_YOLO = importlib.util.find_spec("ultralytics") is not None

# ----------------------------------------------------------------------------
# COLOUR CLASSES  (OpenCV HSV: H 0-179, S 0-255, V 0-255)
# ----------------------------------------------------------------------------
COLOUR_RANGES = {
    "Blue":   ((100, 125), (150, 255), (80, 230)),
    "Red":    ((-6, 6),    (160, 255), (110, 230)),
    "Green":  ((40, 80),   (100, 230), (70, 190)),
    "Yellow": ((22, 32),   (150, 255), (190, 255)),
    "Orange": ((8, 18),    (190, 255), (190, 255)),
    "White":  ((0, 179),   (0, 25),    (205, 255)),
    "Black":  ((0, 179),   (0, 60),    (5, 55)),
    "Silver": ((0, 179),   (0, 30),    (120, 195)),
}
COLOUR_NAMES = list(COLOUR_RANGES.keys())


def sample_bgr(name, rng):
    (h0, h1), (s0, s1), (v0, v1) = COLOUR_RANGES[name]
    h = rng.uniform(h0, h1) % 180
    hsv = np.uint8([[[h, rng.uniform(s0, s1), rng.uniform(v0, v1)]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


# ----------------------------------------------------------------------------
# DRAWING HELPERS (used to create the built-in traffic scenes)
# ----------------------------------------------------------------------------
def _rrect(img, x1, y1, x2, y2, r, color):
    r = int(max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1)


def draw_car(img, cx, cy, w, h, bgr, heading_down, rng):
    """Top-view car. Body in colour `bgr`, dark windows, wheels, lights."""
    x1, x2 = cx - w // 2, cx + w // 2
    y1, y2 = cy - h // 2, cy + h // 2

    def yr(t0, t1):   # t = 0 at the rear, 1 at the front
        if heading_down:
            return int(y1 + t0 * h), int(y1 + t1 * h)
        return int(y2 - t1 * h), int(y2 - t0 * h)

    wheel = (25, 25, 25)
    ww = max(5, w // 9)
    for t0, t1 in ((0.12, 0.30), (0.72, 0.90)):
        ya, yb = yr(t0, t1)
        cv2.rectangle(img, (x1 - 3, ya), (x1 - 3 + ww, yb), wheel, -1)
        cv2.rectangle(img, (x2 + 3 - ww, ya), (x2 + 3, yb), wheel, -1)

    _rrect(img, x1, y1, x2, y2, w // 5, bgr)                       # body

    roof = tuple(int(np.clip(c * rng.uniform(0.9, 1.1), 0, 255)) for c in bgr)
    ya, yb = yr(0.26, 0.58)
    cv2.rectangle(img, (x1 + 4, ya), (x2 - 4, yb), roof, -1)         # roof

    glass = tuple(int(c + rng.integers(-8, 8)) for c in (62, 52, 46))
    ya, yb = yr(0.58, 0.77)                                          # windshield
    cv2.rectangle(img, (x1 + int(0.12 * w), ya), (x2 - int(0.12 * w), yb), glass, -1)
    ya, yb = yr(0.11, 0.25)                                          # rear window
    cv2.rectangle(img, (x1 + int(0.16 * w), ya), (x2 - int(0.16 * w), yb), glass, -1)

    ya, yb = yr(0.94, 0.99)                                          # head-lights
    cv2.rectangle(img, (x1 + 4, ya), (x1 + 4 + w // 5, yb), (200, 235, 250), -1)
    cv2.rectangle(img, (x2 - 4 - w // 5, ya), (x2 - 4, yb), (200, 235, 250), -1)
    ya, yb = yr(0.01, 0.06)                                          # tail-lights
    cv2.rectangle(img, (x1 + 4, ya), (x1 + 4 + w // 6, yb), (30, 30, 200), -1)
    cv2.rectangle(img, (x2 - 4 - w // 6, ya), (x2 - 4, yb), (30, 30, 200), -1)


PERSON_COLOURS = [(40, 40, 200), (200, 80, 40), (40, 170, 60), (30, 200, 230),
                  (20, 120, 120), (150, 60, 160), (30, 120, 230)]


def draw_person(img, cx, cy, rng):
    col = PERSON_COLOURS[int(rng.integers(len(PERSON_COLOURS)))]
    cv2.ellipse(img, (cx, cy), (13, 8), int(rng.integers(0, 180)), 0, 360, col, -1)
    cv2.circle(img, (cx, cy), 5, (25, 30, 45), -1)


def draw_signal(img, x, y):
    cv2.rectangle(img, (x, y), (x + 28, y + 70), (30, 30, 30), -1)
    cv2.circle(img, (x + 14, y + 14), 8, (0, 0, 255), -1)       # RED light ON
    cv2.circle(img, (x + 14, y + 35), 8, (0, 90, 110), -1)
    cv2.circle(img, (x + 14, y + 56), 8, (40, 90, 40), -1)


def make_background():
    img = np.empty((SCENE_H, SCENE_W, 3), np.uint8)
    img[:] = (188, 192, 196)                                        # side-walk
    cv2.rectangle(img, (150, 0), (650, SCENE_H), (82, 82, 86), -1)  # road
    cv2.line(img, (150, 0), (150, SCENE_H), (130, 130, 130), 4)
    cv2.line(img, (650, 0), (650, SCENE_H), (130, 130, 130), 4)
    for x in (396, 404):
        cv2.line(img, (x, 0), (x, SCENE_H), (0, 190, 235), 3)       # double yellow
    for x in (290, 510):
        for y in range(0, SCENE_H, 55):
            cv2.line(img, (x, y), (x, y + 30), (235, 235, 235), 3)  # lane dashes
    cv2.line(img, (180, 335), (395, 335), (240, 240, 240), 6)       # stop lines
    cv2.line(img, (405, 390), (620, 390), (240, 240, 240), 6)
    return img


def generate_scene(rng):
    """Returns (image, empty_background, truth) for one traffic-signal scene."""
    bg = make_background()
    img = bg.copy()
    cars = []

    for lane_x in (235, 345):                       # lanes waiting, heading down
        y_front = 325
        for _ in range(int(rng.integers(0, 5))):
            w, h = int(rng.integers(58, 68)), int(rng.integers(104, 122))
            cy = y_front - h // 2
            if cy - h // 2 < 15:
                break
            name = str(rng.choice(COLOUR_NAMES))
            draw_car(img, lane_x + int(rng.integers(-8, 9)), cy, w, h, sample_bgr(name, rng), True, rng)
            cars.append(name)
            y_front = cy - h // 2 - int(rng.integers(18, 45))

    for lane_x in (455, 565):                       # lanes waiting, heading up
        y_front = 395
        for _ in range(int(rng.integers(0, 5))):
            w, h = int(rng.integers(58, 68)), int(rng.integers(104, 122))
            cy = y_front + h // 2
            if cy + h // 2 > SCENE_H - 15:
                break
            name = str(rng.choice(COLOUR_NAMES))
            draw_car(img, lane_x + int(rng.integers(-8, 9)), cy, w, h, sample_bgr(name, rng), False, rng)
            cars.append(name)
            y_front = cy + h // 2 + int(rng.integers(18, 45))

    if not cars:                                    # always at least one car
        name = str(rng.choice(COLOUR_NAMES))
        draw_car(img, 235, 250, 62, 112, sample_bgr(name, rng), True, rng)
        cars.append(name)

    n_people, placed = int(rng.integers(0, 7)), []
    tries = 0
    while len(placed) < n_people and tries < 200:
        tries += 1
        if rng.random() < 0.5:                      # on the side-walks
            x = int(rng.integers(25, 75)) if rng.random() < 0.5 else int(rng.integers(738, 780))
            y = int(rng.integers(300, 430))
        else:                                       # crossing between the two queues
            x, y = int(rng.integers(200, 600)), int(rng.integers(352, 374))
            if 375 < x < 425:               # keep clear of the double yellow line
                continue
        if all(abs(x - px) > 40 or abs(y - py) > 40 for px, py in placed):
            draw_person(img, x, y, rng)
            placed.append((x, y))

    draw_signal(img, 90, 240)
    draw_signal(img, 682, 410)

    img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img, bg, {"cars": cars, "people": len(placed)}


# ----------------------------------------------------------------------------
# COLOUR FEATURES + ML MODEL
# ----------------------------------------------------------------------------
def colour_features(patch_bgr):
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].ravel().astype(np.float32)
    s = hsv[..., 1].ravel().astype(np.float32)
    v = hsv[..., 2].ravel().astype(np.float32)
    n = max(1, h.size)
    chroma = (s > 80) & (v > 60)
    hist, _ = np.histogram(h[chroma], bins=18, range=(0, 180))
    f = list(hist / n)
    f.append(np.mean(v < 60))
    f.append(np.mean((s < 50) & (v >= 190)))
    f.append(np.mean((s < 50) & (v >= 60) & (v < 190)))
    f.append(s.mean() / 255.0)
    f.append(v.mean() / 255.0)
    if chroma.any():
        ang = h[chroma] * (2 * math.pi / 180.0)
        f.append(float(np.cos(ang).mean()))
        f.append(float(np.sin(ang).mean()))
    else:
        f += [0.0, 0.0]
    return np.array(f, dtype=np.float32)


def render_car_crop(name, rng):
    """Synthetic training image: a car of colour `name` cropped like the detector would."""
    cw, ch = 140, 170
    g = int(rng.integers(70, 100))
    canvas = np.full((ch, cw, 3), (g, g, g + 4), np.uint8)
    w, h = int(rng.integers(56, 70)), int(rng.integers(100, 125))
    draw_car(canvas, cw // 2, ch // 2, w, h, sample_bgr(name, rng), bool(rng.integers(2)), rng)
    canvas = np.clip(canvas.astype(np.float32) * rng.uniform(0.85, 1.08)
                     + rng.normal(0, 4, canvas.shape), 0, 255).astype(np.uint8)
    m = int(rng.integers(0, 6))
    crop = canvas[ch // 2 - h // 2 - m: ch // 2 + h // 2 + m, cw // 2 - w // 2 - m: cw // 2 + w // 2 + m]
    return top_view_patch(crop)


def render_body_patch(name, rng):
    """Synthetic training image: just body paint with shading and some glass."""
    p = np.zeros((40, 50, 3), np.float32)
    p[:] = sample_bgr(name, rng)
    grad = np.linspace(rng.uniform(0.85, 1.05), rng.uniform(0.85, 1.05), 50)[None, :, None]
    p = p * grad
    if rng.random() < 0.6:
        a = int(rng.integers(3, 14))
        p[:a] = (60, 52, 46)
    p = np.clip(p * rng.uniform(0.85, 1.08) + rng.normal(0, 4, p.shape), 0, 255).astype(np.uint8)
    return p


def top_view_patch(crop):
    h, w = crop.shape[:2]
    return crop[int(0.1 * h): int(0.9 * h), int(0.1 * w): int(0.9 * w)]


def body_patch_photo(crop):
    """For real photos: the lower-centre part of the box is mostly body paint."""
    h, w = crop.shape[:2]
    return crop[int(0.4 * h): int(0.9 * h), int(0.2 * w): int(0.8 * w)]


class ColourModel:
    def __init__(self):
        self.clf = None
        self.val_acc = None

    def load_or_train(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as fh:
                    d = pickle.load(fh)
                if d.get("version") == MODEL_VERSION:
                    self.clf, self.val_acc = d["clf"], d["val_acc"]
                    return
            except Exception:
                pass
        self.train()

    def train(self, n_cars=300, n_patch=120, seed=7):
        rng = np.random.default_rng(seed)
        X, y = [], []
        for name in COLOUR_NAMES:
            for _ in range(n_cars):
                X.append(colour_features(render_car_crop(name, rng))); y.append(name)
            for _ in range(n_patch):
                X.append(colour_features(render_body_patch(name, rng))); y.append(name)
        X, y = np.array(X), np.array(y)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        probe = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Xtr, ytr)
        self.val_acc = float(probe.score(Xva, yva))
        self.clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(X, y)
        with open(MODEL_FILE, "wb") as fh:
            pickle.dump({"version": MODEL_VERSION, "clf": self.clf, "val_acc": self.val_acc}, fh)

    def predict(self, patch):
        p = self.clf.predict_proba([colour_features(patch)])[0]
        i = int(np.argmax(p))
        return str(self.clf.classes_[i]), float(p[i])


# ----------------------------------------------------------------------------
# DETECTORS  ->  list of {"kind": "car"/"person", "box": (x, y, w, h)}
# ----------------------------------------------------------------------------
def estimate_background(img):
    """Fallback when no empty-road frame is known: paint every pixel with the nearest
    DOMINANT colour of the image (road / pavement are the biggest colour areas)."""
    h, w = img.shape[:2]
    sm = cv2.resize(img, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    q = (sm // 16).astype(np.int32)
    keys = (q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]).ravel()
    counts = np.bincount(keys, minlength=4096)
    dom = np.where(counts > 0.06 * keys.size)[0]
    if dom.size == 0:
        dom = np.array([int(np.argmax(counts))])
    cents = []
    for k in dom:                       # average real colour of each dominant bin
        cents.append(sm.reshape(-1, 3)[keys == k].mean(axis=0))
    cents = np.array(cents, dtype=np.float32)
    full = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
    d = np.stack([np.abs(full - c).max(axis=2) for c in cents])
    best = np.argmin(d, axis=0)
    return cents[best].astype(np.uint8)


def detect_builtin(img, bg=None):
    H, W = img.shape[:2]
    s = math.sqrt(W * H / float(SCENE_W * SCENE_H))
    if bg is None:
        bg = estimate_background(img)
    diff = cv2.absdiff(cv2.GaussianBlur(img, (5, 5), 0), cv2.GaussianBlur(bg, (5, 5), 0)).max(axis=2)
    mask = (diff > 20).astype(np.uint8) * 255
    k = max(3, int(9 * s)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c) / (s * s)
        ws, hs = w / s, h / s
        if area > 3500 and min(ws, hs) > 40 and max(ws, hs) < 260:
            dets.append({"kind": "car", "box": (x, y, w, h), "src": "builtin"})
        elif 120 < area < 1500 and min(ws, hs) > 10 and max(ws, hs) < 60:
            dets.append({"kind": "person", "box": (x, y, w, h), "src": "builtin"})
    return dets


def detect_yolo(model, img):
    res = model(img, verbose=False, conf=0.35)[0]
    dets = []
    for b in res.boxes:
        cls = int(b.cls[0])
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        box = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        if cls == 2:
            dets.append({"kind": "car", "box": box, "src": "yolo"})
        elif cls == 0:
            dets.append({"kind": "person", "box": box, "src": "yolo"})
    return dets


# ----------------------------------------------------------------------------
# ANALYSIS: colour of each car, counting, drawing
# ----------------------------------------------------------------------------
def analyse(img, dets, colour_model):
    out = img.copy()
    H, W = out.shape[:2]
    fs, th = max(0.45, W / 1400.0), max(2, int(W / 400))
    font = cv2.FONT_HERSHEY_SIMPLEX

    cars = sorted([d for d in dets if d["kind"] == "car"], key=lambda d: (d["box"][1], d["box"][0]))
    people = [d for d in dets if d["kind"] == "person"]
    rows = []

    for i, d in enumerate(cars, 1):
        x, y, w, h = d["box"]
        crop = img[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        patch = top_view_patch(crop) if d["src"] == "builtin" else body_patch_photo(crop)
        if patch.size == 0:
            patch = crop
        name, conf = colour_model.predict(patch)
        is_blue = (name == "Blue")
        col = RED_BOX if is_blue else BLUE_BOX            # RED box for blue car, BLUE box otherwise
        cv2.rectangle(out, (x, y), (x + w, y + h), col, th + 1)
        label = f"#{i} {name}"
        (tw, tht), _ = cv2.getTextSize(label, font, fs, 1)
        top = y - tht - 8
        if top < 0:
            top = y
        cv2.rectangle(out, (x, top), (x + tw + 6, top + tht + 8), col, -1)
        cv2.putText(out, label, (x + 3, top + tht + 3), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append((i, name, conf, is_blue))

    for d in people:
        x, y, w, h = d["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), PERSON_BOX, max(1, th - 1))

    n_blue = sum(1 for r in rows if r[3])
    summary = {"cars": len(rows), "blue": n_blue, "other": len(rows) - n_blue, "people": len(people)}

    lines = [f"Cars at signal: {summary['cars']}",
             f"Blue cars (red box): {summary['blue']}",
             f"Other cars (blue box): {summary['other']}",
             f"People at signal: {summary['people']}"]
    pad, lh = 8, int(26 * max(0.8, fs / 0.55))
    box_w = int(max(cv2.getTextSize(t, font, fs * 1.1, 2)[0][0] for t in lines)) + 2 * pad
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, lh * len(lines) + pad), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    for j, t in enumerate(lines):
        cv2.putText(out, t, (pad, lh * (j + 1)), font, fs * 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return out, summary, rows


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
        root.title("Car Colour Detection & Traffic Signal Counter")
        root.configure(bg="#1e1f26")
        self.rng = np.random.default_rng()
        self.cmodel = ColourModel()
        self.ready = False
        self.yolo = None
        self.samples = []          # list of (img, bg, truth)
        self.thumb_labels = []
        self.cur = None            # dict: img, bg, truth, name
        self.result_img = None
        self.build_ui()
        root.after(200, self.init_model)

    # ---------- UI ----------
    def build_ui(self):
        bg, panel, fg = "#1e1f26", "#2a2c36", "#f0f0f0"
        top = tk.Frame(self.root, bg=bg)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 4))
        ttk.Button(top, text="New Sample Scenes", command=self.new_samples).pack(side="left", padx=3)
        ttk.Button(top, text="Upload Image", command=self.upload).pack(side="left", padx=3)
        ttk.Button(top, text="Detect", command=self.detect).pack(side="left", padx=3)
        ttk.Button(top, text="Save Result", command=self.save_result).pack(side="left", padx=3)
        tk.Label(top, text="  Detector:", bg=bg, fg=fg).pack(side="left")
        self.det_var = tk.StringVar(value="Built-in (offline)")
        self.det_box = ttk.Combobox(top, textvariable=self.det_var, state="readonly", width=24,
                                    values=["Built-in (offline)", "YOLOv8 (real photos)"])
        self.det_box.pack(side="left", padx=3)
        self.det_box.bind("<<ComboboxSelected>>", self.on_detector_change)

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
        self.result_lbl = make_panel(1, "DETECTION RESULT")

        rp = tk.Frame(self.root, bg=panel, padx=10, pady=8)
        rp.grid(row=1, column=2, padx=(4, 10), pady=4, sticky="nsew")
        tk.Label(rp, text="RESULTS", bg=panel, fg="#6cc4ff", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.cars_var = tk.StringVar(value="Cars at signal: -")
        self.blue_var = tk.StringVar(value="Blue cars (RED box): -")
        self.other_var = tk.StringVar(value="Other cars (BLUE box): -")
        self.people_var = tk.StringVar(value="People at signal: -")
        for v, c in ((self.cars_var, "#ffffff"), (self.blue_var, "#ff6b6b"),
                     (self.other_var, "#6cb4ff"), (self.people_var, "#7CFC9A")):
            tk.Label(rp, textvariable=v, bg=panel, fg=c, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=2)
        self.truth_var = tk.StringVar(value="")
        tk.Label(rp, textvariable=self.truth_var, bg=panel, fg="#bbbbbb", wraplength=240,
                 justify="left", font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 6))
        cols = ("#", "Colour", "Conf.", "Box colour")
        self.tree = ttk.Treeview(rp, columns=cols, show="headings", height=9)
        for c, wd in zip(cols, (30, 70, 55, 80)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=wd, anchor="center")
        self.tree.pack(fill="x")

        tf = tk.Frame(self.root, bg=bg)
        tf.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 4))
        tk.Label(tf, text="Built-in sample scenes (click to preview):", bg=bg, fg=fg,
                 font=("Segoe UI", 10)).pack(anchor="w")
        self.thumb_frame = tk.Frame(tf, bg=bg)
        self.thumb_frame.pack(anchor="w")

        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(self.root, textvariable=self.status_var, bg=bg, fg="#aaaaaa", anchor="w",
                 font=("Segoe UI", 9)).grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 8))

    # ---------- startup ----------
    def init_model(self):
        self.status_var.set("Training colour model on built-in generated data (first run only, ~20-60 s)...")
        self.root.update()
        try:
            self.cmodel.load_or_train()
        except Exception as e:
            messagebox.showerror("Error", f"Could not prepare the model:\n{e}")
            return
        self.ready = True
        self.new_samples()
        self.status_var.set(f"Ready. Colour model validation accuracy: {self.cmodel.val_acc * 100:.1f}%"
                            + ("" if HAS_YOLO else "   |   (Optional: pip install ultralytics for real photos)"))

    def new_samples(self):
        self.samples = [generate_scene(self.rng) for _ in range(N_SAMPLES)]
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self.thumb_labels = []
        for i, (img, bg, truth) in enumerate(self.samples):
            lab = tk.Label(self.thumb_frame, bd=3, relief="flat", bg="#1e1f26", cursor="hand2")
            ph = fit_photo(img, THUMB_W, THUMB_H)
            lab.configure(image=ph)
            lab.image = ph
            lab.grid(row=0, column=i, padx=3, pady=3)
            lab.bind("<Button-1>", lambda e, k=i: self.select_sample(k))
            self.thumb_labels.append(lab)
        self.select_sample(0)

    def select_sample(self, i):
        if not self.ready:
            return
        for j, lab in enumerate(self.thumb_labels):
            lab.configure(relief="solid" if j == i else "flat", bg="#6cc4ff" if j == i else "#1e1f26")
        img, bg, truth = self.samples[i]
        self.cur = {"img": img, "bg": bg, "truth": truth, "name": f"sample scene {i + 1}"}
        self.show_preview()
        self.detect()

    # ---------- actions ----------
    def show_preview(self):
        ph = fit_photo(self.cur["img"], PANEL_W, PANEL_H)
        self.preview_lbl.configure(image=ph)
        self.preview_lbl.image = ph

    def on_detector_change(self, _evt=None):
        if self.det_var.get().startswith("YOLO") and not HAS_YOLO:
            messagebox.showinfo("YOLO not installed",
                                "Run:  pip install ultralytics\nthen restart this program.\n"
                                "(It downloads the small yolov8n.pt weights once, ~6 MB.)")
            self.det_var.set("Built-in (offline)")
            return
        if self.cur is not None:
            self.detect()

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
        if max(img.shape[:2]) > 1600:
            s = 1600.0 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self.cur = {"img": img, "bg": None, "truth": None, "name": os.path.basename(path)}
        if HAS_YOLO:
            self.det_var.set("YOLOv8 (real photos)")
        self.show_preview()
        self.detect()
        if not self.det_var.get().startswith("YOLO"):
            self.status_var.set("Uploaded photo with the built-in detector (made for the generated scenes). "
                                "For real photos install ultralytics and choose YOLOv8.")

    def get_yolo(self):
        if self.yolo is None:
            from ultralytics import YOLO
            self.status_var.set("Loading YOLOv8 (downloads ~6 MB weights on first use)...")
            self.root.update()
            self.yolo = YOLO("yolov8n.pt")
        return self.yolo

    def detect(self):
        if not self.ready or self.cur is None:
            return
        img = self.cur["img"]
        try:
            if self.det_var.get().startswith("YOLO"):
                dets = detect_yolo(self.get_yolo(), img)
            else:
                dets = detect_builtin(img, self.cur["bg"])
        except Exception as e:
            messagebox.showerror("Detection error", str(e))
            self.det_var.set("Built-in (offline)")
            return
        out, summ, rows = analyse(img, dets, self.cmodel)
        self.result_img = out
        ph = fit_photo(out, PANEL_W, PANEL_H)
        self.result_lbl.configure(image=ph)
        self.result_lbl.image = ph

        self.cars_var.set(f"Cars at signal: {summ['cars']}")
        self.blue_var.set(f"Blue cars (RED box): {summ['blue']}")
        self.other_var.set(f"Other cars (BLUE box): {summ['other']}")
        self.people_var.set(f"People at signal: {summ['people']}")
        t = self.cur["truth"]
        self.truth_var.set(f"Actual in this built-in scene: {len(t['cars'])} cars "
                           f"({t['cars'].count('Blue')} blue), {t['people']} people" if t else "")
        self.tree.delete(*self.tree.get_children())
        for i, name, conf, is_blue in rows:
            self.tree.insert("", "end", values=(i, name, f"{conf * 100:.0f}%", "RED" if is_blue else "BLUE"))
        self.status_var.set(f"{self.cur['name']}: analysed with {self.det_var.get()}.")

    def save_result(self):
        if self.result_img is None:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", self.result_img)
        if ok:
            buf.tofile(path)
            self.status_var.set(f"Saved: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
