"""
SIGN LANGUAGE DETECTOR  (hand-sign -> word)
===========================================
* NO dataset download needed: the training data is generated inside this script
  (a built-in synthetic hand-skeleton generator -> 3D hand landmarks -> features).
* Model: RandomForest trained on joint-angle / distance features of the 21 hand
  landmarks detected by MediaPipe Hands.
* GUI (Tkinter): Upload Image + Real-time Webcam video.
* Time window: the model only works between 6 PM and 10 PM (change START_HOUR /
  END_HOUR below).  A "ignore time window" checkbox is provided for testing.
* Optional "Calibrate" panel: record a few real samples of YOUR hand to make the
  model more accurate (saved locally, still no download needed).

INSTALL (Python 3.9 - 3.11 recommended):
    pip install opencv-python mediapipe==0.10.14 scikit-learn pillow numpy

RUN:
    python sign_language_detector.py
"""

import math
import os
import pickle
import datetime as dt
from collections import deque, Counter

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
START_HOUR = 18          # 6 PM  (24-hour clock)
END_HOUR = 22            # 10 PM (model works while START_HOUR <= hour < END_HOUR)
CONF_THRESHOLD = 0.45    # below this -> "Unknown"
SMOOTH_FRAMES = 8        # majority vote over the last N video frames
STABLE_FRAMES = 12       # frames a sign must be held to be added to the sentence
DISP_W, DISP_H = 720, 540
MODEL_FILE = "sign_model_cache.pkl"
USER_FILE = "my_calibration_samples.pkl"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ----------------------------------------------------------------------------
# WORDS (signs) THE MODEL KNOWS
# fingers = states of (index, middle, ring, pinky):  E=extended  C=curled  H=half
# thumb   = possible thumb poses: out / up / tuck / touch (touches index tip)
# ----------------------------------------------------------------------------
GESTURES = {
    "HELLO":     dict(fingers="EEEE", thumb=["out", "up"],
                      desc="Open palm, all fingers up"),
    "YES":       dict(fingers="CCCC", thumb=["tuck"],
                      desc="Closed fist, thumb tucked"),
    "GOOD":      dict(fingers="CCCC", thumb=["up"],
                      desc="Thumbs up"),
    "PEACE":     dict(fingers="EECC", thumb=["tuck"], fan=[-0.25, 0.2, 0.1, 0.2],
                      fan_scale=(0.4, 1.2), desc="V sign (index + middle)"),
    "I LOVE YOU": dict(fingers="ECCE", thumb=["out"],
                       desc="Thumb, index & pinky out"),
    "CALL ME":   dict(fingers="CCCE", thumb=["out"],
                      desc="Thumb & pinky out (phone)"),
    "OK":        dict(fingers="HEEE", thumb=["touch"],
                      desc="Thumb touches index tip, other 3 up"),
    "ONE":       dict(fingers="ECCC", thumb=["tuck"],
                      desc="Only index finger up"),
}
WORDS = list(GESTURES.keys())

# ----------------------------------------------------------------------------
# FEATURE EXTRACTION (rotation / scale / mirror invariant)
# ----------------------------------------------------------------------------
def _angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    return math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def extract_features(lm):
    """lm: (21,3) array of hand landmarks -> 1D feature vector."""
    lm = np.asarray(lm, dtype=float)
    d = lambda i, j: np.linalg.norm(lm[i] - lm[j])
    palm = d(0, 9) + 1e-9
    f = []
    for b in (5, 9, 13, 17):                       # four fingers: 3 joint angles each
        f.append(_angle(lm[0], lm[b], lm[b + 1]))
        f.append(_angle(lm[b], lm[b + 1], lm[b + 2]))
        f.append(_angle(lm[b + 1], lm[b + 2], lm[b + 3]))
    f.append(_angle(lm[0], lm[1], lm[2]))          # thumb joint angles
    f.append(_angle(lm[1], lm[2], lm[3]))
    f.append(_angle(lm[2], lm[3], lm[4]))
    for t in (4, 8, 12, 16, 20):                   # tip -> wrist distances
        f.append(d(t, 0) / palm)
    for j in (8, 5, 9, 17, 12):                    # thumb tip relations
        f.append(d(4, j) / palm)
    f.append(d(8, 12) / palm)                      # finger spreads
    f.append(d(12, 16) / palm)
    f.append(d(16, 20) / palm)
    return np.array(f, dtype=float)

# ----------------------------------------------------------------------------
# BUILT-IN SYNTHETIC DATASET GENERATOR (replaces downloading a dataset)
# ----------------------------------------------------------------------------
BASES = {5: (-0.32, 0.95), 9: (-0.10, 1.00), 13: (0.12, 0.95), 17: (0.32, 0.82)}
LENS = {5: (0.45, 0.28, 0.22), 9: (0.50, 0.32, 0.24),
        13: (0.46, 0.30, 0.22), 17: (0.36, 0.22, 0.18)}


def _make_finger(base, lens, curl, spread, rng):
    f = [curl * math.radians(rng.uniform(65, 90)),
         curl * math.radians(rng.uniform(85, 110)),
         curl * math.radians(rng.uniform(50, 75))]
    pos = np.array(base, dtype=float)
    pts, theta = [pos.copy()], 0.0
    for fl, L in zip(f, lens):
        theta += fl
        dvec = np.array([math.sin(spread) * math.cos(theta),
                         math.cos(spread) * math.cos(theta),
                         math.sin(theta)])
        pos = pos + L * dvec
        pts.append(pos.copy())
    return np.array(pts)


def _rotation(rng):
    ax, ay, az = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-math.pi, math.pi)
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def synth_sample(name, rng):
    g = GESTURES[name]
    pts = np.zeros((21, 3))
    fan = np.array(g.get("fan", [-0.15, -0.02, 0.08, 0.2])) * rng.uniform(*g.get("fan_scale", (0.2, 1.4)))
    fan = fan + rng.normal(0, 0.04, 4)
    size = rng.uniform(0.9, 1.1)
    for i, (bid, st) in enumerate(zip((5, 9, 13, 17), g["fingers"])):
        lo, hi = {"E": (0.0, 0.15), "C": (0.7, 1.0), "H": (0.55, 0.8)}[st]
        base = np.array([BASES[bid][0], BASES[bid][1], 0.0]) + rng.normal(0, 0.01, 3)
        pts[bid:bid + 4] = _make_finger(base, np.array(LENS[bid]) * size, rng.uniform(lo, hi), fan[i], rng)

    # thumb
    mode = str(rng.choice(g["thumb"]))
    p1 = np.array([-0.20, 0.22, 0.0]) + rng.normal(0, 0.015, 3)
    p2 = p1 + np.array([-0.20, 0.30, 0.0]) * rng.uniform(0.9, 1.1) + rng.normal(0, 0.015, 3)
    if mode == "touch":
        p4 = pts[8] + rng.normal(0, 0.04, 3)
        p3 = (p2 + p4) / 2 + np.array([rng.uniform(-0.12, 0), rng.uniform(-0.05, 0.05), rng.uniform(0, 0.15)])
    else:
        phi_r, psi_r = {"out": ((40, 80), (0, 20)),
                        "up": ((-5, 30), (0, 15)),
                        "tuck": ((-80, -30), (10, 50))}[mode]
        phi2 = math.radians(rng.uniform(*phi_r))
        phi3 = phi2 + math.radians(rng.uniform(-10, 10))
        psi2 = math.radians(rng.uniform(*psi_r))
        psi3 = psi2 + math.radians(rng.uniform(0, 20))
        dv = lambda ph, ps: np.array([-math.sin(ph) * math.cos(ps), math.cos(ph) * math.cos(ps), math.sin(ps)])
        p3 = p2 + 0.30 * dv(phi2, psi2)
        p4 = p3 + 0.26 * dv(phi3, psi3)
    pts[1:5] = [p1, p2, p3, p4]

    pts = pts @ _rotation(rng).T
    sigma = rng.uniform(0.005, 0.04)
    pts = pts + rng.normal(0, sigma, pts.shape)
    pts[:, 2] += rng.normal(0, sigma * 2.5, 21)
    return pts


def build_synthetic_dataset(n_per_class=1200, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for name in WORDS:
        for _ in range(n_per_class):
            X.append(extract_features(synth_sample(name, rng)))
            y.append(name)
    return np.array(X), np.array(y)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
class SignModel:
    def __init__(self):
        self.clf = None
        self.real_X, self.real_y = [], []
        self.synth = None

    def load_user_samples(self):
        if os.path.exists(USER_FILE):
            try:
                with open(USER_FILE, "rb") as fh:
                    self.real_X, self.real_y = pickle.load(fh)
            except Exception:
                self.real_X, self.real_y = [], []

    def save_user_samples(self):
        with open(USER_FILE, "wb") as fh:
            pickle.dump((self.real_X, self.real_y), fh)

    def load_or_train(self):
        self.load_user_samples()
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as fh:
                    data = pickle.load(fh)
                if data.get("words") == WORDS:
                    self.clf = data["clf"]
                    return
            except Exception:
                pass
        self.train()

    def train(self):
        if self.synth is None:
            self.synth = build_synthetic_dataset()
        X, y = self.synth
        w = np.ones(len(y))
        if self.real_X:
            X = np.vstack([X, np.array(self.real_X)])
            y = np.concatenate([y, np.array(self.real_y)])
            w = np.concatenate([w, np.full(len(self.real_y), 10.0)])
        self.clf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                          n_jobs=-1, random_state=1)
        self.clf.fit(X, y, sample_weight=w)
        with open(MODEL_FILE, "wb") as fh:
            pickle.dump({"clf": self.clf, "words": WORDS}, fh)

    def predict_features(self, feats):
        p = self.clf.predict_proba([feats])[0]
        i = int(np.argmax(p))
        return self.clf.classes_[i], float(p[i])

    def add_real(self, word, feats):
        self.real_X.append(feats)
        self.real_y.append(word)

    def clear_real(self):
        self.real_X, self.real_y = [], []
        if os.path.exists(USER_FILE):
            os.remove(USER_FILE)

# ----------------------------------------------------------------------------
# GUI APPLICATION
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Sign Language Detector")
        root.configure(bg="#1e1f26")
        self.model = SignModel()
        self.ready = False
        self.cap = None
        self.running = False
        self.hands_video = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                          min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.hands_image = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                                          min_detection_confidence=0.5)
        self.history = deque(maxlen=SMOOTH_FRAMES)
        self.cand, self.cand_n = None, 0
        self.sentence = []
        self.rec_word, self.rec_left = None, 0
        self.override = tk.BooleanVar(value=False)
        self.build_ui()
        self.update_clock()
        root.after(200, self.init_model)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- UI ----------------
    def build_ui(self):
        bg, fg, panel = "#1e1f26", "#f0f0f0", "#2a2c36"
        left = tk.Frame(self.root, bg=bg)
        left.grid(row=0, column=0, padx=10, pady=10)
        self.video_label = tk.Label(left, bg="black", width=DISP_W, height=DISP_H)
        self.video_label.pack()
        blank = ImageTk.PhotoImage(Image.new("RGB", (DISP_W, DISP_H), "black"))
        self.video_label.configure(image=blank, width=DISP_W, height=DISP_H)
        self.video_label.image = blank

        right = tk.Frame(self.root, bg=panel, padx=12, pady=10)
        right.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)

        tk.Label(right, text="SIGN LANGUAGE DETECTOR", bg=panel, fg="#6cc4ff",
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        self.clock_var = tk.StringVar()
        tk.Label(right, textvariable=self.clock_var, bg=panel, fg=fg,
                 font=("Segoe UI", 10)).pack(anchor="w")
        self.active_lbl = tk.Label(right, text="", bg=panel, font=("Segoe UI", 11, "bold"))
        self.active_lbl.pack(anchor="w")
        tk.Checkbutton(right, text="Ignore time window (testing)", variable=self.override,
                       bg=panel, fg=fg, selectcolor="#444", activebackground=panel,
                       activeforeground=fg).pack(anchor="w", pady=(0, 8))

        bf = tk.Frame(right, bg=panel)
        bf.pack(fill="x")
        ttk.Button(bf, text="Upload Image", command=self.upload_image).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(bf, text="Start Webcam", command=self.start_webcam).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(bf, text="Stop Webcam", command=self.stop_webcam).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(bf, text="Clear Sentence", command=self.clear_sentence).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        bf.columnconfigure((0, 1), weight=1)

        tk.Label(right, text="Detected word:", bg=panel, fg=fg, font=("Segoe UI", 10)).pack(anchor="w", pady=(10, 0))
        self.word_var = tk.StringVar(value="-")
        tk.Label(right, textvariable=self.word_var, bg=panel, fg="#7CFC9A",
                 font=("Segoe UI", 26, "bold")).pack(anchor="w")
        self.conf_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.conf_var, bg=panel, fg=fg, font=("Segoe UI", 10)).pack(anchor="w")

        tk.Label(right, text="Sentence:", bg=panel, fg=fg, font=("Segoe UI", 10)).pack(anchor="w", pady=(10, 0))
        self.sentence_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.sentence_var, bg="#15161b", fg="#ffd166", wraplength=300,
                 justify="left", anchor="nw", width=36, height=3,
                 font=("Segoe UI", 12)).pack(anchor="w")

        tk.Label(right, text="Known signs:", bg=panel, fg=fg, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        guide = "\n".join(f"- {w}: {GESTURES[w]['desc']}" for w in WORDS)
        tk.Label(right, text=guide, bg=panel, fg="#cfcfcf", justify="left",
                 font=("Segoe UI", 9)).pack(anchor="w")

        tk.Label(right, text="Calibrate (optional, improves accuracy):", bg=panel, fg=fg,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        cf = tk.Frame(right, bg=panel)
        cf.pack(fill="x")
        self.cal_word = ttk.Combobox(cf, values=WORDS, state="readonly", width=14)
        self.cal_word.current(0)
        self.cal_word.grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(cf, text="Record 40 samples", command=self.start_recording).grid(row=0, column=1, padx=2)
        ttk.Button(cf, text="Reset my samples", command=self.reset_samples).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2)

        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(right, textvariable=self.status_var, bg=panel, fg="#aaaaaa",
                 wraplength=300, justify="left", font=("Segoe UI", 9)).pack(anchor="w", pady=(10, 0))

    # ---------------- helpers ----------------
    def init_model(self):
        self.status_var.set("Training model on built-in data (first run only, ~10-30 s)...")
        self.root.update()
        try:
            self.model.load_or_train()
            self.ready = True
            n = len(self.model.real_X)
            self.status_var.set(f"Model ready. ({n} personal samples loaded)")
        except Exception as e:
            messagebox.showerror("Error", f"Could not prepare model:\n{e}")

    def is_active(self):
        if self.override.get():
            return True
        h = dt.datetime.now().hour
        if START_HOUR <= END_HOUR:
            return START_HOUR <= h < END_HOUR
        return h >= START_HOUR or h < END_HOUR

    @staticmethod
    def _fmt(h):
        return dt.time(h % 24).strftime("%I:%M %p")

    def update_clock(self):
        now = dt.datetime.now().strftime("%d %b %Y  %I:%M:%S %p")
        self.clock_var.set(f"{now}   (works {self._fmt(START_HOUR)} - {self._fmt(END_HOUR)})")
        if self.is_active():
            self.active_lbl.config(text="MODEL: ACTIVE", fg="#7CFC9A")
        else:
            self.active_lbl.config(text="MODEL: OFFLINE (outside time window)", fg="#ff6b6b")
        self.root.after(1000, self.update_clock)

    def set_result(self, word, conf=""):
        self.word_var.set(word)
        self.conf_var.set(conf)

    def show(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = min(DISP_W / w, DISP_H / h)
        rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))))
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=img)
        self.video_label.image = img

    def banner(self, bgr, text, color=(0, 0, 255)):
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ---------------- core analysis ----------------
    def analyse(self, bgr, hands, live):
        if not self.ready:
            self.banner(bgr, "Loading model...", (0, 255, 255))
            return bgr
        if not self.is_active():
            self.set_result("OFFLINE", f"Operates {self._fmt(START_HOUR)} - {self._fmt(END_HOUR)}")
            self.banner(bgr, "MODEL OFFLINE - active 6 PM to 10 PM")
            return bgr

        h, w = bgr.shape[:2]
        res = hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            self.set_result("No hand detected", "")
            self.history.clear()
            self.cand, self.cand_n = None, 0
            return bgr

        hl = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(bgr, hl, mp_hands.HAND_CONNECTIONS,
                               mp_styles.get_default_hand_landmarks_style(),
                               mp_styles.get_default_hand_connections_style())
        lm = np.array([[p.x * w, p.y * h, p.z * w] for p in hl.landmark])
        feats = extract_features(lm)

        # optional calibration recording
        if live and self.rec_left > 0:
            self.model.add_real(self.rec_word, feats)
            self.rec_left -= 1
            self.status_var.set(f"Recording '{self.rec_word}' ... {self.rec_left} left")
            if self.rec_left == 0:
                self.finish_recording()

        word, conf = self.model.predict_features(feats)
        if conf < CONF_THRESHOLD:
            word = "Unknown"

        if live:
            self.history.append(word)
            shown = Counter(self.history).most_common(1)[0][0]
            if shown == self.cand:
                self.cand_n += 1
            else:
                self.cand, self.cand_n = shown, 1
            if (self.cand_n == STABLE_FRAMES and shown != "Unknown"
                    and (not self.sentence or self.sentence[-1] != shown)):
                self.sentence.append(shown)
                self.sentence_var.set(" ".join(self.sentence))
            word = shown

        self.set_result(word, f"Confidence: {conf * 100:.0f}%")
        self.banner(bgr, f"{word}  ({conf * 100:.0f}%)", (0, 255, 0) if word != "Unknown" else (0, 200, 255))
        return bgr

    # ---------------- image upload ----------------
    def upload_image(self):
        if not self.ready:
            messagebox.showinfo("Please wait", "Model is still loading.")
            return
        if not self.is_active():
            messagebox.showwarning("Model offline",
                                   f"The model works only between {self._fmt(START_HOUR)} and "
                                   f"{self._fmt(END_HOUR)}.\n(Tick 'Ignore time window' to test.)")
            return
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        self.stop_webcam()
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read that image.")
            return
        out = self.analyse(img, self.hands_image, live=False)
        self.show(out)
        self.status_var.set(f"Image: {os.path.basename(path)}")

    # ---------------- webcam ----------------
    def start_webcam(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam", "Could not open the webcam.")
            self.cap = None
            return
        self.running = True
        self.status_var.set("Webcam running...")
        self.video_loop()

    def stop_webcam(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.rec_left = 0

    def video_loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            self.show(self.analyse(frame, self.hands_video, live=True))
        self.root.after(15, self.video_loop)

    # ---------------- sentence / calibration ----------------
    def clear_sentence(self):
        self.sentence = []
        self.sentence_var.set("")

    def start_recording(self):
        if not self.running:
            messagebox.showinfo("Calibrate", "Start the webcam first, then show the sign and press Record.")
            return
        self.rec_word, self.rec_left = self.cal_word.get(), 40
        self.status_var.set(f"Recording '{self.rec_word}' - hold the sign steady in front of the camera.")

    def finish_recording(self):
        self.status_var.set("Re-training with your samples...")
        self.root.update()
        self.model.save_user_samples()
        self.model.train()
        self.status_var.set(f"Done. Model updated ({len(self.model.real_X)} personal samples).")

    def reset_samples(self):
        self.model.clear_real()
        self.status_var.set("Personal samples deleted. Re-training...")
        self.root.update()
        self.model.train()
        self.status_var.set("Model reset to built-in data.")

    def on_close(self):
        self.stop_webcam()
        self.hands_video.close()
        self.hands_image.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
