"""
app.py  -  Senior Citizen Identification: webcam / video, multi-person tracking, CSV logging, GUI.

Run:  python app.py        (after training:  python train.py --data data/UTKFace)

Pipeline per frame:
  1. Face detection (YuNet if it can be downloaded, otherwise OpenCV Haar cascade) - locating faces only
  2. IoU tracker gives every face a persistent ID
  3. MY trained CNN (models/age_gender_model.keras) predicts age + gender for each face
  4. Predictions are smoothed over frames (median age, mean gender probability)
  5. After MIN_OBS good observations the person is logged ONCE to visitor_log.csv
"""
import csv
import json
import os
import platform
import subprocess
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

try:                                   # GUI libs are optional (demo.py / run_all.py work without)
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk
except ImportError:
    tk = filedialog = messagebox = ttk = ImageTk = None

# ----------------------------------------------------------------------------- settings
MODEL_PATH = "models/age_gender_model.keras"
CONFIG_PATH = "models/config.json"
CSV_PATH = "visitor_log.csv"
YUNET_FILE = "models/face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

IMG_SIZE = 160
SENIOR_AGE = 60                      # senior if smoothed age > 60
USE_CALIBRATED_THRESHOLD = False     # True -> use the threshold tuned on validation data (config.json)
MIN_FACE = 48                        # ignore faces smaller than this (pixels)
MIN_OBS = 8                          # observations required before a person is logged
INFER_EVERY = 3                      # after confirmation, re-run the model every N frames (speed)
MAX_MISSED = 30                      # frames a track survives without a detection
IOU_THRESH = 0.20
BLUR_THRESH = 6.0                    # skip faces blurrier than this (Laplacian variance)
MAX_DISPLAY_W = 960                  # frames are down-scaled to this width for speed

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 215, 255)
CSV_HEADER = ["Person_ID", "Age", "Gender", "Senior_Citizen", "Date", "Time_of_Visit"]


def senior_threshold():
    if USE_CALIBRATED_THRESHOLD and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return float(json.load(f)["senior_threshold_calibrated"])
        except Exception:
            pass
    return float(SENIOR_AGE)


# ----------------------------------------------------------------------------- face detector
class FaceDetector:
    def __init__(self):
        self.yunet, self.haar = None, None
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(YUNET_FILE):
            try:
                urllib.request.urlretrieve(YUNET_URL, YUNET_FILE + ".tmp")
                os.replace(YUNET_FILE + ".tmp", YUNET_FILE)
            except Exception:
                print("YuNet download failed -> using Haar cascade fallback.")
        if os.path.exists(YUNET_FILE) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(YUNET_FILE, "", (320, 320), 0.7, 0.3, 5000)
            except Exception:
                self.yunet = None
        if self.yunet is None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("Face detector:", "YuNet" if self.yunet is not None else "Haar cascade")

    def detect(self, frame):
        h, w = frame.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame)
            raw = [] if faces is None else [tuple(int(v) for v in r[:4]) for r in faces]
        else:
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            raw = [tuple(int(v) for v in r)
                   for r in self.haar.detectMultiScale(gray, 1.1, 6, minSize=(MIN_FACE, MIN_FACE))]
        boxes = []
        for x, y, bw, bh in raw:                      # clamp to frame
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw >= MIN_FACE and bh >= MIN_FACE:
                boxes.append((x, y, bw, bh))
        return boxes


def crop_face(frame, box, margin=0.15):
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(frame.shape[1], x + w + mx), min(frame.shape[0], y + h + my)
    return frame[y1:y2, x1:x2]


def is_sharp(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= BLUR_THRESH


# ----------------------------------------------------------------------------- my CNN
class AgeGenderModel:
    def __init__(self, path=MODEL_PATH):
        import tensorflow as tf                        # imported here so the GUI error is readable
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trained model not found: {path}\nRun train.py first.")
        self.model = tf.keras.models.load_model(path, compile=False)
        self.model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), "float32"), training=False)  # warm-up

    def predict(self, crops):
        batch = np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
                          for c in crops]).astype("float32")
        out = self.model(batch, training=False)
        if isinstance(out, dict):
            age, gender = out["age"], out["gender"]
        else:
            age, gender = out[0], out[1]
        return age.numpy().ravel() * 100.0, gender.numpy().ravel()   # years, P(female)


# ----------------------------------------------------------------------------- tracking
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box):
        self.id, self.box = tid, box
        self.ages, self.gprobs = deque(maxlen=30), deque(maxlen=30)
        self.first_seen = datetime.now()
        self.missed = 0
        self.logged = False

    @property
    def ready(self):
        return len(self.ages) >= MIN_OBS

    @property
    def age(self):
        return float(np.median(self.ages))

    @property
    def gender(self):
        return "Female" if float(np.mean(self.gprobs)) > 0.5 else "Male"


class Tracker:
    """Greedy IoU tracker: each detected face keeps the same ID while it stays on screen."""

    def __init__(self):
        self.tracks, self.next_id = [], 1

    def update(self, boxes):
        pairs = sorted(((iou(t.box, b), ti, bi)
                        for ti, t in enumerate(self.tracks) for bi, b in enumerate(boxes)),
                       reverse=True)
        used_t, used_b, matches = set(), set(), []
        for score, ti, bi in pairs:
            if score < IOU_THRESH:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            t = self.tracks[ti]
            t.box, t.missed = boxes[bi], 0
            matches.append((t, boxes[bi]))
        for ti, t in enumerate(self.tracks):          # unmatched existing tracks
            if ti not in used_t:
                t.missed += 1
        for bi, b in enumerate(boxes):                # unmatched detections -> new people
            if bi not in used_b:
                t = Track(self.next_id, b)
                self.next_id += 1
                self.tracks.append(t)
                matches.append((t, b))
        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED]
        return matches


# ----------------------------------------------------------------------------- logging
class VisitLogger:
    def __init__(self, path=CSV_PATH):
        self.path = path
        self.new_session()

    def new_session(self):
        self.session = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)

    def log(self, track, age, gender, senior):
        row = [f"{self.session}-P{track.id:04d}", int(round(age)), gender,
               "Yes" if senior else "No",
               track.first_seen.strftime("%Y-%m-%d"), track.first_seen.strftime("%H:%M:%S")]
        try:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except PermissionError:                       # e.g. file open in Excel on Windows
            backup = self.path.replace(".csv", "_backup.csv")
            new = not os.path.exists(backup)
            with open(backup, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(CSV_HEADER)
                w.writerow(row)
            print(f"'{self.path}' is locked; wrote to '{backup}' instead.")
        return row


# ----------------------------------------------------------------------------- engine
class SeniorDetector:
    def __init__(self):
        self.detector = FaceDetector()
        self.predictor = AgeGenderModel()
        self.logger = VisitLogger()
        self.threshold = senior_threshold()
        self.reset()

    def reset(self):
        self.tracker = Tracker()
        self.frame_idx = 0
        self.last_boxes = []
        self.stats = {"total": 0, "seniors": 0, "male": 0, "female": 0}
        self.logger.new_session()

    def process(self, frame):
        """Returns (annotated_frame, list_of_new_csv_rows)."""
        self.frame_idx += 1
        matches = self.tracker.update(self.detector.detect(frame))

        todo, crops = [], []
        for t, box in matches:
            if (not t.ready) or self.frame_idx % INFER_EVERY == 0:
                c = crop_face(frame, box)
                if c.size and is_sharp(c):
                    todo.append(t)
                    crops.append(c)
        if crops:
            ages, gprobs = self.predictor.predict(crops)
            for t, a, g in zip(todo, ages, gprobs):
                t.ages.append(float(a))
                t.gprobs.append(float(g))

        new_rows = []
        self.last_boxes = []                       # boxes of the people logged in THIS frame
        for t, (x, y, w, h) in matches:
            if not t.ready:
                color, label = YELLOW, f"#{t.id} Analyzing..."
            else:
                senior = t.age > self.threshold
                color = RED if senior else GREEN
                label = f"#{t.id} {t.gender}, {int(round(t.age))}" + (" SENIOR CITIZEN" if senior else "")
                if not t.logged:
                    t.logged = True
                    new_rows.append(self.logger.log(t, t.age, t.gender, senior))
                    self.last_boxes.append((x, y, w, h))
                    self.stats["total"] += 1
                    self.stats["seniors"] += int(senior)
                    self.stats["male" if t.gender == "Male" else "female"] += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = y - 8 if y - th - 12 > 0 else y + h + th + 8
            cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 4), color, -1)
            cv2.putText(frame, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0) if color == YELLOW else (255, 255, 255), 2, cv2.LINE_AA)
        return frame, new_rows


# ----------------------------------------------------------------------------- GUI
class App:
    def __init__(self, root):
        self.root = root
        root.title("Senior Citizen Identification System")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.cap, self.running, self.fps, self.photo = None, False, 0.0, None

        try:
            self.engine = SeniorDetector()
        except Exception as e:
            messagebox.showerror("Startup error", str(e))
            root.destroy()
            raise SystemExit(1)

        bar = ttk.Frame(root, padding=6)
        bar.pack(side=tk.TOP, fill=tk.X)
        for text, cmd in [("Start Webcam", self.start_webcam), ("Upload Video", self.upload_video),
                          ("Stop", self.stop), ("Open CSV", self.open_csv),
                          ("Export Excel", self.export_excel)]:
            ttk.Button(bar, text=text, command=cmd).pack(side=tk.LEFT, padx=4)

        body = ttk.Frame(root, padding=6)
        body.pack(fill=tk.BOTH, expand=True)
        self.video = tk.Label(body, bg="black", width=96, height=34)
        self.video.pack(side=tk.LEFT)

        side = ttk.Frame(body, padding=(10, 0))
        side.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(side, text="Live counts", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.stat_var = tk.StringVar()
        ttk.Label(side, textvariable=self.stat_var, justify="left", font=("Consolas", 11)).pack(anchor="w", pady=6)
        ttk.Label(side, text="Recent visitors", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 2))
        cols = ("ID", "Age", "Gender", "Senior", "Time")
        self.tree = ttk.Treeview(side, columns=cols, show="headings", height=18)
        for c, wdt in zip(cols, (110, 45, 65, 55, 70)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=wdt, anchor="center")
        self.tree.pack(fill=tk.Y)
        self.tree.tag_configure("senior", background="#ffd6d6")
        self.refresh_stats()

    # --- sources
    def start_webcam(self):
        self._open(0)

    def upload_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")])
        if path:
            self._open(path)

    def _open(self, source):
        self.stop()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open the camera / video file.")
            return
        self.cap = cap
        self.engine.reset()
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.refresh_stats()
        self.running = True
        self._loop()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # --- main loop (runs on Tk's event loop, no threads needed)
    def _loop(self):
        if not self.running:
            return
        t0 = time.time()
        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            messagebox.showinfo("Done", "End of video / camera stream.")
            return
        if frame.shape[1] > MAX_DISPLAY_W:
            s = MAX_DISPLAY_W / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s)

        frame, rows = self.engine.process(frame)

        for r in rows:
            self.tree.insert("", 0, values=(r[0][-5:], r[1], r[2], r[3], r[5]),
                             tags=("senior",) if r[3] == "Yes" else ())
        for i in self.tree.get_children()[50:]:
            self.tree.delete(i)

        dt = max(time.time() - t0, 1e-6)
        self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt
        self.refresh_stats()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.configure(image=self.photo, width=rgb.shape[1], height=rgb.shape[0])
        self.root.after(1, self._loop)

    def refresh_stats(self):
        s = self.engine.stats
        self.stat_var.set(f"Total people : {s['total']}\nSeniors      : {s['seniors']}\n"
                          f"Male         : {s['male']}\nFemale       : {s['female']}\n"
                          f"FPS          : {self.fps:.1f}")

    # --- files
    def open_csv(self):
        if not os.path.exists(CSV_PATH):
            messagebox.showinfo("CSV", "No log file yet.")
            return
        path = os.path.abspath(CSV_PATH)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])

    def export_excel(self):
        try:
            import pandas as pd
            df = pd.read_csv(CSV_PATH)
            out = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
                                               initialfile="visitor_log.xlsx")
            if out:
                df.to_excel(out, index=False)
                messagebox.showinfo("Export", f"Saved {len(df)} rows to\n{out}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    tk_root = tk.Tk()
    App(tk_root)
    tk_root.mainloop()
