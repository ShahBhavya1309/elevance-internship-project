"""
app.py - GUI for "Age and Emotion Detection through Voice".

    python app.py

Rules implemented:
    female voice      -> rejected, message: "Upload male voice."
    male, age > 60    -> Senior Citizen: age + emotion
    male, age <= 60   -> age only

Inputs: Upload a voice note (WAV; mp3/m4a with ffmpeg+pydub), Record from the microphone (needs sounddevice),
or pick one of the BUILT-IN sample voices (no download, no files needed).
On the very first run (if models/voice_models.joblib is missing) the app creates its own built-in dataset and
trains the models automatically.
"""
import csv
import os
import platform
import queue
import subprocess
import tempfile
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import voice_core as vc

LOG_PATH = "voice_log.csv"
C = dict(bg="#f3f5f9", card="#ffffff", ink="#1f2937", muted="#6b7280", ok="#1f9d55", warn="#e67e22",
         bad="#d64545", accent="#2563eb", line="#d1d5db")
EMO_COLORS = {"Neutral": "#6b7280", "Happy": "#f1c40f", "Sad": "#3b82f6", "Angry": "#d64545", "Fearful": "#8e44ad"}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Voice Age & Emotion Detection (male voices)")
        root.geometry("1120x780")
        root.configure(bg=C["bg"])
        self.q = queue.Queue()
        self.analyzer = None
        self.current = None                      # last analysed (trimmed) signal, for playback
        self.samples = vc.ensure_samples()
        self._build_ui()
        self.root.after(100, self._poll)
        self._load_or_train()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=6)

        tk.Label(self.root, text="Age & Emotion Detection through Voice", bg=C["bg"], fg=C["ink"],
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=14, pady=(10, 0))
        tk.Label(self.root, text="Only MALE voices are processed.  Age > 60  =  Senior Citizen  +  emotion.",
                 bg=C["bg"], fg=C["muted"], font=("Segoe UI", 10)).pack(anchor="w", padx=14)

        bar = tk.Frame(self.root, bg=C["bg"])
        bar.pack(fill="x", padx=14, pady=8)
        self.btn_upload = ttk.Button(bar, text="Upload Voice Note", command=self.upload)
        self.btn_record = ttk.Button(bar, text="Record (6 s)", command=self.record)
        self.sample_var = tk.StringVar(value=self.samples[0]["label"])
        self.cmb = ttk.Combobox(bar, textvariable=self.sample_var, state="readonly", width=26,
                                values=[s["label"] for s in self.samples])
        self.btn_sample = ttk.Button(bar, text="Analyze Built-in Sample", command=self.use_sample)
        self.btn_play = ttk.Button(bar, text="Play", command=self.play)
        self.btn_log = ttk.Button(bar, text="Open Log CSV", command=self.open_log)
        self.buttons = [self.btn_upload, self.btn_record, self.btn_sample, self.btn_play, self.btn_log]
        self.btn_upload.pack(side="left", padx=3)
        self.btn_record.pack(side="left", padx=3)
        tk.Label(bar, text="   Built-in sample:", bg=C["bg"], fg=C["muted"]).pack(side="left")
        self.cmb.pack(side="left", padx=3)
        self.btn_sample.pack(side="left", padx=3)
        self.btn_play.pack(side="left", padx=12)
        self.btn_log.pack(side="left", padx=3)

        self.status = tk.StringVar(value="Starting...")
        tk.Label(self.root, textvariable=self.status, bg=C["bg"], fg=C["muted"], anchor="w").pack(fill="x", padx=14)
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")

        mid = tk.Frame(self.root, bg=C["bg"])
        mid.pack(fill="both", expand=True, padx=14, pady=6)

        left = tk.Frame(mid, bg=C["card"], highlightbackground=C["line"], highlightthickness=1)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="Voice waveform", bg=C["card"], fg=C["muted"]).pack(anchor="w", padx=8, pady=(6, 0))
        self.wave = tk.Canvas(left, width=640, height=150, bg="white", highlightthickness=0)
        self.wave.pack(padx=8, pady=2, fill="x")
        tk.Label(left, text="Pitch contour (Hz)", bg=C["card"], fg=C["muted"]).pack(anchor="w", padx=8, pady=(6, 0))
        self.pitch = tk.Canvas(left, width=640, height=130, bg="white", highlightthickness=0)
        self.pitch.pack(padx=8, pady=(2, 8), fill="x")

        right = tk.Frame(mid, bg=C["card"], highlightbackground=C["line"], highlightthickness=1, width=380)
        right.pack(side="left", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self.verdict = tk.Label(right, text="Waiting for a voice...", bg=C["muted"], fg="white",
                                font=("Segoe UI", 16, "bold"), wraplength=340, pady=14)
        self.verdict.pack(fill="x", padx=10, pady=10)
        self.v = {k: tk.StringVar(value="-") for k in ("gender", "age", "category", "emotion")}
        for label, key in [("Gender", "gender"), ("Estimated age", "age"), ("Category", "category"),
                           ("Emotion", "emotion")]:
            row = tk.Frame(right, bg=C["card"])
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=label, width=14, anchor="w", bg=C["card"], fg=C["muted"]).pack(side="left")
            tk.Label(row, textvariable=self.v[key], anchor="w", bg=C["card"], fg=C["ink"],
                     font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Label(right, text="Emotion probabilities (seniors only)", bg=C["card"], fg=C["muted"]).pack(
            anchor="w", padx=14, pady=(10, 0))
        self.bars = tk.Canvas(right, width=350, height=150, bg=C["card"], highlightthickness=0)
        self.bars.pack(padx=14, pady=4)

        cols = ("Time", "Source", "Gender", "Age", "Category", "Emotion")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (150, 260, 90, 70, 130, 100)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="x", padx=14, pady=(4, 12))
        self.tree.tag_configure("senior", background="#ffe9d6")
        self.tree.tag_configure("rejected", background="#fde2e2")

    # ------------------------------------------------------------------ model loading / first-run training
    def _set_busy(self, busy, text=None):
        for b in self.buttons:
            b.state(["disabled"] if busy else ["!disabled"])
        self.cmb.state(["disabled"] if busy else ["!disabled", "readonly"])
        if busy:
            self.progress.pack(fill="x", padx=14, pady=2)
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()
        if text:
            self.status.set(text)

    def _load_or_train(self):
        if os.path.exists(vc.MODEL_PATH):
            try:
                self.analyzer = vc.VoiceAnalyzer()
                m = self.analyzer.bundle["metrics"]
                self._set_busy(False, f"Ready.  Model test results: gender acc {m['gender_accuracy'] * 100:.1f}%, "
                                      f"age MAE {m['age_mae_male']:.1f} yrs, emotion acc "
                                      f"{m['emotion_accuracy_all_males'] * 100:.1f}%  (on built-in test voices)")
                return
            except Exception:
                pass                                            # old/incompatible model file -> retrain
        self._set_busy(True, "First run: creating the built-in voice dataset and training the models "
                             "(a few minutes, only once)...")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        try:
            import train
            train.train_all(n=2000, progress=lambda m: self.q.put(("progress", m)))
            self.q.put(("trained", None))
        except Exception as e:
            self.q.put(("error", f"Training failed: {e}"))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "progress":
                    self.status.set(payload)
                elif kind == "trained":
                    self.analyzer = vc.VoiceAnalyzer()
                    self._set_busy(False, "Training finished. Ready.")
                elif kind == "error":
                    self._set_busy(False, payload)
                    messagebox.showerror("Error", payload)
                elif kind == "recorded":
                    self._set_busy(False, "Recording finished. Analyzing...")
                    self.analyze(payload, vc.SR, "Microphone recording")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ inputs
    def upload(self):
        path = filedialog.askopenfilename(
            initialdir="samples", title="Choose a voice note",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")])
        if not path:
            return
        try:
            y, sr = vc.load_audio(path)
        except ValueError as e:
            messagebox.showerror("Cannot read file", str(e))
            return
        self.analyze(y, sr, os.path.basename(path))

    def use_sample(self):
        item = next(s for s in self.samples if s["label"] == self.sample_var.get())
        y, sr = vc.load_audio(item["path"])
        self.status.set(f"Built-in sample (true values: {item['gender']}, {item['age']} yrs, {item['emotion']})")
        self.analyze(y, sr, "Sample: " + item["label"])

    def record(self):
        try:
            import sounddevice as sd
        except ImportError:
            messagebox.showinfo("Microphone", "Recording needs the 'sounddevice' package:\n\n"
                                              "    pip install sounddevice\n\nYou can still upload files or use the built-in samples.")
            return
        self._set_busy(True, "Recording for 6 seconds - speak now...")

        def work():
            try:
                rec = sd.rec(int(6 * vc.SR), samplerate=vc.SR, channels=1, dtype="float32")
                sd.wait()
                self.q.put(("recorded", rec[:, 0].copy()))
            except Exception as e:
                self.q.put(("error", f"Microphone error: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def play(self):
        if self.current is None:
            messagebox.showinfo("Play", "Analyze a voice first.")
            return
        try:
            import sounddevice as sd
            sd.play(self.current, vc.SR)
            return
        except Exception:
            pass
        if platform.system() == "Windows":
            import winsound
            tmp = os.path.join(tempfile.gettempdir(), "voice_preview.wav")
            vc.save_wav(tmp, self.current)
            winsound.PlaySound(tmp, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            messagebox.showinfo("Play", "Install 'sounddevice' (pip install sounddevice) to play audio.")

    # ------------------------------------------------------------------ analysis + display
    def analyze(self, y, sr, source):
        if self.analyzer is None:
            return
        try:
            res = self.analyzer.analyze(y, sr)
        except ValueError as e:
            self._reset_result()
            self.verdict.config(text="Cannot analyze this recording", bg=C["warn"])
            messagebox.showwarning("Voice problem", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            return

        self.current = res["info"]["y"]
        self._draw_wave(self.current)
        self._draw_pitch(res["info"]["f0"], res["info"]["hop_s"])
        self.v["gender"].set(f"{res['gender']}  ({res['gender_conf'] * 100:.0f}% sure)")

        if not res["accepted"]:
            self.v["age"].set("-")
            self.v["category"].set("-")
            self.v["emotion"].set("-")
            self.verdict.config(text=res["message"], bg=C["bad"])
            self._draw_bars(None, "Not processed (female voice)")
            tag = "rejected"
            messagebox.showwarning("Female voice detected", "Upload male voice.")
        else:
            self.v["age"].set(f"{res['age']} years  (+/- {res['age_margin']})")
            self.v["category"].set(res["category"])
            if res["category"] == "Senior Citizen":
                self.v["emotion"].set(f"{res['emotion']}  ({res['emotion_probs'][res['emotion']] * 100:.0f}%)")
                self.verdict.config(text=f"SENIOR CITIZEN\nAge {res['age']}  |  {res['emotion']}", bg=C["warn"])
                self._draw_bars(res["emotion_probs"], None)
                tag = "senior"
            else:
                self.v["emotion"].set("not required (below 60)")
                self.verdict.config(text=f"Below 60\nAge {res['age']}", bg=C["ok"])
                self._draw_bars(None, "Emotion is detected only for seniors")
                tag = ""

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = (now, source, res["gender"], res["age"] if res["age"] else "-", res["category"] or "REJECTED",
               res["emotion"] or "-")
        self.tree.insert("", 0, values=row, tags=(tag,) if tag else ())
        self._log(now, source, res)

    def _reset_result(self):
        for var in self.v.values():
            var.set("-")
        self.wave.delete("all")
        self.pitch.delete("all")
        self._draw_bars(None, "")

    def _log(self, now, source, res):
        new = not os.path.exists(LOG_PATH)
        try:
            with open(LOG_PATH, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["Time", "Source", "Gender", "Gender_Confidence", "Age", "Category", "Emotion", "Message"])
                w.writerow([now, source, res["gender"], f"{res['gender_conf']:.2f}", res["age"] or "",
                            res["category"] or "REJECTED", res["emotion"] or "", res["message"]])
        except PermissionError:
            pass

    def open_log(self):
        if not os.path.exists(LOG_PATH):
            messagebox.showinfo("Log", "No analyses logged yet.")
            return
        path = os.path.abspath(LOG_PATH)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])

    # ------------------------------------------------------------------ drawing
    def _size(self, canvas):
        canvas.update_idletasks()
        return max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 60)

    def _draw_wave(self, y):
        c = self.wave
        c.delete("all")
        w, h = self._size(c)
        mid = h // 2
        c.create_line(0, mid, w, mid, fill=C["line"])
        bins = np.array_split(y, w)
        for x, b in enumerate(bins):
            if len(b):
                c.create_line(x, mid - float(b.max()) * (mid - 4), x, mid - float(b.min()) * (mid - 4), fill=C["accent"])
        c.create_text(w - 6, 10, anchor="ne", fill=C["muted"], text=f"{len(y) / vc.SR:.1f} s")

    def _draw_pitch(self, f0, hop):
        c = self.pitch
        c.delete("all")
        w, h = self._size(c)
        lo, hi = 60.0, 320.0
        for hz in (100, 150, 200, 250, 300):
            yy = h - 14 - (hz - lo) / (hi - lo) * (h - 24)
            c.create_line(0, yy, w, yy, fill="#eef0f3")
            c.create_text(4, yy - 1, anchor="sw", fill=C["muted"], text=str(hz), font=("Segoe UI", 7))
        n = len(f0)
        prev = None
        for i, v in enumerate(f0):
            if np.isnan(v):
                prev = None
                continue
            pt = (i / max(n - 1, 1) * (w - 1), h - 14 - (min(max(v, lo), hi) - lo) / (hi - lo) * (h - 24))
            if prev is not None:
                c.create_line(prev[0], prev[1], pt[0], pt[1], fill=C["bad"], width=2)
            prev = pt

    def _draw_bars(self, probs, note):
        c = self.bars
        c.delete("all")
        if probs is None:
            c.create_text(175, 70, text=note or "", fill=C["muted"])
            return
        top = max(probs, key=probs.get)
        for i, name in enumerate(vc.EMOTIONS):
            y = 8 + i * 28
            p = probs[name]
            c.create_text(0, y + 10, anchor="w", text=name, fill=C["ink"], font=("Segoe UI", 9, "bold" if name == top else "normal"))
            c.create_rectangle(75, y, 75 + 220, y + 20, fill="#eef0f3", outline="")
            c.create_rectangle(75, y, 75 + 220 * p, y + 20, fill=EMO_COLORS[name], outline="")
            c.create_text(300, y + 10, anchor="w", text=f"{p * 100:.0f}%", fill=C["ink"])


if __name__ == "__main__":
    tk_root = tk.Tk()
    App(tk_root)
    tk_root.mainloop()
