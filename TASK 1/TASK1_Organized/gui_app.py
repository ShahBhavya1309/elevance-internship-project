"""
gui_app.py
----------
Tkinter GUI for the "Long Hair Identification" model, using YOUR OWN
trained models (age_gender_model.h5 + hair_length_model.h5).

Before running this, make sure age_gender_weights.weights.h5 and
hair_length_weights.weights.h5 (trained on Kaggle using
kaggle_train_all.py) are downloaded and placed in this same folder.

Then run:
    python gui_app.py

Flow:
    1. Click "Upload Image" and choose a photo (jpg/png) with a visible face.
    2. Click "Predict".
    3. The GUI shows:
         - the image with a face box drawn on it
         - detected age
         - the model's REAL gender prediction
         - the detected hair length
         - the FINAL gender shown by the app (after applying the
           age-20-30 hair-length override rule)
         - which rule was applied
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

from core import predict, DetectionError, ModelNotFoundError

WINDOW_TITLE = "Long Hair Identification (Age 20-30 Override Demo)"
DISPLAY_SIZE = (420, 420)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("560x760")
        self.root.resizable(False, False)

        self.bgr_image = None
        self.tk_image = None

        tk.Label(
            root, text=WINDOW_TITLE, font=("Helvetica", 14, "bold"), wraplength=520
        ).pack(pady=10)

        self.canvas = tk.Label(root, bg="#dddddd", width=60, height=20)
        self.canvas.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(
            btn_frame, text="Upload Image", command=self.upload_image, width=18
        ).grid(row=0, column=0, padx=5)
        tk.Button(
            btn_frame, text="Predict", command=self.run_prediction, width=18
        ).grid(row=0, column=1, padx=5)

        self.result_frame = tk.Frame(root, bd=1, relief="solid")
        self.result_frame.pack(pady=15, padx=20, fill="x")

        self.result_labels = {}
        fields = [
            ("age", "Estimated Age:"),
            ("real_gender", "Model's Real Gender Prediction:"),
            ("hair_label", "Detected Hair Length:"),
            ("rule_used", "Rule Applied:"),
            ("final_gender", "FINAL Displayed Gender:"),
        ]
        for i, (key, label_text) in enumerate(fields):
            tk.Label(
                self.result_frame, text=label_text, font=("Helvetica", 11, "bold"),
                anchor="w"
            ).grid(row=i, column=0, sticky="w", padx=10, pady=6)
            val_label = tk.Label(
                self.result_frame, text="-", font=("Helvetica", 11), anchor="w"
            )
            val_label.grid(row=i, column=1, sticky="w", padx=10, pady=6)
            self.result_labels[key] = val_label

        self.status_label = tk.Label(root, text="", fg="gray", wraplength=520)
        self.status_label.pack(pady=5)

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        self.bgr_image = cv2.imread(path)
        if self.bgr_image is None:
            messagebox.showerror("Error", "Could not read that image file.")
            return
        self._show_image(self.bgr_image)
        self.status_label.config(text=f"Loaded: {path}")
        self._clear_results()

    def run_prediction(self):
        if self.bgr_image is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        try:
            self.status_label.config(text="Running prediction... please wait.")
            self.root.update_idletasks()
            result = predict(self.bgr_image)
        except ModelNotFoundError as e:
            messagebox.showerror(
                "Model weights not found",
                f"{e}\n\nMake sure age_gender_weights.weights.h5 and "
                "hair_length_weights.weights.h5 are in this same folder.",
            )
            self.status_label.config(text="")
            return
        except DetectionError as e:
            messagebox.showerror("Detection Error", str(e))
            self.status_label.config(text="")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")
            self.status_label.config(text="")
            return

        annotated = self.bgr_image.copy()
        x, y, w, h = result["face_box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 3)
        self._show_image(annotated)

        self.result_labels["age"].config(text=str(result["age"]))
        self.result_labels["real_gender"].config(
            text=f'{result["real_gender"]} ({result["gender_confidence"]}% conf.)'
        )
        self.result_labels["hair_label"].config(
            text=f'{result["hair_label"]} (score={result["hair_score"]})'
        )
        self.result_labels["rule_used"].config(text=result["rule_used"])
        self.result_labels["final_gender"].config(
            text=result["final_gender"], font=("Helvetica", 12, "bold"), fg="#0a5"
        )

        self.status_label.config(text="Done.")

    def _show_image(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail(DISPLAY_SIZE)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.config(image=self.tk_image, width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1])

    def _clear_results(self):
        for lbl in self.result_labels.values():
            lbl.config(text="-", font=("Helvetica", 11), fg="black")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
