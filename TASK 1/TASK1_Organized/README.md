# Long Hair Identification — Full Custom-Model Solution

Detects gender normally for people aged **outside 20–30**, but for people
**aged 20–30**, the displayed gender is decided **purely by hair length**:
long hair → shown as Female, short hair → shown as Male — even if that
contradicts the model's real prediction.

Two custom CNNs are trained (not a third-party pretrained API):
- one **from scratch** for age + real gender (on UTKFace)
- one via **transfer learning** (MobileNetV2 backbone) for hair length

---

## 1. Files in this project

| File | Runs where? | Purpose |
|---|---|---|
| `kaggle_train_all.py` | **Kaggle** (paste into one notebook cell) | Trains both models, saves `age_gender_weights.weights.h5` + `hair_length_weights.weights.h5` |
| `model_defs.py` | Your laptop | Model architectures (shared definitions used by `core.py`) |
| `core.py` | Your laptop | Loads the two weight files, runs the full prediction pipeline, applies the 20–30 override rule |
| `gui_app.py` | Your laptop | Tkinter GUI — upload an image, click Predict |
| `requirements.txt` | Your laptop | Python dependencies |
| `age_gender_weights.weights.h5` | Downloaded from Kaggle | Trained age/gender weights |
| `hair_length_weights.weights.h5` | Downloaded from Kaggle | Trained hair-length weights |

---

## 2. Datasets (Kaggle) — used directly in the cloud, no local download needed

| Purpose | Dataset | Link |
|---|---|---|
| Age + real gender | UTKFace | https://www.kaggle.com/datasets/jangedoo/utkface-new |
| Hair-length proxy | Gender Classification Dataset | https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset |

**Why the second dataset is used for "hair length":** there is no large,
freely available dataset labeled purely by hair length. The Gender
Classification Dataset's `male/` and `female/` folders are used as a stand-in
(`male` → "Short hair" examples, `female` → "Long hair" examples), because in
practice most images in that dataset happen to have short vs. long
hairstyles matching those folders. Mention this if asked — it's a documented
design decision, not a bug.

---

## 3. Train on Kaggle (recommended — zero dataset download to your laptop)

1. Go to https://www.kaggle.com → **Create → New Notebook**
2. On the right sidebar, click **"+ Add Input"**:
   - Search **"UTKFace"** → add the one by **jangedoo**
   - Search **"Gender Classification Dataset"** → add the one by **Ashutosh Chauhan (cashutosh)**
3. Confirm your exact dataset paths by running this in a cell:
   ```python
   import os
   for root, dirs, files in os.walk('/kaggle/input'):
       print(root, len(files), 'files')
   ```
4. Open `kaggle_train_all.py`, check its `AGE_GENDER_DATASET_DIR` and
   `HAIR_DATASET_DIR` near the top match what you saw in step 3 — edit if
   your paths differ.
5. Paste the **entire contents** of `kaggle_train_all.py` into **one**
   notebook cell and run it (Shift+Enter). It trains both models back to
   back and prints progress for each.
6. (Optional but recommended) Turn on GPU first for much faster training:
   **Settings → Accelerator → GPU T4 x2** (top-right of the notebook).
7. Once finished, go to the **Output** panel and download:
   - `age_gender_weights.weights.h5`
   - `hair_length_weights.weights.h5`

   These are small (a few MB each) — only these two files need to reach
   your laptop, not the datasets.

---

## 4. Set up your laptop environment

Use Python **3.11 or 3.12** with TensorFlow **2.19.1** for the supplied Keras 3 weight files. Python 3.14 is not recommended for this setup.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 5. Assemble your project folder

Put all of these in the same folder:
```
model_defs.py
core.py
gui_app.py
requirements.txt
age_gender_weights.weights.h5      <- downloaded from Kaggle
hair_length_weights.weights.h5     <- downloaded from Kaggle
```

---

## 6. Run the GUI

```powershell
python gui_app.py
```

1. Click **Upload Image**, pick a photo with a visible face.
2. Click **Predict**.
3. See: estimated age, the model's real gender prediction, detected hair
   length, which rule fired, and the final displayed gender.

---

## 7. Logic summary (for your report / viva)

```
face = detect_face(image)
age, real_gender = age_gender_model.predict(face)
hair_length = hair_length_model.predict(head_and_shoulders_region)

if 20 <= age <= 30:
    final_gender = "Female" if hair_length == "Long" else "Male"   # override
else:
    final_gender = real_gender                                     # normal
```

---

## 8. Known limitations (worth stating in your submission)

- Age estimation from a single photo is inherently approximate, so the
  20/30 boundary will occasionally misclassify real ages very close to 20
  or 30.
- The hair-length model's "ground truth" is a gender-labeled dataset used
  as a proxy, not a true hair-length-labeled dataset — accuracy on unusual
  hairstyles (short-haired women, long-haired men) will be lower than a
  model trained on genuine hair-length labels.
- Both models expect a single, reasonably front-facing face per image.

---

## 9. Troubleshooting quick reference

| Problem | Fix |
|---|---|
| `No matching distribution found for tensorflow` | You're not using Python 3.11 in your venv |
| `ModuleNotFoundError: No module named 'cv2'` etc. | venv not activated — run `.venv\Scripts\Activate.ps1` |
| Kaggle: `No module named 'model_defs'` | Use `kaggle_train_all.py` instead — it's self-contained, no import needed |
| Kaggle: `filepath must end in .weights.h5` | Already fixed in `kaggle_train_all.py` — filenames use `.weights.h5` |
| GUI: "Model weights not found" | Put both `.weights.h5` files in the same folder as `gui_app.py` |


### Runtime note

The local inference builder uses `weights=None` for the MobileNetV2 initialisation because the supplied `hair_length_weights.weights.h5` already contains the trained backbone weights. This avoids an extra ImageNet model download at inference time.
