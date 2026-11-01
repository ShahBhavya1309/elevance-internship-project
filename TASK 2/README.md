# Senior Citizen Identification System

Detects multiple people in a webcam feed or video, estimates **age and gender** with a custom-trained
CNN, flags anyone **older than 60** as a Senior Citizen, and logs each visitor once to `visitor_log.csv`.


## QUICK START (one command)
```bash
pip install -r requirements.txt
python run_all.py
```
`run_all.py` downloads UTKFace automatically (about 1 GB, first run only), trains the model, builds a sample
multi-person input video from images the model never saw, runs the detector, and prints a TRUE vs PREDICTED
table, then plays the annotated output video. Use `--mode full` for best accuracy, `--input my_video.mp4`
for your own video, `--gui` to open the webcam GUI at the end. Later runs skip download and training.

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## 1. Get the dataset
Download **UTKFace** (aligned & cropped version, e.g. Kaggle "jangedoo/utkface-new") and put all
images in `data/UTKFace/` (files named like `25_0_2_2017....jpg`).

## 2. Train (use a GPU or Google Colab if possible)
```bash
python train.py --data data/UTKFace
python train.py --data data/UTKFace --limit 2000 --head-epochs 1 --tune-epochs 1   # quick smoke test
```
Produces `models/age_gender_model.keras`, `results/metrics.json`, `results/training_curves.png`,
`results/confusion_matrix.png`.

## 3. Run the app
```bash
python app.py
```
Buttons: **Start Webcam**, **Upload Video**, **Stop**, **Open CSV**, **Export Excel**.
Green box = normal visitor, red box = senior citizen, yellow = still analysing.

## CSV format
`Person_ID, Age, Gender, Senior_Citizen, Date, Time_of_Visit`

## How it works
| Stage | Method |
|---|---|
| Face location | YuNet (auto-downloaded) or OpenCV Haar fallback, locating faces only |
| Age + gender | Own MobileNetV2 multi-task CNN (age regression + gender sigmoid), trained on UTKFace |
| Imbalance | Seniors (55+) oversampled 3x in training; stratified split |
| Tracking | IoU tracker with persistent IDs (no duplicate logging) |
| Stability | Median age / mean gender probability over up to 30 frames; logged after 8 good frames |
| Speed | Blurry/small faces skipped; model re-run every 3rd frame once a person is confirmed |

## Tuning
- Seniors being missed? Set `USE_CALIBRATED_THRESHOLD = True` in `app.py` (threshold picked on validation data).
- Slow on CPU? Raise `INFER_EVERY`, lower `MAX_DISPLAY_W`.
- Wrong/duplicate IDs? Adjust `MAX_MISSED` and `IOU_THRESH`.

## Limitations and ethics
- Age estimation has a few years of error, so people near 60 can be misclassified.
- Accuracy drops with side faces, masks, strong shadows and low resolution.
- UTKFace has demographic biases; check performance on your own footage.
- The system stores no images, only age/gender/time. Display a notice in the store and follow local
  privacy rules (consent, data retention, access control).
