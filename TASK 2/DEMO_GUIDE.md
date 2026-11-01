# Demo Guide: explaining the project with Input and Output

## 1. One-line explanation
"The system watches a video or webcam, finds every face, uses my own trained CNN to predict age and
gender, marks anyone above 60 as a **Senior Citizen**, and saves each visitor once in a CSV file."

## 2. Prepare and run the demo (one command)
```bash
python run_all.py
```
This does everything: downloads the dataset (first run), trains the model, creates the sample INPUT
(`demo_input.mp4` + `demo_ground_truth.csv`, using only unseen test images), runs the system, and shows the OUTPUT.
Manual steps if you prefer: `python train.py`, `python make_demo_video.py`, `python demo.py --input demo_input.mp4 --truth demo_ground_truth.csv`.

## 3. Run the demo

| Input type | Command | What you show |
|---|---|---|
| Video (best demo) | `python demo.py --input demo_input.mp4` | annotated video + snapshots + CSV |
| One photo | `python demo.py --input photo.jpg` | annotated photo |
| Folder of photos | `python demo.py --input my_photos/` | table of predictions |
| Live webcam | `python app.py` then **Start Webcam** | GUI with live counts |

### Output 1: console table (example format, your numbers will differ)
```
INPUT : demo_input.mp4
Video time  Person_ID             Age   Gender   Senior
------------------------------------------------------------
     0.8 s  20261030-101502-P0001 34    Male     No
     0.8 s  20261030-101502-P0002 67    Female   Yes
     0.9 s  20261030-101502-P0003 25    Female   No
```
### Output 2: annotated video / snapshots (`demo_output/`)
- Green box with label `#3 Male, 34` = normal visitor
- Red box with label `#2 Female, 67 SENIOR CITIZEN` = senior
- Yellow box `Analyzing...` = person seen for the first few frames, not yet confirmed
- Top-left counter: People / Seniors / Male / Female

### Output 3: CSV file (`demo_output/visitor_log_demo.csv`)
Format is shown in `sample_output/example_visitor_log.csv`:
`Person_ID, Age, Gender, Senior_Citizen, Date, Time_of_Visit`

### Compare input truth vs model output
Open `demo_ground_truth.csv` (true values per scene, with video seconds) next to the printed table
(match using Video time). Point out where the model is correct and where it is off by a few years.
Being honest about small errors, especially for people near 60, is a strength in a viva.

## 4. Explain the flow while the demo runs
1. **Input**: webcam frame or video frame
2. **Face detection**: YuNet / Haar finds faces (only locates them)
3. **Tracking**: each face gets an ID, so one person is not counted every frame
4. **My CNN**: MobileNetV2 trained on UTKFace, two outputs: age (regression) + gender (classification)
5. **Smoothing**: median age over several frames removes flicker
6. **Decision**: age > 60 -> Senior Citizen
7. **Output**: annotated video + one CSV row per person

## 5. Numbers to quote (after training)
Open `results/metrics.json` and show `results/training_curves.png` and `results/confusion_matrix.png`.
Quote YOUR real values: age MAE (years), gender accuracy, and senior precision / recall.

## 6. Likely questions from your sir
- **Why MobileNetV2?** Small and fast enough for real-time on a normal PC, and pretrained features help with limited data.
- **Why one model for age and gender?** Multi-task learning shares face features, so it is faster and often more accurate.
- **Why regression for age?** Age is ordered; regression gives an exact number to compare with 60.
- **Why tracking?** Without it, one person would create hundreds of CSV rows.
- **How did you handle few senior samples?** Oversampling of 55+ images, stratified split, and an optional calibrated threshold.
- **Why smooth over frames?** One frame can be blurry or badly lit; the median of many frames is stable.
- **Limitations?** Error of a few years near 60, side faces, masks, low light, dataset bias. Privacy: only age, gender and time are stored, no images.
- **Future work?** Better detector, larger dataset (IMDB-WIKI), re-identification across cameras, edge deployment.
