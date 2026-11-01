# Age and Emotion Detection through Voice (male voices only)

## Run
```bash
pip install -r requirements.txt
python app.py          # GUI
python demo.py         # console demo: built-in INPUT voices vs model OUTPUT
python demo.py --stress 100   # + test 100 brand-new random voices
```
No dataset download is needed. A trained model is included (`models/voice_models.joblib`). If it is missing or was
made with another scikit-learn version, the app builds the built-in dataset and trains again by itself (3-6 min, once).
Linux may need `sudo apt install python3-tk`.

## Rules
| Input voice | Output |
|---|---|
| Female | Rejected: **"Upload male voice."** |
| Male, age > 60 | **Senior Citizen** + age + emotion (Neutral / Happy / Sad / Angry / Fearful) |
| Male, age <= 60 | Age only |

## GUI
Upload Voice Note (wav), Record (6 s, needs `sounddevice`), or pick a **built-in sample** and click
*Analyze Built-in Sample*. Shows waveform, pitch contour, verdict card, emotion probability bars, history table;
every analysis is appended to `voice_log.csv`.

## How it works
1. `voice_core.py` contains a **built-in voice generator** (source-filter synthesis) that creates labelled male and
   female voices of age 18-85 with 5 emotions, using speech-science trends: pitch, formants, jitter, shimmer,
   breathiness, tremor, speaking rate, prosody; plus random phone/room noise.
2. 63 acoustic features are extracted (pitch statistics, harmonic clarity, CPP, tremor, loudness, rhythm,
   spectrum, 26 MFCC statistics, LPC formants) with NumPy/SciPy.
3. `train.py` trains three models: gender (Random Forest), age (Gradient Boosting regression, male voices),
   emotion (Histogram Gradient Boosting, male voices). Results go to `results/metrics.json` and `results/evaluation.png`.
4. `VoiceAnalyzer` applies the task rules.

## IMPORTANT: honest limits
* The built-in data is **synthetic**. The reported accuracy is measured on held-out synthetic voices and is
  **optimistic**; real human voices will be less accurate (voice-based age estimation is inherently about +/- 8-10 years).
* To improve real-world results, add your own labelled recordings and retrain:
  `python train.py --real-csv real_data/labels.csv` (columns: `file,gender,age,emotion`; age/emotion may be empty).
  Real files get 4x weight. Public sets such as RAVDESS (emotion) or Common Voice (age groups) can be converted this way.
* Do not use for real decisions about people; age and emotion estimates are approximate.
