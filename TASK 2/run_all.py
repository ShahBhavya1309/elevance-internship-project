"""
run_all.py - ONE COMMAND. Does everything and shows the full result.

    python run_all.py                 # quick mode (about 15-30 min on CPU the first time)
    python run_all.py --mode full     # best accuracy (use a GPU / Colab)
    python run_all.py --input my_video.mp4     # use your own video instead of the built-in sample input

STEP 1  Dataset  : UTKFace is downloaded automatically (about 1 GB, public, no login) the first time.
STEP 2  Training : trains MY multi-task CNN (age + gender). Skipped if models/age_gender_model.keras exists.
STEP 3  Input    : builds a sample video of multiple people from images the model has NEVER seen.
STEP 4  Run      : detects, tracks, predicts age/gender, flags seniors (>60), logs CSV.
STEP 5  Output   : TRUE vs PREDICTED table, annotated video, snapshots, CSV, metrics, report image.
Later runs skip steps 1-2 and finish in about a minute.
"""
import argparse
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter

DATA_DIR = "data/UTKFace"
MODEL_PATH = "models/age_gender_model.keras"
HF_REPO = "nu-delta/utkface"
HF_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/data"
FALLBACK_FILES = [f"data/train-0000{i}-of-00003.parquet" for i in range(3)]
OUT_DIR = "demo_output"


def banner(n, total, text):
    print("\n" + "=" * 78 + f"\nSTEP {n}/{total}: {text}\n" + "=" * 78, flush=True)


def check_deps():
    need = [("tensorflow", "tensorflow"), ("cv2", "opencv-python"), ("pyarrow", "pyarrow"),
            ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"), ("PIL", "Pillow"),
            ("pandas", "pandas"), ("numpy", "numpy")]
    missing = [pip for mod, pip in need if importlib.util.find_spec(mod) is None]
    if missing:
        sys.exit(f"Missing packages: {', '.join(missing)}\nRun:  pip install -r requirements.txt")


# ----------------------------------------------------------------------------- dataset
def gender01(v):
    s = str(v).strip().lower()
    if s in ("0", "male", "m", "man", "boy"):
        return 0
    if s in ("1", "female", "f", "woman", "girl"):
        return 1
    return None


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest + ".part", "wb") as f:
        total, done, last = int(r.headers.get("Content-Length", 0)), 0, 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 1:
                last = time.time()
                print(f"\r   {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    os.replace(dest + ".part", dest)
    print()


def list_parquet_files():
    try:
        req = urllib.request.Request(HF_TREE, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            files = sorted(i["path"] for i in json.load(r) if i["path"].endswith(".parquet"))
        if files:
            return files
    except Exception:
        pass
    return FALLBACK_FILES


def convert_parquet(path, start_index, genders_seen):
    """parquet (image bytes + age + gender)  ->  UTKFace-style files  [age]_[gender]_0_[n].jpg"""
    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    counter, skipped = start_index, 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256, columns=["image", "age", "gender"]):
        for row in batch.to_pylist():
            try:
                genders_seen[str(row["gender"])] += 1
                age, g = int(row["age"]), gender01(row["gender"])
                img = row["image"]
                data = img["bytes"] if isinstance(img, dict) else img
                if g is None or not (1 <= age <= 100) or not data:
                    skipped += 1
                    continue
                arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    skipped += 1
                    continue
                counter += 1
                cv2.imwrite(os.path.join(DATA_DIR, f"{age}_{g}_0_{counter:06d}.jpg"), arr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception:
                skipped += 1
    return counter, skipped


def prepare_dataset():
    n = len(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
    if n >= 1000:
        print(f"Dataset found: {n} images in {DATA_DIR}  (skipping download)")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("data/_tmp", exist_ok=True)
    files = list_parquet_files()
    print(f"Downloading UTKFace from Hugging Face ({len(files)} parts, about 1 GB). Please wait...")
    counter, skipped, genders_seen = 0, 0, Counter()
    for i, rel in enumerate(files, 1):
        tmp = os.path.join("data/_tmp", os.path.basename(rel))
        print(f" Part {i}/{len(files)}: {rel}")
        try:
            download(f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{rel}", tmp)
        except Exception as e:
            sys.exit(f"\nDownload failed: {e}\nManual option: download UTKFace from Kaggle "
                     f"(jangedoo/utkface-new) and put the .jpg files in {DATA_DIR}/, then run again.")
        print("   converting to image files ...")
        counter, sk = convert_parquet(tmp, counter, genders_seen)
        skipped += sk
        os.remove(tmp)
    shutil.rmtree("data/_tmp", ignore_errors=True)
    print(f"Prepared {counter} images ({skipped} skipped). Gender labels seen in source: {dict(genders_seen)}")
    if counter < 1000:
        sys.exit("Too few images were prepared. Please download UTKFace manually (see README).")


# ----------------------------------------------------------------------------- steps
def run(cmd):
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        sys.exit(f"\nCommand failed: python {' '.join(cmd)}")


def make_report(out_dir):
    """Tile the first snapshots into one picture you can paste in a report/presentation."""
    import cv2
    import numpy as np
    files = sorted(glob.glob(os.path.join(out_dir, "snapshots", "*.png")))[:6]
    if not files:
        return None
    imgs = [cv2.imread(f) for f in files]
    w = 480
    imgs = [cv2.resize(i, (w, int(i.shape[0] * w / i.shape[1]))) for i in imgs]
    h = max(i.shape[0] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for i in imgs]
    while len(imgs) % 2:
        imgs.append(np.full_like(imgs[0], 40))
    rows = [np.hstack(imgs[i:i + 2]) for i in range(0, len(imgs), 2)]
    report = np.vstack(rows)
    path = os.path.join(out_dir, "report.png")
    cv2.imwrite(path, report)
    return path


def show_windows(video_path, report_path):
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        print("\nPlaying the annotated OUTPUT video... (press Q to skip)")
        while True:
            ok, f = cap.read()
            if not ok:
                break
            cv2.imshow("Senior Citizen Identification - OUTPUT (Q to skip)", f)
            if cv2.waitKey(int(1000 / fps)) & 0xFF in (ord("q"), 27):
                break
        if report_path:
            cv2.imshow("Report - snapshots of detected people (press any key to close)", cv2.imread(report_path))
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("(Could not open a preview window on this machine. Open the files in demo_output/ manually.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--retrain", action="store_true", help="train again even if a model exists")
    ap.add_argument("--input", default=None, help="your own video / image / folder (skips the sample input)")
    ap.add_argument("--no-window", action="store_true", help="do not open preview windows")
    ap.add_argument("--gui", action="store_true", help="open the GUI (webcam / upload) at the end")
    args = ap.parse_args()

    check_deps()
    t0 = time.time()

    banner(1, 5, "Dataset (UTKFace)")
    prepare_dataset()

    banner(2, 5, f"Training my own CNN ({args.mode} mode)")
    if os.path.exists(MODEL_PATH) and not args.retrain:
        print(f"Trained model already exists: {MODEL_PATH}  (skipping; use --retrain to train again)")
    else:
        cmd = ["train.py", "--data", DATA_DIR]
        if args.mode == "quick":
            cmd += ["--limit", "6000", "--head-epochs", "3", "--tune-epochs", "5"]
        run(cmd)

    banner(3, 5, "Input")
    truth = None
    if args.input:
        src = args.input
        print(f"Using your input: {src}")
    else:
        src, truth = "demo_input.mp4", "demo_ground_truth.csv"
        run(["make_demo_video.py", "--images", DATA_DIR, "--out", src, "--truth", truth, "--scenes", "6"])

    banner(4, 5, "Detection + tracking + age/gender prediction + CSV logging")
    cmd = ["demo.py", "--input", src, "--out", OUT_DIR]
    if truth:
        cmd += ["--truth", truth]
    run(cmd)

    banner(5, 5, "Results")
    if os.path.exists("results/metrics.json"):
        m = json.load(open("results/metrics.json"))
        print("Model quality on the held-out TEST set (from training):")
        print(f"  Age MAE            : {m['age_mae_years']:.2f} years")
        print(f"  Gender accuracy    : {100 * m['gender_accuracy']:.1f}%")
        s = m["senior"]["threshold_60"]
        print(f"  Senior (>60)       : precision {s['precision']:.2f}   recall {s['recall']:.2f}   F1 {s['f1']:.2f}")
    report = make_report(OUT_DIR)
    print("\nFiles to show your teacher:")
    for p in [f"{OUT_DIR}/visitor_log_demo.csv", f"{OUT_DIR}/annotated_*.mp4", f"{OUT_DIR}/snapshots/",
              report, "results/training_curves.png", "results/confusion_matrix.png", "results/metrics.json",
              truth]:
        if p:
            print("  ", p)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min.")

    if not args.no_window:
        vids = glob.glob(os.path.join(OUT_DIR, "annotated_*.mp4"))
        if vids:
            show_windows(vids[0], report)
    if args.gui:
        run(["app.py"])


if __name__ == "__main__":
    main()
