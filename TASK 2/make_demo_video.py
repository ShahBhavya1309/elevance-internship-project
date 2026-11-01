"""
make_demo_video.py - Build a SAMPLE INPUT video with several people per frame, plus a ground-truth CSV.

* Uses ONLY images the model never saw during training (models/test_files.txt), so the demo is honest.
* Each "scene" shows 2-3 faces side by side (some seniors, some not).
* Scenes are separated by blank frames (everybody leaves), so every scene = new visitors.

Usage:
    python make_demo_video.py --images data/UTKFace --out demo_input.mp4 --scenes 6
"""
import argparse
import csv
import glob
import os
import random

import cv2
import numpy as np


def parse(path):
    parts = os.path.basename(path).split("_")
    try:
        age, gender = int(parts[0]), int(parts[1])
        return (age, gender) if gender in (0, 1) else (None, None)
    except (ValueError, IndexError):
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/UTKFace")
    ap.add_argument("--test-list", default="models/test_files.txt")
    ap.add_argument("--out", default="demo_input.mp4")
    ap.add_argument("--truth", default="demo_ground_truth.csv")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--per-frame", type=int, default=3, help="faces visible at the same time")
    ap.add_argument("--hold", type=int, default=45, help="frames per scene")
    ap.add_argument("--gap", type=int, default=40, help="blank frames between scenes (must be > 30)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    labelled = []
    for f in glob.glob(os.path.join(args.images, "*.jpg")):
        age, gender = parse(f)
        if age is not None:
            labelled.append((f, age, gender))

    if args.test_list and os.path.exists(args.test_list):
        with open(args.test_list) as fh:
            allowed = {os.path.basename(l.strip()) for l in fh if l.strip()}
        pool = [x for x in labelled if os.path.basename(x[0]) in allowed]
        if any(x[1] > 60 for x in pool) and any(18 <= x[1] <= 55 for x in pool):
            labelled = pool
            print(f"Using {len(pool)} held-out TEST images (never used for training).")
        else:
            print("Warning: test split has too few seniors/adults; using all images.")
    else:
        print("Warning: models/test_files.txt not found; images may overlap with training data.")

    seniors = [x for x in labelled if x[1] > 60]
    others = [x for x in labelled if 18 <= x[1] <= 55]
    if not seniors or not others:
        raise SystemExit("Need UTKFace-style file names ([age]_[gender]_...jpg) with seniors and adults.")

    rng = random.Random(args.seed)
    tile, pad, n = 280, 30, args.per_frame
    W, H = n * tile + (n + 1) * pad, tile + 2 * pad
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    blank = np.full((H, W, 3), 60, np.uint8)

    truth, frame_no = [], 0
    for scene in range(1, args.scenes + 1):
        n_sen = min(rng.choice([0, 1, 1, 2]), n, len(seniors))
        group = rng.sample(seniors, n_sen) + rng.sample(others, n - n_sen)
        rng.shuffle(group)
        imgs = [cv2.resize(cv2.imread(g[0]), (tile, tile)) for g in group]

        start = frame_no / args.fps
        for _ in range(args.hold):
            canvas = blank.copy()
            for i, im in enumerate(imgs):
                x = pad + i * (tile + pad) + rng.randint(-2, 2)     # tiny jitter = natural movement
                y = pad + rng.randint(-2, 2)
                canvas[y:y + tile, x:x + tile] = im
            writer.write(canvas)
            frame_no += 1
        end = frame_no / args.fps

        for pos, (f, age, gender) in enumerate(group, 1):
            x_rel = (pad + (pos - 1) * (tile + pad) + tile / 2) / W
            truth.append([scene, round(start, 2), round(end, 2), pos, round(x_rel, 3), age,
                          "Male" if gender == 0 else "Female", "Yes" if age > 60 else "No",
                          os.path.basename(f)])
        for _ in range(args.gap):                                   # everybody leaves
            writer.write(blank)
            frame_no += 1
    writer.release()

    with open(args.truth, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Scene", "Start_s", "End_s", "Position_L_to_R", "X_Center_Rel", "True_Age",
                    "True_Gender", "True_Senior", "File"])
        w.writerows(truth)
    print(f"Saved input video  : {args.out}  ({W}x{H}, {frame_no / args.fps:.0f}s, {len(truth)} faces)")
    print(f"Saved ground truth : {args.truth}")


if __name__ == "__main__":
    main()
