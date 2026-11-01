"""
demo.py - Run the system on an INPUT (image, folder of images, or video) without the GUI and save
clear OUTPUTS: annotated images/video, snapshots, CSV logs and printed tables.

Usage:
    python demo.py --input photo.jpg
    python demo.py --input my_images_folder/
    python demo.py --input demo_input.mp4 --truth demo_ground_truth.csv   # adds TRUE vs PREDICTED table
Outputs go to demo_output/ (change with --out).
"""
import argparse
import csv
import glob
import os
import shutil

import cv2

from app import (GREEN, RED, MAX_DISPLAY_W, SeniorDetector, Track, VisitLogger, crop_face)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".wmv")


def draw(img, box, label, senior):
    x, y, w, h = box
    color = RED if senior else GREEN
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = y - 8 if y - th - 12 > 0 else y + h + th + 8
    cv2.rectangle(img, (x, ty - th - 4), (x + tw + 4, ty + 4), color, -1)
    cv2.putText(img, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------- images
def run_images(engine, paths, out_dir):
    tid, table = 0, []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  skipped (cannot read): {p}")
            continue
        if img.shape[1] > MAX_DISPLAY_W:
            s = MAX_DISPLAY_W / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s)
        h, w = img.shape[:2]
        boxes = engine.detector.detect(img)
        if not boxes and max(h, w) <= 300:          # already a tight face crop (e.g. UTKFace)
            boxes = [(0, 0, w, h)]
        if not boxes:
            print(f"  {os.path.basename(p)}: no face detected")
            continue
        ages, gprobs = engine.predictor.predict([crop_face(img, b) for b in boxes])
        for box, a, g in zip(boxes, ages, gprobs):
            tid += 1
            gender = "Female" if g > 0.5 else "Male"
            senior = a > engine.threshold
            row = engine.logger.log(Track(tid, box), a, gender, senior)
            draw(img, box, f"{gender}, {int(round(a))}" + (" SENIOR CITIZEN" if senior else ""), senior)
            table.append((os.path.basename(p), row[0][-5:], row[1], row[2], row[3]))
        cv2.imwrite(os.path.join(out_dir, "annotated_" + os.path.basename(p)), img)

    print(f"\n{'Image':28}{'ID':8}{'Age':6}{'Gender':9}{'Senior'}")
    print("-" * 60)
    for r in table:
        print(f"{r[0][:27]:28}{r[1]:8}{r[2]:<6}{r[3]:9}{r[4]}")


# ----------------------------------------------------------------------------- video
def run_video(engine, path, out_dir):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    snaps_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snaps_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "annotated_" + os.path.splitext(os.path.basename(path))[0] + ".mp4")
    writer, n, snaps, preds = None, 0, 0, []

    print(f"\n{'Video time':12}{'Person_ID':24}{'Age':6}{'Gender':9}{'Senior'}")
    print("-" * 62)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if frame.shape[1] > MAX_DISPLAY_W:
            s = MAX_DISPLAY_W / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s)
        frame, rows = engine.process(frame)
        W = frame.shape[1]
        st = engine.stats
        cv2.putText(frame, f"People: {st['total']}  Seniors: {st['seniors']}  Male: {st['male']}  Female: {st['female']}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for r, (bx, by, bw, bh) in zip(rows, engine.last_boxes):
            preds.append({"t": n / fps, "id": r[0], "age": r[1], "gender": r[2], "senior": r[3],
                          "x": (bx + bw / 2) / W})
            print(f"{n / fps:8.1f} s  {r[0]:24}{r[1]:<6}{r[2]:9}{r[3]}")
        if rows and snaps < 12:
            snaps += 1
            cv2.imwrite(os.path.join(snaps_dir, f"snapshot_{snaps:02d}.png"), frame)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    cap.release()
    if writer:
        writer.release()

    with open(os.path.join(out_dir, "predictions_with_video_time.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Video_Time_s", "Person_ID", "Age", "Gender", "Senior_Citizen", "X_Center_Rel"])
        for p in preds:
            w.writerow([round(p["t"], 2), p["id"], p["age"], p["gender"], p["senior"], round(p["x"], 3)])
    print(f"\nAnnotated video : {out_path}\nSnapshots       : {snaps_dir}/")
    return preds


# ----------------------------------------------------------------------------- truth vs output
def compare(preds, truth_csv):
    with open(truth_csv) as f:
        truth = list(csv.DictReader(f))
    print("\n" + "=" * 78)
    print("INPUT (ground truth)  vs  OUTPUT (model prediction)")
    print("=" * 78)
    print(f"{'Scene':6}{'Pos':4}| {'TRUE':^19}| {'PREDICTED':^26}| Result")
    print("-" * 78)

    age_err, g_ok, s_ok, found, total, ts_found, ts_total, extra = [], 0, 0, 0, 0, 0, 0, 0
    for sc in sorted({int(t["Scene"]) for t in truth}):
        T = sorted((t for t in truth if int(t["Scene"]) == sc), key=lambda t: float(t["X_Center_Rel"]))
        start, end = float(T[0]["Start_s"]), float(T[0]["End_s"])
        P = [p for p in preds if start <= p["t"] <= end + 0.3]
        used = set()
        for t in T:
            total += 1
            t_age, t_sen = int(t["True_Age"]), t["True_Senior"] == "Yes"
            ts_total += int(t_sen)
            tx = float(t["X_Center_Rel"])
            cand = sorted((abs(p["x"] - tx), i) for i, p in enumerate(P) if i not in used)
            true_str = f"{t_age:>3} {t['True_Gender']:<6} {'SENIOR' if t_sen else 'no':<6}"
            if cand and cand[0][0] < 0.15:
                i = cand[0][1]
                used.add(i)
                p = P[i]
                found += 1
                p_age, p_sen = int(p["age"]), p["senior"] == "Yes"
                age_err.append(abs(p_age - t_age))
                g_ok += p["gender"] == t["True_Gender"]
                s_ok += p_sen == t_sen
                ts_found += int(t_sen and p_sen)
                res = "OK" if (p["gender"] == t["True_Gender"] and p_sen == t_sen) else "CHECK"
                pred_str = f"{p_age:>3} {p['gender']:<6} {'SENIOR' if p_sen else 'no':<6} ({p_age - t_age:+d})"
            else:
                pred_str, res = f"{'-':>3} {'not detected':<19}", "MISSED"
            print(f"{sc:<6}{t['Position_L_to_R']:<4}| {true_str} | {pred_str:<25} | {res}")
        extra += len(P) - len(used)

    print("-" * 78)
    print(f"Faces in input video        : {total}")
    print(f"Detected and logged         : {found}   (extra false detections: {extra})")
    if found:
        print(f"Age error (mean abs)        : {sum(age_err) / found:.1f} years")
        print(f"Gender correct              : {g_ok}/{found}  ({100 * g_ok / found:.0f}%)")
        print(f"Senior flag correct         : {s_ok}/{found}  ({100 * s_ok / found:.0f}%)")
    if ts_total:
        print(f"Real seniors caught         : {ts_found}/{ts_total}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image, folder of images, or video file")
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--truth", default=None, help="ground-truth CSV from make_demo_video.py (video only)")
    args = ap.parse_args()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    csv_path = os.path.join(args.out, "visitor_log_demo.csv")

    engine = SeniorDetector()
    engine.logger = VisitLogger(csv_path)       # demo has its own clean CSV
    engine.reset()

    src = args.input
    print(f"INPUT : {src}")
    if os.path.isdir(src):
        run_images(engine, sorted(p for p in glob.glob(os.path.join(src, "*"))
                                  if p.lower().endswith(IMG_EXT)), args.out)
    elif src.lower().endswith(VID_EXT):
        preds = run_video(engine, src, args.out)
        if args.truth and os.path.exists(args.truth):
            compare(preds, args.truth)
    else:
        run_images(engine, [src], args.out)

    with open(csv_path) as f:
        n = sum(1 for _ in f) - 1
    print(f"\nCSV log (final) : {csv_path}  ({n} records)")
    s = engine.stats
    print(f"Totals          : people={s['total']}  seniors={s['seniors']}  male={s['male']}  female={s['female']}")


if __name__ == "__main__":
    main()
