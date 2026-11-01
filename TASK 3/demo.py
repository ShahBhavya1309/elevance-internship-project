"""
demo.py - Console demo with BUILT-IN input voices (no dataset, no GUI needed).

    python demo.py                 # runs the 8 built-in sample voices and prints INPUT (truth) vs OUTPUT
    python demo.py --stress 100    # also tests 100 brand-new random voices and prints summary accuracy
    python demo.py --file my_voice.wav     # analyse your own voice note
"""
import argparse
import csv
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import voice_core as vc


def describe(res):
    if not res["accepted"]:
        return f"REJECTED ({res['gender']} {res['gender_conf'] * 100:.0f}%) -> {res['message']}"
    if res["category"] == "Senior Citizen":
        return f"SENIOR CITIZEN | age {res['age']} | emotion {res['emotion']}"
    return f"Below 60 | age {res['age']} | (no emotion needed)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", type=int, default=0, help="also test N new random voices")
    ap.add_argument("--file", default=None, help="analyse one voice note (wav)")
    args = ap.parse_args()

    if not os.path.exists(vc.MODEL_PATH):
        print("No trained model found -> training on the built-in dataset (a few minutes, one time)...")
        import train
        train.train_all(progress=print)
    an = vc.VoiceAnalyzer()

    if args.file:
        print("INPUT :", args.file)
        print("OUTPUT:", describe(an.analyze_file(args.file)))
        return

    items = vc.ensure_samples()
    print("=" * 112)
    print("BUILT-IN INPUT VOICES  vs  MODEL OUTPUT")
    print("=" * 112)
    print(f"{'Input file':26}{'TRUE (hidden from model)':28}{'OUTPUT':54}Check")
    print("-" * 112)
    rows, ok_count = [], 0
    for it in items:
        res = an.analyze_file(it["path"])
        truth = f"{it['gender']}, {it['age']}, {it['emotion']}"
        if it["gender"] == "female":
            ok = not res["accepted"]
        else:
            ok = res["accepted"] and ((res["age"] > vc.SENIOR_AGE) == (it["age"] > vc.SENIOR_AGE)) and \
                 (it["age"] <= vc.SENIOR_AGE or res["emotion"] == it["emotion"])
        ok_count += ok
        print(f"{os.path.basename(it['path']):26}{truth:28}{describe(res):54}{'OK' if ok else 'CHECK'}")
        rows.append([os.path.basename(it["path"]), truth, describe(res), "OK" if ok else "CHECK"])
    print("-" * 112)
    print(f"Fully correct: {ok_count}/{len(items)}   (female rejected, senior flag right, emotion right for seniors)")
    with open("demo_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Input", "True", "Output", "Check"])
        w.writerows(rows)
    print("Saved: demo_results.csv  (input wav files are in samples/)")

    if args.stress:
        rng = np.random.default_rng(12345)
        g_ok = rej = n_f = 0
        age_err, sen_ok, n_m, emo_ok, n_sen = [], 0, 0, 0, 0
        for i in range(args.stress):
            g = "female" if rng.random() < 0.4 else "male"
            age, emo = int(rng.integers(18, 86)), vc.EMOTIONS[int(rng.integers(0, 5))]
            res = an.analyze(vc.synth_voice(g, age, emo, 7_000_000 + i), vc.SR)
            if g == "female":
                n_f += 1
                rej += not res["accepted"]
            else:
                n_m += 1
                if res["accepted"]:
                    age_err.append(abs(res["age"] - age))
                    sen_ok += (res["age"] > vc.SENIOR_AGE) == (age > vc.SENIOR_AGE)
                    if age > vc.SENIOR_AGE:
                        n_sen += 1
                        emo_ok += res["emotion"] == emo
        print("\n" + "=" * 60 + f"\nSTRESS TEST on {args.stress} brand-new random voices\n" + "=" * 60)
        print(f"Female voices rejected     : {rej}/{n_f}")
        print(f"Male voices accepted       : {len(age_err)}/{n_m}")
        if age_err:
            print(f"Age error (mean abs)       : {np.mean(age_err):.1f} years")
            print(f"Senior / not-senior correct: {sen_ok}/{len(age_err)}")
        if n_sen:
            print(f"Emotion correct (seniors)  : {emo_ok}/{n_sen}")


if __name__ == "__main__":
    main()
