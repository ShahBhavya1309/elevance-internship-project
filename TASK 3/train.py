"""
train.py - Build the built-in voice dataset and train MY three models (no download needed).

    python train.py                    # about 4000 built-in voices (a few minutes)
    python train.py --n 8000           # more data
    python train.py --real-csv real_data/labels.csv   # optionally add your own real recordings

Models (scikit-learn):
    gender  : RandomForestClassifier             male / female   (gatekeeper: female -> "Upload male voice.")
    age     : GradientBoostingRegressor          trained on MALE voices only (age in years)
    emotion : HistGradientBoostingClassifier     Neutral / Happy / Sad / Angry / Fearful (male voices)

Optional real data CSV columns:  file,gender,age,emotion   (gender = male/female; age and emotion may be empty)
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score)
from sklearn.model_selection import train_test_split

import voice_core as vc

REAL_WEIGHT = 4.0       # a real recording counts more than a built-in one


def load_real(csv_path, log):
    import csv
    X, gender, age, emo = [], [], [], []
    base = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            path = row["file"] if os.path.isabs(row["file"]) else os.path.join(base, row["file"])
            try:
                y, sr = vc.load_audio(path)
                feats, _ = vc.analyze_signal(vc.to_mono16k(y, sr))
            except Exception as e:
                log(f"  skipped {row['file']}: {e}")
                continue
            X.append(feats)
            gender.append(1 if row["gender"].strip().lower().startswith("f") else 0)
            age.append(int(row["age"]) if row.get("age", "").strip() else -1)
            e = row.get("emotion", "").strip().capitalize()
            emo.append(vc.EMOTIONS.index(e) if e in vc.EMOTIONS else -1)
    log(f"Loaded {len(X)} real recordings from {csv_path}")
    return np.array(X), np.array(gender), np.array(age), np.array(emo)


def train_all(n=4000, seed=0, real_csv=None, out_dir="models", results_dir="results",
              progress=print, workers=None):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    workers = workers or min(4, os.cpu_count() or 1)

    X, g, a, e = vc.build_dataset(n, seed, workers, progress)
    w = np.ones(len(X))
    if real_csv:
        Xr, gr, ar, er = load_real(real_csv, progress)
        if len(Xr):
            X, g, a, e = np.vstack([X, Xr]), np.r_[g, gr], np.r_[a, ar], np.r_[e, er]
            w = np.r_[w, np.full(len(Xr), REAL_WEIGHT)]
    progress(f"Dataset ready: {len(X)} voices x {X.shape[1]} features. Training models...")

    idx = np.arange(len(X))
    strat = g * 10 + np.where(e >= 0, e, 0)
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)

    # ---------------- gender (all voices)
    gm = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    gm.fit(X[tr], g[tr], sample_weight=w[tr])
    gp = gm.predict(X[te])
    progress("Gender model trained.")

    # ---------------- age (male voices with a known age)
    m_tr, m_te = tr[(g[tr] == 0) & (a[tr] > 0)], te[(g[te] == 0) & (a[te] > 0)]
    am = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
                                   random_state=seed)
    am.fit(X[m_tr], a[m_tr], sample_weight=w[m_tr])
    ap = am.predict(X[m_te])
    progress("Age model trained.")

    # ---------------- emotion (male voices with a known emotion)
    e_tr, e_te = tr[(g[tr] == 0) & (e[tr] >= 0)], te[(g[te] == 0) & (e[te] >= 0)]
    em = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, random_state=seed)
    em.fit(X[e_tr], e[e_tr], sample_weight=w[e_tr])
    ep = em.predict(X[e_te])
    progress("Emotion model trained.")

    # ---------------- evaluation on the held-out 20%
    sen_true, sen_pred = a[m_te] > vc.SENIOR_AGE, ap > vc.SENIOR_AGE
    sen_e = e_te[a[e_te] > vc.SENIOR_AGE]
    metrics = {
        "samples_total": int(len(X)), "samples_test": int(len(te)),
        "gender_accuracy": float(accuracy_score(g[te], gp)),
        "gender_confusion_matrix": confusion_matrix(g[te], gp).tolist(),
        "age_mae_male": float(mean_absolute_error(a[m_te], ap)),
        "age_within_5_years": float(np.mean(np.abs(a[m_te] - ap) <= 5)),
        "senior_precision": float(precision_score(sen_true, sen_pred, zero_division=0)),
        "senior_recall": float(recall_score(sen_true, sen_pred, zero_division=0)),
        "senior_f1": float(f1_score(sen_true, sen_pred, zero_division=0)),
        "emotion_accuracy_all_males": float(accuracy_score(e[e_te], ep)),
        "emotion_accuracy_seniors": float(accuracy_score(e[sen_e], em.predict(X[sen_e]))) if len(sen_e) else None,
        "emotion_confusion_matrix": confusion_matrix(e[e_te], ep).tolist(),
        "data_source": "built-in synthetic voices" + (f" + real recordings ({real_csv})" if real_csv else ""),
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    progress(f"Gender acc {metrics['gender_accuracy']:.3f} | Age MAE {metrics['age_mae_male']:.2f} yrs | "
             f"Senior F1 {metrics['senior_f1']:.3f} | Emotion acc {metrics['emotion_accuracy_all_males']:.3f}")

    try:
        make_plots(results_dir, g[te], gp, a[m_te], ap, e[e_te], ep)
    except Exception as ex:                                  # plots are optional
        progress(f"(plots skipped: {ex})")

    import joblib
    joblib.dump({"gender": gm, "age": am, "emotion": em, "emotions": vc.EMOTIONS,
                 "feature_names": vc.FEATURE_NAMES, "metrics": metrics,
                 "sklearn_version": __import__("sklearn").__version__}, os.path.join(out_dir, "voice_models.joblib"))
    progress("Models saved to models/voice_models.joblib")
    print("\nEmotion report (male test voices):\n",
          classification_report(e[e_te], ep, target_names=vc.EMOTIONS, digits=3))
    return metrics


def make_plots(results_dir, g_true, g_pred, a_true, a_pred, e_true, e_pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def cm_plot(ax, y, p, labels, title):
        cm = confusion_matrix(y, p)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=30)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    cm_plot(ax[0], g_true, g_pred, ["Male", "Female"], "Gender")
    ax[1].scatter(a_true, a_pred, s=6, alpha=0.4)
    ax[1].plot([18, 85], [18, 85], "r--")
    ax[1].axhline(vc.SENIOR_AGE, color="gray", ls=":")
    ax[1].axvline(vc.SENIOR_AGE, color="gray", ls=":")
    ax[1].set_title(f"Age (male): MAE {mean_absolute_error(a_true, a_pred):.1f} yrs")
    ax[1].set_xlabel("True age")
    ax[1].set_ylabel("Predicted age")
    cm_plot(ax[2], e_true, e_pred, vc.EMOTIONS, "Emotion (male voices)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "evaluation.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="number of built-in voices to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-csv", default=None, help="optional CSV with your own labelled recordings")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    train_all(args.n, args.seed, args.real_csv, workers=args.workers)
