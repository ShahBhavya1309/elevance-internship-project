"""
voice_core.py - everything the project needs, with NO external dataset:

  1. BUILT-IN DATA GENERATOR  : a source-filter voice synthesizer that creates labelled male/female voices of
                                any age (18-85) and emotion. Parameters follow published speech-science trends
                                (pitch, formants, jitter, shimmer, breathiness, tremor, speaking rate, prosody).
  2. FEATURE EXTRACTION       : 63 acoustic features (pitch, HNR, tremor, energy, rhythm, spectrum, MFCC, formants)
                                written with NumPy/SciPy only.
  3. AUDIO LOADING            : wav (and mp3/ogg/flac when soundfile or pydub+ffmpeg is installed).
  4. VoiceAnalyzer            : the decision logic of the task
        female voice  -> rejected: "Upload male voice."
        male, age > 60  -> Senior Citizen + age + emotion
        male, age <= 60 -> age only
"""
import math
import os

import numpy as np
from scipy import signal
from scipy.fft import dct
from scipy.io import wavfile
from scipy.linalg import solve_toeplitz
from scipy.ndimage import uniform_filter1d

SR = 16000
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fearful"]
SENIOR_AGE = 60
MODEL_PATH = os.path.join("models", "voice_models.joblib")

# =====================================================================================================
# 1. BUILT-IN VOICE GENERATOR (source-filter model)
# =====================================================================================================
# Peterson & Barney style adult-male vowel formants (F1, F2, F3 in Hz)
VOWELS = {"a": (730, 1090, 2440), "e": (530, 1840, 2480), "i": (270, 2290, 3010),
          "o": (570, 840, 2410), "u": (300, 870, 2240)}
VOWEL_KEYS = list(VOWELS)

# emotion -> prosody settings (relative to neutral speech)
EMO = {
    "Neutral": dict(f0=1.00, var=1.4, db=0.0, rate=1.00, tilt=0.55, noise=0.000, pause=0.10, jit=0.000, trem=0.000),
    "Happy":   dict(f0=1.18, var=3.0, db=3.0, rate=1.10, tilt=0.45, noise=0.000, pause=0.08, jit=0.000, trem=0.000),
    "Sad":     dict(f0=0.93, var=0.8, db=-5.0, rate=0.75, tilt=0.70, noise=0.040, pause=0.25, jit=0.001, trem=0.002),
    "Angry":   dict(f0=1.12, var=2.8, db=7.0, rate=1.15, tilt=0.25, noise=-0.010, pause=0.05, jit=0.000, trem=0.000),
    "Fearful": dict(f0=1.25, var=2.4, db=-1.0, rate=1.25, tilt=0.50, noise=0.020, pause=0.12, jit=0.004, trem=0.006),
}


def make_params(rng, gender, age, emotion):
    """Turn (gender, age, emotion) into physical voice parameters, with natural person-to-person variation."""
    e = EMO[emotion]
    k = float(np.clip(rng.normal(1.0, 0.3), 0.4, 1.6))            # how strongly the emotion is expressed
    eff = float(np.clip(age + rng.normal(0, 5), 16, 95))           # "voice age" differs a little from real age

    if gender == "male":
        base = float(np.clip(rng.normal(118, 14), 85, 160))
        base += -0.15 * (min(max(eff, 25), 55) - 25) + 0.55 * max(0.0, eff - 60)   # dips in mid-age, rises after 60
        fscale = rng.normal(1.00, 0.05)
    else:
        base = float(np.clip(rng.normal(208, 22), 150, 290))
        base -= 0.45 * max(0.0, eff - 30)
        fscale = rng.normal(1.18, 0.05)                              # shorter vocal tract -> higher formants
    fscale *= 1 - 0.0007 * (eff - 20)

    return dict(
        dur=rng.uniform(3.0, 5.0),
        f0=float(np.clip(base * (1 + (e["f0"] - 1) * k), 70, 380)),
        var=max(0.4, (1.4 + (e["var"] - 1.4) * k) * rng.uniform(0.8, 1.2)),           # pitch variation (semitones)
        rate=max(1.8, 4.3 * (1 - 0.0025 * max(0.0, eff - 25)) * rng.normal(1, 0.08) * (1 + (e["rate"] - 1) * k)),
        pause=e["pause"] + 0.002 * max(0.0, eff - 30),
        tilt=float(np.clip(0.55 + (e["tilt"] - 0.55) * k + rng.normal(0, 0.05), 0.10, 0.85)),
        jit=(0.004 + 0.00014 * max(0.0, eff - 25) + e["jit"] * k) * rng.uniform(0.7, 1.3),
        shimmer=(0.04 + 0.0012 * max(0.0, eff - 30)) * rng.uniform(0.7, 1.3),
        tremor=(0.002 + 0.0006 * max(0.0, eff - 55) + e["trem"] * k) * rng.uniform(0.7, 1.3),
        tremor_hz=rng.uniform(4.5, 6.5),
        noise=max(0.01, 0.03 + 0.0012 * max(0.0, eff - 40) + e["noise"] * k),
        bw=1 + 0.004 * max(0.0, eff - 30),
        fscale=float(fscale),
        level_db=e["db"] * k,
    )


def _resonator(f, bw, sr):
    r = math.exp(-math.pi * bw / sr)
    c = 2 * r * math.cos(2 * math.pi * f / sr)
    return [1 - c + r * r], [1.0, -c, r * r]


def synthesize(p, rng, sr=SR):
    n = int(p["dur"] * sr)
    t = np.arange(n) / sr

    # ---- syllable timeline (speech-like rhythm with pauses)
    cur, syl = rng.uniform(0.10, 0.30), []
    while True:
        d = (1.0 / p["rate"]) * rng.uniform(0.75, 1.25)
        if cur + d > p["dur"] - 0.10:
            break
        syl.append((cur, cur + d, VOWEL_KEYS[rng.integers(len(VOWEL_KEYS))]))
        cur += d
        if rng.random() < p["pause"]:
            cur += rng.uniform(0.12, 0.35)
    if len(syl) < 3:
        syl = [(0.2, 0.6, "a"), (0.7, 1.1, "e"), (1.2, 1.6, "o")]
    centers = np.array([(s + e) / 2 for s, e, _ in syl])

    # ---- amplitude envelope: raised-cosine syllables, silence in pauses
    env = np.zeros(n)
    for s, e, _ in syl:
        a, b = int(s * sr), min(n, int(e * sr))
        u = np.linspace(0, 1, b - a)
        env[a:b] = np.maximum(env[a:b], (0.15 + 0.85 * np.sin(np.pi * u) ** 2) * rng.uniform(0.9, 1.1))

    # ---- pitch contour (semitones): AR(1) random intonation + declination + tremor
    st = np.zeros(len(centers))
    for i in range(1, len(st)):
        st[i] = 0.5 * st[i - 1] + rng.normal(0, p["var"] * math.sqrt(0.75))
    st = st - 2.0 * centers / p["dur"]
    f0c = p["f0"] * 2 ** (np.interp(t, centers, st) / 12)
    f0c = uniform_filter1d(f0c, int(0.06 * sr), mode="nearest")
    f0c *= 1 + p["tremor"] * np.sin(2 * math.pi * p["tremor_hz"] * t + rng.uniform(0, 6.28))

    # ---- jitter (cycle-to-cycle pitch noise) and shimmer (cycle-to-cycle loudness noise)
    cyc = (np.cumsum(f0c) / sr).astype(int)
    f0j = f0c * (1 + rng.normal(0, p["jit"], cyc[-1] + 2)[cyc])
    phase = np.cumsum(f0j) / sr
    amp = 1 + rng.normal(0, p["shimmer"], int(phase[-1]) + 2)[phase.astype(int)]

    # ---- glottal source (Rosenberg pulse) -> derivative (lip radiation)
    fr = phase % 1.0
    g = np.zeros(n)
    m1 = fr < 0.40
    g[m1] = 0.5 * (1 - np.cos(np.pi * fr[m1] / 0.40))
    m2 = (fr >= 0.40) & (fr < 0.56)
    g[m2] = np.cos(np.pi * (fr[m2] - 0.40) / 0.32)
    src = np.diff(g, prepend=0.0) * amp
    pole = p["tilt"]
    src = signal.lfilter([1 - pole], [1, -pole], src)                # spectral tilt (angry = bright, sad = dark)
    src /= np.sqrt(np.mean(src ** 2)) + 1e-9
    src += rng.normal(size=n) * p["noise"] * 2.5 * (0.3 + g)          # breathiness

    # ---- vocal tract: 4 formant resonators, formants glide from vowel to vowel
    fs = p["fscale"]
    tracks = []
    for k in range(3):
        vals = np.array([VOWELS[v][k] for _, _, v in syl]) * fs * rng.normal(1, 0.03, len(syl))
        tracks.append(np.interp(t, centers, vals))
    tracks.append(np.full(n, 3500.0 * fs))
    bws = np.array([70, 100, 140, 220]) * p["bw"]
    out, zis, blk = np.empty(n), [np.zeros(2) for _ in range(4)], 160
    for s in range(0, n, blk):
        e = min(n, s + blk)
        seg = src[s:e]
        for k in range(4):
            b, a = _resonator(min(tracks[k][(s + e) // 2], 7500), bws[k], sr)
            seg, zis[k] = signal.lfilter(b, a, seg, zi=zis[k])
        out[s:e] = seg
    y = out * env
    y /= np.max(np.abs(y)) + 1e-9
    return y


def add_recording_conditions(y, rng, level_db=0.0, sr=SR):
    """Phone / room effects so the model does not only see 'perfect' audio."""
    if rng.random() < 0.5:                                           # band limiting
        sos = signal.butter(4, rng.uniform(3400, 7000), "low", fs=sr, output="sos")
        y = signal.sosfilt(sos, y)
    if rng.random() < 0.4:
        sos = signal.butter(2, rng.uniform(60, 120), "high", fs=sr, output="sos")
        y = signal.sosfilt(sos, y)
    if rng.random() < 0.3:                                           # small room reverb
        tau = rng.uniform(0.04, 0.12)
        ir = np.exp(-np.arange(int(0.2 * sr)) / (tau * sr)) * rng.normal(size=int(0.2 * sr))
        ir[0] = 1.0
        y = 0.8 * y + 0.2 * signal.fftconvolve(y, ir / np.sqrt(np.sum(ir ** 2)))[:len(y)]
    snr = rng.uniform(18, 38) + level_db                             # louder speech -> better SNR
    noise = rng.normal(size=len(y))
    if rng.random() < 0.5:
        noise = signal.lfilter([1], [1, -0.95], noise)
    noise *= np.sqrt(np.mean(y ** 2) / (10 ** (snr / 10))) / (np.sqrt(np.mean(noise ** 2)) + 1e-9)
    y = y + noise
    return (y / (np.max(np.abs(y)) + 1e-9) * rng.uniform(0.3, 0.95)).astype(np.float32)


def synth_voice(gender, age, emotion, seed):
    rng = np.random.default_rng(seed)
    p = make_params(rng, gender, age, emotion)
    return add_recording_conditions(synthesize(p, rng), rng, p["level_db"])


# =====================================================================================================
# 2. AUDIO LOADING / PRE-PROCESSING
# =====================================================================================================
def _to_float(y):
    if y.dtype == np.uint8:
        return (y.astype(np.float32) - 128) / 128
    if np.issubdtype(y.dtype, np.integer):
        return y.astype(np.float32) / float(np.iinfo(y.dtype).max + 1)
    return y.astype(np.float32)


def load_audio(path):
    """Returns (float32 samples, sample_rate). WAV works everywhere; others need soundfile or pydub+ffmpeg."""
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32")
        return y, sr
    except Exception:
        pass
    if path.lower().endswith(".wav"):
        try:
            sr, y = wavfile.read(path)
            return _to_float(y), sr
        except Exception:
            pass
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        y = np.array(seg.get_array_of_samples(), dtype=np.float32) / float(1 << (8 * seg.sample_width - 1))
        if seg.channels > 1:
            y = y.reshape(-1, seg.channels)
        return y, seg.frame_rate
    except Exception:
        raise ValueError("Cannot read this audio file. Please upload a WAV file "
                         "(for mp3/m4a install ffmpeg + 'pip install pydub').")


def save_wav(path, y, sr=SR):
    wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def to_mono16k(y, sr):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if int(sr) != SR:
        g = math.gcd(int(sr), SR)
        y = signal.resample_poly(y, SR // g, int(sr) // g)
    return (y - y.mean()).astype(np.float32)


def trim_silence(y):
    hop = int(0.02 * SR)
    n = len(y) // hop
    if n < 10 or np.max(np.abs(y)) < 1e-4:
        raise ValueError("The recording is empty or silent.")
    db = 10 * np.log10((y[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    idx = np.where(db > db.max() - 35)[0]
    s, e = max(0, idx[0] - 3) * hop, min(len(y), (idx[-1] + 4) * hop)
    y = y[s:e]
    if len(y) < 1.2 * SR:
        raise ValueError("The voice note is too short. Please speak for at least 2 seconds.")
    return y / (np.max(np.abs(y)) + 1e-9) * 0.95


# =====================================================================================================
# 3. FEATURE EXTRACTION (63 features)
# =====================================================================================================
FRAME, HOP, LMIN, LMAX = 512, 160, 40, 228          # 32 ms frames, 10 ms hop, pitch 70-400 Hz
_HANN = np.hanning(FRAME)


def _mel_fb(n_filt=26):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    bins = np.floor((FRAME + 1) * inv(np.linspace(0, mel(SR / 2), n_filt + 2)) / SR).astype(int)
    fb = np.zeros((n_filt, FRAME // 2 + 1))
    for i in range(1, n_filt + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        fb[i - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[i - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


_FB = _mel_fb()
_FREQ = np.fft.rfftfreq(FRAME, 1 / SR)
_WAC = np.fft.irfft(np.abs(np.fft.rfft(_HANN, 1024)) ** 2)[:FRAME // 2 + 1]
_WAC = np.maximum(_WAC / _WAC[0], 0.05)

FEATURE_NAMES = (
    ["f0_mean", "f0_std", "f0_median", "f0_p5", "f0_p95", "f0_range", "f0_semitone_std", "f0_slope",
     "voiced_ratio", "f0_jitter_proxy", "hnr_mean", "hnr_std", "tremor_ratio", "tremor_amp",
     "db_mean", "db_std", "db_range", "pause_ratio", "syllable_rate", "rhythm_strength",
     "centroid_mean", "centroid_std", "rolloff", "flatness", "spectral_tilt", "hf_lf_ratio", "zcr",
     "harm_low", "harm_mid", "harm_high", "cpp_mean", "cpp_std", "syllable_peak_rate"]
    + [f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)]
    + ["formant_f1", "formant_f2", "formant_f3", "formant_dispersion"])


def _formants(y_pe, voiced_idx, order=16):
    """Median F1-F3 from LPC roots of up to 40 voiced frames."""
    out = []
    for i in np.linspace(0, len(voiced_idx) - 1, min(40, len(voiced_idx))).astype(int):
        start = voiced_idx[i] * HOP
        seg = y_pe[start:start + FRAME]
        if len(seg) < FRAME:
            continue
        seg = seg * _HANN
        r = np.correlate(seg, seg, "full")[FRAME - 1: FRAME + order]
        if r[0] < 1e-9:
            continue
        r[0] *= 1.0001
        try:
            a = solve_toeplitz(r[:order], r[1:order + 1])
        except Exception:
            continue
        roots = np.roots(np.r_[1, -a])
        roots = roots[np.imag(roots) > 0.01]
        fr = np.angle(roots) * SR / (2 * math.pi)
        bw = -np.log(np.abs(roots) + 1e-12) * SR / math.pi
        ok = np.sort(fr[(fr > 200) & (fr < 4500) & (bw < 700)])
        if len(ok) >= 3:
            out.append(ok[:3])
    if not out:
        return np.array([500.0, 1500.0, 2500.0])
    return np.median(np.array(out), axis=0)


def _voice_quality(y, f0_all, voiced_idx):
    """Harmonic clarity per band (breathiness / jitter / shimmer smear the high harmonics) and
    cepstral peak prominence (CPP), measured on 64 ms windows so that low male pitch is resolved."""
    nfft, win = 2048, np.hanning(1024)
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    harm, cpp = [], []
    for i in np.linspace(0, len(voiced_idx) - 1, min(60, len(voiced_idx))).astype(int):
        c = voiced_idx[i] * HOP + FRAME // 2
        seg = y[max(0, c - 512): c + 512]
        if len(seg) < 1024:
            continue
        P = np.abs(np.fft.rfft(seg * win, nfft)) ** 2 + 1e-12
        f0 = f0_all[voiced_idx[i]]
        ph = (freqs / f0) % 1.0
        d = np.minimum(ph, 1 - ph)
        row = []
        for lo, hi in [(100, 1000), (1000, 2000), (2000, 3500)]:
            b = (freqs >= lo) & (freqs < hi)
            row.append(10 * np.log10((P[b & (d < 0.12)].mean() + 1e-12) / (P[b & (d > 0.35)].mean() + 1e-12)))
        harm.append(row)
        ceps = np.fft.irfft(10 * np.log10(P))[:LMAX + 1]
        q = np.arange(LMIN, LMAX + 1)
        line = np.polyfit(q, ceps[LMIN:LMAX + 1], 1)
        k = ceps[LMIN:LMAX + 1].argmax()
        cpp.append(ceps[LMIN + k] - np.polyval(line, LMIN + k))
    if not harm:
        return np.zeros(3), 0.0, 0.0
    return np.mean(harm, 0), float(np.mean(cpp)), float(np.std(cpp))


def analyze_signal(y):
    """y: float32 mono 16 kHz (already loaded). Returns (features[63], info dict for the GUI)."""
    y = trim_silence(y)
    pad = np.pad(y, (0, max(0, FRAME - len(y))))
    fw = np.lib.stride_tricks.sliding_window_view(pad, FRAME)[::HOP] * _HANN
    nf = len(fw)

    rms = np.sqrt((fw ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)
    active = db > db.max() - 30

    # --- pitch by normalised autocorrelation
    ac = np.fft.irfft(np.abs(np.fft.rfft(fw, n=1024, axis=1)) ** 2, axis=1)[:, :FRAME // 2 + 1]
    ac = np.clip(ac / (ac[:, :1] + 1e-12) / _WAC, -1, 1)
    r = ac[:, LMIN - 1: LMAX + 2]
    core, left, right = r[:, 1:-1], r[:, :-2], r[:, 2:]
    cand = np.where((core >= left) & (core > right), core, -1.0)
    first = (cand >= 0.9 * cand.max(1, keepdims=True)).argmax(1)
    ar = np.arange(nf)
    peak = cand[ar, first]
    y0, y1, y2 = r[ar, first], r[ar, first + 1], r[ar, first + 2]
    den = y0 - 2 * y1 + y2
    delta = np.clip(np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0), -1, 1)
    f0_all = SR / (LMIN + first + delta)
    voiced = active & (peak > 0.5)
    if voiced.sum() < 15:
        raise ValueError("No clear voice detected. Please record a clear voice note without much noise.")
    f0 = np.where(voiced, f0_all, np.nan)
    fv = f0[voiced]
    st = 12 * np.log2(f0 / np.median(fv))
    vt = np.where(voiced)[0]

    f = {}
    f["f0_mean"], f["f0_std"], f["f0_median"] = fv.mean(), fv.std(), np.median(fv)
    f["f0_p5"], f["f0_p95"] = np.percentile(fv, 5), np.percentile(fv, 95)
    f["f0_range"] = f["f0_p95"] - f["f0_p5"]
    f["f0_semitone_std"] = np.nanstd(st)
    f["f0_slope"] = np.polyfit(vt * HOP / SR, st[vt], 1)[0]
    f["voiced_ratio"] = voiced.sum() / max(active.sum(), 1)
    both = voiced[1:] & voiced[:-1]
    f["f0_jitter_proxy"] = np.abs(np.diff(st))[both].mean() if both.any() else 0.0
    hnr = 10 * np.log10(np.clip(peak[voiced], 0.01, 0.99) / (1 - np.clip(peak[voiced], 0.01, 0.99)))
    f["hnr_mean"], f["hnr_std"] = hnr.mean(), hnr.std()

    # --- vocal tremor: 4-8 Hz modulation of the pitch contour
    f["tremor_ratio"] = f["tremor_amp"] = 0.0
    if vt[-1] - vt[0] > 60:
        idx = np.arange(vt[0], vt[-1] + 1)
        c = np.interp(idx, vt, st[vt])
        c = (c - uniform_filter1d(c, 31, mode="nearest")) * np.hanning(len(c))
        A = np.abs(np.fft.rfft(c, 1024)) * 2 / np.hanning(len(c)).sum()
        fq = np.fft.rfftfreq(1024, 0.01)
        band, total = (fq >= 4) & (fq <= 8), (fq >= 0.7) & (fq <= 15)
        f["tremor_ratio"] = (A[band] ** 2).sum() / ((A[total] ** 2).sum() + 1e-9)
        f["tremor_amp"] = A[band].max()

    # --- loudness and rhythm
    f["db_mean"], f["db_std"] = db[active].mean() - db.max(), db[active].std()
    f["db_range"] = np.percentile(db, 95) - np.percentile(db, 10)
    f["pause_ratio"] = float((db < db.max() - 25).mean())
    e = (rms - rms.mean()) * np.hanning(nf)
    spec = np.abs(np.fft.rfft(e, 1024))
    fq = np.fft.rfftfreq(1024, 0.01)
    band = (fq >= 1.5) & (fq <= 8)
    f["syllable_rate"] = fq[band][spec[band].argmax()]
    f["rhythm_strength"] = spec[band].max() / (spec[band].sum() + 1e-9)

    # --- spectrum of speech frames
    P = np.abs(np.fft.rfft(fw[active], axis=1)) ** 2 + 1e-12
    cen = (P * _FREQ).sum(1) / P.sum(1)
    f["centroid_mean"], f["centroid_std"] = cen.mean(), cen.std()
    f["rolloff"] = _FREQ[(np.cumsum(P, 1) >= 0.85 * P.sum(1, keepdims=True)).argmax(1)].mean()
    f["flatness"] = np.log10(np.exp(np.log(P).mean(1)) / P.mean(1)).mean()
    L = 10 * np.log10(P.mean(0))
    b = (_FREQ >= 100) & (_FREQ <= 4000)
    f["spectral_tilt"] = np.polyfit(_FREQ[b] / 1000, L[b], 1)[0]
    f["hf_lf_ratio"] = 10 * np.log10(P.mean(0)[(_FREQ >= 2000) & (_FREQ <= 5000)].sum()
                                    / P.mean(0)[_FREQ <= 1000].sum())
    raw = np.lib.stride_tricks.sliding_window_view(pad, FRAME)[::HOP][active]
    f["zcr"] = (np.abs(np.diff(np.sign(raw), axis=1)) > 0).mean()

    mf = dct(np.log(P @ _FB.T + 1e-10), type=2, axis=1, norm="ortho")[:, 1:14]
    for i in range(13):
        f[f"mfcc{i + 1}_mean"], f[f"mfcc{i + 1}_std"] = mf[:, i].mean(), mf[:, i].std()

    # --- formants (vocal tract length -> gender and age cues)
    F = _formants(signal.lfilter([1, -0.97], [1], y), vt)
    f["formant_f1"], f["formant_f2"], f["formant_f3"] = F
    f["formant_dispersion"] = (F[2] - F[0]) / 2

    h, f["cpp_mean"], f["cpp_std"] = _voice_quality(y, f0_all, vt)
    f["harm_low"], f["harm_mid"], f["harm_high"] = h
    from scipy.signal import find_peaks
    smooth = uniform_filter1d(db, 5, mode="nearest")
    pk, _ = find_peaks(smooth, prominence=3.0, distance=8)
    f["syllable_peak_rate"] = len(pk) / (nf * HOP / SR)

    feats = np.nan_to_num(np.array([f[k] for k in FEATURE_NAMES], dtype=np.float64))
    return feats, {"y": y, "f0": f0, "hop_s": HOP / SR}


# =====================================================================================================
# 4. DATASET BUILDER (used by train.py)
# =====================================================================================================
def _build_one(job):
    gender, age, emo, seed = job
    try:
        return analyze_signal(synth_voice(gender, age, emo, seed))[0]
    except ValueError:
        return None


def build_dataset(n, seed=0, workers=1, progress=None):
    """Creates n labelled voices. Returns X, gender(0=male,1=female), age, emotion_index."""
    rng = np.random.default_rng(seed)
    gender = (rng.random(n) < 0.40).astype(int)                 # 60% male (the task is about male voices)
    age = rng.integers(18, 86, n)
    emo = rng.integers(0, len(EMOTIONS), n)
    jobs = [("female" if g else "male", int(a), EMOTIONS[e], seed * 1_000_003 + i)
            for i, (g, a, e) in enumerate(zip(gender, age, emo))]
    feats = []
    if workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, r in enumerate(ex.map(_build_one, jobs, chunksize=16)):
                    feats.append(r)
                    if progress and i % 100 == 0:
                        progress(f"Generating built-in voices... {i}/{n}")
        except Exception:                                   # multiprocessing not available -> one core
            feats = []
    if not feats:
        for i, j in enumerate(jobs):
            feats.append(_build_one(j))
            if progress and i % 100 == 0:
                progress(f"Generating built-in voices... {i}/{n}")
    keep = np.array([x is not None for x in feats])
    X = np.stack([x for x in feats if x is not None])
    return X, gender[keep], age[keep], emo[keep]


# =====================================================================================================
# 5. BUILT-IN DEMO SAMPLES (the "built-in input")
# =====================================================================================================
SAMPLES = [  # (label, gender, age, emotion, seed)
    ("Male, 24 yrs, Happy", "male", 24, "Happy", 9001),
    ("Male, 35 yrs, Neutral", "male", 35, "Neutral", 9002),
    ("Male, 52 yrs, Angry", "male", 52, "Angry", 9003),
    ("Male, 68 yrs, Sad", "male", 68, "Sad", 9004),
    ("Male, 74 yrs, Angry", "male", 74, "Angry", 9005),
    ("Male, 81 yrs, Happy", "male", 81, "Happy", 9006),
    ("Female, 30 yrs, Neutral", "female", 30, "Neutral", 9007),
    ("Female, 66 yrs, Sad", "female", 66, "Sad", 9008),
]


def ensure_samples(folder="samples"):
    """Writes the built-in sample voice notes as WAV files (so you can also test 'Upload')."""
    os.makedirs(folder, exist_ok=True)
    items = []
    for label, g, a, e, seed in SAMPLES:
        path = os.path.join(folder, f"{g}_{a}_{e.lower()}.wav")
        if not os.path.exists(path):
            save_wav(path, synth_voice(g, a, e, seed))
        items.append({"label": label, "path": path, "gender": g, "age": a, "emotion": e})
    return items


# =====================================================================================================
# 6. INFERENCE: THE TASK LOGIC
# =====================================================================================================
class VoiceAnalyzer:
    def __init__(self, path=MODEL_PATH):
        import joblib
        import sklearn
        self.bundle = joblib.load(path)
        if self.bundle.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("Model was trained with another scikit-learn version - retraining is needed.")
        self.emotions = self.bundle["emotions"]
        self.age_margin = int(round(self.bundle["metrics"].get("age_mae_male", 7)))

    def analyze(self, y, sr):
        y = to_mono16k(y, sr)
        feats, info = analyze_signal(y)
        X = feats.reshape(1, -1)
        p_female = float(self.bundle["gender"].predict_proba(X)[0][1])
        res = {"info": info, "p_female": p_female, "gender": "Female" if p_female > 0.5 else "Male",
               "gender_conf": max(p_female, 1 - p_female), "accepted": False, "age": None,
               "category": None, "emotion": None, "emotion_probs": None, "message": ""}

        if p_female > 0.5:                                          # RULE 1: only male voices
            res["message"] = "Upload male voice."
            return res

        res["accepted"] = True
        age = float(np.clip(self.bundle["age"].predict(X)[0], 18, 90))
        res["age"] = int(round(age))
        res["age_margin"] = self.age_margin
        if age > SENIOR_AGE:                                        # RULE 2: > 60 -> senior + emotion
            proba = self.bundle["emotion"].predict_proba(X)[0]
            res["category"] = "Senior Citizen"
            res["emotion_probs"] = {self.emotions[i]: float(p) for i, p in enumerate(proba)}
            res["emotion"] = self.emotions[int(np.argmax(proba))]
            res["message"] = f"Senior Citizen - estimated age {res['age']}, emotion: {res['emotion']}"
        else:                                                       # RULE 3: <= 60 -> age only
            res["category"] = "Below 60"
            res["message"] = f"Estimated age {res['age']} (below 60) - emotion not required"
        return res

    def analyze_file(self, path):
        y, sr = load_audio(path)
        return self.analyze(y, sr)
