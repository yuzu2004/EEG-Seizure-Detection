# src/preprocessing.py

import re
import cmath
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_widths, welch
from scipy.stats import kurtosis, skew
from tqdm import tqdm

ROOT = Path("data/chb-mit")

PATIENTS = [f"chb{i:02d}" for i in range(1, 7)]
OUTPUT_CSV = "data/seizure_event_candidates.csv"

WIN = 15.0
STEP = 5.0
SCORE_THRESHOLD = 15.0

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 7.5),
    "alpha": (8, 13),
    "beta": (13, 25),
    "gamma": (25, 45),
}

# ===== FILTER =====
def bandpass(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

def prefilter_all(sig, fs):
    out = {name: bandpass(sig, fs, lo, hi) for name, (lo, hi) in BANDS.items()}
    out["fast"] = bandpass(sig, fs, 20, 70)
    out["hf"] = bandpass(sig, fs, 40, 80)
    out["lf"] = bandpass(sig, fs, 1, 30)
    return out

# ===== FEATURES =====
def rms(x): return float(np.sqrt(np.mean(x*x)))
def line_length(x): return float(np.sum(np.abs(np.diff(x))))
def zero_crossing_rate(x): return int(((x[:-1]*x[1:])<0).sum())

def is_artifact_fast(seg, hf, lf):
    if np.max(np.abs(seg)) > 600:
        return True
    return rms(hf)/(rms(lf)+1e-6) > 0.6

def spectral_features(seg, fs):
    f, pxx = welch(seg, fs, nperseg=min(512, len(seg)))
    band_powers = {}
    for name, (lo, hi) in BANDS.items():
        idx = (f>=lo)&(f<=hi)
        band_powers[name] = float(np.trapezoid(pxx[idx], f[idx])) if np.any(idx) else 0.0

    psd = pxx/(np.sum(pxx)+1e-10)
    entropy = float(-np.sum(psd*np.log(psd+1e-10)))
    return band_powers, entropy

# ===== SCORE =====
def quantum_score(amps, spikes, fast_peaks, polyspike, spike_wave):
    delta_amp = amps["delta"]/50
    theta_amp = amps["theta"]/20

    spikes_amp = min(len(spikes)/10,1.0)
    fast_amp = min(len(fast_peaks)/10,1.0)

    c = (
        cmath.rect(delta_amp,0)
        + cmath.rect(theta_amp,cmath.pi/4)
        + cmath.rect(spikes_amp,cmath.pi/2)
        + cmath.rect(fast_amp,cmath.pi/3)
        + (1.0 if polyspike else 0.0)
        + (1.0 if spike_wave else 0.0)
    )
    return abs(c)**2

# ===== PROCESS CHANNEL =====
def process_channel(sig, fs, win_samp, step_samp):
    rows = []
    filt = prefilter_all(sig, fs)

    for i in range(0, len(sig)-win_samp+1, step_samp):
        seg = sig[i:i+win_samp]
        hf = filt["hf"][i:i+win_samp]
        lf = filt["lf"][i:i+win_samp]

        if is_artifact_fast(seg, hf, lf):
            continue

        energy = float(np.mean(seg*seg))
        variance = float(np.var(seg))
        skewness = float(skew(seg))
        kurt = float(kurtosis(seg))
        ll = line_length(seg)
        zcr = zero_crossing_rate(seg)

        amps = {k: float(np.std(filt[k][i:i+win_samp])) for k in BANDS}

        spikes,_ = find_peaks(np.abs(seg), height=80)
        fast_peaks,_ = find_peaks(np.abs(filt["fast"][i:i+win_samp]), height=30)

        poly = len(spikes)>=3
        spike_wave = amps["theta"]>12 and amps["beta"]<5

        score = quantum_score(amps, spikes, fast_peaks, poly, spike_wave)
        if score < SCORE_THRESHOLD:
            continue

        band_psd, ent = spectral_features(seg, fs)

        rows.append({
            "start": i/fs,
            "end": (i+win_samp)/fs,
            "score": score,
            "energy": energy,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurt,
            "line_length": ll,
            "zcr": zcr,
            **amps,
            **{f"{k}_psd": v for k,v in band_psd.items()},
            "spectral_entropy": ent,
            "spike_count": len(spikes),
            "fast_spike_count": len(fast_peaks),
            "polyspike": int(poly)
        })

    return rows
