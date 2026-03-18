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

# ================= CONFIG =================
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 7.5),
    "alpha": (8, 13),
    "beta": (13, 25),
    "gamma": (25, 45),
}

# ================= FILTER =================
def bandpass(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

def prefilter_all(sig, fs):
    out = {k: bandpass(sig, fs, *v) for k, v in BANDS.items()}
    out["hf"] = bandpass(sig, fs, 40, 80)
    out["lf"] = bandpass(sig, fs, 1, 30)
    return out

# ================= FEATURES =================
def rms(x): return np.sqrt(np.mean(x*x))
def line_length(x): return np.sum(np.abs(np.diff(x)))
def zero_crossing_rate(x): return ((x[:-1]*x[1:]) < 0).sum()

# ================= SCORE =================
def quantum_score(amps, spikes):
    return (amps["delta"] + amps["theta"] + len(spikes))**2

# ================= MAIN PROCESS =================
def process_channel(sig, fs, win, step):
    rows = []
    filt = prefilter_all(sig, fs)

    for i in range(0, len(sig)-win, step):
        seg = sig[i:i+win]

        amps = {k: np.std(v[i:i+win]) for k, v in filt.items() if k in BANDS}

        spikes, _ = find_peaks(np.abs(seg), height=80)

        score = quantum_score(amps, spikes)
        if score < 10:
            continue

        rows.append({
            "start": i/fs,
            "end": (i+win)/fs,
            "score": score,
            "delta": amps["delta"],
            "theta": amps["theta"],
            "spike_count": len(spikes)
        })

    return rows
