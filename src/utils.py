import numpy as np

def compute_metrics(y_true, probs, th=0.5):
    preds = (probs >= th).astype(int)

    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()

    sens = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)

    return sens, precision
