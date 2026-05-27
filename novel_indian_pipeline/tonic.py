"""Tonic estimation from a 84-bin CQT.

The CQT used in this codebase has 12 bins per octave x 7 octaves = 84 bins.
Bin index k maps to pitch class (k % 12). Tonic estimation reduces to:

    1. Take the magnitude of the CQT and average across time (T axis).
    2. Fold to one octave: pitch_class[c] = sum_k cqt_mean[k] for k % 12 == c.
    3. The tonic is argmax_c pitch_class[c].

This is robust on popular vocals because the tonic dominates the time-averaged
energy. For monophonic separated vocals, an F0-based variant is exposed
behind a flag for ablation.
"""
from __future__ import annotations

import numpy as np

BINS_PER_OCTAVE = 12


def fold_to_pitch_classes(cqt_mean: np.ndarray) -> np.ndarray:
    """Fold a per-bin energy vector of length F (F = 12*octaves) to length 12.

    Bins outside a clean multiple of 12 are silently ignored; for our 84-bin
    CQT the mapping is exact.
    """
    if cqt_mean.ndim != 1:
        raise ValueError(f'expected 1-D vector, got shape {cqt_mean.shape}')
    n_full = (cqt_mean.shape[0] // BINS_PER_OCTAVE) * BINS_PER_OCTAVE
    folded = cqt_mean[:n_full].reshape(-1, BINS_PER_OCTAVE).sum(axis=0)
    return folded


def estimate_tonic_from_cqt(cqt: np.ndarray) -> int:
    """Return the tonic pitch class in {0..11}.

    Args:
        cqt: numpy array of shape (F, T) or (1, F, T). Magnitude is taken
            with abs() so signed/normalised inputs are accepted.
    """
    arr = np.asarray(cqt)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'expected (F, T) CQT, got shape {arr.shape}')

    mag = np.abs(arr).astype(np.float64)
    cqt_mean = mag.mean(axis=1)
    pc = fold_to_pitch_classes(cqt_mean)
    if pc.sum() < 1e-12:
        return 0
    return int(np.argmax(pc))


def tonic_confidence(cqt: np.ndarray) -> float:
    """Ratio of the dominant pitch class's energy to the mean across all 12.

    A flat distribution returns ~1.0 (no clear tonic); a sharp one returns
    much higher. Useful for downstream gating.
    """
    arr = np.asarray(cqt)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    mag = np.abs(arr).astype(np.float64)
    cqt_mean = mag.mean(axis=1)
    pc = fold_to_pitch_classes(cqt_mean)
    if pc.sum() < 1e-12:
        return 0.0
    return float(pc.max() / (pc.mean() + 1e-12))
