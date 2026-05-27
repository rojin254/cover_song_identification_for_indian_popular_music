"""Tonic-relative CQT rotation.

The 84-bin CQT in this codebase is 12 bins/octave x 7 octaves. To put the
tonic at bin 0 of every octave, we roll *within each octave*, not globally:

    bin k  in octave o (= k // 12)  is mapped to
    bin ((k - tonic) % 12) + 12 * o

Equivalently, reshape (84, T) -> (7, 12, T), np.roll along axis=1 by -tonic,
reshape back to (84, T). This preserves octave structure (no leakage between
neighbouring octaves) and is ~milliseconds per track.
"""
from __future__ import annotations

import numpy as np

BINS_PER_OCTAVE = 12


def rotate_cqt(cqt: np.ndarray, tonic_offset: int) -> np.ndarray:
    """Rotate a CQT so the tonic sits at bin 0 of every octave.

    Args:
        cqt: shape (F, T) or (1, F, T). F should be a multiple of 12; if not,
            the trailing tail is left untouched.
        tonic_offset: integer pitch class in {0..11}. The CQT will be rolled
            by -tonic_offset within each octave.

    Returns:
        Same shape as input, dtype preserved.
    """
    if cqt is None:
        return cqt
    arr = np.asarray(cqt)
    had_channel = arr.ndim == 3 and arr.shape[0] == 1
    if had_channel:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'expected (F, T) CQT, got shape {arr.shape}')

    tonic_offset = int(tonic_offset) % BINS_PER_OCTAVE
    if tonic_offset == 0:
        return cqt  # no-op fast path

    F, T = arr.shape
    n_full = (F // BINS_PER_OCTAVE) * BINS_PER_OCTAVE
    head = arr[:n_full].reshape(-1, BINS_PER_OCTAVE, T)
    head = np.roll(head, shift=-tonic_offset, axis=1)
    rotated_head = head.reshape(n_full, T)

    if n_full == F:
        out = rotated_head
    else:
        out = np.concatenate([rotated_head, arr[n_full:]], axis=0)

    if had_channel:
        out = out[None, ...]
    return out.astype(cqt.dtype, copy=False)
