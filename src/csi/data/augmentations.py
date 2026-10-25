import numpy as np
from PIL import Image


def upstream_augment(cqt):
    """Pinned upstream's two masks and quantized PIL bilinear speed resize."""
    data = cqt.copy()
    for _ in range(2):
        width = np.random.randint(24)
        start = np.random.randint(84 - width)
        data[start:start + width] = 0
    data = data.T
    length = max(1, int(data.shape[0] * np.random.uniform(0.7, 1.3)))
    maximum = np.max(data) + 1
    image = Image.fromarray((data * 255.0 / maximum).astype(np.uint8))
    return (np.asarray(image.resize((84, length), Image.Resampling.BILINEAR)) / 255.0 * maximum).T


def crop_pad(cqt, frames, random_crop=False, minimum=200):
    if frames is not None:
        if cqt.shape[1] > frames:
            # Preserve the upstream exclusive upper-bound crop convention.
            start = np.random.randint(cqt.shape[1] - frames) if random_crop else 0
            cqt = cqt[:, start:start + frames]
        elif cqt.shape[1] < frames:
            cqt = np.pad(cqt, ((0, 0), (0, frames - cqt.shape[1])))
    if cqt.shape[1] < minimum:
        cqt = np.pad(cqt, ((0, 0), (0, minimum - cqt.shape[1])))
    return cqt


def preprocess(cqt, settings, profile, training=False, frames=None, tonic=None):
    if cqt.ndim != 2 or cqt.shape[0] != 84 or not cqt.shape[1] or not np.isfinite(cqt).all():
        raise ValueError("Expected finite, nonempty (84, time) CQT")
    if settings["tonic"]["enabled"]:
        from .tonic import rotate_cqt
        if tonic is None:
            raise ValueError("Tonic preprocessing requires a validated cache entry")
        cqt = rotate_cqt(cqt, tonic)
    if settings["augmentation"] and training:
        cqt = upstream_augment(cqt)
    cqt = cqt.astype(np.float32) / (np.max(np.abs(cqt)) + 1e-6)
    length = (frames if training else None) if profile == "upstream" else settings["frames"]
    return crop_pad(cqt, length, random_crop=training and settings["augmentation"])[None].astype(np.float32)
