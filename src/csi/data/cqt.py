import hashlib
import importlib.metadata
import json

import numpy as np


def aggregate_frames(magnitude, size=20):
    if magnitude.ndim != 2 or size <= 0:
        raise ValueError("Expected a frequency/time array and positive aggregation size")
    frames = magnitude.shape[1] // size
    if not frames:
        raise ValueError("Audio yields fewer than one complete CQT aggregation window")
    return magnitude[:, :frames * size].reshape(magnitude.shape[0], frames, size).mean(axis=2).astype(np.float64)


def recipe(settings, audio_sha256):
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "librosa", "scipy", "soundfile", "soxr")}
    spec = {
        "settings": {k: v for k, v in settings.items() if k not in ("frames", "tonic", "augmentation")},
        "versions": versions, "audio_sha256": audio_sha256,
        "mono": True, "res_type": "soxr_hq", "tuning": 0.0,
        "audio_dtype": "float32", "cqt_dtype": "complex64", "offset": 0.0, "duration": None,
        "filter_scale": 1.0, "norm": 1, "sparsity": 0.01, "window": "hann",
        "scale": True, "pad_mode": "constant", "magnitude": True,
        "remainder": "drop", "aggregation_dtype": "float64",
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), spec


def extract(audio_path, settings):
    import librosa
    audio, sr = librosa.load(audio_path, sr=settings["sample_rate"], mono=True, res_type="soxr_hq", dtype=np.float32, offset=0.0, duration=None)
    if len(audio) < 1000:
        raise ValueError(f"Audio too short for historical extraction: {audio_path}")
    magnitude = np.abs(librosa.cqt(
        y=audio, sr=sr, hop_length=settings["hop_length"], fmin=settings["fmin"],
        n_bins=settings["bins"], bins_per_octave=settings["bins_per_octave"],
        tuning=0.0, filter_scale=1.0, norm=1, sparsity=0.01, window="hann",
        scale=True, pad_mode="constant", res_type="soxr_hq", dtype=np.complex64,
    ))
    return aggregate_frames(magnitude, settings["aggregate"])
