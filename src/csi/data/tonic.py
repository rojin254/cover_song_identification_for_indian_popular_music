import numpy as np

ESTIMATOR = "magnitude_pitch_class_mean_argmax_v1"


def estimate_tonic(cqt):
    if cqt.ndim != 2 or cqt.shape[0] != 84 or not cqt.shape[1]:
        raise ValueError("Tonic requires a nonempty (84, time) CQT")
    energy = np.abs(cqt).astype(np.float64).mean(axis=1).reshape(7, 12).sum(axis=0)
    return int(energy.argmax())


def rotate_cqt(cqt, tonic):
    if cqt.ndim != 2 or cqt.shape[0] != 84 or type(tonic) is not int or not 0 <= tonic < 12:
        raise ValueError("Expected (84, time) CQT and integer tonic in [0, 11]")
    return np.roll(cqt.reshape(7, 12, -1), -tonic, axis=1).reshape(cqt.shape)
