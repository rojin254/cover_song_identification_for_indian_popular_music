from .hard_positive import hard_positive
from .mean_positive import mean_positive


def hard_mean_positive(distances, mask):
    hard, valid = hard_positive(distances, mask)
    mean, _ = mean_positive(distances, mask)
    return hard + mean, valid
