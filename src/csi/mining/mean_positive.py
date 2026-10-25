def mean_positive(distances, mask):
    counts = mask.sum(dim=1)
    return distances.masked_fill(~mask, 0).sum(dim=1) / counts.clamp_min(1), counts > 0
