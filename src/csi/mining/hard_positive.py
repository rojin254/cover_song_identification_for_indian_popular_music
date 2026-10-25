def hard_positive(distances, mask):
    valid = mask.any(dim=1)
    values = distances.masked_fill(~mask, float("-inf")).max(dim=1).values
    return values.masked_fill(~valid, 0), valid
