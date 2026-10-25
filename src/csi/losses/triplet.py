import torch
from torch.nn import functional as F


def triplet_loss(anchor, positive, negative, margin=0.3, distance="euclidean", normalize=False):
    if normalize:
        anchor, positive, negative = (F.normalize(x, dim=-1) for x in (anchor, positive, negative))
    if distance == "euclidean":
        return F.triplet_margin_loss(anchor, positive, negative, margin=margin)
    if distance == "cosine":
        return (margin - F.cosine_similarity(anchor, positive) + F.cosine_similarity(anchor, negative)).clamp_min(0).mean()
    raise ValueError(f"Unknown triplet distance: {distance}")
