import torch
from torch.nn import functional as F

from ..mining.masks import first_occurrences


def supcon_loss(embeddings, labels, temperature=0.07, identities=None):
    if len(embeddings) < 2:
        return embeddings.sum() * 0, 0
    eligible = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    if identities is not None:
        eligible &= identities[:, None] != identities[None, :]
        unique = first_occurrences(identities)
        eligible &= unique[:, None] & unique[None, :]
    positive = (labels[:, None] == labels[None, :]) & eligible
    valid = positive.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0, 0
    z = F.normalize(embeddings, dim=1)
    logits = (z @ z.T) / temperature
    # Only valid rows participate, avoiding inf/nan in singleton denominators.
    logits, eligible, positive = logits[valid], eligible[valid], positive[valid]
    denominator = torch.logsumexp(logits.masked_fill(~eligible, float("-inf")), dim=1)
    log_probability = logits - denominator[:, None]
    loss = -(log_probability.masked_fill(~positive, 0).sum(dim=1) / positive.sum(dim=1)).mean()
    return loss, int(valid.sum())
