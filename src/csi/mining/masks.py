import torch


def first_occurrences(identities):
    same_identity = identities[:, None] == identities[None, :]
    earlier = torch.tril(same_identity, diagonal=-1)
    return ~earlier.any(dim=1)
