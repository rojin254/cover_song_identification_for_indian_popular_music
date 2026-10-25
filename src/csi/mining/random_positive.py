import torch


def random_positive(distances, mask):
    valid = mask.any(dim=1)
    weights = mask.float()
    weights[~valid, 0] = 1
    selected = torch.multinomial(weights, 1).squeeze(1)
    return distances.gather(1, selected[:, None]).squeeze(1), valid
