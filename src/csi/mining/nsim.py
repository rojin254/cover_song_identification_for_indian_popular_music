import torch
from torch.nn import functional as F


def nearest_negative(embeddings, labels, chunk_size=256):
    normalized = F.normalize(embeddings, dim=1)
    selections = []
    for start in range(0, len(labels), chunk_size):
        same = labels[start:start + chunk_size, None] == labels[None, :]
        if same.all(dim=1).any():
            raise ValueError("Nearest negatives require at least two training cliques")
        similarity = normalized[start:start + chunk_size] @ normalized.T
        selections.append(similarity.masked_fill(same, float("-inf")).argmax(dim=1))
    return torch.cat(selections)
