import math

import torch
from torch import nn
from torch.nn import functional as F


class SubCenterArcFace(nn.Module):
    def __init__(self, classes, subcenters=3, margin=0.3, scale=30.0, dimensions=300):
        super().__init__()
        self.classes, self.subcenters = classes, subcenters
        self.margin, self.scale = margin, scale
        self.weight = nn.Parameter(torch.empty(classes * subcenters, dimensions))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.normalize(embeddings, dim=1) @ F.normalize(self.weight, dim=1).T
        cosine = cosine.view(-1, self.classes, self.subcenters).max(dim=2).values
        sine = (1 - cosine.square()).clamp_min(1e-12).sqrt()
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        phi = torch.where(
            cosine > math.cos(math.pi - self.margin), phi,
            cosine - math.sin(math.pi - self.margin) * self.margin,
        )
        mask = F.one_hot(labels, self.classes).bool()
        return torch.where(mask, phi, cosine) * self.scale
