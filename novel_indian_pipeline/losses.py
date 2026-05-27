"""Loss components for the novel Indian CSI pipeline.

Three losses, designed to be combined:

  - modified_triplet_loss        (existing; copied verbatim from the
                                  baseline indian pipeline)
  - supcon_loss                  (NEW: Khosla et al., NeurIPS 2020)
  - SubCenterArcFaceHead         (NEW: Deng et al., ECCV 2020)

The dynamic-N_sim refresh used to feed `modified_triplet_loss` lives in
main.py (recompute_nsim_from_model + create_modified_triplets), copied as-is
from the baseline pipeline.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Modified triplet loss (verbatim from indian_new_pipeline/main.py:83-93).
# ---------------------------------------------------------------------------
def modified_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    neg_rand: torch.Tensor,
    neg_sim: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """L = max(0, margin + 2*d(A,P) - [d(A, N_sim) + d(A, N_rand)])  with d = 1 - cos.

    Same definition as the existing `modified_triplet_loss` in
    `indian_new_pipeline/main.py` so row-1 reproduction is exact.
    """
    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    d_rand = 1 - F.cosine_similarity(anchor, neg_rand, dim=-1)
    d_sim = 1 - F.cosine_similarity(anchor, neg_sim, dim=-1)
    loss = torch.clamp(margin + 2 * d_pos - (d_sim + d_rand), min=0.0)
    return loss.mean()


# ---------------------------------------------------------------------------
# 2. Supervised contrastive loss (Khosla et al., 2020).
# ---------------------------------------------------------------------------
def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """SupCon over L2-normalised embeddings.

    For each anchor i, all j != i with labels[j] == labels[i] are positives;
    all other j != i are negatives. Anchors without any positive are skipped.

    Args:
        embeddings: (N, D), un-normalised.
        labels:     (N,)   int64 clique ids.
        temperature: scalar.

    Returns:
        Scalar loss. Returns 0.0 (with grad) if no anchor has a positive.
    """
    if embeddings.numel() == 0:
        return embeddings.sum() * 0.0

    z = F.normalize(embeddings, p=2, dim=1)
    logits = (z @ z.t()) / temperature                    # (N, N)

    # Numerical stability: subtract per-row max from off-diagonal logits.
    logits_max, _ = logits.max(dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    n = z.size(0)
    device = z.device
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = pos_mask & ~self_mask                      # exclude self

    # log(sum_{a != i} exp(logits[i, a]))
    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    log_prob = logits - log_denom                         # (N, N)

    pos_count = pos_mask.float().sum(dim=1)               # (N,)
    valid = pos_count > 0
    if not valid.any():
        return embeddings.sum() * 0.0

    mean_log_prob_pos = (
        (pos_mask.float() * log_prob).sum(dim=1)[valid] / pos_count[valid]
    )
    return -mean_log_prob_pos.mean()


# ---------------------------------------------------------------------------
# 3. SubCenter-ArcFace head (Deng et al., ECCV 2020).
# ---------------------------------------------------------------------------
class SubCenterArcFaceHead(nn.Module):
    """Classification head with K sub-centers per class and additive angular margin.

    Forward returns margin-adjusted logits compatible with nn.CrossEntropyLoss.

    Args:
        num_classes: number of cliques.
        embed_dim:   embedding dimensionality (300 for CQTNet here).
        K:           sub-centers per class.
        margin:      additive angular margin m, in radians (default 0.3 rad).
        scale:       feature scale s (default 30).
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 300,
        K: int = 3,
        margin: float = 0.3,
        scale: float = 30.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.K = K
        self.margin = float(margin)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(num_classes * K, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        self._th = math.cos(math.pi - self.margin)
        self._mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Cosine similarity to every sub-center.
        z = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cos_all = z @ w.t()                               # (N, C*K)

        # Reduce K -> 1 by max over sub-centers.
        cos = cos_all.view(-1, self.num_classes, self.K).max(dim=2).values  # (N, C)

        # Apply additive angular margin to the target class.
        sin = torch.sqrt(torch.clamp(1.0 - cos.pow(2), min=0.0))
        cos_m = cos * self._cos_m - sin * self._sin_m
        # Numerical floor: when cos < th, use cos - mm instead of cos_m.
        cos_m = torch.where(cos > self._th, cos_m, cos - self._mm)

        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * cos_m + (1 - one_hot) * cos) * self.scale
        return logits
