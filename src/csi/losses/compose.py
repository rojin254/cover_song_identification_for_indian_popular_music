import torch
from torch import nn
from torch.nn import functional as F

from ..mining.hard_mean_positive import hard_mean_positive
from ..mining.hard_positive import hard_positive
from ..mining.mean_positive import mean_positive
from ..mining.masks import first_occurrences
from ..mining.random_positive import random_positive
from ..models.subcenter_arcface import SubCenterArcFace
from .modified_triplet import modified_from_distances
from .supcon import supcon_loss
from .triplet import triplet_loss

POSITIVE_MINERS = {"random": random_positive, "hard": hard_positive, "mean": mean_positive, "hard_mean": hard_mean_positive}


class Objectives(nn.Module):
    def __init__(self, config, classes):
        super().__init__()
        self.config = config
        settings = config["objectives"]["arcface"]
        self.arcface = SubCenterArcFace(
            classes, settings["subcenters"], settings["margin"], settings["scale"]
        ) if settings["weight"] else None

    def forward(self, logits, embeddings, labels, originals, identities, negatives=None):
        settings = self.config["objectives"]
        total = embeddings.sum() * 0
        logs = {}
        same = labels[:, None] == labels[None, :]
        distinct = identities[:, None] != identities[None, :]
        unique = first_occurrences(identities)
        positive_mask = same & distinct & unique[:, None] & unique[None, :]
        negative_mask = ~same & unique[None, :]
        anchors = originals.bool() if self.config["mining"]["anchors"] == "originals" else torch.ones_like(labels, dtype=torch.bool)
        metric_valid = positive_mask.any(dim=1) & anchors & unique
        if self.config["mining"]["skip_batch_without_triplet"] and not (metric_valid & negative_mask.any(dim=1)).any():
            return total, {"skipped": 1, "valid_objectives": 0}

        def add(name, loss, count):
            nonlocal total
            weight = settings[name]["weight"]
            total = total + weight * loss
            logs[name] = {"raw": float(loss.detach()), "weighted": float((loss * weight).detach()), "valid": count}

        if settings["ce"]["weight"]:
            add("ce", F.cross_entropy(logits, labels), len(labels))
        if settings["supcon"]["weight"]:
            loss, count = supcon_loss(embeddings, labels, settings["supcon"]["temperature"], identities)
            add("supcon", loss, count)
        if self.arcface is not None:
            add("arcface", F.cross_entropy(self.arcface(embeddings, labels), labels), len(labels))
        if settings["triplet"]["weight"]:
            valid = metric_valid & negative_mask.any(dim=1)
            if valid.any():
                pos = torch.multinomial(positive_mask[valid].float(), 1).squeeze(1)
                neg = torch.multinomial(negative_mask[valid].float(), 1).squeeze(1)
                params = {k: v for k, v in settings["triplet"].items() if k != "weight"}
                loss = triplet_loss(embeddings[valid], embeddings[pos], embeddings[neg], **params)
            else:
                loss = embeddings.sum() * 0
            add("triplet", loss, int(valid.sum()))
        if settings["modified_triplet"]["weight"]:
            if negatives is None:
                raise ValueError("Modified triplet requires training-partition negative embeddings")
            similar, random = negatives
            valid = metric_valid
            if valid.any():
                z = F.normalize(embeddings, dim=1)
                distances = 1 - z @ z.T
                strategy = settings["modified_triplet"]["positive"]
                positive, _ = POSITIVE_MINERS[strategy](distances, positive_mask)
                positive = positive if strategy == "hard_mean" else positive * 2
                loss = modified_from_distances(
                    positive[valid],
                    1 - F.cosine_similarity(embeddings[valid], similar[valid]),
                    1 - F.cosine_similarity(embeddings[valid], random[valid]),
                    settings["modified_triplet"]["margin"],
                )
            else:
                loss = embeddings.sum() * 0
            add("modified_triplet", loss, int(valid.sum()))
        logs["valid_objectives"] = sum(v["valid"] > 0 for v in logs.values())
        logs["skipped"] = int(logs["valid_objectives"] == 0)
        return total, logs
