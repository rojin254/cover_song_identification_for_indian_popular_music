import numpy as np
import torch


def retrieval_metrics(embeddings, labels, no_positive="zero", identities=None):
    if no_positive not in ("zero", "error"):
        raise ValueError("no_positive must be zero or error")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) != len(labels) or not len(labels):
        raise ValueError("Expected nonempty embeddings and matching labels")
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite retrieval embeddings")
    if identities is not None and len(set(identities)) != len(identities):
        raise ValueError("Duplicate evaluation track identities")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero-norm retrieval embedding")
    z = embeddings / norms
    labels = np.asarray(labels)
    ap, top10, first, rankings = [], [], [], []
    for index in range(len(labels)):
        similarity = z @ z[index]
        order = np.argsort(-similarity, kind="stable")
        order = order[order != index]
        relevant = labels[order] == labels[index]
        positions = np.flatnonzero(relevant) + 1
        if not len(positions) and no_positive == "error":
            raise ValueError(f"Query {index} has no relevant result")
        ap.append(float((np.arange(1, len(positions) + 1) / positions).mean()) if len(positions) else 0.0)
        top10.append(float(relevant[:10].sum()) / 10)
        first.append(int(positions[0]) if len(positions) else 0)
        rankings.append(order.tolist())
    return {
        "MAP": float(np.mean(ap)), "Top10": float(np.mean(top10)), "Rank1": float(np.mean(first)),
        "queries": len(labels), "no_positive_queries": sum(not (labels == label).sum() > 1 for label in labels),
        "per_query": [{"AP": a, "Top10": t, "Rank1": r} for a, t, r in zip(ap, top10, first)],
        "rankings": rankings,
    }


@torch.no_grad()
def embed(model, dataset, device, batch_size=1, workers=0):
    from torch.utils.data import DataLoader
    was_training = model.training
    model.eval()
    outputs = []
    try:
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers):
            _, features = model(batch["cqt"].to(device))
            outputs.append(features.cpu())
    finally:
        model.train(was_training)
    if not outputs:
        raise ValueError("Cannot embed an empty dataset")
    return torch.cat(outputs)
