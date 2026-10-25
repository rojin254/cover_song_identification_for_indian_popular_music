from torch.nn import functional as F


def modified_from_distances(positive_term, similar_distance, random_distance, margin=0.3):
    return (margin + positive_term - similar_distance - random_distance).clamp_min(0).mean()


def modified_triplet_loss(anchor, positive, similar, random, margin=0.3):
    return modified_from_distances(
        2 * (1 - F.cosine_similarity(anchor, positive)),
        1 - F.cosine_similarity(anchor, similar),
        1 - F.cosine_similarity(anchor, random), margin,
    )
