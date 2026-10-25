import math
import random

from torch.utils.data import Sampler


class BalancedSampler(Sampler):
    def __init__(self, labels, originals, settings, seed=42):
        self.settings, self.seed, self.epoch = settings, seed, 0
        self.groups = {}
        self.originals = originals
        for index, label in enumerate(labels):
            self.groups.setdefault(label, []).append(index)
        p, k = settings["cliques_per_batch"], settings["tracks_per_clique"]
        if len(self.groups) < p:
            raise ValueError(f"Balanced sampler needs {p} distinct cliques; found {len(self.groups)}")
        if settings["tail"] == "error" and len(self.groups) % p:
            raise ValueError("Clique count is not divisible by cliques_per_batch")
        if settings["shortage"] == "error" and any(len(v) < k for v in self.groups.values()):
            raise ValueError("Clique has fewer than tracks_per_clique tracks")
        if settings["include_original"] and any(not any(originals[i] for i in v) for v in self.groups.values()):
            raise ValueError("Original inclusion requested but a clique has no original")

    def __len__(self):
        return len(self.groups) // self.settings["cliques_per_batch"]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        cliques = sorted(self.groups)
        rng.shuffle(cliques)
        p, k = self.settings["cliques_per_batch"], self.settings["tracks_per_clique"]
        for start in range(0, len(self) * p, p):
            batch = []
            for clique in cliques[start:start + p]:
                pool = list(self.groups[clique])
                rng.shuffle(pool)
                selected = []
                if self.settings["include_original"]:
                    original = rng.choice([i for i in pool if self.originals[i]])
                    selected.append(original)
                    pool.remove(original)
                selected.extend(pool[:k - len(selected)])
                while len(selected) < k:
                    selected.append(rng.choice(self.groups[clique]))
                batch.extend(selected)
            yield batch

    def state_dict(self):
        return {"epoch": self.epoch, "seed": self.seed}

    def load_state_dict(self, state):
        if state["seed"] != self.seed:
            raise ValueError("Sampler seed mismatch")
        self.epoch = state["epoch"]
