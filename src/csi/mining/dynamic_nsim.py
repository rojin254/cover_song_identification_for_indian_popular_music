from .nsim import nearest_negative


def due(epoch, previous, interval):
    return previous is None or (interval is not None and epoch - previous >= interval)


class NegativeCache:
    def __init__(self):
        self.identities = None
        self.embeddings = None
        self.selection_epoch = None
        self.embedding_epoch = None

    def refresh(self, epoch, embeddings, labels, settings):
        events = []
        select = due(epoch, self.selection_epoch, settings["selection_refresh_epochs"])
        update = due(epoch, self.embedding_epoch, settings["embedding_refresh_epochs"])
        if select:
            self.identities = nearest_negative(embeddings, labels).cpu()
            self.selection_epoch = epoch
            events.append("negative_identities")
        if update:
            self.embeddings = embeddings.detach().cpu().clone()
            self.embedding_epoch = epoch
            events.append("negative_embeddings")
        return events

    def state_dict(self):
        return dict(vars(self))

    def load_state_dict(self, state):
        if set(state) != set(vars(self)):
            raise ValueError("Invalid negative cache state")
        self.__dict__.update(state)
