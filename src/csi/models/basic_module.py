from torch import nn


class BasicModule(nn.Module):
    """Runtime base; legacy filesystem save/load helpers live in checkpoints."""

    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))
