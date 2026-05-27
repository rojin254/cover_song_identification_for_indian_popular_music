import os
import time

import torch


class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        prefix = 'check_points/' + self.model_name + (name or '') + '/'
        os.makedirs(prefix, exist_ok=True)
        path = time.strftime(prefix + '%m%d_%H%M%S.pth')
        torch.save(self.state_dict(), path)
        torch.save(self.state_dict(), prefix + 'latest.pth')
        return path
